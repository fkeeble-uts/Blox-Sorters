import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
from pathlib import Path
from collections import deque
import subprocess
import sys

# --- NEW FUNCTION TO DRAW CAMERA ORIGIN ---
def draw_camera_origin(image):
    """Draws a crosshair at the center of the image (the camera's (0,0,0) origin)."""
    H, W, _ = image.shape
    center_x = W // 2
    center_y = H // 2
    
    CENTER_COLOR = (255, 0, 255)  # Magenta for high visibility
    LINE_LENGTH = 30 # Length of the axis lines in pixels
    THICKNESS = 1

    # Draw the exact center (origin)
    cv2.circle(image, (center_x, center_y), 5, CENTER_COLOR, -1)
    cv2.circle(image, (center_x-100, center_y-100), 5, CENTER_COLOR, -1)
    cv2.circle(image, (center_x+100, center_y-100), 5, CENTER_COLOR, -1)
    cv2.circle(image, (center_x-100, center_y+100), 5, CENTER_COLOR, -1)
    cv2.circle(image, (center_x+100, center_y+100), 5, CENTER_COLOR, -1)

    # Draw the X-axis (Right is Positive X_C)
    cv2.line(image, 
             (center_x - LINE_LENGTH, center_y), 
             (center_x + LINE_LENGTH, center_y), 
             CENTER_COLOR, THICKNESS)

    # Draw the Y-axis (Down is Positive Y_C)
    cv2.line(image, 
             (center_x, center_y - LINE_LENGTH), 
             (center_x, center_y + LINE_LENGTH), 
             CENTER_COLOR, THICKNESS)
    
    # Label the axes
    cv2.putText(image, "X_C", (center_x + LINE_LENGTH + 5, center_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)
    cv2.putText(image, "Y_C", (center_x + 5, center_y + LINE_LENGTH + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)


class BlockDetector:
    """Handles block detection and pose estimation with temporal stabilization"""
    
    def __init__(self, config):
        self.config = config
        self.shape_tolerance = 0.04
        
        # Temporal stabilization - track blocks over multiple frames
        self.detection_history = deque(maxlen=10)  # Keep last 10 frames
        
        # Color ranges (HSV) - tune these for your lighting
        self.color_ranges = {
            "red": [([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [179, 255, 255])],
            "green": [([35, 70, 50], [85, 255, 255])],
            "blue": [([100, 70, 50], [130, 255, 255])],
            "yellow": [([20, 70, 50], [35, 255, 255])],
        }
    
    def classify_color(self, bgr_color):
        """Classify color from BGR pixel"""
        bgr = np.uint8([[bgr_color]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        H, S, V = hsv[0], hsv[1], hsv[2]
        
        if S < 50 or V < 50:
            return "gray"
        
        for color_name, ranges in self.color_ranges.items():
            for (lower, upper) in ranges:
                if (lower[0] <= H <= upper[0] and 
                    lower[1] <= S <= upper[1] and 
                    lower[2] <= V <= upper[2]):
                    return color_name
        return "unknown"
    
    def detect_blocks_single_frame(self, depth_image, color_image, depth_frame, depth_scale):
        """Detect blocks in a single frame (internal method)"""
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Get calibration parameters
        depth_range = self.config["depth_range_mm"]
        min_units = int(depth_range[0] / 1000 / depth_scale)
        max_units = int(depth_range[1] / 1000 / depth_scale)
        
        # Create depth mask
        depth_mask = cv2.inRange(depth_image, min_units, max_units)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        depth_mask = cv2.erode(depth_mask, kernel, iterations=2)
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filters
            if self.config["min_area"] < area < self.config["max_area"]:
                # Shape fitting
                perimeter = cv2.arcLength(contour, True)
                epsilon = self.shape_tolerance * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Rectangularity check
                rect = cv2.minAreaRect(contour)
                (_, (w, h), angle) = rect
                rect_area = w * h
                area_ratio = area / rect_area if rect_area > 0 else 0
                
                # Validation: 4 corners, convex, rectangular
                if (len(approx) == 4 and 
                    cv2.isContourConvex(approx) and 
                    area_ratio >= self.config["min_rectangularity"]):
                    
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # Normalize angle (width is longest side)
                        if w < h:
                            angle += 90
                        
                        # Get 3D position in camera frame
                        depth_val = depth_image[cY, cX]
                        if depth_val > 0:
                            z_m = depth_val * depth_scale
                            
                            # Project 2D pixel to 3D point (in meters)
                            point_3d = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [cX, cY], z_m
                            )
                            
                            # Convert to millimeters
                            x_mm = point_3d[0] * 1000
                            y_mm = point_3d[1] * 1000
                            z_mm = z_m * 1000
                            
                            # Classify color
                            color_bgr = color_image[cY, cX]
                            color = self.classify_color(color_bgr)
                            
                            # Calculate height above table if calibrated
                            height_above_table = None
                            if self.config.get("table_height_mm") is not None:
                                height_above_table = round(
                                    self.config["table_height_mm"] - z_mm, 2
                                )
                            
                            blocks.append({
                                'x_mm': x_mm,
                                'y_mm': y_mm,
                                'z_mm': z_mm,
                                'height_above_table_mm': height_above_table,
                                'rotation_angle_deg': angle,
                                'color': color,
                                'confidence': area_ratio,
                                'pixel_x': cX,
                                'pixel_y': cY
                            })
        
        return blocks
    
    def stabilize_detections(self, current_blocks):
        """Average detections over multiple frames to reduce jitter"""
        self.detection_history.append(current_blocks)
        
        if len(self.detection_history) < 5:
            # Not enough history yet
            return current_blocks
        
        # Group nearby blocks across frames (within 20mm)
        stable_blocks = []
        
        # Start with most recent frame
        for block in current_blocks:
            matching_blocks = [block]
            
            # Find matching blocks in previous frames
            for past_frame in list(self.detection_history)[:-1]:
                for past_block in past_frame:
                    dist = np.sqrt((block['x_mm'] - past_block['x_mm'])**2 + 
                                  (block['y_mm'] - past_block['y_mm'])**2)
                    if dist < 20:  # Same block if within 20mm
                        matching_blocks.append(past_block)
            
            # Need at least 3 detections to be considered stable
            if len(matching_blocks) >= 3:
                # Average the measurements
                avg_block = {
                    'x_mm': np.mean([b['x_mm'] for b in matching_blocks]),
                    'y_mm': np.mean([b['y_mm'] for b in matching_blocks]),
                    'z_mm': np.mean([b['z_mm'] for b in matching_blocks]),
                    'rotation_angle_deg': np.mean([b['rotation_angle_deg'] for b in matching_blocks]),
                    'color': block['color'],  # Use most recent color
                    'confidence': np.mean([b['confidence'] for b in matching_blocks]),
                    'pixel_x': block['pixel_x'],
                    'pixel_y': block['pixel_y'],
                    'height_above_table_mm': block['height_above_table_mm']
                }
                
                # Check if already added (avoid duplicates)
                is_duplicate = False
                for existing in stable_blocks:
                    dist = np.sqrt((avg_block['x_mm'] - existing['x_mm'])**2 + 
                                  (avg_block['y_mm'] - existing['y_mm'])**2)
                    if dist < 15:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    stable_blocks.append(avg_block)
        
        return stable_blocks
    
    def detect_blocks(self, depth_image, color_image, depth_frame, depth_scale, stabilize=True):
        """Detect blocks with optional stabilization"""
        current_blocks = self.detect_blocks_single_frame(
            depth_image, color_image, depth_frame, depth_scale
        )
        
        if stabilize:
            blocks = self.stabilize_detections(current_blocks)
        else:
            blocks = current_blocks
        
        # Add block IDs and round values
        for i, block in enumerate(blocks):
            block['block_id'] = i + 1
            block['x_mm'] = round(block['x_mm'], 2)
            block['y_mm'] = round(block['y_mm'], 2)
            block['z_mm'] = round(block['z_mm'], 2)
            block['rotation_angle_deg'] = round(block['rotation_angle_deg'], 2)
            block['confidence'] = round(block['confidence'], 3)
        
        # Create visualization
        debug_image = color_image.copy()
        for block in blocks:
            cX, cY = block['pixel_x'], block['pixel_y']
            
            # Draw circle
            cv2.circle(debug_image, (cX, cY), 5, (0, 255, 0), -1)
            
            # Draw cross for center
            cv2.line(debug_image, (cX-10, cY), (cX+10, cY), (0, 255, 0), 2)
            cv2.line(debug_image, (cX, cY-10), (cX, cY+10), (0, 255, 0), 2)
            
            # Label
            text = f"#{block['block_id']}: {block['color']}"
            cv2.putText(debug_image, text, (cX - 40, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            coord_text = f"({block['x_mm']:.0f}, {block['y_mm']:.0f})"
            cv2.putText(debug_image, coord_text, (cX - 35, cY + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return blocks, debug_image


def load_calibration():
    """Load calibration from file"""
    if Path('camera_calibration.json').exists():
        with open('camera_calibration.json', 'r') as f:
            config = json.load(f)
            if config.get('validated', False):
                return config
    return None


def capture_snapshot(pipeline, align, depth_scale, detector, num_samples=30):
    """Capture a stable snapshot by averaging multiple frames"""
    print("\nCapturing stable snapshot...")
    print("Please ensure:")
    print("  ✓ Robot arm is NOT in camera view")
    print("  ✓ All blocks are stationary")
    print("  ✓ Lighting is consistent")
    print("\nSampling frames", end="", flush=True)
    
    # Clear history
    detector.detection_history.clear()
    
    # Collect samples
    for i in range(num_samples):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if depth_frame and color_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect without stabilization for history building
            detector.detect_blocks_single_frame(
                depth_image, color_image, depth_frame, depth_scale
            )
        
        if i % 5 == 0:
            print(".", end="", flush=True)
        time.sleep(0.05)  # 50ms between samples
    
    print(" Done!")
    
    # Final detection with full stabilization
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    blocks, debug_img = detector.detect_blocks(
        depth_image, color_image, depth_frame, depth_scale, stabilize=True
    )

    # ADD ORIGIN MARKER TO SNAPSHOT RESULT
    draw_camera_origin(debug_img)
    
    return blocks, debug_img


def run_diagnostic_calibration():
    """Launch the diagnostic calibration tool"""
    print("\n" + "="*70)
    print("LAUNCHING DIAGNOSTIC CALIBRATION TOOL")
    print("="*70)
    print("\nThis will open a multi-view window showing detection stages.")
    print("Adjust parameters until blocks are detected correctly.")
    print("Press SPACE to save parameters, Q to quit.")
    print("\nStarting in 2 seconds...")
    time.sleep(2)
    
    # Create diagnostic script (MODIFIED TO INCLUDE DRAW_CAMERA_ORIGIN)
    diagnostic_code = '''import pyrealsense2 as rs
import numpy as np
import cv2
import json

def draw_camera_origin(image):
    """Draws a crosshair at the center of the image (the camera's (0,0,0) origin)."""
    H, W, _ = image.shape
    center_x = W // 2
    center_y = H // 2
    
    CENTER_COLOR = (255, 0, 255)  # Magenta for high visibility
    LINE_LENGTH = 30 # Length of the axis lines in pixels
    THICKNESS = 2

    # Draw the exact center (origin)
    cv2.circle(image, (center_x, center_y), 5, CENTER_COLOR, -1)

    # Draw the X-axis (Right is Positive X_C)
    cv2.line(image, 
             (center_x - LINE_LENGTH, center_y), 
             (center_x + LINE_LENGTH, center_y), 
             CENTER_COLOR, THICKNESS)

    # Draw the Y-axis (Down is Positive Y_C)
    cv2.line(image, 
             (center_x, center_y - LINE_LENGTH), 
             (center_x, center_y + LINE_LENGTH), 
             CENTER_COLOR, THICKNESS)
    
    # Label the axes
    cv2.putText(image, "X_C", (center_x + LINE_LENGTH + 5, center_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)
    cv2.putText(image, "Y_C", (center_x + 5, center_y + LINE_LENGTH + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    
    min_depth_mm = 250
    max_depth_mm = 273
    min_area = 4000
    max_area = 6000
    min_rect = 0.5
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            min_units = int(min_depth_mm / 1000 / depth_scale)
            max_units = int(max_depth_mm / 1000 / depth_scale)
            
            depth_mask = cv2.inRange(depth_image, min_units, max_units)
            depth_mask_display = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
            
            kernel = np.ones((5, 5), np.uint8)
            depth_mask_clean = cv2.erode(depth_mask, kernel, iterations=2)
            depth_mask_clean = cv2.dilate(depth_mask_clean, kernel, iterations=2)
            depth_mask_clean_display = cv2.cvtColor(depth_mask_clean, cv2.COLOR_GRAY2BGR)
            
            contours, _ = cv2.findContours(depth_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour_debug = color_image.copy()
            final_debug = color_image.copy()
            detected_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                cv2.drawContours(contour_debug, [contour], -1, (255, 0, 255), 2)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(contour_debug, f"A:{int(area)}", (cX-30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                if min_area < area < max_area:
                    cv2.drawContours(contour_debug, [contour], -1, (0, 255, 255), 2)
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.04 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    rect = cv2.minAreaRect(contour)
                    (_, (w, h), angle) = rect
                    rect_area = w * h
                    area_ratio = area / rect_area if rect_area > 0 else 0
                    cv2.drawContours(contour_debug, [approx], -1, (255, 255, 0), 2)
                    corners = len(approx)
                    is_convex = cv2.isContourConvex(approx)
                    
                    if M["m00"] != 0:
                        cv2.putText(contour_debug, f"C:{corners} R:{area_ratio:.2f}", (cX-40, cY+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                    
                    if corners == 4 and is_convex and area_ratio >= min_rect:
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        cv2.drawContours(final_debug, [box], 0, (0, 255, 0), 3)
                        cv2.circle(final_debug, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.putText(final_debug, f"BLOCK {detected_count+1}", (cX-40, cY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_count += 1
            
            # --- Draw the origin marker on the final view ---
            draw_camera_origin(final_debug)
            
            info_bg = np.zeros((100, 1920, 3), dtype=np.uint8)
            cv2.putText(info_bg, f"Depth:[{min_depth_mm},{max_depth_mm}]mm Area:[{min_area},{max_area}] Rect:{min_rect:.2f} | Detected:{detected_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(color_image, "1.COLOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_mask_display, "2.DEPTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_mask_clean_display, "3.MORPH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(contour_debug, "4.CONTOURS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(final_debug, "5.FINAL (Origin: Magenta)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            top_row = np.hstack([color_image, depth_mask_display, depth_mask_clean_display])
            bottom_row = np.hstack([contour_debug, final_debug, np.zeros_like(color_image)])
            display = np.vstack([info_bg, top_row, bottom_row])
            
            cv2.namedWindow('Diagnostic', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Diagnostic', 1920, 800)
            cv2.imshow('Diagnostic', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'): min_depth_mm += 1
            elif key == ord('s'): min_depth_mm -= 1
            elif key == ord('a'): max_depth_mm -= 1
            elif key == ord('d'): max_depth_mm += 1
            elif key == ord('e'): min_area += 10
            elif key == ord('r'): min_area = max(10, min_area - 10)
            elif key == ord('t'): max_area -= 1000
            elif key == ord('g'): max_area += 1000
            elif key == ord('y'): min_rect = min(1.0, min_rect + 0.05)
            elif key == ord('h'): min_rect = max(0.1, min_rect - 0.05)
            elif key == ord(' '):
                params = {"depth_range_mm": [min_depth_mm, max_depth_mm], "min_area": min_area, "max_area": max_area, "min_rectangularity": round(min_rect, 2), "validated": True}
                with open('camera_calibration.json', 'w') as f:
                    json.dump(params, f, indent=4)
                print("\\n✓ Saved!")
            elif key == ord('q'): break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''
    
    with open('diagnostic_calibration.py', 'w', encoding='utf-8') as f:
        f.write(diagnostic_code)
    
    subprocess.call([sys.executable, 'diagnostic_calibration.py'])


def main():
    print("\n" + "="*70)
    print("LEGO BLOCK DETECTION SYSTEM - ROBOT OPERATION MODE")
    print("="*70)
    
    # Try to load existing calibration
    config = load_calibration()
    
    if config is None:
        print("\n⚠ No valid calibration found!")
        input("\nPress ENTER to start calibration...")
        run_diagnostic_calibration()
        config = load_calibration()
        if config is None:
            print("\n❌ Calibration not completed. Exiting.")
            return
    
    print("\n✓ Calibration loaded:")
    print(f"  Depth range: {config['depth_range_mm']} mm")
    print(f"  Area range: [{config['min_area']}, {config['max_area']}]")
    print(f"  Min rectangularity: {config['min_rectangularity']}")
    
    # Initialize detector
    detector = BlockDetector(config)
    
    # Setup RealSense
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    align = rs.align(rs.stream.color)
    
    print("\nStarting camera...")
    try:
        profile = pipeline.start(rs_config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"✓ Camera ready")
    except RuntimeError as e:
        print(f"\n❌ ERROR starting camera: {e}")
        print("Camera may be in use by another process. Exiting.")
        return # Exit if camera fails to start.
    
    print("\n" + "="*70)
    print("OPERATION MODES:")
    print("="*70)
    print("  1. LIVE PREVIEW - See detections in real-time (with stabilization)")
    print("  2. SNAPSHOT MODE - Capture one stable scan for robot")
    print("  3. RECALIBRATE - Adjust detection parameters")
    print("  4. QUIT")
    print("="*70)
    
    mode = input("\nSelect mode (1/2/3/4): ").strip()
    
    try:
        if mode == '1':
            # Live preview mode
            print("\n" + "="*70)
            print("LIVE PREVIEW MODE (Stabilized)")
            print("  Press 'Q' to return to menu")
            print("="*70)
            
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                blocks, debug_img = detector.detect_blocks(
                    depth_image, color_image, depth_frame, depth_scale, stabilize=True
                )
                
                # --- ADD ORIGIN MARKER TO LIVE PREVIEW ---
                draw_camera_origin(debug_img)
                
                cv2.putText(debug_img, f"Stable blocks: {len(blocks)} (Q=Quit) | Origin is Magenta", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Live Preview', debug_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        elif mode == '2':
            # Snapshot mode - for robot operation
            print("\n" + "="*70)
            print("SNAPSHOT MODE - Robot Operation Workflow")
            print("="*70)
            print("\nRECOMMENDED WORKFLOW:")
            print("  1. Ensure robot arm is NOT blocking camera view")
            print("  2. Capture snapshot → saves blocks.json")
            print("  3. Robot reads blocks.json and picks first block")
            print("  4. Repeat from step 1 for remaining blocks")
            print("  5. Press Ctrl+C to exit")
            print("="*70)
            
            while True:
                input("\nPress ENTER to capture snapshot (or Ctrl+C to quit)...")
                
                blocks, debug_img = capture_snapshot(
                    pipeline, align, depth_scale, detector
                )
                
                if blocks:
                    # Save to JSON
                    output = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'num_blocks': len(blocks),
                        'camera_frame': 'RealSense D435',
                        'units': 'millimeters',
                        'coordinate_system': {
                            'x': 'right (from camera view)',
                            'y': 'down (from camera view)',
                            'z': 'distance from camera lens'
                        },
                        'workflow_note': 'Pick blocks sequentially. Re-scan after each pick.',
                        'blocks': blocks
                    }
                    
                    with open('blocks.json', 'w') as f:
                        json.dump(output, f, indent=4)
                    
                    print(f"\n{'='*70}")
                    print(f"✓ SAVED {len(blocks)} blocks to blocks.json")
                    print(f"{'='*70}")
                    for block in blocks:
                        print(f"  Block {block['block_id']}: {block['color']:7s} at "
                              f"X:{block['x_mm']:7.1f} Y:{block['y_mm']:7.1f} Z:{block['z_mm']:6.1f} mm, "
                              f"Angle:{block['rotation_angle_deg']:6.1f}°")
                    print(f"{'='*70}")
                    print("\n→ Robot can now read blocks.json and pick block #1")
                    print("→ After pick complete, return here for next scan")
                    
                    # Show preview
                    cv2.imshow('Snapshot Result', debug_img)
                    print("\nPress any key to close preview...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("\n⚠ No blocks detected in snapshot!")
        
        elif mode == '3':
            # Recalibrate
            print("\nStopping camera for recalibration...")
            pipeline.stop()
            cv2.destroyAllWindows()
            run_diagnostic_calibration()
            print("\n✓ Recalibration complete. Restart program to use new settings.")
        
        else:
            print("\nGoodbye!")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    finally:
        try:
            pipeline.stop()
            print("\n✓ Camera stopped.")
        except RuntimeError:
            pass 
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ensure subprocess is run outside of an existing try block to prevent
    # issues with global cleanup.
    if len(sys.argv) > 1 and sys.argv[1] == 'diagnostic':
        # This branch is for the subprocess run, which is no longer needed 
        # since run_diagnostic_calibration handles the subprocess and self-contained code.
        pass
    else:
        main()