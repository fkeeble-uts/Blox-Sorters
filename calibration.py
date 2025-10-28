import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
from pathlib import Path
from collections import deque
import subprocess
import sys


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def draw_camera_origin(image):
    """Draws a crosshair at the center of the image (the camera's (0,0,0) origin)."""
    H, W, _ = image.shape
    center_x = W // 2
    center_y = H // 2
    
    CENTER_COLOR = (255, 0, 255)  # Magenta for high visibility
    LINE_LENGTH = 30
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


def load_calibration():
    """Load calibration from file"""
    if Path('camera_calibration.json').exists():
        with open('camera_calibration.json', 'r') as f:
            config = json.load(f)
            if config.get('validated', False):
                return config
    return None


# ============================================================================
# PERSISTENT BLOCK TRACKER
# ============================================================================

class PersistentBlockTracker:
    """Tracks blocks across frames with persistence and stable IDs"""
    
    def __init__(self, timeout_seconds=2.0, match_distance_mm=25.0):
        """
        Args:
            timeout_seconds: How long to remember a block after it disappears
            match_distance_mm: Maximum distance (mm) to consider same block
        """
        self.tracked_blocks = {}  # {block_id: BlockInfo}
        self.next_id = 1
        self.timeout_seconds = timeout_seconds
        self.match_distance_mm = match_distance_mm
    
    def update(self, detected_blocks):
        """
        Update tracker with new detections, return stabilized blocks
        
        Args:
            detected_blocks: List of block dicts from current frame
        
        Returns:
            List of stable tracked blocks with consistent IDs
        """
        current_time = time.time()
        
        # Mark all existing blocks as "not seen this frame"
        for block_info in self.tracked_blocks.values():
            block_info['seen_this_frame'] = False
        
        # Match detections to existing tracked blocks
        matched_ids = set()
        unmatched_detections = []
        
        for detection in detected_blocks:
            best_match_id = None
            best_match_dist = self.match_distance_mm
            
            # Find closest tracked block
            for block_id, block_info in self.tracked_blocks.items():
                if block_id in matched_ids:
                    continue
                
                dist = np.sqrt(
                    (detection['x_mm'] - block_info['x_mm'])**2 +
                    (detection['y_mm'] - block_info['y_mm'])**2
                )
                
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = block_id
            
            if best_match_id is not None:
                # Update existing block
                self._update_block(best_match_id, detection, current_time)
                matched_ids.add(best_match_id)
            else:
                # New block detected
                unmatched_detections.append(detection)
        
        # Add new blocks
        for detection in unmatched_detections:
            self._add_new_block(detection, current_time)
        
        # Remove timed-out blocks
        self._remove_stale_blocks(current_time)
        
        # Return all active blocks (sorted by ID)
        stable_blocks = []
        for block_id in sorted(self.tracked_blocks.keys()):
            block_info = self.tracked_blocks[block_id]
            stable_blocks.append(block_info['current_state'].copy())
        
        return stable_blocks
    
    def _add_new_block(self, detection, current_time):
        """Add a newly detected block to tracking"""
        block_id = self.next_id
        self.next_id += 1
        
        self.tracked_blocks[block_id] = {
            'block_id': block_id,
            'first_seen': current_time,
            'last_seen': current_time,
            'detection_count': 1,
            'seen_this_frame': True,
            'history': deque(maxlen=20),
            'current_state': None,
            'x_mm': detection['x_mm'],
            'y_mm': detection['y_mm']
        }
        
        self._update_block(block_id, detection, current_time)
    
    def _update_block(self, block_id, detection, current_time):
        """Update an existing block with new measurement"""
        block_info = self.tracked_blocks[block_id]
        
        # Add measurement to history
        block_info['history'].append({
            'x_mm': detection['x_mm'],
            'y_mm': detection['y_mm'],
            'z_mm': detection['z_mm'],
            'rotation_angle_deg': detection['rotation_angle_deg'],
            'color': detection['color'],
            'confidence': detection['confidence'],
            'timestamp': current_time
        })
        
        block_info['last_seen'] = current_time
        block_info['detection_count'] += 1
        block_info['seen_this_frame'] = True
        
        # Compute filtered state (weighted average of recent measurements)
        history = list(block_info['history'])
        weights = np.linspace(0.5, 1.0, len(history))
        weights = weights / weights.sum()
        
        avg_x = sum(h['x_mm'] * w for h, w in zip(history, weights))
        avg_y = sum(h['y_mm'] * w for h, w in zip(history, weights))
        avg_z = sum(h['z_mm'] * w for h, w in zip(history, weights))
        avg_angle = self._average_angles([h['rotation_angle_deg'] for h in history], weights)
        avg_confidence = sum(h['confidence'] * w for h, w in zip(history, weights))
        
        # Update position for matching
        block_info['x_mm'] = avg_x
        block_info['y_mm'] = avg_y
        
        # Most recent color
        latest_color = history[-1]['color']
        
        # Get latest pixel position
        pixel_x = detection.get('pixel_x', 0)
        pixel_y = detection.get('pixel_y', 0)
        height_above_table = detection.get('height_above_table_mm')
        
        # Update current state
        block_info['current_state'] = {
            'block_id': block_id,
            'x_mm': round(avg_x, 2),
            'y_mm': round(avg_y, 2),
            'z_mm': round(avg_z, 2),
            'height_above_table_mm': height_above_table,
            'rotation_angle_deg': round(avg_angle, 2),
            'color': latest_color,
            'confidence': round(avg_confidence, 3),
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'detection_count': block_info['detection_count'],
            'age_seconds': round(current_time - block_info['first_seen'], 2)
        }
    
    def _average_angles(self, angles, weights):
        """Average angles properly (handle wraparound at 0/360)"""
        angles_rad = np.array(angles) * np.pi / 180.0
        weights = np.array(weights)
        
        sin_avg = np.average(np.sin(angles_rad), weights=weights)
        cos_avg = np.average(np.cos(angles_rad), weights=weights)
        
        avg_rad = np.arctan2(sin_avg, cos_avg)
        avg_deg = avg_rad * 180.0 / np.pi
        
        if avg_deg < 0:
            avg_deg += 360
        
        return avg_deg
    
    def _remove_stale_blocks(self, current_time):
        """Remove blocks that haven't been seen recently"""
        stale_ids = []
        
        for block_id, block_info in self.tracked_blocks.items():
            time_since_seen = current_time - block_info['last_seen']
            if time_since_seen > self.timeout_seconds:
                stale_ids.append(block_id)
        
        for block_id in stale_ids:
            del self.tracked_blocks[block_id]
    
    def get_stable_blocks(self, min_detections=5):
        """
        Get only blocks that have been seen enough times to be confident
        
        Args:
            min_detections: Minimum number of detections required
        
        Returns:
            List of stable blocks
        """
        stable = []
        for block_info in self.tracked_blocks.values():
            if block_info['detection_count'] >= min_detections:
                stable.append(block_info['current_state'].copy())
        
        return stable
    
    def clear(self):
        """Clear all tracked blocks"""
        self.tracked_blocks.clear()
        self.next_id = 1


# ============================================================================
# ENHANCED BLOCK DETECTOR
# ============================================================================

class BlockDetector:
    """Enhanced block detection with persistent tracking"""
    
    def __init__(self, config):
        self.config = config
        self.shape_tolerance = 0.04
        
        # Persistent tracker (NEW)
        self.tracker = PersistentBlockTracker(
            timeout_seconds=2.0,
            match_distance_mm=25.0
        )
        
        # Legacy detection history (kept for compatibility)
        self.detection_history = deque(maxlen=10)
        
        # Color ranges (HSV)
        self.color_ranges = {
            "red": [([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [179, 255, 255])],
            "green": [([35, 70, 50], [85, 255, 255])],
            "blue": [([100, 70, 50], [130, 255, 255])],
            "yellow": [([20, 70, 50], [35, 255, 255])],
            "orange": [([10, 70, 50], [20, 255, 255])],
            "purple": [([130, 70, 50], [160, 255, 255])],
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
        
        # Morphological operations
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
                        
                        # Normalize angle
                        if w < h:
                            angle += 90
                        
                        # Get 3D position
                        depth_val = depth_image[cY, cX]
                        if depth_val > 0:
                            z_m = depth_val * depth_scale
                            
                            # Project to 3D
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
                            
                            # Calculate height above table
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
    
    def detect_blocks(self, depth_image, color_image, depth_frame, depth_scale, stabilize=True):
        """
        Detect blocks with persistent tracking (ENHANCED)
        
        Args:
            stabilize: If True, use persistent tracker (recommended)
        
        Returns:
            (blocks, debug_image)
        """
        # Get raw detections
        current_detections = self.detect_blocks_single_frame(
            depth_image, color_image, depth_frame, depth_scale
        )
        
        if stabilize:
            # Use persistent tracker (NEW)
            blocks = self.tracker.update(current_detections)
        else:
            # No stabilization
            blocks = current_detections
            for i, block in enumerate(blocks):
                block['block_id'] = i + 1
        
        # Create visualization
        debug_image = color_image.copy()
        
        for block in blocks:
            cX, cY = block['pixel_x'], block['pixel_y']
            
            # Color based on stability
            detection_count = block.get('detection_count', 1)
            if detection_count < 5:
                circle_color = (0, 165, 255)  # Orange - new/unstable
                status = "NEW"
            else:
                circle_color = (0, 255, 0)  # Green - stable
                status = "STABLE"
            
            # Draw circle
            cv2.circle(debug_image, (cX, cY), 8, circle_color, -1)
            
            # Draw cross
            cv2.line(debug_image, (cX-15, cY), (cX+15, cY), circle_color, 2)
            cv2.line(debug_image, (cX, cY-15), (cX, cY+15), circle_color, 2)
            
            # Label with ID and color
            label = f"#{block['block_id']}: {block['color']}"
            cv2.putText(debug_image, label, (cX - 40, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 2)
            
            # Coordinates
            coord_text = f"({block['x_mm']:.0f}, {block['y_mm']:.0f})"
            cv2.putText(debug_image, coord_text, (cX - 35, cY + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Detection count for debugging
            if detection_count < 10:
                count_text = f"{status} n={detection_count}"
            else:
                count_text = f"{status}"
            cv2.putText(debug_image, count_text, (cX - 30, cY + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return blocks, debug_image
    
    def clear_tracker(self):
        """Clear tracking history (call before snapshot)"""
        self.tracker.clear()
        self.detection_history.clear()


# ============================================================================
# SNAPSHOT CAPTURE
# ============================================================================

def capture_snapshot(pipeline, align, depth_scale, detector, num_samples=50, min_detections=10):
    """
    Capture a stable snapshot with persistent tracking (ENHANCED)
    
    Args:
        num_samples: Number of frames to collect (~1.7 seconds at 30fps)
        min_detections: Minimum detections to consider block stable
    """
    print("\nCapturing enhanced stable snapshot...")
    print("Please ensure:")
    print("  ✓ Robot arm is NOT in camera view")
    print("  ✓ All blocks are stationary")
    print("  ✓ Lighting is consistent")
    print(f"\nCollecting {num_samples} frames (~{num_samples/30:.1f} seconds)...", end="", flush=True)
    
    # Clear previous tracking
    detector.clear_tracker()
    
    # Collect samples
    for i in range(num_samples):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if depth_frame and color_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Update tracker
            detector.detect_blocks(depth_image, color_image, depth_frame, 
                                  depth_scale, stabilize=True)
        
        if i % 10 == 0:
            print(".", end="", flush=True)
        time.sleep(0.033)  # ~30fps
    
    print(" Done!")
    
    # Final frame with stability filtering
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # Get all tracked blocks
    all_blocks, debug_img = detector.detect_blocks(
        depth_image, color_image, depth_frame, depth_scale, stabilize=True
    )
    
    # Filter for stability
    stable_blocks = [
        block for block in all_blocks 
        if block.get('detection_count', 0) >= min_detections
    ]
    
    # Mark stable blocks on image
    for block in stable_blocks:
        cX, cY = block['pixel_x'], block['pixel_y']
        cv2.putText(debug_img, "✓ STABLE", (cX - 30, cY - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    # Add camera origin
    draw_camera_origin(debug_img)
    
    print(f"\n✓ Found {len(stable_blocks)} stable blocks (from {len(all_blocks)} total detections)")
    
    return stable_blocks, debug_img


# ============================================================================
# DIAGNOSTIC CALIBRATION
# ============================================================================

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
    
    diagnostic_code = '''import pyrealsense2 as rs
import numpy as np
import cv2
import json

def draw_camera_origin(image):
    """Draws a crosshair at the center of the image (the camera's (0,0,0) origin)."""
    H, W, _ = image.shape
    center_x = W // 2
    center_y = H // 2
    
    CENTER_COLOR = (255, 0, 255)
    LINE_LENGTH = 30
    THICKNESS = 2

    cv2.circle(image, (center_x, center_y), 5, CENTER_COLOR, -1)
    cv2.line(image, (center_x - LINE_LENGTH, center_y), (center_x + LINE_LENGTH, center_y), CENTER_COLOR, THICKNESS)
    cv2.line(image, (center_x, center_y - LINE_LENGTH), (center_x, center_y + LINE_LENGTH), CENTER_COLOR, THICKNESS)
    cv2.putText(image, "X_C", (center_x + LINE_LENGTH + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)
    cv2.putText(image, "Y_C", (center_x + 5, center_y + LINE_LENGTH + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)

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


# ============================================================================
# MAIN (Standalone Testing)
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ENHANCED BLOCK DETECTION SYSTEM - STANDALONE MODE")
    print("="*70)
    
    # Load calibration
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
        return
    
    print("\n" + "="*70)
    print("OPERATION MODES:")
    print("="*70)
    print("  1. LIVE PREVIEW - See detections with persistent tracking")
    print("  2. SNAPSHOT MODE - Capture stable blocks for robot")
    print("  3. RECALIBRATE - Adjust detection parameters")
    print("  4. QUIT")
    print("="*70)
    
    mode = input("\nSelect mode (1/2/3/4): ").strip()
    
    try:
        if mode == '1':
            # Live preview mode
            print("\n" + "="*70)
            print("LIVE PREVIEW MODE (Enhanced Persistent Tracking)")
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
                
                # Add origin marker
                draw_camera_origin(debug_img)
                
                # Show status
                stable_count = sum(1 for b in blocks if b.get('detection_count', 0) >= 5)
                status_text = f"Blocks: {len(blocks)} ({stable_count} stable) | Q=Quit | Orange=New Green=Stable"
                cv2.putText(debug_img, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Live Preview', debug_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        elif mode == '2':
            # Snapshot mode
            print("\n" + "="*70)
            print("SNAPSHOT MODE - Enhanced Stability")
            print("="*70)
            print("\nRECOMMENDED WORKFLOW:")
            print("  1. Ensure robot arm is NOT blocking camera view")
            print("  2. Capture snapshot → saves blocks.json")
            print("  3. Robot reads blocks.json and picks blocks")
            print("  4. Press Ctrl+C to exit")
            print("="*70)
            
            while True:
                input("\nPress ENTER to capture snapshot (or Ctrl+C to quit)...")
                
                blocks, debug_img = capture_snapshot(
                    pipeline, align, depth_scale, detector,
                    num_samples=50, min_detections=10
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
                        'workflow_note': 'Enhanced tracking - blocks filtered for stability',
                        'blocks': blocks
                    }
                    
                    with open('blocks.json', 'w') as f:
                        json.dump(output, f, indent=4)
                    
                    print(f"\n{'='*70}")
                    print(f"✓ SAVED {len(blocks)} STABLE BLOCKS to blocks.json")
                    print(f"{'='*70}")
                    for block in blocks:
                        print(f"  Block {block['block_id']}: {block['color']:7s} at "
                              f"X:{block['x_mm']:7.1f} Y:{block['y_mm']:7.1f} Z:{block['z_mm']:6.1f} mm, "
                              f"Angle:{block['rotation_angle_deg']:6.1f}° (n={block.get('detection_count', 0)})")
                    print(f"{'='*70}")
                    
                    # Show preview
                    cv2.imshow('Snapshot Result', debug_img)
                    print("\nPress any key to close preview...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("\n⚠ No stable blocks detected!")
        
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
    main()