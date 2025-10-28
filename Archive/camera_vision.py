"""
Vision system module - handles camera initialization, block detection, and visualization
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
from pathlib import Path
from collections import deque
import threading


def draw_camera_origin(image):
    """Draws a crosshair at the center of the image (camera origin)"""
    H, W = image.shape[:2]
    center_x = W // 2
    center_y = H // 2
    
    CENTER_COLOR = (255, 0, 255)  # Magenta
    LINE_LENGTH = 30
    THICKNESS = 2

    cv2.circle(image, (center_x, center_y), 5, CENTER_COLOR, -1)
    
    # Draw calibration grid points
    for dx, dy in [(0, 0), (100, 100), (100, -100), (-100, -100), (-100, 100)]:
        cv2.circle(image, (center_x + dx, center_y + dy), 3, CENTER_COLOR, -1)
    
    cv2.line(image, (center_x - LINE_LENGTH, center_y), 
             (center_x + LINE_LENGTH, center_y), CENTER_COLOR, THICKNESS)
    cv2.line(image, (center_x, center_y - LINE_LENGTH), 
             (center_x, center_y + LINE_LENGTH), CENTER_COLOR, THICKNESS)
    
    cv2.putText(image, "X_C", (center_x + LINE_LENGTH + 5, center_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)
    cv2.putText(image, "Y_C", (center_x + 5, center_y + LINE_LENGTH + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 1)


class BlockDetector:
    """Handles block detection with temporal stabilization"""
    
    def __init__(self, config):
        self.config = config
        self.shape_tolerance = 0.04
        self.detection_history = deque(maxlen=10)
        
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
        """Detect blocks in a single frame"""
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        depth_range = self.config["depth_range_mm"]
        min_units = int(depth_range[0] / 1000 / depth_scale)
        max_units = int(depth_range[1] / 1000 / depth_scale)
        
        depth_mask = cv2.inRange(depth_image, min_units, max_units)
        
        kernel = np.ones((5, 5), np.uint8)
        depth_mask = cv2.erode(depth_mask, kernel, iterations=2)
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        blocks = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.config["min_area"] < area < self.config["max_area"]:
                perimeter = cv2.arcLength(contour, True)
                epsilon = self.shape_tolerance * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                rect = cv2.minAreaRect(contour)
                (_, (w, h), angle) = rect
                rect_area = w * h
                area_ratio = area / rect_area if rect_area > 0 else 0
                
                if (len(approx) == 4 and 
                    cv2.isContourConvex(approx) and 
                    area_ratio >= self.config["min_rectangularity"]):
                    
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        if w < h:
                            angle += 90
                        
                        depth_val = depth_image[cY, cX]
                        if depth_val > 0:
                            z_m = depth_val * depth_scale
                            
                            point_3d = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [cX, cY], z_m
                            )
                            
                            x_mm = point_3d[0] * 1000
                            y_mm = point_3d[1] * 1000
                            z_mm = z_m * 1000
                            
                            color_bgr = color_image[cY, cX]
                            color = self.classify_color(color_bgr)
                            
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
        """Average detections over multiple frames"""
        self.detection_history.append(current_blocks)
        
        if len(self.detection_history) < 5:
            return current_blocks
        
        stable_blocks = []
        
        for block in current_blocks:
            matching_blocks = [block]
            
            for past_frame in list(self.detection_history)[:-1]:
                for past_block in past_frame:
                    dist = np.sqrt((block['x_mm'] - past_block['x_mm'])**2 + 
                                  (block['y_mm'] - past_block['y_mm'])**2)
                    if dist < 20:
                        matching_blocks.append(past_block)
            
            if len(matching_blocks) >= 3:
                avg_block = {
                    'x_mm': np.mean([b['x_mm'] for b in matching_blocks]),
                    'y_mm': np.mean([b['y_mm'] for b in matching_blocks]),
                    'z_mm': np.mean([b['z_mm'] for b in matching_blocks]),
                    'rotation_angle_deg': np.mean([b['rotation_angle_deg'] for b in matching_blocks]),
                    'color': block['color'],
                    'confidence': np.mean([b['confidence'] for b in matching_blocks]),
                    'pixel_x': block['pixel_x'],
                    'pixel_y': block['pixel_y'],
                    'height_above_table_mm': block['height_above_table_mm']
                }
                
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
    
    def visualize_detections(self, color_image, blocks):
        """Create visualization of detected blocks"""
        debug_image = color_image.copy()
        
        for block in blocks:
            cX, cY = block['pixel_x'], block['pixel_y']
            
            cv2.circle(debug_image, (cX, cY), 5, (0, 255, 0), -1)
            cv2.line(debug_image, (cX-10, cY), (cX+10, cY), (0, 255, 0), 2)
            cv2.line(debug_image, (cX, cY-10), (cX, cY+10), (0, 255, 0), 2)
            
            text = f"#{block['block_id']}: {block['color']}"
            cv2.putText(debug_image, text, (cX - 40, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            coord_text = f"({block['x_mm']:.0f}, {block['y_mm']:.0f})"
            cv2.putText(debug_image, coord_text, (cX - 35, cY + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        draw_camera_origin(debug_image)
        
        return debug_image


class VisionSystem:
    """Main vision system class"""
    
    def __init__(self):
        self.config = self.load_calibration()
        self.detector = BlockDetector(self.config)
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.last_detection_image = None
        self.live_view_active = False
        self.live_view_window = 'Live Camera View'
        self.live_view_thread = None
        
        self.initialize_camera()


    def _live_view_loop(self):
        """The core blocking loop to run in a separate thread."""
        print("Live view thread started. Press 'q' to stop.")
        
        try:
            while self.live_view_active:
                # Capture and process frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames) # Use aligned frames for consistency
                color_frame = aligned_frames.get_color_frame()
                
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                
                # Draw the camera origin on the live feed for reference
                draw_camera_origin(color_image) 
                
                cv2.imshow(self.live_view_window, color_image)
                
                # Check for 'q' key press. waitKey(1) is non-blocking here.
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_live_view()
        except Exception as e:
            print(f"Error in live view thread: {e}")
            self.stop_live_view()
        finally:
            cv2.destroyWindow(self.live_view_window) # Ensure the window closes when loop ends
            self.live_view_active = False # Explicitly set state to False
            print("Live view thread terminated.")

    def toggle_live_view(self):
        """Toggle the live camera view on/off by managing the thread."""
        if self.live_view_active:
            self.stop_live_view()
        else:
            self.live_view_active = True
            # Create a new thread that runs the _live_view_loop method
            self.live_view_thread = threading.Thread(target=self._live_view_loop)
            self.live_view_thread.daemon = True # Allows the program to exit even if the thread is running
            self.live_view_thread.start()
            print("Live view started in background thread.")

    def stop_live_view(self):
        """Safely stop the live view thread."""
        if self.live_view_active:
            self.live_view_active = False
            # We don't need to explicitly join the thread here since it will 
            # naturally exit when the loop condition (live_view_active) becomes False.


    def load_calibration(self):
        """Load camera calibration"""
        if Path('camera_calibration.json').exists():
            with open('camera_calibration.json', 'r') as f:
                config = json.load(f)
                if config.get('validated', False):
                    return config
        
        # Default config if none exists
        return {
            'depth_range_mm': [250, 273],
            'min_area': 4000,
            'max_area': 6000,
            'min_rectangularity': 0.5,
            'validated': False
        }
    
    def initialize_camera(self):
        """Initialize RealSense camera"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.align = rs.align(rs.stream.color)
        
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        # Warm up camera
        for _ in range(30):
            self.pipeline.wait_for_frames()
    
    def capture_stable_snapshot(self, num_samples=30):
        """Capture and detect blocks with temporal stabilization"""
        print("  Sampling frames", end="", flush=True)
        
        self.detector.detection_history.clear()
        
        # Collect samples for stabilization
        for i in range(num_samples):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if depth_frame and color_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                self.detector.detect_blocks_single_frame(
                    depth_image, color_image, depth_frame, self.depth_scale
                )
            
            if i % 5 == 0:
                print(".", end="", flush=True)
            time.sleep(0.05)
        
        print(" Done!")
        
        # Final detection with stabilization
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        blocks = self.detector.stabilize_detections(
            self.detector.detect_blocks_single_frame(
                depth_image, color_image, depth_frame, self.depth_scale
            )
        )
        
        # Add block IDs and round values
        for i, block in enumerate(blocks):
            block['block_id'] = i + 1
            block['x_mm'] = round(block['x_mm'], 2)
            block['y_mm'] = round(block['y_mm'], 2)
            block['z_mm'] = round(block['z_mm'], 2)
            block['rotation_angle_deg'] = round(block['rotation_angle_deg'], 2)
            block['confidence'] = round(block['confidence'], 3)
        
        # Create and save visualization
        self.last_detection_image = self.detector.visualize_detections(color_image, blocks)
        
        return blocks
    
    def show_last_detection(self, window_name='Block Detection'):
        """Display the last detection result"""
        if self.last_detection_image is not None:
            cv2.imshow(window_name, self.last_detection_image)
            print("\nPress any key to close preview...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
    
    def cleanup(self):
        """Clean up camera resources and stop the live view thread."""
        self.stop_live_view()  # Set the flag to False
        
        # Wait for the thread to finish if it's running
        if self.live_view_thread and self.live_view_thread.is_alive():
            print("Waiting for live view thread to finish...")
            # Give it a short timeout to prevent application hanging
            self.live_view_thread.join(timeout=1.0) 
        
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()