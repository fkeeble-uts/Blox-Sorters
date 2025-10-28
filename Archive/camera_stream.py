import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time

# --- CONFIGURATION CONSTANTS (ADJUST THESE) ---
MIN_DEPTH_M = 0.25      # Minimum distance (m) to consider an object
MAX_DEPTH_M = 0.28     # Maximum distance (m) to consider an object
# The paper is small, so keeping these areas is okay.
MIN_BLOCK_AREA = 30     # Minimum contour area in pixels to consider (pre-filter)
MAX_BLOCK_AREA = 80000  # Maximum contour area in pixels to consider (eliminates large table/background remnants)

# --- NEW SHAPE/FEATURE DETECTION CONSTANTS ---
SHAPE_FIT_TOLERANCE = 0.04 # Epsilon factor for cv2.approxPolyDP (4% of contour perimeter is a good start)
EXPECTED_CORNERS = 4       # Number of corners expected for a rectangular block
# CRITICAL CONFIDENCE CRITERIA: Ratio of (Contour Area / Rotated Bounding Box Area).
MIN_RECTANGULARITY_RATIO = 0.6 

# --- NEW COLOR CONFIGURATION (HSV RANGES) ---
# HUE (H: 0-179), SATURATION (S: 0-255), VALUE (V: 0-255)
# NOTE: YOU MUST FINE-TUNE THESE RANGES IN YOUR LIGHTING ENVIRONMENT.
COLOR_RANGES = {
    # Red requires two ranges because Hue wraps around
    "red": [([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [179, 255, 255])],
    "green": [([35, 70, 50], [85, 255, 255])],
    "blue": [([100, 70, 50], [130, 255, 255])],
    "yellow": [([20, 70, 50], [35, 255, 255])],
}

# --- CRITICAL ROBOT COORDINATE SYSTEM CONFIGURATION (mm) ---
# 1. CALIBRATION OFFSETS: Replace these with values found during the hand-eye calibration procedure.
# These are the absolute shifts (in mm) between the camera's origin and the robot's origin.
OFFSET_X_MM = 300.0     # Delta X: Offset needed to map camera X to robot X (TEST AND REPLACE!)
OFFSET_Y_MM = 100.0     # Delta Y: Offset needed to map camera Y to robot Y (TEST AND REPLACE!)

# 2. Z-AXIS HEIGHT: Measured physically from the camera lens to the robot's Z=0 plane (the table top).
TABLE_HEIGHT_CAM_M = 0.277 # The average Z-distance from camera to the table surface (in meters)
BLOCK_HEIGHT_MM = 12.0      # The standard height of the blocks (for pick Z coordinate)


# --- COLOR CLASSIFICATION FUNCTION ---
def classify_color(bgr_color):
    """Converts BGR pixel value to HSV and classifies it based on predefined ranges."""
    # Convert single BGR pixel to numpy array and then to HSV
    bgr = np.uint8([[bgr_color]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    
    H, S, V = hsv[0], hsv[1], hsv[2]
    
    # Simple check for very dark/white/gray blocks (low saturation/value)
    if S < 50 or V < 50:
        return "unknown/gray"

    for color_name, ranges in COLOR_RANGES.items():
        for (lower, upper) in ranges:
            # Check if H, S, V fall within the range
            if H >= lower[0] and H <= upper[0] and \
               S >= lower[1] and S <= upper[1] and \
               V >= lower[2] and V <= upper[2]:
                return color_name
    
    return "unknown"


# 1. Configure streams
pipeline = rs.pipeline()
config = rs.config()

# Enable the streams you need
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Optional: Enable alignment between color and depth frames
align_to = rs.stream.color
align = rs.align(align_to)

# 2. Start streaming
print("Starting RealSense stream...")
profile = pipeline.start(config)

# Get depth scale (converts Z16 units to meters)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"Depth Scale is: {depth_scale:.5f} meters per unit")

# Convert meter constants to 16-bit depth units (Z16 format)
min_depth_units = int(MIN_DEPTH_M / depth_scale)
max_depth_units = int(MAX_DEPTH_M / depth_scale)


try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get camera intrinsics (needed for 3D projection)
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics


        # ----------------------------------------------------------------
        # --- STEP 1: DEPTH-BASED SEGMENTATION (Isolating the Blocks) ---
        # ----------------------------------------------------------------

        # Create a depth mask for objects in the desired range (Z16 units)
        depth_mask = cv2.inRange(depth_image, min_depth_units, max_depth_units)

        # Apply some morphological operations to clean up the mask for better contour stability
        kernel = np.ones((5, 5), np.uint8)
        # Increase iterations for more aggressive noise reduction (Erode) and gap filling (Dilate)
        depth_mask = cv2.erode(depth_mask, kernel, iterations=2)
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=2)
        
        # Visualize the mask by merging it with the color image for debugging
        masked_color = cv2.bitwise_and(color_image, color_image, mask=depth_mask)


        # ----------------------------------------------------------
        # --- STEP 2: CONTOUR DETECTION AND ROBIN SHAPE FITTING ---
        # ----------------------------------------------------------

        # Find external contours in the cleaned depth mask
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store final robot-ready coordinates (as per data_flow_guide.md)
        blocks_data_mm = []

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter 1: Basic area check (to eliminate small noise)
            if area > MIN_BLOCK_AREA:
                
                # Filter 1.1 (NEW): Max area check (to eliminate large table remnants)
                if area < MAX_BLOCK_AREA:
                
                    # Calculate the perimeter
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Approximate the contour to a simpler polygon (shape fitting)
                    epsilon = SHAPE_FIT_TOLERANCE * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Get the rotated bounding box properties
                    rect = cv2.minAreaRect(contour)
                    (_, (w, h), angle) = rect
                    rect_area = w * h
                    
                    # Avoid division by zero
                    if rect_area > 0:
                        area_ratio = area / rect_area
                    else:
                        area_ratio = 0.0

                    # --- START DEBUG OUTPUT ---
                    # Draw a magenta outline for any object that passes the area filters
                    cv2.drawContours(color_image, [contour], -1, (255, 0, 255), 1)
                    # Print the core shape metrics for analysis
                    print(f"DEBUG_FILTER: Area={area:.1f}, Corners={len(approx)}, Ratio={area_ratio:.2f}")
                    # --- END DEBUG OUTPUT ---
                    
                    # Filter 2: Check if the approximation has the expected number of corners (4 for a block)
                    if len(approx) == EXPECTED_CORNERS:
                        
                        # Filter 3: Check if the shape is convex (a solid block)
                        if cv2.isContourConvex(approx):
                            
                            # Filter 4: High Confidence Check (Must be nearly rectangular)
                            if area_ratio >= MIN_RECTANGULARITY_RATIO:
                                
                                # Block is Confirmed
                                
                                # Get the moments to find the centroid (center of mass)
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
    
                                    # Normalize angle so that the width (w) is always the longest side.
                                    if w < h:
                                        angle += 90
                                    
                                    rotation_angle_degrees = round(angle, 2)
                                    # ----------------------------------------------------------
                                    
                                    
                                    # ----------------------------------------------------------
                                    # --- STEP 3: 3D LOCALIZATION & COLOR CLASSIFICATION ---
                                    # ----------------------------------------------------------
    
                                    # 3A. Get the raw depth value at the center pixel
                                    depth_val_units = depth_image[cY, cX]
    
                                    if depth_val_units > 0:
                                        # Convert depth value from units to meters
                                        Z_coord_m = depth_val_units * depth_scale
    
                                        # 3B. Project the 2D pixel to a 3D point (X, Y, Z in meters)
                                        point_3D = rs.rs2_deproject_pixel_to_point(
                                            depth_intrin, 
                                            [cX, cY], 
                                            Z_coord_m
                                        )
    
                                        X_coord_m, Y_coord_m = point_3D[0], point_3D[1]
    
                                        # 3C. Get the BGR color at the center pixel and classify
                                        color_bgr = color_image[cY, cX]
                                        block_color_name = classify_color(color_bgr)
    
    
                                        # 3D. Apply Coordinate Transformation (Camera Frame [m] -> Robot Frame [mm])
    
                                        # 1. Scale from meters to millimeters (S=1000)
                                        X_cam_mm = X_coord_m * 1000
                                        Y_cam_mm = Y_coord_m * 1000
                                        Z_cam_mm = Z_coord_m * 1000 # Distance from camera to block top (mm)
    
                                        # 2. Apply Rotations/Flips and Offsets (The 90-degree twist)
                                        # NOTE: This assumes a typical 90-degree rotation needed for overhead camera to DoBot base.
                                        # YOU MUST VERIFY THE SIGNS (+/-) during calibration!
                                        X_robot_mm = (-Y_cam_mm + OFFSET_X_MM) # Flipped axes, inverted Y_cam, plus offset
                                        Y_robot_mm = (-X_cam_mm + OFFSET_Y_MM) # Flipped axes, inverted X_cam, plus offset
    
                                        # 3. Calculate Z_robot (Z is height above the table, in mm)
                                        # Z_robot = (Distance from Camera to Table) - (Distance from Camera to Block Top)
                                        Z_table_offset_mm = TABLE_HEIGHT_CAM_M * 1000
                                        Z_block_top_mm = Z_table_offset_mm - Z_cam_mm
    
                                        # Store data in the format the DoBot script expects
                                        blocks_data_mm.append({
                                            'block_id': len(blocks_data_mm) + 1,
                                            'x': round(X_robot_mm, 2),
                                            'y': round(Y_robot_mm, 2),
                                            # Add half the block height to ensure the Z position is the picking surface (block top)
                                            'z': round(Z_block_top_mm + BLOCK_HEIGHT_MM / 2, 2), 
                                            'color': block_color_name,
                                            'rotation_angle': rotation_angle_degrees # NEW: Angle for Rz movement
                                        })
    
                                        # Visualize on the color image
                                        text_3D = f"R_mm=[{X_robot_mm:.1f}, {Y_robot_mm:.1f}], A={rotation_angle_degrees:.1f}°, C={area_ratio:.2f} ({block_color_name})"
                                        
                                        # Draw the rotated box (in red)
                                        box_points = cv2.boxPoints(rect)
                                        box_points = np.intp(box_points) # Convert to integer coordinates
                                        cv2.drawContours(color_image, [box_points], 0, (0, 0, 255), 2)
                                        
                                        # Draw the contour outline (in green)
                                        cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 1)
    
                                        cv2.circle(color_image, (cX, cY), 5, (255, 0, 0), -1) # Blue center dot
                                        cv2.putText(color_image, text_3D, (cX - 70, cY - 20), 
                                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                        
                                        # Print to console for real-time debugging
                                        print(f"Block {len(blocks_data_mm)}: Color={block_color_name}, Pos=({X_robot_mm:.1f}, {Y_robot_mm:.1f}), Angle={rotation_angle_degrees:.1f}°, Confidence={area_ratio:.2f}")
        
        
        # --------------------------------------------------------------------------
        # --- STEP 4: DISPLAY AND DATA OUTPUT (JSON FILE GENERATION) ---
        # --------------------------------------------------------------------------

        # Apply colormap on depth image (for visualization)
        depth_colormap = cv2.applyColorMap(
             cv2.convertScaleAbs(depth_image, alpha=0.03), 
             cv2.COLORMAP_JET
        )
        
        # Stack all images horizontally for side-by-side view
        images = np.hstack((color_image, masked_color, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense Block Detection (Press S to Save/Q to Exit)', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Block Detection (Press S to Save/Q to Exit)', images)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Exit loop if 'q' key is pressed
        if key == ord('q'):
            break
            
        # Save data if 's' key is pressed
        if key == ord('s'):
            if blocks_data_mm:
                with open('pick_list.json', 'w') as f:
                    json.dump(blocks_data_mm, f, indent=4)
                print("\n[SUCCESS] Block data saved to pick_list.json.")
                print(f"File contents for DoBot:\n{json.dumps(blocks_data_mm, indent=4)}")
                time.sleep(1) # Pause to see the output confirmation

finally:
    # 4. Stop streaming
    print("Stopping RealSense stream.")
    pipeline.stop()
    cv2.destroyAllWindows()
