import pyrealsense2 as rs
import numpy as np
import cv2
import time

# --- CONFIGURATION CONSTANTS (ADJUST THESE) ---
# NOTE: These depth values assume your work surface (e.g., table) is around 0.7m away.
# Adjust MIN_DEPTH_M and MAX_DEPTH_M based on your actual setup and block height.
# MIN_BLOCK_AREA filters out noise; adjust based on block size and camera distance.
MIN_DEPTH_M = 0.65    # Minimum distance (m) to consider an object
MAX_DEPTH_M = 0.75    # Maximum distance (m) to consider an object
MIN_BLOCK_AREA = 500  # Minimum contour area in pixels to be considered a block

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
        # --- NEW STEP 1: DEPTH-BASED SEGMENTATION (Isolating the Blocks) ---
        # ----------------------------------------------------------------

        # Create a depth mask for objects in the desired range (Z16 units)
        depth_mask = cv2.inRange(depth_image, min_depth_units, max_depth_units)

        # Apply some morphological operations to clean up the mask (optional cleanup)
        kernel = np.ones((5, 5), np.uint8)
        depth_mask = cv2.erode(depth_mask, kernel, iterations=1)
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=1)
        
        # Visualize the mask by merging it with the color image for debugging
        masked_color = cv2.bitwise_and(color_image, color_image, mask=depth_mask)


        # ----------------------------------------------------------
        # --- NEW STEP 2: CONTOUR DETECTION (Finding Block Shapes) ---
        # ----------------------------------------------------------

        # Find external contours in the cleaned depth mask
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks_data = []

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter contours based on size
            if area > MIN_BLOCK_AREA:
                
                # Get the moments to find the centroid (center of mass)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # ----------------------------------------------------------
                    # --- NEW STEP 3: 3D LOCALIZATION & COLOR CLASSIFICATION ---
                    # ----------------------------------------------------------

                    # 3A. Get the raw depth value at the center pixel
                    depth_val_units = depth_image[cY, cX]

                    if depth_val_units > 0:
                        # Convert depth value from units to meters
                        depth_m = depth_val_units * depth_scale

                        # 3B. Project the 2D pixel to a 3D point (X, Y, Z in meters)
                        # X is lateral, Y is vertical, Z is depth (distance from camera)
                        point_3D = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, 
                            [cX, cY], 
                            depth_m
                        )

                        X_coord, Y_coord, Z_coord = point_3D[0], point_3D[1], point_3D[2]

                        # 3C. Get the BGR color at the center pixel
                        color_bgr = color_image[cY, cX]

                        # Store and print data
                        blocks_data.append({
                            'center_pixel': (cX, cY),
                            'position_3D': (X_coord, Y_coord, Z_coord),
                            'color_bgr': color_bgr
                        })

                        # Visualize on the color image
                        text_3D = f"[{X_coord:.3f}, {Y_coord:.3f}, {Z_coord:.3f}]"
                        cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1) # Red center dot
                        cv2.putText(color_image, text_3D, (cX - 70, cY - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Print to console for robotic system integration
                        print(f"Block found: Pxl=({cX}, {cY}), Pos_m=({X_coord:.4f}, {Y_coord:.4f}, {Z_coord:.4f}), Color={color_bgr}")
        
        
        # 3. Display the streams
        # Apply colormap on depth image (for visualization)
        depth_colormap = cv2.applyColorMap(
             cv2.convertScaleAbs(depth_image, alpha=0.03), 
             cv2.COLORMAP_JET
        )
        
        # Stack all images horizontally for side-by-side view
        # Showing Color, Masked Color, and Depth
        images = np.hstack((color_image, masked_color, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense Block Detection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Block Detection', images)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 4. Stop streaming
    print("Stopping RealSense stream.")
    pipeline.stop()
    cv2.destroyAllWindows()
