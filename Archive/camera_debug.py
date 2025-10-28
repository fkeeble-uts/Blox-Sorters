import pyrealsense2 as rs
import numpy as np
import cv2

# This diagnostic script shows you EXACTLY what the camera sees at each processing step

def main():
    # Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    align = rs.align(rs.stream.color)
    
    print("Starting diagnostic mode...")
    print("\nControls:")
    print("  W/S - Adjust min depth")
    print("  A/D - Adjust max depth")
    print("  E/R - Adjust min area")
    print("  T/G - Adjust max area")
    print("  Y/H - Adjust rectangularity")
    print("  SPACE - Save parameters to file")
    print("  Q - Quit")
    print("\n" + "="*70)
    
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale:.5f} meters per unit")
    
    # Adjustable parameters
    min_depth_mm = 250
    max_depth_mm = 280
    min_area = 30
    max_area = 80000
    min_rect = 0.6
    
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
            
            # Convert depth range to units
            min_units = int(min_depth_mm / 1000 / depth_scale)
            max_units = int(max_depth_mm / 1000 / depth_scale)
            
            # === STAGE 1: DEPTH MASK ===
            depth_mask = cv2.inRange(depth_image, min_units, max_units)
            depth_mask_display = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
            
            # Count pixels in depth range
            pixels_in_range = np.sum(depth_mask > 0)
            
            # === STAGE 2: MORPHOLOGICAL OPERATIONS ===
            kernel = np.ones((5, 5), np.uint8)
            depth_mask_clean = cv2.erode(depth_mask, kernel, iterations=2)
            depth_mask_clean = cv2.dilate(depth_mask_clean, kernel, iterations=2)
            depth_mask_clean_display = cv2.cvtColor(depth_mask_clean, cv2.COLOR_GRAY2BGR)
            
            # === STAGE 3: CONTOUR DETECTION ===
            contours, _ = cv2.findContours(depth_mask_clean, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            contour_debug = color_image.copy()
            final_debug = color_image.copy()
            
            detected_count = 0
            
            print(f"\rContours found: {len(contours)} | In range pixels: {pixels_in_range:6d} | ", end="")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Draw ALL contours in magenta (to see what was found)
                cv2.drawContours(contour_debug, [contour], -1, (255, 0, 255), 2)
                
                # Show area as text
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(contour_debug, f"A:{int(area)}", (cX-30, cY),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                # === STAGE 4: AREA FILTER ===
                if min_area < area < max_area:
                    # Draw contours that pass area filter in yellow
                    cv2.drawContours(contour_debug, [contour], -1, (0, 255, 255), 2)
                    
                    # === STAGE 5: SHAPE FITTING ===
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.04 * perimeter
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Get rectangularity
                    rect = cv2.minAreaRect(contour)
                    (_, (w, h), angle) = rect
                    rect_area = w * h
                    area_ratio = area / rect_area if rect_area > 0 else 0
                    
                    # Draw approximated polygon in cyan
                    cv2.drawContours(contour_debug, [approx], -1, (255, 255, 0), 2)
                    
                    corners = len(approx)
                    is_convex = cv2.isContourConvex(approx)
                    
                    # Show shape info
                    if M["m00"] != 0:
                        cv2.putText(contour_debug, f"C:{corners} R:{area_ratio:.2f}", 
                                   (cX-40, cY+15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                    
                    # === STAGE 6: FINAL VALIDATION ===
                    if corners == 4 and is_convex and area_ratio >= min_rect:
                        # PASSED! Draw in GREEN
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        cv2.drawContours(final_debug, [box], 0, (0, 255, 0), 3)
                        cv2.circle(final_debug, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.putText(final_debug, f"BLOCK {detected_count+1}", (cX-40, cY-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_count += 1
            
            print(f"Detected: {detected_count}", end="")
            
            # === CREATE MULTI-VIEW DISPLAY ===
            # Add parameter info to images (width must be 1920 to match 3 stacked images)
            info_bg = np.zeros((100, 1920, 3), dtype=np.uint8)
            cv2.putText(info_bg, f"Depth: [{min_depth_mm}, {max_depth_mm}] mm", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(info_bg, f"Area: [{min_area}, {max_area}]", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(info_bg, f"Rectangularity: {min_rect:.2f}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(info_bg, f"Controls: W/S=MinDepth | A/D=MaxDepth | E/R=MinArea | T/G=MaxArea | Y/H=Rect | SPACE=Save | Q=Quit", 
                       (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Label each view
            cv2.putText(color_image, "1. COLOR INPUT", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_mask_display, "2. DEPTH MASK", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_mask_clean_display, "3. AFTER MORPHOLOGY", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(contour_debug, "4. CONTOUR ANALYSIS", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(contour_debug, "Magenta=Found, Yellow=PassArea, Cyan=Shape", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(final_debug, "5. FINAL DETECTIONS", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Stack images in grid
            top_row = np.hstack([color_image, depth_mask_display, depth_mask_clean_display])
            bottom_row = np.hstack([contour_debug, final_debug, np.zeros_like(color_image)])
            
            display = np.vstack([info_bg, top_row, bottom_row])
            
            # Show in large window
            cv2.namedWindow('Diagnostic View', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Diagnostic View', 1920, 800)
            cv2.imshow('Diagnostic View', display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('w'):
                min_depth_mm += 1
            elif key == ord('s'):
                min_depth_mm -= 1
            elif key == ord('a'):
                max_depth_mm -= 1
            elif key == ord('d'):
                max_depth_mm += 1
            elif key == ord('e'):
                min_area += 10
            elif key == ord('r'):
                min_area = max(10, min_area - 10)
            elif key == ord('t'):
                max_area -= 1000
            elif key == ord('g'):
                max_area += 1000
            elif key == ord('y'):
                min_rect += 0.05
                min_rect = min(1.0, min_rect)
            elif key == ord('h'):
                min_rect -= 0.05
                min_rect = max(0.1, min_rect)
            elif key == ord(' '):  # SPACE - Save parameters
                import json
                params = {
                    "depth_range_mm": [min_depth_mm, max_depth_mm],
                    "min_area": min_area,
                    "max_area": max_area,
                    "min_rectangularity": round(min_rect, 2),
                    "validated": True
                }
                with open('camera_calibration.json', 'w') as f:
                    json.dump(params, f, indent=4)
                print(f"\n✓ Parameters saved to camera_calibration.json")
                print(f"  Depth: [{min_depth_mm}, {max_depth_mm}] mm")
                print(f"  Area: [{min_area}, {max_area}]")
                print(f"  Rectangularity: {min_rect:.2f}")
            elif key == ord('q'):
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n\nDiagnostic complete.")
        print(f"\nFinal parameters:")
        print(f"  Depth range: [{min_depth_mm}, {max_depth_mm}] mm")
        print(f"  Area range: [{min_area}, {max_area}]")
        print(f"  Min rectangularity: {min_rect:.2f}")


if __name__ == "__main__":
    main()