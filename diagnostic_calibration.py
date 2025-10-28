import pyrealsense2 as rs
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
                print("\n✓ Saved!")
            elif key == ord('q'): break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
