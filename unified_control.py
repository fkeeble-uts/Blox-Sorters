#!/usr/bin/env python3
"""
Unified Robot Control System
Integrates camera detection, robot calibration, and autonomous pick & place
"""

import threading
import time
import json
import os
import sys
import numpy as np
import cv2
import roslibpy
from pathlib import Path
import math

# Import from existing modules
from transform_camera_to_robot import transform_camera_to_robot, load_transformation_config
from dobot_control import (
    get_end_effector_pose,
    send_target_end_effector_pose,
    dobot_gripper_open,
    dobot_gripper_close,
    dobot_gripper_release,
)

# Import from renamed files
try:
    import robot_calibrator as robot_cal
    import camera_calibrator as block_det
except ImportError as e:
    print(f" Import Error: {e}")
    print("\nPlease ensure these files exist:")
    print("  - robot_calibration.py")
    print("  - block_detection.py")
    print("  - transform_camera_to_robot.py")
    print("  - dobot_control.py")
    sys.exit(1)

# --- CONFIGURATION ---
ROBOT_CAL_FILE = 'robot_transform_matrix.json'
CAMERA_CAL_FILE = 'camera_calibration.json'
BLOCKS_FILE = 'blocks.json'
DOBOT_IP = '10.42.0.1'
DOBOT_PORT = 9090

SAFE_Z_HEIGHT_M = 0.03
GRIP_Z_HEIGHT_M = -0.022
DROP_LOCATION_MM = (-175.0,-75)
SAFE_LOCATION_MM = (-50,-50)
DROP_HEIGHT_M = 0.05
# ---------------------


# ============================================================================
# ROBOT MOVEMENT FUNCTIONS
# ============================================================================

def go_to_transformed_pose(client, cam_x_mm, cam_y_mm, z_m, rpy_radians):
    """Transforms camera coordinates to robot coordinates and sends move command"""
    print(f"--- Transforming Camera Target ({cam_x_mm:.2f}mm, {cam_y_mm:.2f}mm)")

    try:
        rob_x_m, rob_y_m = transform_camera_to_robot(cam_x_mm, cam_y_mm)
    except FileNotFoundError as e:
        print(f" Error: {e}")
        return False
    except ValueError as e:
        print(f" Error during transformation: {e}")
        return False

    target_pose = [rob_x_m, rob_y_m, z_m]
    
    print(f"--- Sending Robot Target (X:{rob_x_m:.4f}m, Y:{rob_y_m:.4f}m, Z:{z_m:.4f}m)")
    send_target_end_effector_pose(client, target_pose, rpy=rpy_radians)
    time.sleep(2.0)
    
    current_pose = get_end_effector_pose(client, timeout=2.0)
    print(f"--- Robot At: X:{current_pose['position']['x']:.4f}m, Y:{current_pose['position']['y']:.4f}m")
    
    return True


def autonomous_pick_and_place(client, cam_x_mm, cam_y_mm, drop_x_mm, drop_y_mm, block_angle_rad):
    """Executes a full pick and place cycle for a single target"""
    print("\n" + "="*50)
    print(f"STARTING PICK CYCLE: Cam Target ({cam_x_mm:.0f}, {cam_y_mm:.0f}), Angle: {block_angle_rad:.2f} rad")
    print("="*50)

    PICK_RPY = [block_angle_rad, -math.pi / 2.0, 0.0]
    DROP_RPY = [0.0, -math.pi / 2.0, 0.0]

    # 1. Move to safe hover position
    if not go_to_transformed_pose(client, cam_x_mm, cam_y_mm, SAFE_Z_HEIGHT_M, rpy_radians=PICK_RPY):
        return

    # 2. Open Gripper
    print("→ Opening Gripper")
    dobot_gripper_open(client)
    time.sleep(1.0)
    
    # 3. Descend to grip height
    print("→ Descending to grip height")
    if not go_to_transformed_pose(client, cam_x_mm, cam_y_mm, GRIP_Z_HEIGHT_M, rpy_radians=PICK_RPY):
        return
    time.sleep(1.0)

    # 4. Close Gripper
    print("→ Closing Gripper to grip")
    dobot_gripper_close(client)
    time.sleep(1.5)

    # 5. Lift to safe height
    print("→ Lifting to safe height")
    if not go_to_transformed_pose(client, cam_x_mm, cam_y_mm, SAFE_Z_HEIGHT_M, rpy_radians=PICK_RPY):
        return

    # # 6. Move to drop location
    # print("\n" + "-"*50)
    # print(f"STARTING PLACE CYCLE: Drop Target ({drop_x_mm:.0f}, {drop_y_mm:.0f})")
    # print("-"*50)

    # if not go_to_transformed_pose(client, drop_x_mm, drop_y_mm, SAFE_Z_HEIGHT_M):
    #     return

    # # 7. Descend to release height
    # print("→ Descending to release height")
    # if not go_to_transformed_pose(client, drop_x_mm, drop_y_mm, DROP_HEIGHT_M):
    #     return
    # time.sleep(1.0)

    # # 8. Release Gripper
    # print("→ Releasing object")
    # dobot_gripper_open(client)
    # time.sleep(1.5)

    # # 9. Lift back to safe height
    # print("→ Lifting back to safe height")
    # go_to_transformed_pose(client, drop_x_mm, drop_y_mm, SAFE_Z_HEIGHT_M)

    # # 10. Back to waypoint for next pickup 
    # print('Returning to safe waypoint')
    # go_to_transformed_pose(client, -50, -50, SAFE_Z_HEIGHT_M)


    
    
    print("\n✓ CYCLE COMPLETE")


# ============================================================================
# CAMERA FEED THREAD
# ============================================================================

class CameraFeedThread(threading.Thread):
    """Non-blocking camera feed with live detection visualization"""
    
    def __init__(self, detector, config, pipeline, align, depth_scale):
        super().__init__(daemon=True)
        self.detector = detector
        self.config = config
        self.pipeline = pipeline
        self.align = align
        self.depth_scale = depth_scale
        
        self.running = True
        self.paused = False
        self.lock = threading.Lock()
        
        self.latest_blocks = []
        self.latest_debug_image = None
        
    def run(self):
        """Main camera loop"""
        window_name = 'Live Camera Feed'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                blocks, debug_img = self.detector.detect_blocks(
                    depth_image, color_image, depth_frame, self.depth_scale, stabilize=True
                )
                
                block_det.draw_camera_origin(debug_img)
                
                status_text = f"Blocks: {len(blocks)} | Press 'Q' to hide feed"
                cv2.putText(debug_img, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                with self.lock:
                    self.latest_blocks = blocks
                    self.latest_debug_image = debug_img.copy()
                
                cv2.imshow(window_name, debug_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.pause()
                    cv2.destroyWindow(window_name)
                    
            except Exception as e:
                print(f"Camera thread error: {e}")
                time.sleep(0.5)
        
        cv2.destroyAllWindows()
    
    def pause(self):
        with self.lock:
            self.paused = True
    
    def resume(self):
        with self.lock:
            self.paused = False
        cv2.namedWindow('Live Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Camera Feed', 800, 600)
    
    def stop(self):
        self.running = False
        self.join(timeout=2.0)
    
    def get_latest_blocks(self):
        with self.lock:
            return self.latest_blocks.copy()
    
    def get_latest_image(self):
        with self.lock:
            if self.latest_debug_image is not None:
                return self.latest_debug_image.copy()
        return None


# ============================================================================
# CALIBRATION FUNCTIONS
# ============================================================================

def check_calibrations():
    """Check if calibration files exist and are valid"""
    robot_cal_ok = os.path.exists(ROBOT_CAL_FILE)
    camera_cal_ok = False
    
    if os.path.exists(CAMERA_CAL_FILE):
        try:
            with open(CAMERA_CAL_FILE, 'r') as f:
                config = json.load(f)
                camera_cal_ok = config.get('validated', False)
        except:
            pass
    
    return robot_cal_ok, camera_cal_ok


def calibration_wizard(client, camera_system):
    """Interactive calibration wizard"""
    print("\n" + "="*70)
    print("CALIBRATION WIZARD")
    print("="*70)
    
    robot_cal_ok, camera_cal_ok = check_calibrations()
    
    if not camera_cal_ok:
        print("\n📷 CAMERA CALIBRATION REQUIRED")
        response = input("Start camera calibration now? (Y/n): ").strip().lower()
        
        if response != 'n':
            camera_system['thread'].pause()
            camera_system['pipeline'].stop()
            cv2.destroyAllWindows()
            
            print("\nLaunching camera calibration tool...")
            block_det.run_diagnostic_calibration()
            
            print("\nReinitializing camera...")
            pipeline, align, depth_scale, config, detector = initialize_camera()
            camera_system['pipeline'] = pipeline
            camera_system['align'] = align
            camera_system['depth_scale'] = depth_scale
            camera_system['config'] = config
            camera_system['detector'] = detector
            
            camera_system['thread'].stop()
            camera_system['thread'] = CameraFeedThread(detector, config, pipeline, align, depth_scale)
            camera_system['thread'].start()
            
            camera_cal_ok = os.path.exists(CAMERA_CAL_FILE)
    
    if not robot_cal_ok:
        print("\n🤖 ROBOT CALIBRATION REQUIRED")
        response = input("Start robot calibration now? (Y/n): ").strip().lower()
        
        if response != 'n':
            print("\nStarting robot calibration...")
            robot_cal.collect_calibration_data(client)
            robot_cal_ok = os.path.exists(ROBOT_CAL_FILE)
            
            if robot_cal_ok:
                load_transformation_config(ROBOT_CAL_FILE)
    
    robot_cal_ok, camera_cal_ok = check_calibrations()
    
    if robot_cal_ok and camera_cal_ok:
        print("\n✅ ALL CALIBRATIONS COMPLETE!")
    else:
        print("\n⚠️  Some calibrations incomplete.")
    
    input("\nPress ENTER to continue...")

    
def recalibrate_camera_properly(client, camera_system):
    """Properly recalibrate camera with full resource cleanup"""
    print("\n" + "="*70)
    print("CAMERA RECALIBRATION")
    print("="*70)
    
    # Step 1: Stop camera feed thread
    print("Stopping camera feed thread...")
    camera_system['thread'].stop()
    time.sleep(0.5)
    
    # Step 2: Stop pipeline
    print("Releasing camera hardware...")
    try:
        camera_system['pipeline'].stop()
    except Exception as e:
        print(f"Warning during pipeline stop: {e}")
    
    # Step 3: Close all OpenCV windows
    cv2.destroyAllWindows()
    time.sleep(1.0)  # Give system time to release resources
    
    # Step 4: Launch diagnostic tool
    print("\nLaunching diagnostic calibration tool...")
    print("(This will open in a new window)")
    
    # Import here to ensure clean module state
    import camera_calibrator as block_det
    
    try:
        block_det.run_diagnostic_calibration()
    except Exception as e:
        print(f"Error during calibration: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Wait for diagnostic tool to finish
    time.sleep(2.0)
    
    # Step 6: Reinitialize camera system
    print("\nReinitializing camera system...")
    try:
        pipeline, align, depth_scale, config, detector = initialize_camera()
        
        camera_system['pipeline'] = pipeline
        camera_system['align'] = align
        camera_system['depth_scale'] = depth_scale
        camera_system['config'] = config
        camera_system['detector'] = detector
        
        # Step 7: Start new camera feed thread
        camera_thread = CameraFeedThread(detector, config, pipeline, align, depth_scale)
        camera_thread.start()
        camera_system['thread'] = camera_thread
        
        print("✓ Camera recalibrated and reinitialized")
        
    except Exception as e:
        print(f"✗ Failed to reinitialize camera: {e}")
        import traceback
        traceback.print_exc()
        print("\nYou may need to restart the entire program.")
    
    input("\nPress ENTER to continue...")


def initialize_camera():
    """Initialize camera system"""
    import pyrealsense2 as rs
    
    config = block_det.load_calibration()
    if config is None:
        config = {
            'depth_range_mm': [270, 277],
            'min_area': 1000,
            'max_area': 4000,
            'min_rectangularity': 0.7,
            'validated': False
        }
    
    detector = block_det.BlockDetector(config)
    
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    align = rs.align(rs.stream.color)
    
    try:
        profile = pipeline.start(rs_config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    except RuntimeError as e:
        print(f"\n❌ Failed to start camera: {e}")
        sys.exit(1)
    
    return pipeline, align, depth_scale, config, detector


def capture_stable_snapshot(camera_system):
    """Capture a stable snapshot"""
    print("\n" + "="*70)
    print("CAPTURING STABLE SNAPSHOT")
    print("="*70)
    
    input("\nPress ENTER when ready to capture...")
    
    blocks, debug_img = block_det.capture_snapshot(
        camera_system['pipeline'],
        camera_system['align'],
        camera_system['depth_scale'],
        camera_system['detector'],
        num_samples=50,
        min_detections=10
    )
    
    if blocks:
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_blocks': len(blocks),
            'blocks': blocks
        }
        
        with open(BLOCKS_FILE, 'w') as f:
            json.dump(output, f, indent=4)
        
        print(f"\n{'='*70}")
        print(f"✓ CAPTURED {len(blocks)} blocks")
        print(f"{'='*70}")
        for block in blocks:
            print(f"  Block {block['block_id']}: {block['color']:7s} at "
                  f"X:{block['x_mm']:7.1f} Y:{block['y_mm']:7.1f} mm")
        print(f"{'='*70}")
        
        cv2.namedWindow('Snapshot Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Snapshot Result', 800, 600)
        cv2.imshow('Snapshot Result', debug_img)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow('Snapshot Result')
        
        return blocks
    else:
        print("\n⚠️  No blocks detected!")
        return []


def test_robot_connection(client):
    """Test robot connection"""
    print("\n🔍 Testing robot connection...")
    
    try:
        if not client.is_connected:
            print("❌ ROS bridge not connected")
            return False
        print("✓ ROS bridge connected")
        
        print("  Attempting to read pose...")
        pose = get_end_effector_pose(client, timeout=5.0)
        print(f"✓ Pose: X={pose['position']['x']:.4f}m, Y={pose['position']['y']:.4f}m")
        return True
        
    except TimeoutError:
        print("❌ Timeout: No pose data received")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_robot_movement(client):
    """Test basic robot movement"""
    print("\n" + "="*70)
    print("ROBOT MOVEMENT TEST")
    print("="*70)
    
    try:
        pose = get_end_effector_pose(client, timeout=5.0)
        print(f"✓ Current: X={pose['position']['x']:.4f}m, Y={pose['position']['y']:.4f}m")
        
        target = [pose['position']['x'], pose['position']['y'], pose['position']['z'] + 0.01]
        print(f"  Moving up 10mm...")
        send_target_end_effector_pose(client, target, rpy=(0.0, 0.0, 0.0))
        time.sleep(3.0)
        
        new_pose = get_end_effector_pose(client, timeout=5.0)
        delta_z = new_pose['position']['z'] - pose['position']['z']
        print(f"\nDelta Z: {delta_z*1000:.2f}mm")
        
        if abs(delta_z) > 0.005:
            print("✓ Robot is responding!")
        else:
            print("⚠️  Robot may not be responding")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    input("\nPress ENTER to continue...")


# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================

def autonomous_pick_place_workflow(client, camera_system):
    """Full autonomous pick and place workflow"""
    while True:
        print("\n" + "="*70)
        print("AUTONOMOUS PICK & PLACE WORKFLOW")
        print("="*70)
        print("1. Capture Snapshot")
        print("2. Preview Blocks")
        print("3. Execute Pick & Place")
        print("4. Test Robot Movement")
        print("5. Return to Main Menu")
        print("="*70)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            capture_stable_snapshot(camera_system)
            
        elif choice == '2':
            if not os.path.exists(BLOCKS_FILE):
                print("\n⚠️  No snapshot data.")
                continue
            
            with open(BLOCKS_FILE, 'r') as f:
                data = json.load(f)
            
            blocks = data.get('blocks', [])
            print(f"\n{'='*70}")
            print(f"DETECTED BLOCKS ({len(blocks)} total)")
            print(f"{'='*70}")
            
            if blocks:
                for block in blocks:
                    print(f"  #{block['block_id']}: {block['color']:7s} | "
                          f"X:{block['x_mm']:7.1f} Y:{block['y_mm']:7.1f} mm")
            else:
                print("  (No blocks)")
            print(f"{'='*70}")
            
        elif choice == '3':
            if not os.path.exists(BLOCKS_FILE):
                print("\n⚠️  No snapshot data.")
                continue
            
            with open(BLOCKS_FILE, 'r') as f:
                data = json.load(f)
            
            blocks = data.get('blocks', [])
            
            if not blocks:
                print("\n⚠️  No blocks to pick!")
                continue
            
            print(f"\n{'='*70}")
            print(f"EXECUTING PICK & PLACE FOR {len(blocks)} BLOCKS")
            print(f"{'='*70}")
            
            confirm = input("\nProceed? (Y/n): ").strip().lower()
            if confirm == 'n':
                continue
            
            for i, block in enumerate(blocks, 1):
                print(f"\n{'='*70}")
                print(f"BLOCK {i}/{len(blocks)}: {block['color']}")
                print(f"{'='*70}")
                
                try:
                    block_rotation_rad = block.get('angle_rad', 0.0)
                    time.sleep(2)

                    send_target_end_effector_pose(client, [0.2, 0.2, 0.03], rpy=(0.0, 0.0, 1.8))
                    time.sleep(4)

                    autonomous_pick_and_place(
                        client, block['x_mm'], block['y_mm'], 
                        DROP_LOCATION_MM[0], DROP_LOCATION_MM[1], block_rotation_rad
                    )

                    send_target_end_effector_pose(client, [0.2, 0.2, 0.03], rpy=(0.0, 0.0, 1.8))
                    time.sleep(4)

                    color = block['color']

                    if color == "blue":
                        send_target_end_effector_pose(client, [0.2, 0, 0.03], rpy=(0.0, 0.0, 1.8))
                        time.sleep(4)
                    elif color == "green":
                        send_target_end_effector_pose(client, [0.2, -0.075, 0.03], rpy=(0.0, 0.0, 1.8))
                        time.sleep(4)
                    else:
                        send_target_end_effector_pose(client, [0.2, 0.075, 0.03], rpy=(0.0, 0.0, 1.8))
                        time.sleep(4)

                    dobot_gripper_open(client);   time.sleep(1.5)
                    dobot_gripper_close(client);  time.sleep(1.5)
                    dobot_gripper_release(client);time.sleep(1.0)

                    print(f"✓ Block {i} complete")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    if input("Continue? (Y/n): ").strip().lower() == 'n':
                        break
                
                if i < len(blocks):
                    time.sleep(1.0)
            
            print(f"\n{'='*70}")
            print(" WORKFLOW COMPLETE")
            print(f"{'='*70}")
            input("Press ENTER...")
            
        elif choice == '4':
            test_robot_movement(client)
            
        elif choice == '5':
            break
        
        else:
            print("Invalid option.")


def manual_move_test(client):
    """Manual movement test"""
    print("\n" + "="*70)
    print("MANUAL MOVE TEST")
    print("="*70)
    
    try:
        cam_x = float(input("Camera X (mm): ").strip())
        cam_y = float(input("Camera Y (mm): ").strip())
        
        go_to_transformed_pose(client, cam_x, cam_y, SAFE_Z_HEIGHT_M)
        print("✓ Move complete")
        
    except ValueError:
        print("❌ Invalid input")
    except Exception as e:
        print(f"❌ Move failed: {e}")
    
    input("\nPress ENTER...")


def main_menu(client, camera_system):
    """Main control menu"""
    while True:
        robot_cal_ok, camera_cal_ok = check_calibrations()
        
        cam_status = "✓" if camera_cal_ok else "✗"
        robot_status = "✓" if robot_cal_ok else "✗"
        feed_status = "ON" if not camera_system['thread'].paused else "OFF"
        
        print("\n" + "="*70)
        print("UNIFIED ROBOT CONTROL SYSTEM")
        print("="*70)
        print(f"[Feed: {feed_status} | Camera Cal: {cam_status} | Robot Cal: {robot_status}]")
        print("="*70)
        print("1. Autonomous Pick & Place")
        print("2. Recalibrate Camera")
        print("3. Recalibrate Robot")
        print("4. Manual Move Test")
        print("5. Toggle Camera Feed")
        print("6. Exit")
        print("="*70)
        
        choice = input("Select (1-6): ").strip()
        
        if choice == '1':
            if not robot_cal_ok or not camera_cal_ok:
                print("\n  Calibrations incomplete!")
                if input("Run wizard? (Y/n): ").strip().lower() != 'n':
                    calibration_wizard(client, camera_system)
                continue
            
            autonomous_pick_place_workflow(client, camera_system)
            
        elif choice == '2':
            recalibrate_camera_properly(client, camera_system)
            
        elif choice == '3':
            robot_cal.collect_calibration_data(client)
            
            if os.path.exists(ROBOT_CAL_FILE):
                load_transformation_config(ROBOT_CAL_FILE)
                print("✓ Robot recalibrated")
            
            input("Press ENTER...")
            
        elif choice == '4':
            manual_move_test(client)
            
        elif choice == '5':
            if camera_system['thread'].paused:
                camera_system['thread'].resume()
                print("✓ Camera feed resumed")
            else:
                camera_system['thread'].pause()
                print("✓ Camera feed paused")
            time.sleep(1)
            
        elif choice == '6':
            print("\nShutting down...")
            break
        
        else:
            print("Invalid option.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("UNIFIED ROBOT CONTROL SYSTEM - STARTUP")
    print("="*70)
    
    client = roslibpy.Ros(host=DOBOT_IP, port=DOBOT_PORT)
    
    try:
        client.run()
        if not client.is_connected:
            print("❌ Failed to connect")
            sys.exit(1)
        print("✓ Robot connected")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    if not test_robot_connection(client):
        if input("Continue anyway? (y/N): ").strip().lower() != 'y':
            client.terminate()
            sys.exit(1)
    
    print("\nInitializing camera...")
    try:
        pipeline, align, depth_scale, config, detector = initialize_camera()
        print("✓ Camera initialized")
    except Exception as e:
        print(f" Camera failed: {e}")
        client.terminate()
        sys.exit(1)
    
    robot_cal_ok, camera_cal_ok = check_calibrations()
    
    print("\nCalibration Status:")
    print(f"  Camera: {'✓' if camera_cal_ok else '✗'}")
    print(f"  Robot:  {'✓' if robot_cal_ok else '✗'}")
    
    camera_system = {
        'pipeline': pipeline,
        'align': align,
        'depth_scale': depth_scale,
        'config': config,
        'detector': detector,
        'thread': None
    }
    
    camera_thread = CameraFeedThread(detector, config, pipeline, align, depth_scale)
    camera_thread.start()
    camera_system['thread'] = camera_thread
    
    print("\n✓ Camera feed started")
    time.sleep(1)
    
    if not robot_cal_ok or not camera_cal_ok:
        print("\n  Incomplete calibrations")
        if input("Run wizard? (Y/n): ").strip().lower() != 'n':
            calibration_wizard(client, camera_system)
    
    try:
        main_menu(client, camera_system)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    finally:
        print("\nCleaning up...")
        camera_thread.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
        client.terminate()
        print("✓ Shutdown complete")


if __name__ == '__main__':
    main()