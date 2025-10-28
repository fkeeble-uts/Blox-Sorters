
"""
Main orchestrator for the pick-and-place robot system.
Streamlined workflow: Calibrate -> Detect -> Pick -> Place
"""
import time
import sys
from pathlib import Path
import roslibpy
import cv2

# Import custom modules
from Archive.camera_vision import VisionSystem
from calibration import CameraCalibration, RobotCalibration
from transform_camera_to_robot import transform_camera_to_robot
from calibration_utils import run_interactive_calibration
from dobot_control import (
    get_end_effector_pose,
    send_target_end_effector_pose,
    dobot_gripper_open,
    dobot_gripper_close,
    dobot_gripper_release,
)

# Configuration
DOBOT_IP = '10.42.0.1'
DOBOT_PORT = 9090
SAFE_Z_HEIGHT_M = 0.05  # 50mm above workspace
GRIP_Z_HEIGHT_M = -0.025  # 25mm below base for gripping
DROP_LOCATION_MM = (100.0, 100.0)  # Default drop-off location in camera coords


def move_to_camera_coords(client, cam_x_mm, cam_y_mm, z_m, rpy=(0.0, 0.0, 0.0)):
    """Transform camera coords to robot coords and move"""
    try:
        rob_x_m, rob_y_m = transform_camera_to_robot(cam_x_mm, cam_y_mm)
        
        if rob_x_m is None or rob_y_m is None:
            print("✗ Coordinate transformation failed")
            return False
        
        print(f"   Camera ({cam_x_mm:.1f}, {cam_y_mm:.1f})mm → Robot ({rob_x_m:.4f}, {rob_y_m:.4f})m")
        
        target_pose = [rob_x_m, rob_y_m, z_m]
        send_target_end_effector_pose(client, target_pose, rpy=rpy)
        time.sleep(2.0)
        return True
        
    except Exception as e:
        print(f"✗ Movement error: {e}")
        return False


def pick_and_place_block(client, block, drop_x_mm, drop_y_mm):
    """Execute pick and place for a single block"""
    cam_x = block['x_mm']
    cam_y = block['y_mm']
    
    print(f"\n{'='*50}")
    print(f"PICKING Block {block['block_id']} ({block['color']})")
    print(f"{'='*50}")
    
    # Move to hover position above block
    print("→ Moving to hover position...")
    if not move_to_camera_coords(client, cam_x, cam_y, SAFE_Z_HEIGHT_M):
        print("✗ Failed to move to hover position")
        return False
    
    # Open gripper
    print("→ Opening gripper...")
    dobot_gripper_open(client)
    time.sleep(1.0)
    
    # Descend to grip
    print("→ Descending to grip...")
    if not move_to_camera_coords(client, cam_x, cam_y, GRIP_Z_HEIGHT_M):
        print("✗ Failed to descend")
        return False
    time.sleep(1.0)
    
    # Close gripper
    print("→ Gripping...")
    dobot_gripper_close(client)
    time.sleep(1.5)
    
    # Lift to safe height
    print("→ Lifting...")
    move_to_camera_coords(client, cam_x, cam_y, SAFE_Z_HEIGHT_M)
    
    # Move to drop location
    print(f"\n{'-'*50}")
    print(f"PLACING at ({drop_x_mm:.0f}, {drop_y_mm:.0f})")
    print(f"{'-'*50}")
    
    print("→ Moving to drop position...")
    if not move_to_camera_coords(client, drop_x_mm, drop_y_mm, SAFE_Z_HEIGHT_M):
        print("✗ Failed to move to drop position")
        return False
    
    # Descend to release
    print("→ Descending...")
    move_to_camera_coords(client, drop_x_mm, drop_y_mm, GRIP_Z_HEIGHT_M)
    time.sleep(1.0)
    
    # Release
    print("→ Releasing...")
    dobot_gripper_release(client)
    time.sleep(1.5)
    
    # Lift back up
    print("→ Lifting...")
    move_to_camera_coords(client, drop_x_mm, drop_y_mm, SAFE_Z_HEIGHT_M)
    
    print("\n✓ PICK AND PLACE COMPLETE")
    return True


def capture_blocks(vision):
    """Capture and detect blocks"""
    print("\n" + "="*70)
    print("BLOCK DETECTION")
    print("="*70)
    print("\nInstructions:")
    print("  1. Ensure robot arm is NOT blocking camera view")
    print("  2. Place blocks in the workspace")
    print("  3. Press ENTER when ready")
    
    input("\nPress ENTER to capture...")
    
    blocks = vision.capture_stable_snapshot()
    
    if not blocks:
        print("✗ No blocks detected")
        return []
    
    print(f"\n✓ Detected {len(blocks)} blocks:")
    for block in blocks:
        print(f"  Block {block['block_id']}: {block['color']:7s} at "
              f"X:{block['x_mm']:7.1f} Y:{block['y_mm']:7.1f} mm, "
              f"Angle:{block['rotation_angle_deg']:6.1f}°")
    
    # Show preview
    #vision.show_last_detection()
    
    return blocks


def run_calibration_workflow(client):
    """Complete calibration workflow"""
    print("\n" + "="*70)
    print("CALIBRATION WORKFLOW")
    print("="*70)
    
    camera_calibrated = Path('camera_calibration.json').exists()
    robot_calibrated = Path('robot_transform_matrix.json').exists()
    
    # Step 1: Camera calibration
    if not camera_calibrated:
        print("\n[STEP 1/2] Camera Calibration")
        print("This will launch the diagnostic tool to tune detection parameters.")
        input("Press ENTER to continue...")
        
        cam_cal = CameraCalibration()
        if cam_cal.run_diagnostic_calibration():
            print("✓ Camera calibration complete")
        else:
            print("✗ Camera calibration failed")
            return False
    else:
        print("\n[STEP 1/2] Camera Calibration - SKIPPED (already calibrated)")
    
    # Step 2: Robot calibration
    if not robot_calibrated:
        print("\n[STEP 2/2] Robot Calibration")
        print("You'll guide the robot to 5 known positions to calculate the transform.")
        input("Press ENTER to continue...")
        
        rob_cal = RobotCalibration(client)
        if rob_cal.collect_and_solve():
            print("✓ Robot calibration complete")
        else:
            print("✗ Robot calibration failed")
            return False
    else:
        print("\n[STEP 2/2] Robot Calibration - SKIPPED (already calibrated)")
    
    print("\n" + "="*70)
    print("✓ ALL CALIBRATIONS COMPLETE")
    print("="*70)
    return True


def run_autonomous_cycle(client, vision):
    """Run complete autonomous pick and place cycle"""
    print("\n" + "="*70)
    print("AUTONOMOUS PICK-AND-PLACE CYCLE")
    print("="*70)
    
    # Get drop location
    try:
        drop_x = float(input(f"\nEnter drop X coordinate (mm) [{DROP_LOCATION_MM[0]}]: ") 
                      or DROP_LOCATION_MM[0])
        drop_y = float(input(f"Enter drop Y coordinate (mm) [{DROP_LOCATION_MM[1]}]: ") 
                      or DROP_LOCATION_MM[1])
    except ValueError:
        print("Invalid input, using defaults")
        drop_x, drop_y = DROP_LOCATION_MM
    
    while True:
        # Detect blocks
        blocks = capture_blocks(vision)
        
        if not blocks:
            print("\nNo more blocks detected. Cycle complete!")
            break
        
        # Pick the first block
        block = blocks[0]
        
        confirm = input(f"\nPick block {block['block_id']} ({block['color']})? (y/n/q): ").lower()
        if confirm == 'q':
            print("Cycle cancelled by user")
            break
        elif confirm != 'y':
            print("Skipping block")
            continue
        
        # Execute pick and place
        success = pick_and_place_block(client, block, drop_x, drop_y)
        
        if not success:
            retry = input("\nOperation failed. Retry? (y/n): ").lower()
            if retry != 'y':
                break
        
        # Ask if user wants to continue
        cont = input("\nContinue to next block? (y/n): ").lower()
        if cont != 'y':
            break
    
    print("\n✓ Autonomous cycle finished")


def run_manual_mode(client, vision):
    """Manual control mode"""
    print("\n" + "="*70)
    print("MANUAL CONTROL MODE")
    print("="*70)
    
    while True:
        print("\nOptions:")
        print("  1. Capture blocks (no action)")
        print("  2. Move to camera coordinates")
        print("  3. Pick and place single block")
        print("  4. Test gripper")
        print("  5. Return to main menu")
        
        choice = input("\nSelect (1-5): ").strip()
        
        if choice == '1':
            capture_blocks(vision)
        
        elif choice == '2':
            try:
                cam_x = float(input("Camera X (mm): "))
                cam_y = float(input("Camera Y (mm): "))
                move_to_camera_coords(client, cam_x, cam_y, SAFE_Z_HEIGHT_M)
            except ValueError:
                print("Invalid input")
        
        elif choice == '3':
            blocks = capture_blocks(vision)
            if blocks:
                try:
                    block_num = int(input(f"Pick which block (1-{len(blocks)}): "))
                    if 1 <= block_num <= len(blocks):
                        drop_x, drop_y = DROP_LOCATION_MM
                        pick_and_place_block(client, blocks[block_num-1], drop_x, drop_y)
                    else:
                        print("Invalid block number")
                except ValueError:
                    print("Invalid input")
        
        elif choice == '4':
            print("Testing gripper...")
            dobot_gripper_open(client)
            time.sleep(1)
            dobot_gripper_close(client)
            time.sleep(1)
            dobot_gripper_release(client)
            print("✓ Gripper test complete")
        
        elif choice == '5':
            break
        
        else:
            print("Invalid selection")


def main():
    # Connect to robot
    client = roslibpy.Ros(host=DOBOT_IP, port=DOBOT_PORT)
    print(f"Attempting to connect to ROS bridge at ws://{DOBOT_IP}:{DOBOT_PORT}...")
    
    try:
        client.run()
        if client.is_connected:
            print("✓ Robot connection successful")
        else:
            print("✗ Failed to connect to ROS bridge. Check IP/port and service status.")
            return 1
    except Exception as e:
        print(f"✗ Robot connection failed: {e}")
        return 1
    
    # Initialize vision system
    vision = None
    try:
        print("Initializing camera...")
        vision = VisionSystem()
        print("✓ Camera initialized")
    except Exception as e:
        print(f"⚠ Camera initialization failed: {e}")
        print("Vision features will be unavailable")
    
    # Check calibration status
    print("\n" + "="*70)
    print("PICK-AND-PLACE ROBOT SYSTEM")
    print("="*70)
    
    camera_calibrated = Path('camera_calibration.json').exists()
    robot_calibrated = Path('robot_transform_matrix.json').exists()
    
    print("\nSystem Status:")
    print(f"  Camera Calibration: {'✓ Found' if camera_calibrated else '✗ Missing'}")
    print(f"  Robot Calibration:  {'✓ Found' if robot_calibrated else '✗ Missing'}")
    
    # Main menu loop
    try:
        while True:
            print("\n" + "="*70)
            print("MAIN MENU")
            print("="*70)
            print("  1. Run Calibration (camera + robot)")
            print("  2. Autonomous Pick-and-Place Cycle")
            print("  3. Manual Control Mode")
            print("  4. Recalibrate Camera Only")
            print("  5. Recalibrate Robot Only")
            if vision:
                status = "ON" if vision.live_view_active else "OFF"
                print(f"  6. Toggle Live Camera View (Currently: {status})")
            print("  7. Exit")
            print("="*70)
            
            choice = input("\nSelect operation (1-7): ").strip()
            
            if choice == '1':
                if vision:
                    vision.stop_live_view()  # Stop live view during calibration
                run_calibration_workflow(client)
                # Refresh calibration status
                camera_calibrated = Path('camera_calibration.json').exists()
                robot_calibrated = Path('robot_transform_matrix.json').exists()
                # Reload vision if camera was calibrated
                if camera_calibrated and vision:
                    vision.cleanup()
                    vision = VisionSystem()
            
            elif choice == '2':
                if not (camera_calibrated and robot_calibrated):
                    print("\n✗ System not calibrated. Run calibration first (Option 1)")
                    continue
                if not vision:
                    print("\n✗ Camera not available")
                    continue
                run_autonomous_cycle(client, vision)
            
            elif choice == '3':
                if not (camera_calibrated and robot_calibrated):
                    print("\n✗ System not calibrated. Run calibration first (Option 1)")
                    continue
                if not vision:
                    print("\n✗ Camera not available")
                    continue
                run_manual_mode(client, vision)
            
            elif choice == '4':
                if vision:
                    vision.stop_live_view()  # Stop live view during calibration
                cam_cal = CameraCalibration()
                if cam_cal.run_diagnostic_calibration():
                    camera_calibrated = True
                    # Reload vision system with new calibration
                    if vision:
                        vision.cleanup()
                    vision = VisionSystem()
            
            elif choice == '5':
                rob_cal = RobotCalibration(client)
                if rob_cal.collect_and_solve():
                    robot_calibrated = True
            
            elif choice == '6':
                if vision:
                    vision.toggle_live_view()
                else:
                    print("\n✗ Camera not available")
            
            elif choice == '7':
                print("\nShutting down...")
                break
            
            else:
                print("Invalid selection")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        if vision:
            vision.cleanup()
        if client.is_connected:
            client.terminate()
        print("✓ System shutdown complete")

        cv2.destroyAllWindows()
    
    return 0



if __name__ == '__main__':
    sys.exit(main())