#!/usr/bin/env python3
import argparse
import time
import roslibpy
import json
import os
import numpy as np

# Assuming dobot_control.py is available and contains the necessary functions
from dobot_control import (
    get_end_effector_pose,
    send_target_end_effector_pose,
    dobot_gripper_open,
    dobot_gripper_close,
    dobot_gripper_release,
)

# Crucial: This script must be present and contain the transformation logic
# from transform_camera_to_robot.py (which you should have saved).
try:
    from transform_camera_to_robot import transform_camera_to_robot
except ImportError:
    print("FATAL ERROR: transform_camera_to_robot.py not found.")
    print("Please ensure that file is in the same directory and contains the required function.")
    exit(1)

# --- CONFIGURATION CONSTANTS ---
SAFE_Z_HEIGHT_M = 0.05  # Safe height above the workspace in meters (50mm)
GRIP_Z_HEIGHT_M = -0.025 # Gripping height in meters (e.g., -20mm below base plane)
DOBOT_IP = '10.42.0.1'
DOBOT_PORT = 9090
BLOCKS_JSON_FILE = 'blocks.json'
# --- END CONFIGURATION ---

def go_to_transformed_pose(client, cam_x_mm, cam_y_mm, z_m, rpy_radians=(0.0, 0.0, 0.0)):
    """
    Transforms camera coordinates (mm) to robot coordinates (meters) and sends the move command.
    """
    print(f"--- Transforming Camera Target ({cam_x_mm:.2f}mm, {cam_y_mm:.2f}mm)")

    try:
        rob_x_m, rob_y_m = transform_camera_to_robot(cam_x_mm, cam_y_mm)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    except ValueError as e:
        print(f"❌ Error during transformation: {e}")
        return False

    target_pose = [rob_x_m, rob_y_m, z_m]
    
    print(f"--- Sending Robot Target (X:{rob_x_m:.4f}m, Y:{rob_y_m:.4f}m, Z:{z_m:.4f}m)")
    send_target_end_effector_pose(client, target_pose, rpy=rpy_radians)
    time.sleep(2.0)
    
    current_pose = get_end_effector_pose(client, timeout=2.0)
    print(f"--- Robot At: X:{current_pose['position']['x']:.4f}m, Y:{current_pose['position']['y']:.4f}m")
    
    return True


def autonomous_pick_and_place(client, cam_x_mm, cam_y_mm, drop_x_mm, drop_y_mm):
    """
    Executes a full pick and place cycle for a single target.
    (Requires accurate Z-height, which should be adjusted based on block thickness.)
    """
    print("\n" + "="*50)
    print(f"STARTING PICK CYCLE: Cam Target ({cam_x_mm:.0f}, {cam_y_mm:.0f})")
    print("="*50)

    # 1. Move to a safe hover position over the object
    if not go_to_transformed_pose(client, cam_x_mm, cam_y_mm, SAFE_Z_HEIGHT_M):
        return

    # 2. Open Gripper
    print("→ Opening Gripper")
    dobot_gripper_open(client)
    time.sleep(1.0)
    
    # 3. Descend to grip height
    print("→ Descending to grip height")
    if not go_to_transformed_pose(client, cam_x_mm, cam_y_mm, GRIP_Z_HEIGHT_M):
        return
    time.sleep(1.0)

    # 4. Close Gripper (Pick up)
    print("→ Closing Gripper to grip")
    dobot_gripper_close(client)
    time.sleep(1.5)

    # 5. Lift to safe Z-height
    print("→ Lifting to safe height")
    if not go_to_transformed_pose(client, cam_x_mm, cam_y_mm, SAFE_Z_HEIGHT_M):
        return

    # --- DROP OFF SEQUENCE ---
    print("\n" + "-"*50)
    print(f"STARTING PLACE CYCLE: Drop Target ({drop_x_mm:.0f}, {drop_y_mm:.0f})")
    print("-"*50)

    # 6. Move to safe hover position over the drop location
    if not go_to_transformed_pose(client, drop_x_mm, drop_y_mm, SAFE_Z_HEIGHT_M):
        return

    # 7. Descend to release height (using GRIP_Z_HEIGHT for stability)
    print("→ Descending to release height")
    if not go_to_transformed_pose(client, drop_x_mm, drop_y_mm, GRIP_Z_HEIGHT_M):
        return
    time.sleep(1.0)

    # 8. Release Gripper (Place down)
    print("→ Releasing object")
    dobot_gripper_release(client)
    time.sleep(1.5)

    # 9. Lift back to safe Z-height
    print("→ Lifting back to safe height")
    go_to_transformed_pose(client, drop_x_mm, drop_y_mm, SAFE_Z_HEIGHT_M)
    
    print("\n✓ CYCLE COMPLETE")


def process_blocks_from_json(client):
    """
    Reads block data from the blocks.json file and moves the robot to each target X/Y.
    """
    if not os.path.exists(BLOCKS_JSON_FILE):
        print(f"❌ Error: Block data file '{BLOCKS_JSON_FILE}' not found.")
        return

    print(f"\n--- Loading block data from {BLOCKS_JSON_FILE} ---")
    
    try:
        with open(BLOCKS_JSON_FILE, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to parse JSON in {BLOCKS_JSON_FILE}. Check file integrity.")
        return

    blocks = data.get('blocks', [])
    if not blocks:
        print("⚠️ Warning: JSON file contains no 'blocks' to process.")
        return
    
    print(f"Found {len(blocks)} blocks. Processing movement to X/Y locations...")

    for i, block in enumerate(blocks):
        try:
            cam_x = block['x_mm']
            cam_y = block['y_mm']
            block_id = block.get('block_id', i + 1)
            color = block.get('color', 'unknown')
            
            print(f"\n[BLOCK {block_id}] Color: {color}, Camera X/Y: ({cam_x:.2f}, {cam_y:.2f}) mm")
            
            # Use go_to_transformed_pose to move to the X/Y location at a safe Z
            success = go_to_transformed_pose(client, cam_x, cam_y, SAFE_Z_HEIGHT_M)
            
            if success:
                print(f"✓ Moved successfully to block {block_id} X/Y position.")
            else:
                print(f"❌ Movement failed for block {block_id}.")

            time.sleep(1.0) # Pause between block movements

        except KeyError as e:
            print(f"❌ Data Error in block entry {i+1}: Missing key {e}. Skipping.")
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing block {i+1}: {e}")
            
    print("\n--- Finished processing all blocks in JSON file. ---")


def main():
    client = roslibpy.Ros(host=DOBOT_IP, port=DOBOT_PORT)
    print(f"Attempting to connect to ROS bridge at ws://{DOBOT_IP}:{DOBOT_PORT}...")
    
    try:
        client.run()
        if client.is_connected:
            print("✓ Connection successful.")
        else:
            print("❌ Failed to connect to ROS bridge. Check IP/port and service status.")
            return
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    while True:
        print("\n" + "="*40)
        print("DOBOT AUTOMATION TOOL MENU")
        print("="*40)
        print(" 1. Manual Move (Input Camera X/Y)")
        print(f" 2. Process Blocks from JSON ({BLOCKS_JSON_FILE})")
        print(" 3. Exit")
        print("="*40)
        
        mode = input("Select an operation (1-3): ").strip()
        
        if mode == '1':
            try:
                cam_x = float(input("Enter Target Camera X (mm): "))
                cam_y = float(input("Enter Target Camera Y (mm): "))
                # Move only to the X/Y position at a safe height
                go_to_transformed_pose(client, cam_x, cam_y, SAFE_Z_HEIGHT_M)
            except ValueError:
                print("Invalid input. Please enter numbers only.")
        
        elif mode == '2':
            process_blocks_from_json(client)
            
        elif mode == '3':
            print("Exiting application.")
            break
            
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")

    client.terminate()


if __name__ == '__main__':
    main()
