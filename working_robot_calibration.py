#!/usr/bin/env python3
import time
import roslibpy
import os
import json
import numpy as np
import pandas as pd # Used for efficient matrix setup and data processing

# Ensure you have a working dobot_control.py module accessible
from dobot_control import (
    get_end_effector_pose,
    send_target_end_effector_pose,
    dobot_gripper_open,
    dobot_gripper_close,
    dobot_gripper_release,
)

# --- CONFIGURATION ---
CALIBRATION_TARGETS_MM = [
    (0.0, 0.0),      # P0: Camera Origin (Base Point)
    (50.0, 50.0),  # P1
    (50.0, -50.0), # P2
    (-50.0, -50.0),# P3
    (-50.0, 50.0)  # P4
]

OUTPUT_CONFIG_FILE = 'robot_transform_matrix.json'
# ---------------------

def solve_affine_transformation(raw_data):
    """
    Performs the 2D-to-2D Affine transformation calculation on the collected data.
    
    Args:
        raw_data (list): List of lists containing the header and all collected points.
    """
    
    # 1. Convert raw data (list of lists) into a Pandas DataFrame
    header = raw_data[0]
    data_points = raw_data[1:]
    
    if len(data_points) < 5:
        print("❌ ERROR: Only collected a header or not enough points (need 5 total). Aborting solver.")
        return

    df = pd.DataFrame(data_points, columns=header)
    
    # 2. Establish Base Origin (P0) and Calculate Displacements
    origin_data = df[df['Point_ID'] == 0].iloc[0]
    robot_x_base = origin_data['Robot_X_Pos_mm']
    robot_y_base = origin_data['Robot_Y_Pos_mm']
    
    print(f"\n✓ Robot Base Origin (from P0): X={robot_x_base:.3f}, Y={robot_y_base:.3f}")
    
    # Calculate Delta (Displacement) for all non-origin points (P1-P4)
    df_fit = df[df['Point_ID'] != 0].copy()
    
    df_fit['Delta_X_Robot'] = df_fit['Robot_X_Pos_mm'] - robot_x_base
    df_fit['Delta_Y_Robot'] = df_fit['Robot_Y_Pos_mm'] - robot_y_base
    
    # 3. Define the Matrices for Least Squares
    
    # C: Input Camera Coordinates (X_C, Y_C) for P1-P4
    C = df_fit[['X_Camera_Target_mm', 'Y_Camera_Target_mm']].values 

    # X_R, Y_R: Target Robot Displacements (for P1-P4)
    X_R_vector = df_fit['Delta_X_Robot'].values 
    Y_R_vector = df_fit['Delta_Y_Robot'].values 

    # 4. Solve the System (Least Squares)
    C_pinv = np.linalg.pinv(C)

    # Solve for M_X coefficients (A, B)
    M_X_coeffs = C_pinv @ X_R_vector
    A, B = M_X_coeffs

    # Solve for M_Y coefficients (C, D)
    M_Y_coeffs = C_pinv @ Y_R_vector
    C_coeff, D = M_Y_coeffs

    # 5. Form the Final Transformation Matrix M
    M_transform = np.array([[A, B], [C_coeff, D]])

    # 6. Save the results
    output_config = {
        'transformation_matrix': M_transform.tolist(),
        'A': round(A, 6), 'B': round(B, 6),
        'C': round(C_coeff, 6), 'D': round(D, 6),
        'robot_base_x_mm': round(robot_x_base, 3),
        'robot_base_y_mm': round(robot_y_base, 3),
        'calibration_points_used': C.shape[0]
    }
    
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        json.dump(output_config, f, indent=4)

    print("\n" + "="*80)
    print("✅ AFFINE TRANSFORMATION SOLVED AND SAVED")
    print(f"Configuration saved to: {os.path.abspath(OUTPUT_CONFIG_FILE)}")
    print("\nCalculated Transformation Matrix (M):")
    print(M_transform.round(6))
    print("\n--- Final Robot Translation Formulas ---")
    print(f"Robot_X_Target = Robot_X_Base ({robot_x_base:.3f} mm) + (({A:.6f} * X_Camera) + ({B:.6f} * Y_Camera))")
    print(f"Robot_Y_Target = Robot_Y_Base ({robot_y_base:.3f} mm) + (({C_coeff:.6f} * X_Camera) + ({D:.6f} * Y_Camera))")
    print("="*80)


def collect_calibration_data(client):
    """
    Guides the user to position the Dobot and collects data in memory.
    """
    
    if not client.is_connected:
        print("❌ CRITICAL ERROR: ROS bridge client is not connected.")
        print("Please ensure the ROS bridge is running and the IP address is correct.")
        return

    data = []
    
    print("\n" + "="*80)
    print("STARTING 2D-TO-2D AFFINE CALIBRATION DATA COLLECTION (5 POINTS)")
    print("="*80)
    
    # Header for the data structure
    data.append([
        "Point_ID", 
        "X_Camera_Target_mm", 
        "Y_Camera_Target_mm", 
        "Robot_X_Pos_mm", 
        "Robot_Y_Pos_mm",
        "Robot_Z_Pos_mm" 
    ])

    all_data_collected = True
    for i, (cam_x, cam_y) in enumerate(CALIBRATION_TARGETS_MM):
        point_id = i
        
        print(f"\n--- Point {point_id} (P{point_id}) ---")
        print(f"REQUIRED CAMERA POSITION: X={cam_x:.1f} mm, Y={cam_y:.1f} mm")
        
        input(f"Press Enter when the Dobot TCP is perfectly centered on the target P{point_id}...")
        
        try:
            pose_data = get_end_effector_pose(client, timeout=5.0)
            
            # --- FIX: Check type and access dictionary keys instead of list indices ---
            if not isinstance(pose_data, dict) or 'position' not in pose_data:
                 raise ValueError("get_end_effector_pose returned invalid or unusable data structure.")

            # Extract data from the dictionary (converts meters to millimeters)
            robot_x_mm = pose_data['position']['x'] * 1000.0
            robot_y_mm = pose_data['position']['y'] * 1000.0
            robot_z_mm = pose_data['position']['z'] * 1000.0
            # -------------------------------------------------------------------------

            print(f"-> RECORDED ROBOT POSITION (mm): X={robot_x_mm:.3f}, Y={robot_y_mm:.3f}, Z={robot_z_mm:.3f}")

            # Store the data point
            data.append([
                point_id, 
                cam_x, 
                cam_y, 
                robot_x_mm, 
                robot_y_mm,
                robot_z_mm
            ])

        except Exception as e:
            # We explicitly check for common failures above, so any remaining '0' or low-level error 
            # likely points to a network or ROS communication issue.
            print(f"❌ FAILED TO RECORD POSITION for P{point_id}. Error: {e}")
            all_data_collected = False
            break # Stop collection if we can't read a single point

    
    if all_data_collected and len(data) == len(CALIBRATION_TARGETS_MM) + 1:
        print("\nDATA COLLECTION SUCCESSFUL. Starting calculation...")
        solve_affine_transformation(data)
    else:
        print("\n" + "="*80)
        print("⚠️ CALIBRATION FAILED: Not all data points were collected due to errors.")
        print("If this error persists, ensure your ROS bridge is running and publishing poses to the correct topic.")
        print("="*80)


def main():
    client = roslibpy.Ros(host='10.42.0.1', port=9090) 
    client.run()

    try:
        collect_calibration_data(client) 
        
    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
    finally:
        if client.is_connected:
            client.terminate()

if __name__ == '__main__':
    main()
