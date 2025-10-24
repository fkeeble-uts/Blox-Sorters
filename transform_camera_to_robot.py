import json
import numpy as np
import os

# --- GLOBAL VARIABLES TO HOLD THE CALIBRATION DATA ---
M = None
BASE_OFFSET = None
CONFIG_FILE = 'robot_transform_matrix.json'
# ----------------------------------------------------

def load_transformation_config(filepath):
    """Loads the transformation matrix and base offset from the JSON file."""
    global M, BASE_OFFSET
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Load the 2x2 matrix and base offset
        M = np.array(config['transformation_matrix'])
        base_offset = np.array([config['robot_base_x_mm'], config['robot_base_y_mm']])
        BASE_OFFSET = base_offset
        
        print(f"✓ Calibration data loaded successfully from {os.path.basename(filepath)}")
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{filepath}' not found. Please run calibration first.")
        M = None
        BASE_OFFSET = None
        return False
    except KeyError as e:
        print(f"❌ Error: Configuration file '{filepath}' is corrupt or missing key: {e}")
        M = None
        BASE_OFFSET = None
        return False

# Load the configuration immediately when this module is imported
load_transformation_config(CONFIG_FILE)


def transform_camera_to_robot(camera_x_mm, camera_y_mm):
    """
    Applies the affine transformation (M) and translation (BASE_OFFSET)
    to convert camera coordinates (mm) to robot base coordinates (m).
    
    Returns: (target_x_m, target_y_m) or (None, None) on failure.
    """
    if M is None:
        # Configuration loading failed, try loading again just in case
        if not load_transformation_config(CONFIG_FILE):
             return None, None
    
    # 1. Input vector from the camera (displacement from camera origin)
    camera_vector = np.array([camera_x_mm, camera_y_mm])
    
    # 2. Apply the transformation (rotation, scale, skew)
    # R_disp = M * C_disp
    robot_displacement_vector = M @ camera_vector
    
    # 3. Add the robot's base offset (translation)
    # R_target = R_base + R_disp
    robot_target_position_mm = BASE_OFFSET + robot_displacement_vector
    
    # 4. Convert millimeters to meters (as ROS commands expect meters)
    target_x_m = robot_target_position_mm[0] / 1000.0
    target_y_m = robot_target_position_mm[1] / 1000.0
    
    return target_x_m, target_y_m
