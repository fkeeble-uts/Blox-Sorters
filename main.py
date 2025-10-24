#!/usr/bin/env python3
import argparse
import time
import roslibpy

from dobot_control import (
    get_end_effector_pose,
    send_target_end_effector_pose,
    dobot_gripper_open,
    dobot_gripper_close,
    dobot_gripper_release,
    move_dobot_joint_positions,   # available if/when you want joint moves
)

# If you have a separate camera module, expose a clean function there and import it here:
# from camera_control import find_target_xyz_rpy  # (xyz in metres, rpy in radians)

def run_demo_sequence(client):
    """Replicates your previous demo sequence using the shared dobot_control library."""
    pose = get_end_effector_pose(client, timeout=2.0)
    print('[EE Pose] Start:', pose)
    time.sleep(1.0)

    print('----------------------------------------------')
    print('position 1')
    send_target_end_effector_pose(client, [0.10, 0.0, -0.02], rpy=(0.0, 0.0, 1.8))
    time.sleep(2.0)
    print('[EE Pose] After pos1:', get_end_effector_pose(client, timeout=2.0))

    print('----------------------------------------------')
    print('position 2')
    send_target_end_effector_pose(client, [0.10, -0.24, -0.02], rpy=(0.0, 0.0, -0.02))
    time.sleep(2.0)
    print('[EE Pose] After pos2:', get_end_effector_pose(client, timeout=2.0))

    print('----------------------------------------------')
    print('position 3')
    send_target_end_effector_pose(client, [0.25, 0.0, -0.02], rpy=(0.0, 0.0, -0.02))
    time.sleep(2.0)
    print('[EE Pose] After pos3:', get_end_effector_pose(client, timeout=2.0))

    print('----------------------------------------------')
    print('position 4')
    send_target_end_effector_pose(client, [0.10, 0.24, -0.02], rpy=(0.0, 0.0, -0.02))
    time.sleep(2.0)
    print('[EE Pose] After pos4:', get_end_effector_pose(client, timeout=2.0))

    print('----------------------------------------------')
    print('position 1 (return)')
    send_target_end_effector_pose(client, [0.10, 0.0, -0.02], rpy=(0.0, 0.0, -0.02))
    time.sleep(2.0)
    print('[EE Pose] After return:', get_end_effector_pose(client, timeout=2.0))

    # Gripper demo
    print('Gripper: OPEN → CLOSE → RELEASE')
    dobot_gripper_open(client);   time.sleep(1.5)
    dobot_gripper_close(client);  time.sleep(1.5)
    dobot_gripper_release(client);time.sleep(1.0)


def pickUpLego(client):
    """Replicates your previous demo sequence using the shared dobot_control library."""
    pose = get_end_effector_pose(client, timeout=2.0)
    print('[EE Pose] Start:', pose)
    time.sleep(1.0)

    print('----------------------------------------------')
    print('position 1')
    send_target_end_effector_pose(client, [0.10, 0.0, 0.03], rpy=(0.0, 0.0, 1.8))
    time.sleep(2.0)
    print('[EE Pose] After pos1:', get_end_effector_pose(client, timeout=2.0))

    dobot_gripper_open(client);   time.sleep(1.5)

    print('----------------------------------------------')
    print('position 2')
    send_target_end_effector_pose(client, [0.10, 0, -0.02], rpy=(0.0, 0.0, -0.02))
    time.sleep(2.0)
    print('[EE Pose] After pos2:', get_end_effector_pose(client, timeout=2.0))

    dobot_gripper_close(client);  time.sleep(1.5)

    print('----------------------------------------------')
    print('position 3')
    send_target_end_effector_pose(client, [0.10, 0.0, 0.03], rpy=(0.0, 0.0, 1.8))
    time.sleep(2.0)
    print('[EE Pose] After pos1:', get_end_effector_pose(client, timeout=2.0))

    print('----------------------------------------------')
    print('position 4')
    send_target_end_effector_pose(client, [0.10, 0.24, 0.03], rpy=(0.0, 0.0, -0.02))
    time.sleep(3.0)
    print('[EE Pose] After pos4:', get_end_effector_pose(client, timeout=2.0))

    dobot_gripper_open(client);   time.sleep(1.5)
    dobot_gripper_close(client);  time.sleep(1.5)
    dobot_gripper_release(client);time.sleep(1.0)

    print('----------------------------------------------')
    print('position 1 (return)')
    send_target_end_effector_pose(client, [0.10, 0.0, 0.03], rpy=(0.0, 0.0, -0.02))
    time.sleep(2.0)
    print('[EE Pose] After return:', get_end_effector_pose(client, timeout=2.0))

    # # Gripper demo
    # print('Gripper: OPEN → CLOSE → RELEASE')
    # dobot_gripper_open(client);   time.sleep(1.5)
    # dobot_gripper_close(client);  time.sleep(1.5)
    # dobot_gripper_release(client);time.sleep(1.0)

def main():
    client = roslibpy.Ros(host='10.42.0.1', port=9090)  # Replace with your ROS bridge IP
    client.run()

    try:
        # run_demo_sequence(client)
        # print('[EE Pose] After return:', get_end_effector_pose(client, timeout=2.0))
        pickUpLego(client)
        pickUpLego(client)

    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")

if __name__ == '__main__':
    main()
