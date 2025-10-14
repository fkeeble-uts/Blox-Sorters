#!/usr/bin/env python3
import time
import roslibpy
import threading
from spatialmath import SE3
from math import pi

current_pose = None

# --- Dobot Magician gripper helpers (roslibpy) ---
TOOL_TOPIC = '/dobot_magician/target_tool_state'
TOOL_MSG_TYPE = 'std_msgs/Int32MultiArray'  # or 'std_msgs/Int8MultiArray'

def _tool_topic(client):
    return roslibpy.Topic(client, TOOL_TOPIC, TOOL_MSG_TYPE)

def dobot_gripper_set(client, pump: int, grip: int):
    topic = _tool_topic(client)
    topic.advertise()
    topic.publish(roslibpy.Message({'data': [int(pump), int(grip)]}))
    topic.unadvertise()

def dobot_gripper_open(client):
    dobot_gripper_set(client, 1, 0)

def dobot_gripper_close(client):
    dobot_gripper_set(client, 1, 1)

def dobot_gripper_release(client):
    dobot_gripper_set(client, 0, 0)

# -----------------------------------------------------------------
# --- End-effector pose subscriber and monitor ---

def end_effector_pose_cb(message):
    """Callback for /dobot_magician/end_effector_poses topic."""
    global current_pose
    # Extract pose data from message
    pose = message['pose']
    pos = pose['position']
    ori = pose['orientation']

    current_pose = {
        'x': pos['x'],
        'y': pos['y'],
        'z': pos['z'],
        'qx': ori['x'],
        'qy': ori['y'],
        'qz': ori['z'],
        'qw': ori['w'],
    }

def print_pose(pose):
    """Prints position + orientation (converted to Euler)."""
    try:
        # Construct SE3 from quaternion + translation
        T = SE3.Quaternion([pose['qw'], pose['qx'], pose['qy'], pose['qz']], t=[pose['x'], pose['y'], pose['z']])
        eul = T.rpy('deg')
        print(f"[POSE] Position (m): x={pose['x']:.3f}, y={pose['y']:.3f}, z={pose['z']:.3f}")
        print(f"[POSE] Orientation (deg): roll={eul[0]:.1f}, pitch={eul[1]:.1f}, yaw={eul[2]:.1f}")
        print("-" * 50)
    except Exception as e:
        print(f"[POSE ERROR] {e}")

def start_pose_monitor(interval=0.5):
    """Thread that prints the latest end-effector pose every `interval` seconds."""
    def monitor():
        while True:
            if current_pose:
                print_pose(current_pose)
            time.sleep(interval)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

# -----------------------------------------------------------------

def move_dobot_joint_positions(client, joint_positions):
    """Send one JointTrajectoryPoint to /dobot_magician/target_joint_states."""
    pub = roslibpy.Topic(client,
                         '/dobot_magician/target_joint_states',
                         'trajectory_msgs/JointTrajectory')
    pub.advertise()
    msg = {'points': [{'positions': [float(x) for x in joint_positions]}]}
    pub.publish(roslibpy.Message(msg))
    pub.unadvertise()

# -----------------------------------------------------------------

if __name__ == '__main__':
    client = roslibpy.Ros(host='10.42.0.1', port=9090)
    client.run()

    # Subscribe to end effector pose
    sub_pose = roslibpy.Topic(client, '/dobot_magician/end_effector_poses', 'geometry_msgs/PoseStamped')
    sub_pose.subscribe(end_effector_pose_cb)

    # Start periodic pose printing
    start_pose_monitor(interval=0.5)

    steps = 10

    try:
        # Example sequence
        move_dobot_joint_positions(client, [1.5, pi/6, 0.3, 0.0])
        #for i in range(steps):
            #move_dobot_joint_positions(client, [-i/steps * pi/3, pi/6-i/steps * pi/3, 0.3, 0.0])
        #for i in range(steps):
            #move_dobot_joint_positions(client, [-pi/3 + i/steps * pi/3, pi/6 - pi/3 - i/steps * pi/3, 0.3, 0.0])
        #for i in range(steps):
            #move_dobot_joint_positions(client, [i/steps * pi/3, pi/6 - 2*pi/3 + i/steps * pi/3, 0.3, 0.0])
        #for i in range(steps):
            #move_dobot_joint_positions(client, [pi/3 - i/steps * pi/3, pi/6 -pi/3 + i/steps * pi/3, 0.3, 0.0])
        time.sleep(2)

        dobot_gripper_open(client)
        time.sleep(2)

        dobot_gripper_close(client)
        time.sleep(2)

        dobot_gripper_release(client)
        time.sleep(2)

        move_dobot_joint_positions(client, [pi/2, 0.4, 0.3, 0.0])
        time.sleep(2)

        dobot_gripper_open(client)
        time.sleep(2)

        dobot_gripper_close(client)
        time.sleep(2)

        dobot_gripper_release(client)
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
        sub_pose.unsubscribe()