#!/usr/bin/env python3
import time
import roslibpy
import numpy as np
from math import pi

current_pos = None

# --- Dobot Magician gripper helpers (roslibpy) ---
# Verify the message type once with:
#   rostopic info /dobot_magician/target_tool_state
# If it reports std_msgs/Int8MultiArray, change TOOL_MSG_TYPE accordingly.
TOOL_TOPIC = '/dobot_magician/target_tool_state'
TOOL_MSG_TYPE = 'std_msgs/Int32MultiArray'   # or 'std_msgs/Int8MultiArray'

def _tool_topic(client):
    return roslibpy.Topic(client, TOOL_TOPIC, TOOL_MSG_TYPE)

def dobot_gripper_set(client, pump: int, grip: int):
    """Publish [pump, grip] to the tool state.
       pump: 0=off, 1=on
       grip: 0=open, 1=close
    """
    topic = _tool_topic(client)
    topic.advertise()
    topic.publish(roslibpy.Message({'data': [int(pump), int(grip)]}))
    topic.unadvertise()

def dobot_gripper_open(client):
    """Pump ON, gripper OPEN — handy before picking or to release while keeping vacuum ready."""
    dobot_gripper_set(client, 1, 0)

def dobot_gripper_close(client):
    """Pump ON, gripper CLOSE — grasp an object."""
    dobot_gripper_set(client, 1, 1)

def dobot_gripper_release(client):
    """Pump OFF, gripper OPEN — fully release and stop the pump."""
    dobot_gripper_set(client, 0, 0)

#-----------------------------------------------------------------

def call_service(client, service_name):
    """Call a std_srvs/Trigger service and print the response."""
    service = roslibpy.Service(client, service_name, 'std_srvs/Trigger')
    request = roslibpy.ServiceRequest({})  # Trigger service takes no arguments

    print(f"[ROS] Calling service: {service_name}")
    result = service.call(request)
    print(f"[ROS] Response: success={result['success']}, message='{result['message']}'")

def joint_state_cb(message):
    global current_pos
    # The JointState message contains 'position' array
    current_pos = list(message['position'])

def move_dobot_joint_positions(client, joint_positions):
    """
    Publish a single JointTrajectoryPoint to /dobot_magician/target_joint_states.
    joint_positions: iterable of 4 floats (radians) → [j1, j2, j3, j4]
    """
    if len(joint_positions) != 4:
        raise ValueError(f"Dobot expects 4 joints, got {len(joint_positions)}")

    pub = roslibpy.Topic(client,
                         '/dobot_magician/target_joint_states',
                         'trajectory_msgs/JointTrajectory')
    pub.advertise()

    # Minimal message (matches the MATLAB example: just one point with Positions)
    msg = {
        # joint_names is optional for this driver; include if your setup requires it:
        # 'joint_names': ['joint_1', 'joint_2', 'joint_3', 'joint_4'],
        'points': [
            {
                'positions': [float(x) for x in joint_positions]
                # No time_from_start → driver executes immediately (same as MATLAB snippet)
            }
        ]
    }

    pub.publish(roslibpy.Message(msg))
    pub.unadvertise()


if __name__ == '__main__':
    client = roslibpy.Ros(host='10.42.0.1', port=9090)  # Replace with your ROS bridge IP
    client.run()
    try:
        # Example: open gripper, move, close gripper

        # move to a joint position
        target_joint_positions = [0.0, 0.80, 0.30, 0.0]
        move_dobot_joint_positions(client, target_joint_positions)
        time.sleep(2)


        # Open gripper
        dobot_gripper_open(client)
        time.sleep(2)

        # Close gripper
        dobot_gripper_close(client)
        time.sleep(2)

        dobot_gripper_release(client)
        time.sleep(2)

        # move to a joint position
        target_joint_positions = [0.0, 0.40, 0.30, 0.0]
        move_dobot_joint_positions(client, target_joint_positions)
        time.sleep(2)

        # Open gripper
        dobot_gripper_open(client)
        time.sleep(2)

        # Close gripper
        dobot_gripper_close(client)
        time.sleep(2)

        dobot_gripper_release(client)
        time.sleep(2)
        

    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
