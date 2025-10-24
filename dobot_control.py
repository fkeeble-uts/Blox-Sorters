#!/usr/bin/env python3
import time
import math
import roslibpy

# --- Configuration -----------------------------------------------------------
# Verify once with:  rostopic info /dobot_magician/target_tool_state
# Change to 'std_msgs/Int8MultiArray' if your setup uses Int8.
TOOL_TOPIC = '/dobot_magician/target_tool_state'
TOOL_MSG_TYPE = 'std_msgs/Int32MultiArray'   # or 'std_msgs/Int8MultiArray'

# ---------------------------------------------------------------------------

def _tool_topic(client):
    return roslibpy.Topic(client, TOOL_TOPIC, TOOL_MSG_TYPE)

def dobot_gripper_set(client, pump: int, grip: int):
    """Publish [pump, grip] to the tool state topic.
       pump: 0=off, 1=on
       grip: 0=open, 1=close
    """
    topic = _tool_topic(client)
    topic.advertise()
    try:
        topic.publish(roslibpy.Message({'data': [int(pump), int(grip)]}))
    finally:
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

def call_trigger_service(client, service_name: str):
    """Call a std_srvs/Trigger service and return dict result."""
    service = roslibpy.Service(client, service_name, 'std_srvs/Trigger')
    req = roslibpy.ServiceRequest({})
    return service.call(req)

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
    try:
        msg = {
            # 'joint_names': ['joint_1','joint_2','joint_3','joint_4'],  # include if your driver requires
            'points': [{'positions': [float(x) for x in joint_positions]}]
        }
        pub.publish(roslibpy.Message(msg))
    finally:
        pub.unadvertise()

def _quat_to_euler_rpy(x, y, z, w):
    """Quaternion (x,y,z,w) → Euler roll, pitch, yaw (radians)."""
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w*y - z*x)
    t2 = max(min(t2, +1.0), -1.0)
    pitch = math.asin(t2)

    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw

def get_end_effector_pose(client, timeout=2.0):
    """
    One-shot read from /dobot_magician/end_effector_poses (geometry_msgs/PoseStamped).

    Returns:
        {
          'position':   {'x': float, 'y': float, 'z': float},
          'quaternion': {'x': float, 'y': float, 'z': float, 'w': float},
          'euler_rpy':  {'roll': float, 'pitch': float, 'yaw': float}
        }
    Raises:
        TimeoutError if no message within 'timeout' seconds.
    """
    latest = {'msg': None}

    sub = roslibpy.Topic(client,
                         '/dobot_magician/end_effector_poses',
                         'geometry_msgs/PoseStamped')

    def _cb(m):
        latest['msg'] = m

    sub.subscribe(_cb)
    t0 = time.time()
    try:
        while latest['msg'] is None and (time.time() - t0) < timeout:
            time.sleep(0.02)

        if latest['msg'] is None:
            raise TimeoutError('No end effector pose received within timeout.')

        pose = latest['msg'].get('pose', {})
        pos = pose.get('position', {})
        ori = pose.get('orientation', {})

        x = float(pos.get('x', 0.0))
        y = float(pos.get('y', 0.0))
        z = float(pos.get('z', 0.0))

        qx = float(ori.get('x', 0.0))
        qy = float(ori.get('y', 0.0))
        qz = float(ori.get('z', 0.0))
        qw = float(ori.get('w', 1.0))

        roll, pitch, yaw = _quat_to_euler_rpy(qx, qy, qz, qw)

        return {
            'position':   {'x': x, 'y': y, 'z': z},
            'quaternion': {'x': qx, 'y': qy, 'z': qz, 'w': qw},
            'euler_rpy':  {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        }
    finally:
        sub.unsubscribe()

def _rpy_to_quaternion(roll, pitch, yaw):
    # ZYX (yaw→pitch→roll) convention
    cy, sy = math.cos(yaw*0.5),   math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5),  math.sin(roll*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return qx, qy, qz, qw

def send_target_end_effector_pose(client, xyz, rpy=(0.0, 0.0, 0.0), degrees=False):
    """Publish geometry_msgs/Pose to /dobot_magician/target_end_effector_pose.
       xyz in metres, rpy in radians (set degrees=True to pass degrees)."""
    if degrees:
        rpy = tuple(math.radians(v) for v in rpy)
    qx, qy, qz, qw = _rpy_to_quaternion(*rpy)

    pub = roslibpy.Topic(client,
                         '/dobot_magician/target_end_effector_pose',
                         'geometry_msgs/Pose')
    pub.advertise()
    try:
        msg = {
            'position':    {'x': float(xyz[0]), 'y': float(xyz[1]), 'z': float(xyz[2])},
            'orientation': {'x': float(qx), 'y': float(qy), 'z': float(qz), 'w': float(qw)}
        }
        pub.publish(roslibpy.Message(msg))
    finally:
        pub.unadvertise()
