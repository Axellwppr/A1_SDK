import rospy
import numpy as np
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control
from typing import List, Tuple
import threading
import xml.etree.ElementTree as ET
from transforms3d import affines, euler


# set print precision
np.set_printoptions(precision=3, suppress=True)


class A1ArmInterface:
    def __init__(
        self,
        control_frequency: int = 50,
        kp: List[float] = [40, 40, 40, 20, 20, 20],
        kd: List[float] = [40, 40, 40, 1, 1, 1],
        urdf_path: str = "/home/elijah/Documents/a1/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
    ):
        self.urdf_path = urdf_path
        self.joint_info = self.parse_urdf(urdf_path)

        self.pub = rospy.Publisher(
            "/arm_joint_command_host", arm_control, queue_size=10
        )
        self.rate = rospy.Rate(control_frequency)
        self.arm_control_msg = arm_control()
        self.arm_control_msg.header.seq = 0
        self.arm_control_msg.header.stamp = rospy.Time.now()
        self.arm_control_msg.header.frame_id = "world"
        self.arm_control_msg.kp = kp
        self.arm_control_msg.kd = kd
        self.arm_control_msg.p_des = [0, 0, 0, 0, 0, 0]
        self.arm_control_msg.v_des = [0, 0, 0, 0, 0, 0]
        self.arm_control_msg.t_ff = [0, 0, 0, 0, 0, 0]

        self.running = False
        self.thread = None

        # Joint state variables
        self.joint_positions = [0.0] * 6
        self.joint_velocities = [0.0] * 6
        self.joint_state_lock = threading.Lock()

        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states_host", JointState, self._joint_state_callback
        )

    def parse_urdf(self, urdf_path: str):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joint_info = {}

        for joint in root.findall(".//joint[@type='revolute']"):
            name = joint.attrib["name"]
            origin = joint.find("origin")
            axis = joint.find("axis")

            if origin is not None and axis is not None:
                xyz = [float(x) for x in origin.attrib["xyz"].split()]
                rpy = [float(r) for r in origin.attrib["rpy"].split()]
                axis_xyz = [float(a) for a in axis.attrib["xyz"].split()]

                joint_info[name] = {"xyz": xyz, "rpy": rpy, "axis": axis_xyz}

        return joint_info

    def _joint_state_callback(self, msg: JointState):
        with self.joint_state_lock:
            # Assuming the first 6 joints are the arm joints
            self.joint_positions = list(msg.position[:6])
            self.joint_velocities = list(msg.velocity[:6])

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._control_loop)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()

    def _control_loop(self):
        while self.running:
            self.arm_control_msg.header.seq += 1
            self.arm_control_msg.header.stamp = rospy.Time.now()
            self.pub.publish(self.arm_control_msg)
            self.rate.sleep()

    def set_targets(self, positions: List[float], velocities: List[float]):
        if len(positions) != 6 or len(velocities) != 6:
            raise ValueError("Both positions and velocities must have 6 elements")

        self.arm_control_msg.p_des = positions
        self.arm_control_msg.v_des = velocities

    def set_feed_forward_torques(self, torques: List[float]):
        if len(torques) != 6:
            raise ValueError("Torques must have 6 elements")

        self.arm_control_msg.t_ff = torques

    def get_joint_states(self) -> Tuple[List[float], List[float]]:
        with self.joint_state_lock:
            return self.joint_positions, self.joint_velocities

    def get_forward_kinematics(self) -> Tuple[List[float], List[float]]:
        """
        Compute forward kinematics and return the end-effector pose.

        Returns:
            Tuple[List[float], List[float]]: Translation [x, y, z] and orientation [roll, pitch, yaw]
        """
        T = np.eye(4)

        for joint_name, joint_angle in zip(
            self.joint_info.keys(), self.joint_positions
        ):
            joint = self.joint_info[joint_name]

            # Create transformation matrix for joint
            R = euler.euler2mat(*joint["rpy"])
            t = joint["xyz"]
            T_joint = affines.compose(t, R, np.ones(3))

            # Apply joint rotation
            axis = joint["axis"]
            R_joint = euler.axangle2mat(axis, joint_angle)
            T_rot = affines.compose([0, 0, 0], R_joint, np.ones(3))

            # Combine transformations
            T = T.dot(T_joint).dot(T_rot)

        # Extract final translation and orientation
        translation = T[:3, 3]
        orientation = euler.mat2euler(T[:3, :3])

        return translation, orientation


if __name__ == "__main__":
    try:
        rospy.init_node("a1_arm_interface", anonymous=True)
        arm_interface = A1ArmInterface(
            # kp=[40, 40, 40, 20, 20, 20],
            kp=[0, 0, 0, 0, 0, 0],
            kd=[40, 40, 40, 1, 1, 1],
            urdf_path="/home/elijah/Documents/a1/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
        )
        arm_interface.start()

        # Example usage
        steps = 1000
        for step in range(steps):
            positions = [
                1.0 * np.sin(2 * np.pi * step / steps),
                0.4 * (1 - np.cos(2 * np.pi * step / steps)),
                0,
                0,
                0,
                0,
            ]
            velocities = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]  # You may want to calculate proper velocities
            arm_interface.set_targets(positions, velocities)

            # Read and print current joint states
            current_positions, current_velocities = arm_interface.get_joint_states()
            print(f"Current positions: {np.array(current_positions)}")
            rospy.sleep(0.1)  # Sleep for 100ms between updates

        arm_interface.stop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
