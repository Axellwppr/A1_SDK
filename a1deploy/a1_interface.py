import rospy
import numpy as np
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control
from typing import List, Tuple
import threading
import xml.etree.ElementTree as ET
from transforms3d import affines, euler
from live_plot_client import LivePlotClient

class A1ArmInterface:
    def __init__(
        self,
        control_frequency: int = 500,
        kp: List[float] = [40, 40, 40, 20, 20, 20],
        kd: List[float] = [40, 40, 40, 1, 1, 1],
        urdf_path: str = "",
    ):
        print("init Arm interface")
        self.urdf_path = urdf_path
        self.joint_info = self.parse_urdf(urdf_path)

        self.pub = rospy.Publisher(
            "/arm_joint_command_host", arm_control, queue_size=10
        )
        self.plot = LivePlotClient(ip="127.0.0.1", port=9999)
        self.rate = rospy.Rate(control_frequency)
        self.wait_init = True
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
        self.arm_control_msg_lock = threading.Lock()

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
            
            if self.wait_init:
                # Set initial joint positions to the current joint positions
                self.arm_control_msg.p_des = self.joint_positions.copy()
                self.wait_init = False
        print(self.joint_positions[:3])
        self.plot.update(self.joint_positions[:3])

    def start(self):
        if not self.running:
            self.running = True
            # self.wait_init = True
            self.thread = threading.Thread(target=self._control_loop)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.wait_init = True
            if self.thread:
                self.thread.join()

    def _control_loop(self):
        while self.running:
            if self.wait_init:
                print("Waiting for initial joint positions...")
            else:
                with self.arm_control_msg_lock:
                    # print("p_des", np.array(self.arm_control_msg.p_des))
                    self.arm_control_msg.header.seq += 1
                    self.arm_control_msg.header.stamp = rospy.Time.now()
                    self.pub.publish(self.arm_control_msg)
            self.rate.sleep()

    def set_targets(self, positions: List[float], velocities: List[float]):
        if self.wait_init:
            print("Waiting for initial joint positions...")
            return
        if len(positions) != 6 or len(velocities) != 6:
            raise ValueError("Both positions and velocities must have 6 elements")
        with self.arm_control_msg_lock:
            self.arm_control_msg.p_des = positions
            self.arm_control_msg.v_des = velocities

    def set_feed_forward_torques(self, torques: List[float]):
        if len(torques) != 6:
            raise ValueError("Torques must have 6 elements")

        self.arm_control_msg.t_ff = torques

    def get_joint_states(self) -> Tuple[List[float], List[float]]:
        with self.joint_state_lock:
            return self.joint_positions, self.joint_velocities

    def get_forward_kinematics(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute forward kinematics and return the end-effector pose and linear velocity.

        Returns:
            Tuple[List[float], List[float], List[float]]: Translation [x, y, z],
            orientation [roll, pitch, yaw], and linear velocity [vx, vy, vz]
        """
        T = np.eye(4)
        positions = []
        axes = []
        joint_types = []

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

            # Store the joint type (assuming 'revolute' or 'prismatic')
            joint_type = joint.get('type', 'revolute')
            joint_types.append(joint_type)

            # Store the position of the joint (origin)
            o_i = T[:3, 3]
            positions.append(o_i)

            # Compute the joint axis in the base frame
            z_i = T[:3, :3].dot(axis)
            axes.append(z_i)

        # Extract final translation and orientation
        translation = T[:3, 3]
        orientation = list(euler.mat2euler(T[:3, :3]))

        # Compute the Jacobian matrix
        n = len(self.joint_positions)
        Jv = np.zeros((3, n))

        o_n = translation  # End-effector position

        for i in range(n):
            z_i = axes[i]
            o_i = positions[i]

            if joint_types[i] == 'revolute':
                Jv[:, i] = np.cross(z_i, o_n - o_i)
            elif joint_types[i] == 'prismatic':
                Jv[:, i] = z_i

        # Compute linear velocity of the end-effector
        q_dot = np.array(self.joint_velocities)
        linear_velocity = Jv.dot(q_dot)

        return np.array(translation), np.array(orientation), np.array(linear_velocity)
        


if __name__ == "__main__":
    # set print precision
    np.set_printoptions(precision=3, suppress=True)
    try:
        rospy.init_node("a1_arm_interface", anonymous=True)
        arm_interface = A1ArmInterface(
            kp=[60, 60, 60, 20, 20, 20],
            # kp=[0, 0, 0, 0, 0, 0],
            kd=[2, 2, 2, 1, 1, 1],
            urdf_path="/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
        )
        # nkp = np.array([40, 40, 40, 20, 20, 20])
        # nkd = np.array([2, 2, 2, 1, 1, 1])
        arm_interface.start()

        freq = 500
        rate = rospy.Rate(freq)
        # Example usage
        steps = freq * 10
        for step in range(steps):
            positions = [
                # 1.0 * np.sin(2 * np.pi * step / steps),
                0,
                0.8 * (1 - np.cos(2 * np.pi * step / steps)),
                # 0,
                # 0,
                -0.6,
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

            # if step > 100:
                # positions = [0.0, 0.5, 0, 0., 0., 0.0]
            # current_positions, current_velocities = arm_interface.get_joint_states()
            # print("pos: ", np.array(current_positions))
            # print("vel: ", np.array(current_velocities))
            # torque = (nkp * (np.array(positions) - np.array(current_positions)) + nkd * (np.array(velocities) - np.array(current_velocities))).clip(-20, 20)
            # torque[-3:] = 0
            
            arm_interface.set_targets(positions, velocities)
            # arm_interface.set_feed_forward_torques(torque.tolist())
            # Read and print current joint states
            
            # print(f"Current positions: {np.array(current_positions)}")
            # print(f"Current positions: {arm_interface.get_forward_kinematics()}")
            rate.sleep()  # Sleep for 100ms between updates

        arm_interface.stop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
