import rospy
import torch
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control
from typing import List, Tuple
import threading
from live_plot_client import LivePlotClient
import time


class A1ArmInterface:
    def __init__(
        self,
        control_frequency: int = 1000,
        kp: List[float] = [40, 40, 40, 20, 20, 20],
        kd: List[float] = [40, 40, 40, 1, 1, 1],
        urdf_path: str = "",
    ):
        self.urdf_path = urdf_path

        self.pub = rospy.Publisher(
            "/arm_joint_command_host", arm_control, queue_size=10
        )
        # self.plot = LivePlotClient(ip="127.0.0.1", port=9999)
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

        self.count = 0
        self.start_time = time.perf_counter()

        # Joint state variables
        self.joint_positions = torch.zeros(6)
        # self.joint_positions = torch.tensor([0.0, 0.2, -0.3, 0.0, 0.0, 0.0])
        self.joint_velocities = torch.zeros(6)
        self.joint_state_lock = threading.Lock()
        self.arm_control_msg_lock = threading.Lock()

        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber(
            "/joint_states_host", JointState, self._joint_state_callback
        )

    def _joint_state_callback(self, msg: JointState):
        with self.joint_state_lock:
            self.count += 1
            # Assuming the first 6 joints are the arm joints
            self.joint_positions[:6] = torch.as_tensor(msg.position[:6])
            self.joint_velocities[:6] = torch.as_tensor(msg.velocity[:6])

            if self.wait_init:
                self.arm_control_msg.p_des = self.joint_positions.tolist()
                self.start_time = time.perf_counter()
                self.wait_init = False

    def start(self):
        if not self.running:
            self.running = True
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
                pass
            else:
                with self.arm_control_msg_lock:
                    # print("p_des", np.array(self.arm_control_msg.p_des))
                    self.arm_control_msg.header.seq += 1
                    self.arm_control_msg.header.stamp = rospy.Time.now()
                    self.pub.publish(self.arm_control_msg)
            self.rate.sleep()

    def set_targets(self, positions: torch.Tensor, velocities: torch.Tensor):
        if self.wait_init:
            return
        if positions.size(0) != 6 or velocities.size(0) != 6:
            raise ValueError("Both positions and velocities must have 6 elements")
        with self.arm_control_msg_lock:
            self.arm_control_msg.p_des = positions.tolist()
            self.arm_control_msg.v_des = velocities.tolist()

    def set_feed_forward_torques(self, torques: torch.Tensor):
        if torques.size(0) != 6:
            raise ValueError("Torques must have 6 elements")
        self.arm_control_msg.t_ff = torques.tolist()

    def get_joint_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with self.joint_state_lock:
            return self.joint_positions.clone(), self.joint_velocities.clone()


if __name__ == "__main__":
    # set print precision
    try:
        rospy.init_node("a1_arm_interface", anonymous=True)
        arm_interface = A1ArmInterface(
            # kp=[60, 60, 60, 20, 20, 20],
            kp=[0, 0, 0, 0, 0, 0],
            # kd=[2, 2, 2, 1, 1, 1],
            kd=[40, 40, 40, 1, 1, 1],
        )
        arm_interface.start()

        freq = 500
        rate = rospy.Rate(freq)
        # Example usage
        steps = freq * 100000
        for step in range(steps):
            positions = [
                # 1.0 * np.sin(2 * np.pi * step / steps),
                0,
                0.8 * (1 - np.cos(2 * np.pi * step / steps)),
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

            # arm_interface.set_targets(positions, velocities)
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
