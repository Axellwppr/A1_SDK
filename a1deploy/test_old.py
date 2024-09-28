import torch
import itertools
import argparse
import time
import datetime
import argparse
import rospy
import itertools
import math
from a1_interface import A1ArmInterface
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from setproctitle import setproctitle
from command_manager import FixedCommandManager, KeyboardCommandManager
from live_plot_client import LivePlotClient
import pytorch_kinematics as pk
from play import Arm


class Arm_test():
    def __init__(self,arm=None,dt=0.05):
        print("init Arm")
        self.default_joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.dt = dt
        self.pos = torch.zeros(6)
        self.vel = torch.zeros(6)
        self._arm = arm
        self._arm.start()
        # self.plot = LivePlotClient(zmq_addr="tcp://127.0.0.1:5555")

        while self._arm.wait_init:
            print("waiting for arm to be ready")
            time.sleep(0.02)

    def close(self):
        self._arm.stop()

    def update(self):
        pos_t, vel_t = self._arm.get_joint_states()
        self.pos = pos_t
        self.vel = vel_t
        return pos_t, vel_t

    def step(self, joint_pos=None, joint_vel=None, tau_ff=None):
        # self.update()
        joint_pos = joint_pos.clip(-np.pi, np.pi)
        target = (joint_pos - self.pos).clip(-0.2, 0.2) + self.pos
        # print(target)
        self._arm.set_targets(
            target.tolist(),
            # self.pos.tolist(),
            [0, 0, 0, 0, 0, 0],
        )
        if tau_ff is not None:
            self._arm.set_feed_forward_torques(tau_ff.tolist())
        self.update()
        return self.pos, self.vel


class IK:
    def __init__(self, robot, chain):
        self.robot = robot
        self.chain = chain

    def compute(self, target_pos, target_vel, target_force=None):
        self.robot.update()
        joint_pos, joint_vel = self.robot.pos, self.robot.vel

        # t_start_a = time.perf_counter()
        ret = chain.forward_kinematics(joint_pos, end_only=True)
        J = chain.jacobian(joint_pos)[0, :3]
        # print("ik_time", time.perf_counter() - t_start_a)

        ee_pos = ret.get_matrix()[0, :3, 3]
        ee_vel = (J @ joint_vel.unsqueeze(1)).squeeze(1)
        # rot = pk.matrix_to_quatenion(m[:, :3, :3])
        ee_pos_diff = target_pos - ee_pos
        ee_vel_diff = target_vel - ee_vel
        # print("ee_pos_diff", ee_pos_diff)
        # print("ee_vel", ee_vel)
        desired_vel = 40 * ee_pos_diff + 8 * ee_vel_diff
        nomial_error = - joint_pos

        J_T = J.T

        lambda_matrix = torch.eye(J.shape[1]) * 0.1
        A = J_T @ J + lambda_matrix
        b = J_T @ desired_vel.unsqueeze(-1)
        delta_q = torch.linalg.lstsq(A, b).solution.squeeze(-1) + 0.1 * nomial_error
        if target_force is not None:
            tau_ff = torch.pinverse(J) @ target_force
        else:
            tau_ff = None
        return joint_pos + delta_q * self.robot.dt, delta_q, ee_pos, tau_ff

def lemniscate(t: float, c: float):
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    sin2p1 = sin_t ** 2 + 1
    x = torch.tensor([c * sin_t, cos_t, sin_t * cos_t]) / sin2p1
    return x

from pathgen import PathGenerator

if __name__ == "__main__":
    rospy.init_node("a1_arm_interface", anonymous=True)
    path = "/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
    arm = A1ArmInterface(
        kp=[120, 120, 80, 30, 30, 30],
        # kp=[140, 200, 120, 20, 20, 20],
        kd=[2, 2, 2, 1, 1, 0.4],
        urdf_path=path,
    )
    chain = pk.build_serial_chain_from_urdf(open(path, "rb").read(), "arm_seg6")
    dt = 0.02
    robot = Arm(arm=arm, dt=dt)

    ik = IK(robot, chain)

    r = 0.15
    default_target_pos = torch.tensor([0.3, 0, 0.4])
    # print(default_target_pos.dtype)
    length = 4000
    for _ in range(10):
        pg = PathGenerator()
        his_pos = np.zeros((length, 3))
        t = 0
        for i in itertools.count():
            if i % 100 == 0:
                print("rate = ", arm.count / (time.perf_counter() - arm.start_time))
            if i >= length:
                break
            t_start = time.perf_counter()
            if i % 1 == 0:
                t = i * dt
            target_pos = default_target_pos  + torch.tensor([0.0, r *math.cos(t), r * math.sin(t)])
            # target_pos = default_target_pos  + lemniscate(t, 0.2) * torch.tensor([1.0, 0.4, 0.5])
            # target_vel = torch.tensor([0., r * -math.sin(t), r * math.cos(t)])
            # target_pos = pg.get_position(t)
            # target_pos = default_target_pos
            print(target_pos)
            target_vel = torch.tensor([0., 0., 0.])
            joint_pos, joint_vel, pos, tau_ff = ik.compute(target_pos, target_vel)
            his_pos[i] = target_pos
            # print(joint_pos)
            
            robot.step(joint_pos)
            step_time = time.perf_counter() - t_start
            # print("step_time", step_time)
            time.sleep(max(0, dt - step_time))
        
        # 获取当前时间并格式化
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

        np.save(f"./his/his_pos_{formatted_time}.npy", his_pos)