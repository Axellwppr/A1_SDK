from a1_interface import A1ArmInterface
from collections import deque
from command_manager import KeyboardCommandManager
from live_plot_client import LivePlotClient
from setproctitle import setproctitle
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from typing import Tuple
import argparse
import gc
import itertools
import pytorch_kinematics as pk
import rospy
import time
import torch

try:
    profile
except NameError:
    # 如果 @profile 未定义，创建一个空的装饰器
    def profile(func):
        return func


class Arm:
    def __init__(
        self,
        dt=0.02,
        prev_steps=3,
        arm=None,
        command_manager=None,
        urdf_path="",
        debug=True,
    ):
        print("init Arm")
        self.default_joint_pos = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf_path, "rb").read(), "arm_seg6"
        )
        self.dt = dt
        self.pos = torch.zeros(6)
        self.vel = torch.zeros(6)
        self.ee_pos = torch.zeros(3)
        self.ee_vel = torch.zeros(3)
        self.prev_actions = torch.zeros((5, prev_steps))
        self.last_action = torch.zeros(5)
        self.command = torch.zeros(10)
        self.obs = torch.zeros(25)
        self.step_count = 0
        self._arm = arm
        self.command_manager = command_manager
        self.debug = debug

        # self.plot = LivePlotClient(zmq_addr="tcp://127.0.0.1:5555")
        if not self.debug:
            self._arm.start()
            while self._arm.wait_init:
                print("waiting for arm to be ready")
                time.sleep(1)

    def close(self):
        self._arm.stop()
        self.command_manager.close()

    def reset(self):
        self.command_manager.reset()

    @profile
    def take_action(self, action: torch.Tensor):
        self.step_count += 1
        action.clip_(-2 * torch.pi, 2 * torch.pi)

        self.prev_actions[:, 1:] = self.prev_actions[:, :-1]
        self.prev_actions[:, 0] = action
        self.last_action.mul_(0.8).add_(action * 0.2)

        target = self.default_joint_pos.clone()
        target[:5].add_(self.last_action * 0.5).clip_(-torch.pi, torch.pi)
        target.sub_(self.pos).clip_(-0.2, 0.2).add_(self.pos)

        if not self.debug:
            self._arm.set_targets(
                target,
                torch.zeros(6),
            )

    @profile
    def compute_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.update_fk()
        self.command = self.command_manager.update(self.ee_pos, self.ee_vel)
        self.obs[:5] = self.pos[:5]
        self.obs[5:10] = self.vel[:5]
        self.obs[10:25] = self.prev_actions.flatten()
        # print("now_pos", self.pos)
        return self.command, self.obs

    @profile
    def update_fk(self):
        self.pos, self.vel = self._arm.get_joint_states()
        ret = self.chain.forward_kinematics(self.pos, end_only=True)
        self.J = self.chain.jacobian(self.pos)[0, :3]

        self.ee_pos = ret.get_matrix()[0, :3, 3]
        self.ee_vel = (self.J @ self.vel.unsqueeze(1)).squeeze(1)

        # print("ee_pos", self.ee_pos)
        # print("ee_vel", self.ee_vel)


def main():
    rospy.init_node("a1_arm_interface", anonymous=True)
    setproctitle("play_a1")

    path = "policy-10-01_15-59.pt"
    # path = "policy-a1-427.pt"
    policy = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
    # po = TensorDict(policy)

    # print(policy)
    # breakpoint()
    # return
    policy.module[0].set_missing_tolerance(True)
    torch.set_grad_enabled(False)

    try:
        arm = A1ArmInterface(kp=[80, 80, 80, 30, 30, 30], kd=[2, 2, 2, 1, 1, 1])
        dt = 0.02
        robot = Arm(
            dt=dt,
            arm=arm,
            command_manager=KeyboardCommandManager(),
            urdf_path="/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
        )
        robot.reset()
        cmd, obs = robot.compute_obs()

        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            td = TensorDict(
                {
                    "policy": obs,
                    "is_init": torch.tensor(1, dtype=bool),
                    "adapt_hx": torch.zeros(128),
                    "command": cmd,
                },
                [],
            ).unsqueeze(0)

            # 初始化一个 deque 用于存储最近的 100 次循环的 elapsed 时间
            elapsed_times = deque(maxlen=100)

            for i in itertools.count(1):  # 从 1 开始计数，方便后续的模运算
                start = time.perf_counter()

                # 循环体
                cmd, obs = robot.compute_obs()
                td["next", "policy"] = obs.unsqueeze(0)
                td["next", "command"] = cmd.unsqueeze(0)
                td["next", "is_init"] = torch.tensor([0], dtype=bool)
                td = td["next"]

                policy(td)
                action = td["action"][0]

                # 执行动作
                robot.take_action(action)

                # 手动触发垃圾回收
                # gc.collect()

                # 计算本次循环的耗时
                end = time.perf_counter()
                elapsed = end - start
                elapsed_times.append(elapsed)

                if i % 400 == 0:
                    avg_elapsed = sum(elapsed_times) / len(elapsed_times)
                    min_elapsed = min(elapsed_times)
                    max_elapsed = max(elapsed_times)
                    print(
                        f"Iteration {i}: Last 100 loops - Avg: {avg_elapsed:.6f}s, Min: {min_elapsed:.6f}s, Max: {max_elapsed:.6f}s"
                    )

                if i % 100 == 0:
                    print("now_pos", robot.ee_pos, "pos_diff", cmd[:3])

                # 维持循环时间
                elapsed_total = time.perf_counter() - start
                sleep_time = max(0, dt - elapsed_total)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        robot.close()
        print("End")


if __name__ == "__main__":
    main()
