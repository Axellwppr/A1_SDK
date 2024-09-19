import torch
import itertools
import argparse
import time
import numpy as np
import argparse
import rospy
from a1_interface import A1ArmInterface
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from setproctitle import setproctitle
from command_manager import FixedCommandManager, KeyboardCommandManager

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


class Arm:
    def __init__(self, prev_steps=3, arm=None, command_manager=None):
        print("init Arm")
        self.default_joint_pos = np.array([0.0, 0.2, -0.3, 0.0, 0.0, 0.0])
        self.dt = 0.02
        self.pos = np.zeros(6)
        self.vel = np.zeros(6)
        self.prev_actions = np.zeros((prev_steps, 5))
        self.last_action = np.zeros(5)
        self.command = np.zeros(10)
        self.obs = np.zeros(25)
        self.step_count = 0
        self._arm = arm
        self._arm.start()
        self.command_manager = command_manager

        while self._arm.wait_init:
            print("waiting for arm to be ready")
            time.sleep(1)

    def close(self):
        self._arm.stop()
        self.command_manager.close()

    def reset(self):
        self.command_manager.reset()
        self.update()
        return self._compute_obs()

    def update(self):
        pos_t, vel_t = self._arm.get_joint_states()
        self.pos = pos_t.copy()
        self.vel = vel_t.copy()

    def step(self, action=None):
        self.step_count += 1
        if action is not None:
            action = action.clip(-2 * np.pi, 2 * np.pi)
            self.prev_actions[1:, :] = self.prev_actions[:-1, :]
            self.prev_actions[0] = action
            self.last_action = self.last_action * 0.5 + action * 0.5
            target = self.default_joint_pos.copy()
            target[:5] += self.last_action * 0.5
            target = target.clip(-np.pi, np.pi)
            target = (target - self.pos).clip(-0.2, 0.2) + self.pos
            self._arm.set_targets(
                target.tolist(),
                [0, 0, 0, 0, 0, 0],
            )
        self.update()
        self.command = self.command_manager.update()
        return self._compute_obs()

    def _compute_obs(self):
        self.obs[:5] = self.pos[:5]
        self.obs[5:10] = self.vel[:5]
        self.obs[10:25] = self.prev_actions.flatten()
        return self.command, self.obs

def main():
    rospy.init_node("a1_arm_interface", anonymous=True)

    parser = argparse.ArgumentParser()
    # parser.add_argument("-l", "--log", action="store_true", default=False)
    args = parser.parse_args()
    setproctitle("play_a1")
    
    print("load policy")
    path = "policy-a1-401.pt"
    policy = torch.load(path, weights_only=False)
    policy.module[0].set_missing_tolerance(True)

    arm = A1ArmInterface(
        kp=[80, 80, 80, 20, 20, 20],
        kd=[2, 2, 2, 1, 1, 1],
        urdf_path="/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
    )
    robot = Arm(arm=arm, command_manager=FixedCommandManager(arm=arm, compliant=True))

    cmd, obs = robot.reset()
    cmd, obs = robot._compute_obs()
    
    action_record = np.load('record3.npy')
    print(action_record.shape)
    
    obs_his = np.zeros([1000, 25])
    act_his = np.zeros([1000, 5])
    cmd_his = np.zeros([1000, 10])
    try:
        td = TensorDict(
            {
                "policy": torch.as_tensor(obs, dtype=torch.float32),
                "is_init": torch.tensor(1, dtype=bool),
                "adapt_hx": torch.zeros(128),
                "command": torch.as_tensor(cmd, dtype=torch.float32),
            },
            [],
        ).unsqueeze(0)
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for i in itertools.count():
                print(f"iter {i}")
                start = time.perf_counter()
                policy(td)
                action = td["action"].cpu().numpy()[0]
                                
                # if i > 999:
                #     np.save("obs.npy", obs_his)
                #     np.save("act.npy", act_his)
                #     np.save("cmd.npy", cmd_his)
                #     break

                # obs_his[i, :] = obs
                # act_his[i, :] = action
                # cmd_his[i, :] = cmd

                
                # # breakpoint()
                # action = action_record[1, i, :]
                # print(i, action)
                # print(action)
                # print(td["state_value"].item())
                # print(processed_actions)
                # print(robot._robot.get_joint_pos_target())
                # obs = torch.as_tensor(robot._compute_obs())

                cmd, obs = robot.step(action)
                cmd = torch.as_tensor(cmd, dtype=torch.float32)
                obs = torch.as_tensor(obs, dtype=torch.float32)
                td["next", "policy"] = obs.unsqueeze(0)
                td["next", "command"] = cmd.unsqueeze(0)
                td["next", "is_init"] = torch.tensor([0], dtype=bool)
                td = td["next"]
                time.sleep(max(0, 0.02 - (time.perf_counter() - start)))

    except KeyboardInterrupt:
        robot.close()
        print("End")


if __name__ == "__main__":
    main()
