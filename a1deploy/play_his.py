import torch
import itertools
import argparse
import time
import datetime
import numpy as np
# import h5py
import argparse
import rospy
from pynput import keyboard
from a1_interface import A1ArmInterface
from abc import ABC, abstractmethod
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType
from setproctitle import setproctitle

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


class CommandManagerInterface(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, current_ee_pos):
        pass

    @abstractmethod
    def close(self):
        pass


class FixedCommandManager(CommandManagerInterface):

    def __init__(self, command=np.array([0.2, 0.2, 0.5, 50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 1.0])):
    # def __init__(self, command=np.array([0.2, 0.2, 0.5, 0, 0, 0, 0, 0, 0, 1.0])):
        self.command = command

    def reset(self):
        pass

    def update(self, current_ee_pos):
        print("ee_pos: ", current_ee_pos)
        command = self.command.copy()
        command[:3] -= current_ee_pos
        # command[:3] = 0.0
        return command
    
    def close(self):
        pass


class KeyboardCommandManager(CommandManagerInterface):

    # default_setpoint_pos_ee_b = np.array([0.1, 0.0, 0.4])

    def __init__(self, step_size=0.01):
        self.step_size = step_size

        self.command = np.zeros(10)

        self.command_setpoint_pos_ee_b = np.zeros(3)
        self.command_kp = np.array([100.0, 100.0, 100.0])  # Default values
        self.command_kd = 2 * np.sqrt(self.command_kp)
        self.compliant_ee = True
        self.mass = 1.0

        self.key_pressed = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
            "w": False,
            "s": False,
            "k": False,
            "l": False,
            "c": False,
        }

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()
        print("[KeyboardCommandManager]: Keyboard listener started")

    def _on_press(self, key):
        try:
            if key.char.lower() in self.key_pressed:
                self.key_pressed[key.char.lower()] = True
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_pressed["up"] = True
            elif key == keyboard.Key.down:
                self.key_pressed["down"] = True
            elif key == keyboard.Key.left:
                self.key_pressed["left"] = True
            elif key == keyboard.Key.right:
                self.key_pressed["right"] = True

    def _on_release(self, key):
        try:
            if key.char.lower() in self.key_pressed:
                self.key_pressed[key.char.lower()] = False
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_pressed["up"] = False
            elif key == keyboard.Key.down:
                self.key_pressed["down"] = False
            elif key == keyboard.Key.left:
                self.key_pressed["left"] = False
            elif key == keyboard.Key.right:
                self.key_pressed["right"] = False

    def reset(self):
        self.command_setpoint_pos_ee_b = np.zeros(3)
        self.command_kp = np.array([100.0, 100.0, 100.0])

    def update(self, current_ee_pos):
        delta = np.zeros(3)

        if self.key_pressed["up"]:
            delta[0] += self.step_size
        if self.key_pressed["down"]:
            delta[0] -= self.step_size
        if self.key_pressed["right"]:
            delta[1] += self.step_size
        if self.key_pressed["left"]:
            delta[1] -= self.step_size
        if self.key_pressed["w"]:
            delta[2] += self.step_size
        if self.key_pressed["s"]:
            delta[2] -= self.step_size

        self.command_setpoint_pos_ee_b += delta

        # Clamp the values to a reasonable range
        self.command_setpoint_pos_ee_b = np.clip(
            self.command_setpoint_pos_ee_b, [0.2, -0.2, 0.2], [0.6, 0.2, 0.6]
        )

        # Update kp and kd
        if self.key_pressed["k"]:
            self.command_kp += 10
        if self.key_pressed["l"]:
            self.command_kp -= 10
        self.command_kp = np.clip(self.command_kp, 50, 200)
        self.command_kd = 2 * np.sqrt(self.command_kp)

        # Toggle compliance
        if self.key_pressed["c"]:
            self.compliant_ee = not self.compliant_ee
            self.key_pressed["c"] = False  # Reset to avoid continuous toggling

        # print(
        #     f"Command: {self.command_setpoint_pos_ee_b}, kp: {self.command_kp}, kd: {self.command_kd}, compliant: {self.compliant_ee}"
        # )

        # Compute the difference
        command_setpoint_pos_ee_diff_b = self.command_setpoint_pos_ee_b - current_ee_pos

        # Populate command array
        self.command[0:3] = command_setpoint_pos_ee_diff_b * (not self.compliant_ee)
        self.command[3:6] = self.command_kp * (not self.compliant_ee)
        self.command[6:9] = self.command_kd
        self.command[9] = self.mass

        return self.command

    def close(self):
        self.listener.stop()


class Arm:
    def __init__(self, log_file = None, prev_steps=3, command_manager=None):
        print("init Arm")
        self.log_file = log_file

        self.default_joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.dt = 0.02
        self.latency = 0.0
        self.pos = np.zeros(6)
        self.vel = np.zeros(6)
        self.prev_actions = np.zeros((prev_steps, 5))
        self.last_action = np.zeros(5)
        self.command = np.zeros(10)
        self.ee_pos = np.zeros(3)
        self.ee_ori = np.zeros(3)

        self.start_t = time.perf_counter()
        self.timestamp = time.perf_counter()
        self.step_count = 0

        self._arm = A1ArmInterface(
            kp=[60, 60, 60, 20, 20, 20],
            kd=[2, 2, 2, 1, 1, 1],
            urdf_path="/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
        )

        self._arm.start()

        if log_file is not None:
            default_len = 50 * 60
            log_file.attrs["cursor"] = 0
            log_file.create_dataset(
                "observation",
                (default_len, 6),
                maxshape=(None, 6),
            )
            log_file.create_dataset("action", (default_len, 6), maxshape=(None, 6))

            log_file.create_dataset("jpos", (default_len, 6), maxshape=(None, 6))
            log_file.create_dataset("jvel", (default_len, 6), maxshape=(None, 6))

        self.command_manager = command_manager

    def close(self):
        self._arm.stop()
        self.command_manager.close()
        if self.log_file is not None:
            self.log_file.close()

    def reset(self):
        self.command_manager.reset()
        self.start_t = time.perf_counter()
        self.update()
        return self._compute_obs()

    def update(self):
        self.prev_pos = self.pos
        self.prev_vel = self.vel
        self.pos, self.vel = self._arm.get_joint_states()
        self.ee_pos, self.ee_ori = self._arm.get_forward_kinematics()
        self.latency = (time.perf_counter() - self.timestamp)
        self.timestamp = time.perf_counter()

    def update_command(self):
        self.command = self.command_manager.update(self.ee_pos)

    def step(self, action=None):
        self.step_count += 1
        if action is not None:
            action = action.clip(-2 * np.pi, 2 * np.pi)
            self.prev_actions[1:, :] = self.prev_actions[:-1, :]
            self.prev_actions[0, :] = action
            self.last_action = self.last_action * 0.5 + action * 0.5
            target = (self.last_action * 0.5 + self.default_joint_pos).clip(-np.pi, np.pi)
            target = np.concatenate([target, np.zeros(1)])
            target = (target - self.pos).clip(-0.2, 0.2) + self.pos
            target = target.tolist()
            self._arm.set_targets(
                target,
                [0, 0, 0, 0, 0, 0],
            )
        self.update()
        self.update_command()
        self._maybe_log()
        return self._compute_obs()

    def _compute_obs(self):
        obs = [
            self.pos,
            np.zeros(2),
            self.vel,
            np.zeros(2),
            self.prev_actions.flatten(),
        ]
        return self.command, np.concatenate(obs)

    def _maybe_log(self):
        if self.log_file is None:
            return
        self.log_file["action"][self.step_count] = self.prev_actions[:, 0]
        self.log_file["jpos"][self.step_count] = self.pos
        self.log_file["jvel"][self.step_count] = self.vel
        self.log_file.attrs["cursor"] = self.step_count

        if self.step_count == self.log_file["jpos"].len() - 1:
            new_len = self.step_count + 1 + 3000
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, value.shape[1]))


def main():
    rospy.init_node("a1_arm_interface", anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log", action="store_true", default=False)
    args = parser.parse_args()

    timestr = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    setproctitle("play_a1")

    path = "policy-a1-295.pt"
    policy = torch.load(path, weights_only=False)
    policy.module[0].set_missing_tolerance(True)
    # print(policy)

    if args.log:
        log_file = h5py.File(f"log_{timestr}.h5py", "a")
    else:
        log_file = None

    robot = Arm(log_file, command_manager=FixedCommandManager())

    # policy = lambda td: torch.zeros(12)
    cmd, obs = robot.reset()
    cmd, obs = robot._compute_obs()
    print(obs.shape)
    print(policy)
    
    # action_record = np.load('record.npy')
    # action_record = torch.from_numpy(npy_array)
    # print(action_record.shape)
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
                start = time.perf_counter()
                policy(td)
                action = td["action"].cpu().numpy()[0]
                # breakpoint()
                # action = action_record[i, 1, :]
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

                # if i % 25 == 0:
                    # print(robot.projected_gravity)
                    # print(action)
                    # print(robot.pos)
                    # print(robot.jpos_sdk.reshape(4, 3))
                    # print(robot.sdk_to_orbit(robot.jpos_sdk).reshape(3, 4))

                td = td["next"]
                time.sleep(max(0, 0.02 - (time.perf_counter() - start)))

    except KeyboardInterrupt:
        print("End")


if __name__ == "__main__":
    main()
