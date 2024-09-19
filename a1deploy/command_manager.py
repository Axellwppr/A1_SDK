from pynput import keyboard
from abc import ABC, abstractmethod
import numpy as np

class CommandManagerInterface(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def close(self):
        pass


class FixedCommandManager(CommandManagerInterface):
    def __init__(self, arm=None, command=None, apply_scaling=True, compliant=False):
        if command is None:
            command = np.array([0.2, 0.2, 0.5, 80.0, 80.0, 80.0, 20.0, 20.0, 20.0, 1.0])
        if compliant:
            command[3:6] = 0
        self.compliant = compliant
        self.apply_scaling = apply_scaling
        self.command = command
        self.arm = arm

    def reset(self):
        pass

    def update(self):
        command = self.command.copy()
        ee_pos, ee_ori, ee_vel = self.arm.get_forward_kinematics()
        print("ee_pos", ee_pos, "ee_vel", ee_vel)
        command[:3] -= ee_pos
        if self.apply_scaling:
            command[3:6] *= command[:3]
            command[6:9] *= - ee_vel
        if self.compliant:
            command[:3] = 0
        print("command", command[:3])
        return command
    
    def close(self):
        pass


class KeyboardCommandManager(CommandManagerInterface):

    def __init__(self, arm=None, step_size=0.01, apply_scaling=False):
        super().__init__(apply_scaling)
        self.step_size = step_size
        self.arm = arm

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

    def update(self):
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

        # Compute the difference
        command_setpoint_pos_ee_diff_b = self.command_setpoint_pos_ee_b - current_ee_pos

        # Populate command array
        self.command[0:3] = command_setpoint_pos_ee_diff_b * (not self.compliant_ee)
        self.command[3:6] = self.command_kp * (not self.compliant_ee)
        self.command[6:9] = self.command_kd
        self.command[9] = self.mass

        # Return command, scaling will be handled in the parent class
        return self.command

    def close(self):
        self.listener.stop()
