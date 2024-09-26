from pynput import keyboard
from abc import ABC, abstractmethod
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

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
            command = np.array([0.2, 0.2, 0.5, 40.0, 40.0, 40.0, 20.0, 20.0, 20.0, 1.0])
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
        # print("ee_pos", ee_pos, "ee_vel", ee_vel)
        command[:3] -= ee_pos
        if self.apply_scaling:
            command[3:6] *= command[:3]
            command[6:9] *= - ee_vel
        if self.compliant:
            command[:3] = 0
        # print("command", command[:3])
        return command
    
    def close(self):
        pass

class FixedCommandManager_test(CommandManagerInterface):
    def __init__(self, arm=None, command=None):
        if command is None:
            command = np.array([0.2, 0.2, 0.5])
        self.command = command
        self.arm = arm

    def reset(self):
        pass

    def update(self):
        command = self.command.copy()
        return command
    
    def close(self):
        pass

import threading
import sys
import termios
import tty
import numpy as np
import time

class KeyboardCommandManager(CommandManagerInterface):

    def __init__(self, arm=None, step_size=0.01, apply_scaling=True):
        self.step_size = step_size
        self.arm = arm

        self.command = np.zeros(10)

        self.command_setpoint_pos_ee_b = np.array([0.2, 0.0, 0.5])
        self.command_kp = np.array([40.0, 40.0, 4+0.0])  # 默认值
        self.command_kd = 2 * np.sqrt(self.command_kp)
        self.compliant_ee = False
        self.apply_scaling = apply_scaling
        self.mass = 1.0

        self.running = True
        self.input_thread = threading.Thread(target=self._input_listener)
        self.input_thread.daemon = True
        self.input_thread.start()
        print("[KeyboardCommandManager]: 控制台输入监听已启动")

    def _input_listener(self):
        while self.running:
            ch = self._getch()
            if ch == '\x1b':  # 转义字符
                # 读取接下来的两个字符
                next1 = self._getch()
                next2 = self._getch()
                if next1 == '[':
                    if next2 == 'A':
                        self._on_key('up')
                    elif next2 == 'B':
                        self._on_key('down')
                    elif next2 == 'C':
                        self._on_key('right')
                    elif next2 == 'D':
                        self._on_key('left')
            else:
                self._on_key(ch)

    def _getch(self):
        """从标准输入获取单个字符。"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
        except Exception as e:
            print("读取字符时出错: ", e)
            ch = ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def _on_key(self, key):
        key = key.lower()
        if key == 'up':
            self.command_setpoint_pos_ee_b[0] += self.step_size
        elif key == 'down':
            self.command_setpoint_pos_ee_b[0] -= self.step_size
        elif key == 'right':
            self.command_setpoint_pos_ee_b[1] += self.step_size
        elif key == 'left':
            self.command_setpoint_pos_ee_b[1] -= self.step_size
        elif key == 'w':
            self.command_setpoint_pos_ee_b[2] += self.step_size
        elif key == 's':
            self.command_setpoint_pos_ee_b[2] -= self.step_size
        elif key == 'k':
            self.command_kp += 10
            self.command_kd = 2 * np.sqrt(self.command_kp)
        elif key == 'l':
            self.command_kp -= 10
            self.command_kd = 2 * np.sqrt(self.command_kp)
        elif key == 'c':
            self.compliant_ee = not self.compliant_ee
            print(f"Compliant EE 设置为 {self.compliant_ee}")

    def reset(self):
        self.command_setpoint_pos_ee_b = np.zeros(3)
        self.command_kp = np.array([100.0, 100.0, 100.0])

    def update(self):
        # 将值限制在合理范围内
        self.command_setpoint_pos_ee_b = np.clip(
            self.command_setpoint_pos_ee_b, [0.2, -0.2, 0.2], [0.6, 0.2, 0.6]
        )

        # 确保 command_kp 在限制范围内
        self.command_kp = np.clip(self.command_kp, 50, 200)
        self.command_kd = 2 * np.sqrt(self.command_kp)

        # ee_pos, ee_ori, ee_vel = self.arm.get_forward_kinematics()
        ee_pos, ee_ori, ee_vel = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])

        # 计算差异
        command_setpoint_pos_ee_diff_b = self.command_setpoint_pos_ee_b - ee_pos

        self.command[0:3] = command_setpoint_pos_ee_diff_b
        self.command[3:6] = self.command_kp
        self.command[6:9] = self.command_kd
        self.command[9] = self.mass

        if self.compliant_ee:
            # 填充命令数组
            self.command[0:6] = 0
        if self.apply_scaling:
            # 将位置和速度缩放到合理范围
            self.command[3:6] *= self.command[0:3]
            self.command[6:9] *= -ee_vel
        # 返回命令，缩放将在父类中处理
        return self.command

    def close(self):
        self.running = False
        self.input_thread.join()

if __name__ == "__main__":
    CommandManager = KeyboardCommandManager()
    while True:
        print(CommandManager.update())
        time.sleep(0.02)
