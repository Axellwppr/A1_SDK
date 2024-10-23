from play import Arm
from a1_interface import A1ArmInterface
from command_manager import KeyboardCommandManager
import rospy
import torch
import math
from time import sleep


def wave(t: float):
    t = t % 2
    if t < 1:
        return t
    else:
        return 2 - t


def main():
    rospy.init_node("a1_arm_interface", anonymous=True)
    data = []
    n = 5
    try:
        arm = A1ArmInterface(kp=[80, 80, 80, 30, 30, 30], kd=[2, 2, 2, 1, 1, 1])
        dt = 0.02
        robot = Arm(
            dt=dt,
            arm=arm,
            command_manager=KeyboardCommandManager(),
            urdf_path="/home/axell/桌面/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf",
            debug=False,
            default_joint_pos=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        freq = 50
        rate = rospy.Rate(freq)
        # Example usage
        steps = freq * 20
        # arm.set_targets(torch.zeros(6), torch.zeros(6))
        sleep(0)
        for step in range(steps):
            # print("OK")
            t = step / freq
            positions = torch.tensor(
                [
                    0,
                    0.5,
                    -0.5,
                    0,
                    0,
                    0,
                ],
                dtype=torch.float32,
            )
            if n == 2:
                positions[n] = wave(t * 2) * (-1.5) - 1.0  # 2
            elif n == 1:
                positions[n] = wave(t * 2) * (1.0) + 1.0  # 2
            elif n == 0:
                positions[n] = wave(t * 2) * 1.8 - 0.8  # 0
            elif n == 3:
                positions[n] = wave(t * 2) * 1.8 - 0.8  # 0
            elif n == 4:
                positions[n] = wave(t * 2) * 1.8 - 0.8  # 0
            elif n == 5:
                positions[n] = wave(t * 2) - 1.0

            # positions[n] = -(math.sin(math.pi * t) * 0.5 + 1.0)  # 2
            # positions[n] = math.sin(math.pi * t) * 0.5 + 1.0  # 1
            # positions[n] = math.sin(2 * math.pi * t) * 0.5  # 0
            # print(positions)
            arm.set_targets(positions * 0.5, torch.zeros(6, dtype=torch.float32))
            # robot.update_fk()
            # robot.take_action(positions[:5])
            j_pos, j_vel = arm.get_joint_states()
            print(j_pos)
            robot.plot.send([j_pos[n], arm.arm_control_msg.p_des[n]])
            data.append([j_pos[n], j_vel[n], arm.arm_control_msg.p_des[n]])
            rate.sleep()
    except KeyboardInterrupt:
        robot.close()
        print("End")
    datat = torch.tensor(data)
    torch.save(datat, f"{n}.pt")


if __name__ == "__main__":
    main()
