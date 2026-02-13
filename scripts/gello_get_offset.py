import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tyro

from gello.dynamixel.driver import DynamixelDriver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MENAGERIE_ROOT: Path = Path(__file__).parent / "third_party" / "mujoco_menagerie"


@dataclass
class Args:
    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[float, ...] = (1, 1, -1, 1, 1, 1)
    """The joint angles that the GELLO is placed in at (in radians)."""

    gripper: bool = True
    """Whether or not the gripper is attached."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints


def get_config(args: Args) -> None:
    joint_ids = list(range(1, args.num_joints + 1))
    driver = DynamixelDriver(joint_ids, port=args.port, baudrate=57600)

    # assume that the joint state shouold be args.start_joints
    # find the offset, which is a multiple of np.pi/2 that minimizes the error between the current joint state and args.start_joints
    # this is done by brute force, we seach in a range of +/- 8pi

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = args.joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = args.start_joints[index]
        return np.abs(joint_i - start_i)

    for _ in range(10):
        driver.get_joints()  # warmup

    for _ in range(1):
        best_offsets = []
        curr_joints = driver.get_joints()
        for i in range(args.num_robot_joints):
            best_offset = 0
            best_error = 1e6
            for offset in np.linspace(
                -8 * np.pi, 8 * np.pi, 8 * 4 + 1
            ):  # intervals of pi/2
                error = get_error(offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            best_offsets.append(best_offset)
        print()
        print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
        def wrap_pi(x):
            return (x + np.pi) % (2*np.pi) - np.pi

        fine_offsets = []
        for i in range(args.num_robot_joints):
            sign  = args.joint_signs[i]
            meas  = curr_joints[i]       # 当前从 Dynamixel 读到的原始弧度
            start = args.start_joints[i] # 你定义的 home（弧度）
            # 连续解
            offset_cont = meas - sign*start
            # 对齐到与粗解最接近的 2π 分支
            k = round((best_offsets[i] - offset_cont) / (2*np.pi))
            offset_final = offset_cont + k*2*np.pi
            fine_offsets.append(offset_final)

        print("fine offsets (continuous):  ", [f"{x:.6f}" for x in fine_offsets])

        # 对比残差（粗 vs 细）
        residual_coarse, residual_fine = [], []
        for i in range(args.num_robot_joints):
            sign  = args.joint_signs[i]
            meas  = curr_joints[i]
            start = args.start_joints[i]
            pred_c = sign*(meas - best_offsets[i])
            pred_f = sign*(meas - fine_offsets[i])
            residual_coarse.append(float(wrap_pi(pred_c - start)))
            residual_fine.append(float(wrap_pi(pred_f - start)))

        print("residual (coarse offsets): ", [f"{np.rad2deg(x):.2f}°" for x in residual_coarse])
        print("residual (fine   offsets): ", [f"{np.rad2deg(x):.2f}°" for x in residual_fine])

        print(
            "best offsets function of pi: ["
            + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in best_offsets])
            + " ]",
        )
        if args.gripper:
            print(
                "gripper open (degrees)       ",
                np.rad2deg(driver.get_joints()[-1]) - 0.2,
            )
            print(
                "gripper close (degrees)      ",
                np.rad2deg(driver.get_joints()[-1]) - 42,
            )


def main(args: Args) -> None:
    get_config(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
