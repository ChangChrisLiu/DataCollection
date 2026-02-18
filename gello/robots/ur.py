from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation

from gello.robots.robot import Robot


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = False):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")
            # gripper.activate()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper
        self._gripper_pos = 0  # Tracked locally (0-255), like joysticktst.py

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        """Get gripper position as normalized 0-1 from local tracking."""
        return self._gripper_pos / 255.0

    def get_tcp_pose_raw(self) -> np.ndarray:
        """Get the current TCP pose as [x, y, z, rx, ry, rz] (UR rotation vector).

        Returns:
            np.ndarray: (6,) TCP pose in base frame.
        """
        return np.array(self.r_inter.getActualTCPPose())

    def move_joints(
        self,
        joints: list,
        speed: float = 0.5,
        accel: float = 0.3,
    ) -> None:
        """Move to joint positions via moveJ (blocking).

        Args:
            joints: 6 joint angles in radians.
            speed: Joint speed in rad/s.
            accel: Joint acceleration in rad/s^2.
        """
        self.robot.moveJ(list(joints[:6]), speed, accel)

    def move_linear(
        self,
        pose: np.ndarray,
        speed: float = 0.1,
        accel: float = 0.5,
        asynchronous: bool = False,
    ) -> None:
        """Move the TCP linearly to a target pose.

        Uses UR moveL which handles IK internally. When asynchronous=False
        (default), blocks until motion completes. When asynchronous=True,
        returns immediately — use speed_stop() or stopL() to interrupt.

        Args:
            pose: (6,) target pose [x, y, z, rx, ry, rz] in base frame.
            speed: TCP speed in m/s.
            accel: TCP acceleration in m/s^2.
            asynchronous: If True, return immediately (non-blocking).
        """
        self.robot.moveL(list(pose), speed, accel, asynchronous)

    def move_linear_path(
        self,
        path: list,
        asynchronous: bool = False,
    ) -> None:
        """Move the TCP through a sequence of waypoints with blending.

        Each element in path is a list of 9 values:
            [x, y, z, rx, ry, rz, speed, accel, blend_radius]

        The UR controller blends between consecutive waypoints using the
        specified blend radius (meters). First and last waypoints should
        have blend_radius=0.

        Args:
            path: List of 9-element waypoints.
            asynchronous: If True, return immediately (non-blocking).
        """
        self.robot.moveL(path, asynchronous)

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state via servoJ.

        Args:
            joint_state (np.ndarray): The state to command the robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints,
            velocity,
            acceleration,
            dt,
            lookahead_time,
            gain,
        )
        if self._use_gripper:
            gripper_pos = joint_state[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)

    def command_cartesian_velocity(
        self,
        velocity: np.ndarray,
        acceleration: float = 0.5,
        time_running: float = 0.1,
        gripper_vel: float = 0.0,
    ) -> None:
        """Command TCP velocity via speedL.

        UR handles IK internally. Gripper position is tracked locally
        (no round-trip read) matching joysticktst.py behavior.

        Args:
            velocity: 6D Cartesian velocity [vx, vy, vz, wx, wy, wz]
                in base frame.
            acceleration: TCP acceleration (m/s^2).
            time_running: Time the command is active before safety
                stop (watchdog).
            gripper_vel: Gripper velocity in normalized [0,1] range
                per step. Positive = close, negative = open.
        """
        assert len(velocity) == 6, f"Expected 6D velocity, got {len(velocity)}"
        self.robot.speedL(list(velocity), acceleration, time=time_running)
        if self._use_gripper and abs(gripper_vel) > 0.001:
            self._gripper_pos = max(0, min(255, self._gripper_pos + int(gripper_vel * 255)))
            self.gripper.move(self._gripper_pos)

    def speed_stop(self) -> None:
        """Stop speedL motion immediately."""
        self.robot.speedStop()

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable, False to disable.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        gripper_pos = np.array([joints[-1]])

        # TCP pose: convert UR rotation vector to [x,y,z,qx,qy,qz,qw]
        tcp_raw = self.r_inter.getActualTCPPose()  # [x,y,z,rx,ry,rz]
        pos = np.array(tcp_raw[:3])
        quat = Rotation.from_rotvec(tcp_raw[3:6]).as_quat()  # [qx,qy,qz,qw]
        pos_quat = np.concatenate([pos, quat])

        # Joint velocities from RTDE (6D, excludes gripper)
        joint_vel = np.array(self.r_inter.getActualQd())

        return {
            "joint_positions": joints,
            "joint_velocities": joint_vel,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }


def main():
    robot_ip = "10.125.145.89"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
