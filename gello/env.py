# gello/env.py (最终、正确、异步版)
import time
from typing import Any, Dict, Optional

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        # [已修复] 恢复到原始的 __init__ 
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """
        [已修复] 步骤环境。
        恢复到 gello 的原始逻辑：
        它将完整的 (N)-DOF 动作向量（例如 7-DOF）发送到机器人。
        """
        # (这是原始的、正确的断言)
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"输入动作长度 ({len(joints)}) 与机器人总自由度 ({self._robot.num_dofs()}) 不匹配"
        
        # [关键修复] 恢复到原始逻辑。
        # T1 (URRobot) 显然期望一个 7-DOF (6+1) 的完整数组。
        self._robot.command_joint_state(joints)
        
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """
        [已修改] 获取观测。
        (此函数已在方案 B 中被修改，以处理异步相机时间戳)
        """
        observations = {}
        
        # <--- [方案 B 的唯一修改] ---
        if self._camera_dict:
            for name, camera in self._camera_dict.items():
                # 我们新的 ZMQClientCamera.read() 是非阻塞的 (100Hz)
                # 它返回 (timestamp, image, depth)
                timestamp, image, depth = camera.read()
                
                # 存储所有三个，以便后续进行时间戳对齐
                # (使用原始的 _rgb 命名)
                observations[f"{name}_timestamp"] = timestamp
                observations[f"{name}_rgb"] = image
                observations[f"{name}_depth"] = depth
        # <--- [修改结束] ---

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()


