# gello/env.py
"""Robot environment with async camera support."""
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
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        return self._robot

    def __len__(self):
        return 0

    def step(self, action) -> Dict[str, Any]:
        """Send a command to the robot and return observations.

        Args:
            action: Either a joint-position np.ndarray (for servoJ agents like
                    GELLO/SpaceMouse), or a dict with 'type' key for velocity
                    agents (joystick HOSAS):
                      {'type': 'velocity', 'velocity': ndarray(6), ...}
                      {'type': 'skill', 'skill': 'home'|'reorient'}
        """
        if isinstance(action, dict):
            action_type = action.get("type")
            if action_type == "velocity":
                self._robot.command_cartesian_velocity(
                    velocity=np.array(action["velocity"]),
                    acceleration=action.get("acceleration", 0.5),
                    time_running=action.get("time", 0.1),
                    gripper_vel=action.get("gripper_vel", 0.0),
                )
            elif action_type == "skill":
                self._handle_skill(action["skill"])
            else:
                raise ValueError(f"Unknown action type: {action_type}")
        else:
            # Joint-position command (servoJ)
            assert len(action) == self._robot.num_dofs(), (
                f"Action length ({len(action)}) != robot DOF ({self._robot.num_dofs()})"
            )
            self._robot.command_joint_state(action)

        self._rate.sleep()
        return self.get_obs()

    def _handle_skill(self, skill: str) -> None:
        """Execute a robot skill (blocking motion)."""
        from gello.agents.joystick_agent import HOME_JOINTS_RAD

        if skill == "home":
            print("[SKILL] Moving to home position...")
            self._robot.speed_stop()
            # Move to home via servoJ in small steps
            current = self._robot.get_joint_state()[:6]
            home = np.array(HOME_JOINTS_RAD)
            steps = max(int(np.abs(current - home).max() / 0.01), 50)
            gripper_pos = self._robot.get_joint_state()[-1]
            for jnt in np.linspace(current, home, steps):
                full_cmd = np.append(jnt, gripper_pos)
                self._robot.command_joint_state(full_cmd)
                import time as _time
                _time.sleep(0.002)
            print("[SKILL] Home reached.")

        elif skill == "reorient":
            print("[SKILL] Vertical reorient requested.")
            # Reorient is robot-specific; for now just stop motion
            self._robot.speed_stop()
            print("[SKILL] Reorient: use moveL externally or extend this method.")

        else:
            print(f"[SKILL] Unknown skill: {skill}")

    def get_obs(self) -> Dict[str, Any]:
        """Get observations from all cameras and robot state.

        Camera observations include timestamps for synchronization.
        ZMQClientCamera.read() returns (timestamp, image, depth).
        """
        observations = {}

        if self._camera_dict:
            for name, camera in self._camera_dict.items():
                timestamp, image, depth = camera.read()
                observations[f"{name}_timestamp"] = timestamp
                observations[f"{name}_rgb"] = image
                observations[f"{name}_depth"] = depth

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
