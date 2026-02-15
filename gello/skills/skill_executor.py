# gello/skills/skill_executor.py
"""Skill executor for replaying pre-recorded trajectories.

Loads relative (gripper-local) waypoints and transforms them to the base frame
using the current TCP pose at execution time. Uses UR moveL for Cartesian
linear motion and polls observations from a separate read-only ZMQ port.
"""

import pickle
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

from gello.utils.transform_utils import (
    homogeneous_to_pose6d,
    pose6d_to_homogeneous,
)


class SkillExecutor:
    """Execute a pre-recorded skill trajectory with coordinate transforms.

    Args:
        skill_path: Path to the skill .pkl file containing:
            - "waypoints": list of (4, 4) relative homogeneous transforms
            - "gripper_positions": list of float gripper positions [0, 1]
            - "fps": recording frame rate (for timing reference)
        robot_client: ZMQ client on port 6001 (control -- blocking moveL).
        obs_client: ZMQ client on port 6002 (read-only -- observations).
            If None, observations are not polled during execution.
    """

    def __init__(
        self,
        skill_path: str,
        robot_client: Any,
        obs_client: Optional[Any] = None,
        move_speed: float = 0.1,
        move_accel: float = 0.5,
    ):
        self._robot_client = robot_client
        self._obs_client = obs_client
        self._move_speed = move_speed
        self._move_accel = move_accel

        # Load skill data
        with open(skill_path, "rb") as f:
            skill_data = pickle.load(f)

        self._waypoints = skill_data["waypoints"]  # list of 4x4 np.ndarray
        self._gripper_positions = skill_data.get("gripper_positions", [])
        self._fps = skill_data.get("fps", 30)

        print(
            f"[SkillExecutor] Loaded skill from {skill_path}: "
            f"{len(self._waypoints)} waypoints, "
            f"{len(self._gripper_positions)} gripper frames"
        )

    @property
    def num_waypoints(self) -> int:
        return len(self._waypoints)

    def execute(
        self,
        current_tcp_pose_raw: np.ndarray,
    ) -> Generator[Tuple[Optional[Dict[str, Any]], np.ndarray], None, None]:
        """Execute the skill trajectory, yielding (obs, target_pose) at each step.

        Transforms each relative waypoint to the base frame:
            T_target_base = T_current_base @ T_step_local

        Each moveL call blocks until the UR completes the motion.
        Observations are polled from the read-only obs server (port 6002)
        after each waypoint completes.

        Args:
            current_tcp_pose_raw: (6,) TCP pose [x,y,z,rx,ry,rz] at trigger moment.

        Yields:
            Tuple of (obs_dict or None, target_pose_6d) after each waypoint.
        """
        T_current = pose6d_to_homogeneous(current_tcp_pose_raw)
        total = len(self._waypoints)

        print(f"[SkillExecutor] Executing {total} waypoints...")

        for i, T_step_local in enumerate(self._waypoints):
            # Transform to base frame
            T_target = T_current @ T_step_local
            target_pose = homogeneous_to_pose6d(T_target)

            # Execute moveL (blocking)
            try:
                self._robot_client.move_linear(
                    pose=target_pose,
                    speed=self._move_speed,
                    accel=self._move_accel,
                )
            except Exception as e:
                print(f"[SkillExecutor] moveL failed at waypoint {i}: {e}")
                break

            # Apply gripper position if available
            if i < len(self._gripper_positions):
                grip_pos = self._gripper_positions[i]
                try:
                    # Get current joints and update gripper
                    joints = self._robot_client.get_joint_state()
                    joints_cmd = joints.copy()
                    joints_cmd[-1] = grip_pos
                    self._robot_client.command_joint_state(joints_cmd)
                except Exception as e:
                    print(f"[SkillExecutor] Gripper command failed: {e}")

            # Poll observations from read-only server
            obs = None
            if self._obs_client is not None:
                try:
                    obs = self._obs_client.get_observations()
                except Exception as e:
                    print(f"[SkillExecutor] Obs poll failed: {e}")

            # Update current pose for next step
            T_current = T_target

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"[SkillExecutor] Waypoint {i + 1}/{total} complete")

            yield obs, target_pose

        print("[SkillExecutor] Skill execution complete.")
