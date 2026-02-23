# gello/agents/safety.py
"""Workspace and motion safety checks for VLA inference."""

import numpy as np


class SafetyMonitor:
    """Clamp actions and enforce workspace bounds for VLA inference.

    Joint limits are the UR5e defaults (±2π). Workspace bounds should be
    tuned per setup — the defaults here are conservative.
    """

    # UR5e joint limits (radians) — symmetric ±2π
    JOINT_LIMITS_LOW = np.array([-2 * np.pi] * 6)
    JOINT_LIMITS_HIGH = np.array([2 * np.pi] * 6)

    def __init__(
        self,
        workspace_bounds: dict | None = None,
        max_joint_delta: float = 0.05,
        max_eef_delta: float = 0.01,
        max_rotation_delta: float = 0.05,
    ):
        """
        Args:
            workspace_bounds: dict with 'low' and 'high' keys, each (3,) array
                for [x, y, z] in meters.  None = no workspace check.
            max_joint_delta: max change per joint per step (rad).
            max_eef_delta: max EEF translation per step (m).
            max_rotation_delta: max EEF rotation per step (rad).
        """
        if workspace_bounds is not None:
            self.ws_low = np.asarray(workspace_bounds["low"], dtype=np.float64)
            self.ws_high = np.asarray(workspace_bounds["high"], dtype=np.float64)
        else:
            self.ws_low = None
            self.ws_high = None
        self.max_joint_delta = max_joint_delta
        self.max_eef_delta = max_eef_delta
        self.max_rotation_delta = max_rotation_delta

    def check_joint_action(
        self, delta_joints: np.ndarray, current_joints: np.ndarray
    ) -> np.ndarray:
        """Clamp joint deltas and enforce joint limits.

        Args:
            delta_joints: (6,) proposed joint deltas.
            current_joints: (6,) current joint positions.

        Returns:
            (6,) safe joint deltas (clamped).
        """
        clamped = np.clip(delta_joints, -self.max_joint_delta, self.max_joint_delta)
        target = current_joints + clamped
        target = np.clip(target, self.JOINT_LIMITS_LOW, self.JOINT_LIMITS_HIGH)
        return target - current_joints

    def check_eef_action(
        self, delta_pos: np.ndarray, delta_rot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clamp EEF deltas.

        Args:
            delta_pos: (3,) proposed position delta [dx, dy, dz] in meters.
            delta_rot: (3,) proposed rotation delta [dr, dp, dy] in radians.

        Returns:
            (safe_delta_pos, safe_delta_rot) tuple.
        """
        norm_pos = np.linalg.norm(delta_pos)
        if norm_pos > self.max_eef_delta:
            delta_pos = delta_pos * (self.max_eef_delta / norm_pos)

        norm_rot = np.linalg.norm(delta_rot)
        if norm_rot > self.max_rotation_delta:
            delta_rot = delta_rot * (self.max_rotation_delta / norm_rot)

        return delta_pos, delta_rot

    def check_target_pose(self, target_pose: np.ndarray) -> bool:
        """Check if target pose is within workspace bounds.

        Args:
            target_pose: (6,) [x, y, z, rx, ry, rz].

        Returns:
            True if within bounds (or no bounds configured).
        """
        if self.ws_low is None:
            return True
        pos = target_pose[:3]
        return bool(np.all(pos >= self.ws_low) and np.all(pos <= self.ws_high))
