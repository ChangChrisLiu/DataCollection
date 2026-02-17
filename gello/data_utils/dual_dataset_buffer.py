# gello/data_utils/dual_dataset_buffer.py
"""Dual-dataset buffer for VLA data collection pipeline.

Manages frame recording across two phases (teleop + skill execution) and
exports two datasets from a single round of data collection:

  Dataset A (VLA Planner):
    Teleop frames + 3 stop-signal frames (same pose, gripper=255).
    Teaches the VLA model WHEN to call a skill.

  Dataset B (VLA Executor):
    Teleop frames (no stop signals) + skill execution frames.
    Teaches the VLA model the COMPLETE motion end-to-end.

Frame format (each element in the lists):
    {
        "timestamp": float,
        "joint_positions": list[float],   # 6 joint angles
        "tcp_pose": list[float],          # [x,y,z,rx,ry,rz]
        "gripper_pos": int,               # 0-255
        "wrist_rgb": np.ndarray,          # (H, W, 3) uint8
        "wrist_depth": np.ndarray,        # (H, W, 1) uint16
        "base_rgb": np.ndarray,           # (H, W, 3) uint8
        "base_depth": np.ndarray,         # (H, W, 1) uint16
    }
"""

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Gripper value used as stop signal (out-of-range for 0-255 gripper)
STOP_SIGNAL_GRIPPER = 255


class DualDatasetBuffer:
    """Buffer for dual-dataset VLA data collection."""

    def __init__(self):
        self._teleop_frames: List[Dict[str, Any]] = []
        self._stop_frames: List[Dict[str, Any]] = []
        self._skill_frames: List[Dict[str, Any]] = []
        self._phase: str = "idle"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def num_teleop_frames(self) -> int:
        return len(self._teleop_frames)

    @property
    def num_skill_frames(self) -> int:
        return len(self._skill_frames)

    def start(self) -> None:
        """Begin a new recording round. Clears all buffers."""
        self._teleop_frames = []
        self._stop_frames = []
        self._skill_frames = []
        self._phase = "teleop"
        print("[Buffer] Recording started (phase: teleop)")

    def record_teleop_frame(self, frame: Dict[str, Any]) -> None:
        """Record a single teleop frame."""
        if self._phase != "teleop":
            return
        self._teleop_frames.append(frame)

    def insert_stop_signal(self, num_repeats: int = 3) -> None:
        """Insert stop-signal frames at the end of teleop.

        Creates num_repeats copies of the last teleop frame with gripper=255.
        Transitions phase from teleop to skill.
        """
        if self._phase != "teleop" or len(self._teleop_frames) == 0:
            print("[Buffer] WARNING: Cannot insert stop signal (no teleop frames)")
            return

        last_frame = self._teleop_frames[-1]
        for _ in range(num_repeats):
            stop_frame = copy.deepcopy(last_frame)
            stop_frame["gripper_pos"] = STOP_SIGNAL_GRIPPER
            self._stop_frames.append(stop_frame)

        self._phase = "skill"
        print(
            f"[Buffer] Inserted {num_repeats} stop-signal frames "
            f"(gripper={STOP_SIGNAL_GRIPPER}). Phase: skill"
        )

    def record_skill_frame(self, frame: Dict[str, Any]) -> None:
        """Record a single skill-execution frame."""
        if self._phase not in ("skill", "post_skill"):
            return
        self._skill_frames.append(frame)

    def finish_skill(self) -> None:
        """Mark skill execution as complete. Recording continues for post-skill."""
        if self._phase == "skill":
            self._phase = "post_skill"
            print(
                f"[Buffer] Skill done. {len(self._skill_frames)} skill frames. "
                f"Phase: post_skill (waiting for home)"
            )

    def export(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Export two datasets and reset.

        Returns:
            (ds_planner, ds_executor):
              ds_planner: teleop + stop signals (VLA high-level planner)
              ds_executor: teleop + skill frames (VLA full executor)
        """
        ds_planner = list(self._teleop_frames) + list(self._stop_frames)
        ds_executor = list(self._teleop_frames) + list(self._skill_frames)

        print(
            f"[Buffer] Exported: "
            f"planner={len(ds_planner)} frames "
            f"({len(self._teleop_frames)} teleop + {len(self._stop_frames)} stop), "
            f"executor={len(ds_executor)} frames "
            f"({len(self._teleop_frames)} teleop + {len(self._skill_frames)} skill)"
        )

        # Reset
        self._teleop_frames = []
        self._stop_frames = []
        self._skill_frames = []
        self._phase = "idle"

        return ds_planner, ds_executor

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current recording."""
        return {
            "phase": self._phase,
            "num_teleop_frames": len(self._teleop_frames),
            "num_stop_frames": len(self._stop_frames),
            "num_skill_frames": len(self._skill_frames),
            "stop_signal_value": STOP_SIGNAL_GRIPPER,
        }
