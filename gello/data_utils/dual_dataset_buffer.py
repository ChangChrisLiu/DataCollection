# gello/data_utils/dual_dataset_buffer.py
"""Dual-dataset buffer manager for the VLA training pipeline.

Manages the state machine (idle -> teleop -> skill -> done) and
generates two strictly aligned datasets from a single teleoperation
episode:

  Dataset 1 (VLA+Skill): Teleop trajectory with the final frame's
      gripper action overwritten to SKILL_SIGNAL (2.0) as a discrete
      skill-trigger token for VLA models.

  Dataset 2 (Full VLA): Seamless concatenation of un-hijacked teleop
      trajectory + skill execution trajectory.

At 30Hz control rate (matching camera frame rate), every loop iteration
is a frame -- no timestamp gating is needed.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Out-of-bounds gripper value to signal "trigger skill" to the VLA model.
# Valid gripper range is [0.0, 1.0], so 2.0 is clearly out-of-bounds.
SKILL_SIGNAL = 2.0


class DualDatasetBuffer:
    """Buffer manager for dual-dataset generation.

    Usage:
        buffer = DualDatasetBuffer()
        buffer.start_teleop()

        # Phase 1: Teleop
        while teleoperating:
            obs = env.get_obs()
            action = agent.act(obs)
            buffer.record_teleop_frame(obs, action)

        # Phase 2: Trigger
        tcp_pose = robot.get_tcp_pose_raw()
        buffer.trigger_skill(tcp_pose)

        # Phase 3: Skill
        for obs, pose in skill_executor.execute(tcp_pose):
            buffer.record_skill_frame(obs, pose)

        # Phase 4: Export
        buffer.finish()
        ds1, ds2 = buffer.export_datasets()
    """

    def __init__(self):
        self._teleop_frames: List[Dict[str, Any]] = []
        self._skill_frames: List[Dict[str, Any]] = []
        self._trigger_tcp_pose: Optional[np.ndarray] = None
        self._trigger_frame_index: int = -1
        self._phase: str = "idle"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def trigger_tcp_pose(self) -> Optional[np.ndarray]:
        return self._trigger_tcp_pose

    @property
    def num_teleop_frames(self) -> int:
        return len(self._teleop_frames)

    @property
    def num_skill_frames(self) -> int:
        return len(self._skill_frames)

    def start_teleop(self) -> None:
        """Begin recording a new episode. Clears all buffers."""
        self._teleop_frames = []
        self._skill_frames = []
        self._trigger_tcp_pose = None
        self._trigger_frame_index = -1
        self._phase = "teleop"
        print("[DualBuffer] Phase: TELEOP (recording started)")

    def record_teleop_frame(
        self,
        obs: Dict[str, Any],
        action: Any,
    ) -> None:
        """Record a single teleop frame (obs + action).

        At 30Hz control rate, every frame should be recorded.

        Args:
            obs: Observation dict from env.get_obs().
            action: Action dict or array from agent.act().
        """
        if self._phase != "teleop":
            return

        self._teleop_frames.append(
            {
                "obs": copy.deepcopy(obs),
                "action": copy.deepcopy(action),
            }
        )

    def trigger_skill(self, tcp_pose_raw: np.ndarray) -> None:
        """Signal skill trigger. Captures TCP pose and transitions state.

        Args:
            tcp_pose_raw: (6,) TCP pose [x,y,z,rx,ry,rz] at trigger moment.
        """
        if self._phase != "teleop":
            print(
                f"[DualBuffer] WARNING: trigger_skill called in "
                f"phase '{self._phase}', expected 'teleop'"
            )
            return

        self._trigger_tcp_pose = np.array(tcp_pose_raw, dtype=np.float64)
        self._trigger_frame_index = len(self._teleop_frames)
        self._phase = "skill"
        print(
            f"[DualBuffer] Phase: SKILL (triggered at frame "
            f"{self._trigger_frame_index}, TCP: {tcp_pose_raw[:3]})"
        )

    def record_skill_frame(
        self,
        obs: Optional[Dict[str, Any]],
        action: Any,
    ) -> None:
        """Record a single skill execution frame.

        Args:
            obs: Observation dict (may be None if obs polling failed).
            action: Target pose or action from skill executor.
        """
        if self._phase != "skill":
            return

        self._skill_frames.append(
            {
                "obs": copy.deepcopy(obs) if obs is not None else None,
                "action": copy.deepcopy(action),
            }
        )

    def finish(self) -> None:
        """Mark the episode as complete."""
        self._phase = "done"
        print(
            f"[DualBuffer] Phase: DONE "
            f"(teleop: {len(self._teleop_frames)} frames, "
            f"skill: {len(self._skill_frames)} frames)"
        )

    def export_datasets(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Export the two datasets.

        Returns:
            Tuple of (dataset1, dataset2):
              dataset1 (VLA+Skill): Teleop frames with hijacked gripper signal.
              dataset2 (Full VLA): Teleop + skill frames concatenated.
        """
        if self._phase != "done":
            print(
                f"[DualBuffer] WARNING: export_datasets called in "
                f"phase '{self._phase}', expected 'done'"
            )

        # Dataset 1: VLA+Skill (hijacked gripper)
        dataset1 = []
        for i, frame in enumerate(self._teleop_frames):
            frame_copy = copy.deepcopy(frame)
            # Hijack the last frame's gripper action
            if i == len(self._teleop_frames) - 1 and len(self._teleop_frames) > 0:
                action = frame_copy["action"]
                if isinstance(action, dict):
                    # Velocity-mode action: inject gripper signal
                    action["gripper_vel"] = SKILL_SIGNAL
                elif isinstance(action, np.ndarray) and len(action) > 0:
                    # Joint-position action: overwrite last element (gripper)
                    action[-1] = SKILL_SIGNAL
                frame_copy["action"] = action
            dataset1.append(frame_copy)

        # Dataset 2: Full VLA (teleop + skill, no hijack)
        dataset2 = []
        for frame in self._teleop_frames:
            dataset2.append(copy.deepcopy(frame))
        for frame in self._skill_frames:
            dataset2.append(copy.deepcopy(frame))

        print(
            f"[DualBuffer] Exported: DS1={len(dataset1)} frames "
            f"(last gripper={SKILL_SIGNAL}), "
            f"DS2={len(dataset2)} frames (seamless)"
        )
        return dataset1, dataset2

    def get_episode_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current episode."""
        return {
            "phase": self._phase,
            "num_teleop_frames": len(self._teleop_frames),
            "num_skill_frames": len(self._skill_frames),
            "trigger_frame_index": self._trigger_frame_index,
            "trigger_tcp_pose": (
                self._trigger_tcp_pose.tolist()
                if self._trigger_tcp_pose is not None
                else None
            ),
            "skill_signal_value": SKILL_SIGNAL,
        }
