# gello/data_utils/episode_buffer.py
"""Unified episode buffer with phase-labeled recording.

Replaces the dual-dataset buffer with a single frame stream annotated
by phase labels. Five recording phases are supported:

  - ``armed``:        Recording ready, waiting for joystick input
  - ``teleop``:       Human joystick control (approach)
  - ``skill``:        Autonomous skill execution
  - ``correction``:   Human correction after failed grasp
  - ``skill_resume``: Skill resumes absolute waypoints after correction

Stop signals are NOT stored here — they are synthesized during conversion
(3 copies of the last frame in the relevant phase with gripper=255).

Frame format (each element in the frame list):
    {
        "timestamp": float,
        "phase": str,                  # set by record_frame()
        "joint_positions": list[float],
        "tcp_pose": list[float],
        "gripper_pos": int,            # 0-255
        "wrist_rgb": np.ndarray,       # (256, 256, 3) uint8
        "wrist_timestamp": float,
        "base_rgb": np.ndarray,        # (256, 256, 3) uint8
        "base_timestamp": float,
    }
"""

from typing import Any, Dict, List, Tuple

VALID_PHASES = {"idle", "armed", "teleop", "skill", "correction", "skill_resume"}


class EpisodeBuffer:
    """Unified buffer for phase-labeled episode recording."""

    def __init__(self):
        self._frames: List[Dict[str, Any]] = []
        self._phase: str = "idle"
        self._phase_segments: List[Dict[str, Any]] = []
        self._segment_start: int = 0

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def start(self) -> None:
        """Begin a new recording episode. Clears all buffers."""
        self._frames = []
        self._phase_segments = []
        self._phase = "armed"
        self._segment_start = 0
        print("[Buffer] Recording armed (waiting for joystick input)")

    def set_phase(self, phase: str) -> None:
        """Transition to a new phase, closing the current segment."""
        if phase not in VALID_PHASES:
            raise ValueError(f"Invalid phase '{phase}'. Must be one of {VALID_PHASES}")
        if phase == self._phase:
            return

        # Close current segment if we have frames in it
        if self._frames and self._segment_start < len(self._frames):
            self._phase_segments.append(
                {
                    "phase": self._phase,
                    "start": self._segment_start,
                    "end": len(self._frames) - 1,
                }
            )

        self._phase = phase
        self._segment_start = len(self._frames)
        print(f"[Buffer] Phase -> {phase}")

    def record_frame(self, frame: Dict[str, Any]) -> None:
        """Record a single frame with the current phase label."""
        if self._phase in ("idle", "armed"):
            return
        frame["phase"] = self._phase
        self._frames.append(frame)

    def export(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Export all frames and phase segments, then reset.

        Returns:
            (frames, phase_segments) tuple.
        """
        # Close final segment
        if self._frames and self._segment_start < len(self._frames):
            self._phase_segments.append(
                {
                    "phase": self._phase,
                    "start": self._segment_start,
                    "end": len(self._frames) - 1,
                }
            )

        frames = list(self._frames)
        segments = list(self._phase_segments)

        print(
            f"[Buffer] Exported {len(frames)} frames across "
            f"{len(segments)} phase segments"
        )

        # Reset
        self._frames = []
        self._phase_segments = []
        self._phase = "idle"
        self._segment_start = 0

        return frames, segments

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current recording (call BEFORE export)."""
        # Build a summary of frames per phase
        phase_counts: Dict[str, int] = {}
        for f in self._frames:
            p = f.get("phase", "unknown")
            phase_counts[p] = phase_counts.get(p, 0) + 1

        return {
            "phase": self._phase,
            "num_frames": len(self._frames),
            "phase_counts": phase_counts,
        }
