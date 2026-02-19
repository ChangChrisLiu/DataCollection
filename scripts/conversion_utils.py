#!/usr/bin/env python3
# scripts/conversion_utils.py
"""Shared utilities for converting .pkl episodes to RLDS and LeRobot formats.

Supports both unified episodes (phase-labeled, single directory) and
legacy dual-dataset episodes (vla_planner/vla_executor sub-directories).

Functions:
    discover_episodes       - Find all episode dirs (unified or legacy)
    load_episode_frames     - Load all .pkl frames sorted by filename
    load_episode_metadata   - Load episode_meta.json
    filter_frames_by_phase  - Keep only frames matching given phases
    synthesize_stop_signals  - Deep-copy last frame N times with gripper=255
    synthesize_trigger_signals - Amplify last N frames as trigger signal (3x repeat)
    remove_noop_frames       - Remove frames where robot state hasn't changed
    resize_rgb_pil          - Resize RGB image using PIL LANCZOS (no cv2)
    normalize_gripper       - Convert 0-255 int -> 0.0-1.0 float
    rotvec_to_rpy           - UR rotation vector -> roll-pitch-yaw Euler angles
    compute_delta_eef       - Delta EEF between consecutive TCP poses
    compute_delta_joints    - Delta joints between consecutive frames
    validate_episode_alignment - Temporal/spatial alignment checks
"""

import copy
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Episode discovery & loading
# ---------------------------------------------------------------------------


def discover_episodes(
    data_dir: str,
    dataset_type: Optional[str] = None,
) -> List[Path]:
    """Find all episode directories.

    Supports two layouts:
      1. Unified (new): frame_*.pkl directly in episode dir
      2. Legacy: frame_*.pkl inside episode_dir/<dataset_type>/

    Args:
        data_dir: Root data directory (e.g. "data/vla_dataset").
        dataset_type: Legacy sub-directory name ("vla_planner" or "vla_executor").
                      If None, looks for unified format.

    Returns:
        Sorted list of episode directory paths.
    """
    root = Path(data_dir)
    if not root.exists():
        return []

    episodes = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("episode_"):
            continue
        # Skip failed episodes
        if d.name.endswith("_Failed"):
            continue

        if dataset_type is None:
            # Unified format: frames at top level
            if any(d.glob("frame_*.pkl")):
                episodes.append(d)
        else:
            # Legacy format: frames in sub-directory
            ds_dir = d / dataset_type
            if ds_dir.exists() and any(ds_dir.glob("frame_*.pkl")):
                episodes.append(d)

    return episodes


def load_episode_frames(
    episode_path: Path,
    dataset_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load all .pkl frames from an episode, sorted by filename.

    Args:
        episode_path: Path to episode directory.
        dataset_type: Legacy sub-directory name. If None, loads from top level.

    Returns:
        List of frame dicts sorted by filename (chronological order).
    """
    if dataset_type is not None:
        frame_dir = episode_path / dataset_type
    else:
        frame_dir = episode_path

    frame_files = sorted(frame_dir.glob("frame_*.pkl"))
    frames = []
    for fp in frame_files:
        with open(fp, "rb") as f:
            frames.append(pickle.load(f))
    return frames


def load_episode_metadata(episode_path: Path) -> Dict[str, Any]:
    """Load episode_meta.json from an episode directory.

    Returns empty dict if not found.
    """
    meta_path = episode_path / "episode_meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Phase filtering, stop signals, no-op removal
# ---------------------------------------------------------------------------


def filter_frames_by_phase(
    frames: List[Dict[str, Any]],
    phases: Set[str],
) -> List[Dict[str, Any]]:
    """Keep only frames whose 'phase' field is in the given set.

    Args:
        frames: List of phase-labeled frame dicts.
        phases: Set of phase names to keep (e.g. {"teleop", "correction"}).

    Returns:
        Filtered list preserving original order.
    """
    return [f for f in frames if f.get("phase", "teleop") in phases]


def synthesize_stop_signals(
    frames: List[Dict[str, Any]],
    num_repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Append stop-signal frames at the end.

    Deep-copies the last frame num_repeats times with gripper_pos=255.
    Used during conversion to mark "call skill now" for the planner.

    Args:
        frames: Original frame list.
        num_repeats: Number of stop-signal copies.

    Returns:
        New list with stop signals appended (original list unchanged).
    """
    if not frames:
        return []

    result = list(frames)
    last = frames[-1]
    for _ in range(num_repeats):
        stop = copy.deepcopy(last)
        stop["gripper_pos"] = 255
        stop["phase"] = "stop_signal"
        result.append(stop)

    return result


def synthesize_trigger_signals(
    frames: List[Dict[str, Any]],
    n_tail: int = 15,
    n_repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Preserve and amplify the last n_tail frames as trigger signals.

    Splits frames into main body and tail. Removes no-ops from the body only.
    Triples the tail frames and marks them with phase="trigger_signal".

    Args:
        frames: List of phase-filtered frame dicts.
        n_tail: Number of tail frames to preserve (0.5s at 30Hz = 15).
        n_repeats: How many times to repeat the tail.

    Returns:
        cleaned_main + (tail x n_repeats)
    """
    if not frames:
        return []

    if len(frames) <= n_tail:
        # Not enough frames — use all as tail
        main, tail = [], list(frames)
    else:
        main = frames[:-n_tail]
        tail = frames[-n_tail:]

    # Remove no-ops from main only (tail preserved as-is)
    main = remove_noop_frames(main)

    # Repeat the tail, mark as trigger signal
    trigger_frames = []
    for _ in range(n_repeats):
        for f in tail:
            sig = copy.deepcopy(f)
            sig["phase"] = "trigger_signal"
            trigger_frames.append(sig)

    return main + trigger_frames


def remove_noop_frames(
    frames: List[Dict[str, Any]],
    joint_threshold: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Remove consecutive frames where the robot state hasn't changed.

    Always keeps the first frame and any frame where the max joint
    position delta exceeds the threshold.

    Args:
        frames: List of frame dicts.
        joint_threshold: Max per-joint delta (radians) below which
                         the frame is considered a no-op.

    Returns:
        Filtered list with no-op frames removed.
    """
    if not frames:
        return []

    filtered = [frames[0]]
    for f in frames[1:]:
        prev = filtered[-1]
        prev_j = np.array(prev["joint_positions"][:6])
        curr_j = np.array(f["joint_positions"][:6])
        j_delta = np.max(np.abs(curr_j - prev_j))
        if j_delta > joint_threshold:
            filtered.append(f)

    return filtered


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def resize_rgb_pil(img: np.ndarray, size: tuple) -> np.ndarray:
    """Resize an RGB uint8 image using PIL LANCZOS resampling.

    Args:
        img: (H, W, 3) uint8 RGB numpy array.
        size: Target (width, height) tuple.

    Returns:
        (height, width, 3) uint8 RGB numpy array.
    """
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)


# ---------------------------------------------------------------------------
# State / action helpers
# ---------------------------------------------------------------------------


def normalize_gripper(val: int) -> float:
    """Convert gripper position 0-255 -> 0.0-1.0."""
    return float(val) / 255.0


def rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
    """Convert a UR rotation vector to roll-pitch-yaw Euler angles.

    Args:
        rotvec: (3,) rotation vector [rx, ry, rz].

    Returns:
        (3,) array [roll, pitch, yaw] in radians (XYZ extrinsic convention).
    """
    return Rotation.from_rotvec(rotvec).as_euler("xyz")


def compute_delta_eef(tcp_t: List[float], tcp_t1: List[float]) -> np.ndarray:
    """Compute delta EEF between consecutive TCP poses.

    Args:
        tcp_t:  [x, y, z, rx, ry, rz] at time t.
        tcp_t1: [x, y, z, rx, ry, rz] at time t+1.

    Returns:
        (6,) array [dx, dy, dz, drx, dry, drz].
    """
    return np.array(tcp_t1, dtype=np.float64) - np.array(tcp_t, dtype=np.float64)


def compute_delta_joints(
    joints_t: List[float],
    joints_t1: List[float],
) -> np.ndarray:
    """Compute delta joints between consecutive frames.

    Args:
        joints_t:  [j0..j5] at time t.
        joints_t1: [j0..j5] at time t+1.

    Returns:
        (6,) array [dj0..dj5].
    """
    return np.array(joints_t1[:6], dtype=np.float64) - np.array(
        joints_t[:6], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Alignment validation
# ---------------------------------------------------------------------------


@dataclass
class AlignmentReport:
    """Result of temporal/spatial alignment validation for one episode."""

    episode_path: str
    num_frames: int
    num_timing_warnings: int = 0
    num_camera_warnings: int = 0
    num_action_warnings: int = 0
    flagged_frame_indices: List[int] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.num_timing_warnings == 0
            and self.num_camera_warnings == 0
            and self.num_action_warnings == 0
        )


def validate_episode_alignment(
    frames: List[Dict[str, Any]],
    episode_path: str = "",
    expected_hz: float = 30.0,
    max_dt_deviation: float = 0.5,
    max_camera_offset_s: float = 0.012,
    max_joint_jump_rad: float = 0.5,
    max_tcp_jump_m: float = 0.05,
) -> AlignmentReport:
    """Validate temporal and spatial alignment of an episode's frames.

    Checks three categories:
    1. Frame timing consistency (inter-frame dt vs expected 1/hz).
    2. Camera-to-robot timestamp offset (if camera timestamps present).
    3. Action smoothness (joint and TCP jumps between consecutive frames).

    Args:
        frames: List of frame dicts from load_episode_frames().
        episode_path: Episode path string for the report.
        expected_hz: Expected recording rate in Hz.
        max_dt_deviation: Max fractional deviation from expected dt (0.5 = 50%).
        max_camera_offset_s: Max camera-robot timestamp offset in seconds.
        max_joint_jump_rad: Max per-joint position jump in radians.
        max_tcp_jump_m: Max TCP position jump in meters.

    Returns:
        AlignmentReport with warning counts and flagged frame indices.
    """
    report = AlignmentReport(episode_path=episode_path, num_frames=len(frames))

    if len(frames) < 2:
        return report

    expected_dt = 1.0 / expected_hz
    flagged = set()
    has_camera_ts = "wrist_timestamp" in frames[0] or "base_timestamp" in frames[0]

    if not has_camera_ts:
        report.details.append(
            "No camera timestamps found (old format) — skipping camera offset checks."
        )

    for i in range(len(frames)):
        frame = frames[i]

        # --- Check 1: Frame timing (needs consecutive pair) ---
        if i < len(frames) - 1:
            next_frame = frames[i + 1]
            ts_curr = frame.get("timestamp", 0.0)
            ts_next = next_frame.get("timestamp", 0.0)
            if ts_curr > 0 and ts_next > 0:
                dt = ts_next - ts_curr
                if dt > 0 and abs(dt - expected_dt) / expected_dt > max_dt_deviation:
                    report.num_timing_warnings += 1
                    flagged.add(i)
                    report.details.append(
                        f"  Frame {i}: dt={dt*1000:.1f}ms "
                        f"(expected {expected_dt*1000:.1f}ms)"
                    )

        # --- Check 2: Camera-robot timestamp offset ---
        if has_camera_ts:
            frame_ts = frame.get("timestamp", 0.0)
            for cam_name in ("wrist", "base"):
                cam_ts = frame.get(f"{cam_name}_timestamp")
                if cam_ts is not None and frame_ts > 0 and cam_ts > 0:
                    offset = abs(cam_ts - frame_ts)
                    if offset > max_camera_offset_s:
                        report.num_camera_warnings += 1
                        flagged.add(i)
                        report.details.append(
                            f"  Frame {i}: {cam_name} camera offset "
                            f"{offset*1000:.1f}ms > {max_camera_offset_s*1000:.0f}ms"
                        )

        # --- Check 3: Action smoothness (needs consecutive pair) ---
        if i < len(frames) - 1:
            next_frame = frames[i + 1]

            joints_curr = np.array(frame.get("joint_positions", [0] * 6))
            joints_next = np.array(next_frame.get("joint_positions", [0] * 6))
            joint_delta = np.max(np.abs(joints_next - joints_curr))
            if joint_delta > max_joint_jump_rad:
                report.num_action_warnings += 1
                flagged.add(i)
                report.details.append(
                    f"  Frame {i}: joint jump {np.degrees(joint_delta):.1f}deg "
                    f"> {np.degrees(max_joint_jump_rad):.1f}deg"
                )

            tcp_curr = np.array(frame.get("tcp_pose", [0] * 6)[:3])
            tcp_next = np.array(next_frame.get("tcp_pose", [0] * 6)[:3])
            tcp_delta = np.linalg.norm(tcp_next - tcp_curr)
            if tcp_delta > max_tcp_jump_m:
                report.num_action_warnings += 1
                flagged.add(i)
                report.details.append(
                    f"  Frame {i}: TCP jump {tcp_delta*1000:.1f}mm "
                    f"> {max_tcp_jump_m*1000:.0f}mm"
                )

    report.flagged_frame_indices = sorted(flagged)
    return report


def print_alignment_report(report: AlignmentReport) -> None:
    """Print a human-readable summary of an alignment report."""
    total_warnings = (
        report.num_timing_warnings
        + report.num_camera_warnings
        + report.num_action_warnings
    )
    status = "PASSED" if report.passed else "CHECK NEEDED"
    print(
        f"  {Path(report.episode_path).name}: "
        f"{report.num_frames} frames, "
        f"{total_warnings} warnings — {status}"
    )
    if not report.passed:
        print(
            f"    Timing: {report.num_timing_warnings}, "
            f"Camera: {report.num_camera_warnings}, "
            f"Action: {report.num_action_warnings}"
        )
        for detail in report.details[:10]:
            print(detail)
        if len(report.details) > 10:
            print(f"    ... and {len(report.details) - 10} more warnings")
