# gello/skills/csv_skill_executor.py
"""CSV-based skill executor with path blending and grasp verification.

Loads waypoints from a CSV file and executes them using UR moveL path
blending for smooth trajectories. Waypoints are segmented by gripper
value — each segment of consecutive same-gripper waypoints is executed
as a single blended path, with gripper changes between segments.

Key features:
  - **Path blending**: Groups waypoints by gripper value into sub-paths
    executed with moveL(path) and blend radii for smooth motion.
  - **Grasp verification**: After the verification waypoint (marked in
    CSV), checks actual gripper position. If the grasp failed (gripper
    near open), triggers auto-interrupt for human correction.
  - **Interrupt/resume**: An interrupt_event (threading.Event) can stop
    execution mid-path. Resume skips relative waypoints.

Waypoint layout:
  Relative section (rows 0..relative_count-1):
    Manipulation waypoints applied RELATIVE to the trigger TCP pose.
  Absolute section (remaining rows):
    Transfer waypoints in ABSOLUTE base-frame coordinates.
"""

import csv
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gello.utils.transform_utils import (
    align_pose6d_rotvecs,
    homogeneous_to_pose6d,
    pose6d_to_homogeneous,
)

# Thresholds for detecting that async moveL has reached its target
_POS_ARRIVED_M = 0.002  # 2mm position tolerance
_ROT_ARRIVED_RAD = 0.02  # ~1.1 deg rotation tolerance
_POLL_HZ = 50  # polling rate during async moveL
_MOVE_TIMEOUT_S = 30.0  # max time to wait for a single waypoint

# Blend radius defaults (meters)
_DEFAULT_BLEND_M = 0.002  # 2mm
_SENSITIVE_BLEND_M = 0.0001  # 0.1mm for contact approach


@dataclass
class SkillWaypoint:
    """A single waypoint from the skill CSV."""

    joints: np.ndarray  # (6,) joint angles
    tcp_pose: np.ndarray  # (6,) [x,y,z,rx,ry,rz]
    gripper_pos: int  # 0-255
    is_verification: bool = False  # True for grasp verification WP


def load_skill_csv(csv_path: str) -> List[SkillWaypoint]:
    """Load waypoints from a skill CSV file.

    Expected columns: timestamp, j0-j5, tcp_x-tcp_rz, gripper_pos, skill_id, image_file
    Verification waypoints have image_file == "verification_wp".
    """
    waypoints = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 14:
                continue
            joints = np.array([float(row[i]) for i in range(1, 7)])
            tcp = np.array([float(row[i]) for i in range(7, 13)])
            gripper = int(float(row[13]))
            is_verif = len(row) > 15 and row[15].strip() == "verification_wp"
            waypoints.append(
                SkillWaypoint(
                    joints=joints,
                    tcp_pose=tcp,
                    gripper_pos=gripper,
                    is_verification=is_verif,
                )
            )
    return waypoints


@dataclass
class GripperSegment:
    """A group of consecutive waypoints with the same gripper value."""

    start_idx: int  # index in full waypoint list
    end_idx: int  # inclusive
    gripper_pos: int
    waypoints: List[SkillWaypoint] = field(default_factory=list)
    has_verification: bool = False  # True if any WP is a verification WP


def _segment_by_gripper(waypoints: List[SkillWaypoint]) -> List[GripperSegment]:
    """Split waypoints into segments of consecutive same-gripper values."""
    if not waypoints:
        return []

    segments = []
    seg_start = 0
    seg_gripper = waypoints[0].gripper_pos

    for i in range(1, len(waypoints)):
        if waypoints[i].gripper_pos != seg_gripper or waypoints[i - 1].is_verification:
            seg = GripperSegment(
                start_idx=seg_start,
                end_idx=i - 1,
                gripper_pos=seg_gripper,
                waypoints=waypoints[seg_start:i],
                has_verification=any(
                    wp.is_verification for wp in waypoints[seg_start:i]
                ),
            )
            segments.append(seg)
            seg_start = i
            seg_gripper = waypoints[i].gripper_pos

    # Final segment
    seg = GripperSegment(
        start_idx=seg_start,
        end_idx=len(waypoints) - 1,
        gripper_pos=seg_gripper,
        waypoints=waypoints[seg_start:],
        has_verification=any(wp.is_verification for wp in waypoints[seg_start:]),
    )
    segments.append(seg)
    return segments


def _compute_blend_radii(
    segment: GripperSegment,
    is_first_segment: bool,
    is_last_segment: bool,
) -> List[float]:
    """Compute blend radii for waypoints in a segment.

    Rules:
      - First/last WP in the segment: blend = 0
      - Waypoints near gripper change boundaries (±1 of segment edge): blend = 0
      - First two WPs of the first segment: 0.1mm (sensitive contact approach)
      - Default: 2mm
    """
    n = len(segment.waypoints)
    radii = [_DEFAULT_BLEND_M] * n

    # First and last WP of each sub-path must have blend = 0
    radii[0] = 0.0
    if n > 1:
        radii[-1] = 0.0

    # Sensitive blend for first segment WPs 0-1 (contact approach)
    if is_first_segment:
        for i in range(min(2, n)):
            if radii[i] != 0.0:
                radii[i] = _SENSITIVE_BLEND_M

    return radii


class CSVSkillExecutor:
    """Execute skills from CSV files with path blending and grasp verification.

    Args:
        skill_csvs: Dict mapping skill name -> CSV file path.
        robot_client: ZMQ client (port 6001) for move_linear + gripper.
        obs_client: ZMQ client (port 6002) for observation polling.
        relative_counts: Dict mapping skill name -> number of relative waypoints.
        move_speed: moveL speed (m/s).
        move_accel: moveL acceleration (m/s^2).
    """

    def __init__(
        self,
        skill_csvs: Dict[str, str],
        robot_client: Any,
        obs_client: Optional[Any] = None,
        relative_counts: Optional[Dict[str, int]] = None,
        grasp_thresholds: Optional[Dict[str, int]] = None,
        move_speed: float = 0.1,
        move_accel: float = 0.5,
    ):
        self._robot_client = robot_client
        self._obs_client = obs_client
        self._move_speed = move_speed
        self._move_accel = move_accel
        self._grasp_thresholds = grasp_thresholds or {}

        # Load all skill CSVs
        self._skills: Dict[str, List[SkillWaypoint]] = {}
        self._relative_counts: Dict[str, int] = {}
        for name, path in skill_csvs.items():
            waypoints = load_skill_csv(path)
            self._skills[name] = waypoints
            rc = 20  # default
            if relative_counts and name in relative_counts:
                rc = relative_counts[name]
            self._relative_counts[name] = rc
            n_verif = sum(1 for wp in waypoints if wp.is_verification)
            print(
                f"[SkillExecutor] Loaded '{name}' from {path}: "
                f"{len(waypoints)} waypoints "
                f"({rc} relative + {max(0, len(waypoints) - rc)} absolute"
                f", {n_verif} verification)"
            )

    @property
    def available_skills(self) -> List[str]:
        return list(self._skills.keys())

    def has_skill(self, name: str) -> bool:
        return name in self._skills

    def _move_and_wait(
        self,
        target_pose: np.ndarray,
        interrupt_event: Optional[threading.Event] = None,
    ) -> bool:
        """Async moveL to a single waypoint with interrupt polling.

        Returns True if arrived, False if interrupted.
        """
        target = np.asarray(target_pose, dtype=np.float64)

        self._robot_client.move_linear(
            pose=target,
            speed=self._move_speed,
            accel=self._move_accel,
            asynchronous=True,
        )

        dt = 1.0 / _POLL_HZ
        t0 = time.time()
        while time.time() - t0 < _MOVE_TIMEOUT_S:
            if interrupt_event is not None and interrupt_event.is_set():
                self._robot_client.stop_linear()
                time.sleep(0.05)
                return False

            actual = np.asarray(self._robot_client.get_tcp_pose_raw(), dtype=np.float64)
            pos_err = np.linalg.norm(actual[:3] - target[:3])
            rot_err = np.linalg.norm(actual[3:] - target[3:])
            if pos_err < _POS_ARRIVED_M and rot_err < _ROT_ARRIVED_RAD:
                self._robot_client.stop_linear()
                return True

            time.sleep(dt)

        print(
            f"[SkillExecutor] WARNING: moveL timeout ({_MOVE_TIMEOUT_S}s). "
            f"Robot may not have reached target."
        )
        self._robot_client.stop_linear()
        return True

    def _move_path_and_wait(
        self,
        path: list,
        last_target: np.ndarray,
        interrupt_event: Optional[threading.Event] = None,
    ) -> bool:
        """Execute a blended path and wait for the last waypoint.

        Uses move_linear_path for paths with >1 waypoint, falls back to
        single moveL for single-waypoint paths.

        Args:
            path: List of 9-element waypoint lists for moveL(path).
            last_target: (6,) TCP pose of the last waypoint for arrival check.
            interrupt_event: If set, stops the robot.

        Returns:
            True if arrived, False if interrupted.
        """
        if len(path) == 1:
            # Single waypoint — use regular async moveL
            return self._move_and_wait(last_target, interrupt_event)

        # Multi-waypoint blended path
        self._robot_client.move_linear_path(path, asynchronous=True)

        target = np.asarray(last_target, dtype=np.float64)
        dt = 1.0 / _POLL_HZ
        t0 = time.time()
        # Longer timeout for multi-waypoint paths
        timeout = _MOVE_TIMEOUT_S * len(path)

        while time.time() - t0 < timeout:
            if interrupt_event is not None and interrupt_event.is_set():
                self._robot_client.stop_linear()
                time.sleep(0.05)
                return False

            actual = np.asarray(self._robot_client.get_tcp_pose_raw(), dtype=np.float64)
            pos_err = np.linalg.norm(actual[:3] - target[:3])
            rot_err = np.linalg.norm(actual[3:] - target[3:])
            if pos_err < _POS_ARRIVED_M and rot_err < _ROT_ARRIVED_RAD:
                self._robot_client.stop_linear()
                return True

            time.sleep(dt)

        self._robot_client.stop_linear()
        print(
            f"[SkillExecutor] WARNING: path timeout ({timeout:.0f}s). "
            f"Robot may not have reached target."
        )
        return True

    def _set_gripper(self, gripper_pos: int) -> None:
        """Set gripper to the given 0-255 position via direct socket command.

        Uses the dedicated set_gripper method which controls the gripper
        via its socket interface, avoiding RTDE servoJ conflicts with
        moveL/speedL.
        """
        try:
            self._robot_client.set_gripper(gripper_pos)
            time.sleep(0.05)
        except Exception as e:
            print(f"[SkillExecutor] Gripper command failed: {e}")

    def _set_gripper_speed(self, speed: int) -> None:
        """Set gripper finger speed (0-255) via ZMQ."""
        try:
            self._robot_client.set_gripper_speed(speed)
        except Exception as e:
            print(f"[SkillExecutor] Gripper speed command failed: {e}")

    def _check_grasp(self, threshold: int) -> Tuple[bool, int, int]:
        """Check if the grasp succeeded after verification waypoint.

        Reads the ACTUAL gripper position from hardware (GET POS). If an
        object blocks the gripper, the actual position will be below the
        threshold (object prevented full closure).

        Returns:
            (success, threshold, actual_pos_255)
        """
        actual_255 = self._robot_client.get_actual_gripper_pos()
        grasp_ok = actual_255 < threshold
        return grasp_ok, threshold, actual_255

    def execute(
        self,
        skill_name: str,
        trigger_tcp_raw: Optional[np.ndarray] = None,
        interrupt_event: Optional[threading.Event] = None,
        resume_absolute_only: bool = False,
        on_grasp_failed: Optional[Callable[[], None]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute a skill with path blending and grasp verification.

        Args:
            skill_name: Name of the skill (e.g. "cpu").
            trigger_tcp_raw: (6,) current TCP pose at trigger time.
            interrupt_event: If set by another thread, execution stops.
            resume_absolute_only: If True, skip relative waypoints.
            on_grasp_failed: Callback when grasp verification fails.
                Called BEFORE the method returns.

        Returns:
            (completed, grasp_info) tuple:
              completed: True if all waypoints done, False if interrupted.
              grasp_info: Dict with grasp_verified, grasp_commanded, grasp_actual.
                          Empty dict if no verification waypoint was reached.
        """
        if skill_name not in self._skills:
            print(f"[SkillExecutor] Unknown skill: {skill_name}")
            return True, {}

        waypoints = self._skills[skill_name]
        rel_count = self._relative_counts[skill_name]
        grasp_threshold = self._grasp_thresholds.get(skill_name, 200)
        total = len(waypoints)
        grasp_info: Dict[str, Any] = {}

        if total == 0:
            print("[SkillExecutor] No waypoints to execute.")
            return True, grasp_info

        # Slow gripper for controlled grasp
        self._set_gripper_speed(128)

        # Clear any stale interrupt
        if interrupt_event is not None:
            interrupt_event.clear()

        # Determine start index and compute transform offset
        T_offset = None
        if resume_absolute_only:
            start_idx = rel_count
            print(
                f"[SkillExecutor] RESUMING '{skill_name}' (absolute only): "
                f"waypoints {rel_count + 1}-{total}"
            )
        else:
            start_idx = 0
            T_trigger = pose6d_to_homogeneous(trigger_tcp_raw)
            T_skill_origin = pose6d_to_homogeneous(waypoints[0].tcp_pose)
            T_offset = T_trigger @ np.linalg.inv(T_skill_origin)
            print(
                f"[SkillExecutor] Executing '{skill_name}': "
                f"{total} waypoints ({rel_count} relative + "
                f"{total - rel_count} absolute)"
            )

        # Compute all target poses
        active_wps = waypoints[start_idx:]
        target_poses = []
        for i, wp in enumerate(active_wps):
            global_idx = start_idx + i
            if global_idx < rel_count and T_offset is not None:
                T_wp = pose6d_to_homogeneous(wp.tcp_pose)
                T_target = T_offset @ T_wp
                target_poses.append(homogeneous_to_pose6d(T_target))
            else:
                target_poses.append(wp.tcp_pose.copy())

        # Align rotation vectors to prevent near-pi flips.
        # The trigger TCP (or first waypoint) serves as the reference direction.
        if target_poses:
            ref_rv = np.asarray(target_poses[0][3:6], dtype=np.float64)
            target_poses = align_pose6d_rotvecs(target_poses, ref_rv)

        # Segment active waypoints by gripper value
        segments = _segment_by_gripper(active_wps)

        wp_counter = 0  # tracks progress through active_wps
        for seg_idx, seg in enumerate(segments):
            is_first = seg_idx == 0
            is_last = seg_idx == len(segments) - 1

            # Check interrupt before starting segment
            if interrupt_event is not None and interrupt_event.is_set():
                print(f"\n[SkillExecutor] *** INTERRUPTED before segment {seg_idx} ***")
                return False, grasp_info

            # Set gripper for this segment (before moving)
            gripper_changed = False
            if not is_first:
                self._set_gripper(seg.gripper_pos)
                gripper_changed = True
                print(f"[SkillExecutor] Gripper -> {seg.gripper_pos}")

            seg_poses = target_poses[wp_counter : wp_counter + len(seg.waypoints)]

            if gripper_changed:
                # After gripper change (grasp/release): use individual moveL
                # per waypoint, matching joysticktst.py. Path blending can
                # fail on short post-grasp segments (e.g. 5cm lift) because
                # the controller pre-plans the entire trajectory and the
                # actual load/position may differ from planned.
                arrived = True
                for j, pose in enumerate(seg_poses):
                    if not self._move_and_wait(pose, interrupt_event):
                        arrived = False
                        break
            else:
                # First segment: use path blending for smooth approach
                blend_radii = _compute_blend_radii(seg, is_first, is_last)
                path = []
                for j, pose in enumerate(seg_poses):
                    entry = list(pose) + [
                        self._move_speed,
                        self._move_accel,
                        blend_radii[j],
                    ]
                    path.append(entry)
                last_target = seg_poses[-1]
                arrived = self._move_path_and_wait(path, last_target, interrupt_event)

            if not arrived:
                global_idx = start_idx + wp_counter + len(seg.waypoints) - 1
                print(
                    f"\n[SkillExecutor] *** INTERRUPTED during segment "
                    f"{seg_idx} (WP {global_idx + 1}/{total}) ***"
                )
                return False, grasp_info

            wp_counter += len(seg.waypoints)
            print(
                f"[SkillExecutor] Segment {seg_idx + 1}/{len(segments)} complete "
                f"({len(seg.waypoints)} WPs, gripper={seg.gripper_pos})"
            )

            # Grasp verification after verification segment
            if seg.has_verification:
                grasp_ok, threshold, actual = self._check_grasp(grasp_threshold)
                grasp_info = {
                    "grasp_verified": grasp_ok,
                    "grasp_threshold": threshold,
                    "grasp_actual": actual,
                }
                if grasp_ok:
                    print(
                        f"[SkillExecutor] Grasp VERIFIED "
                        f"(actual={actual}, threshold={threshold})"
                    )
                else:
                    print(
                        f"\n[SkillExecutor] *** GRASP FAILED *** "
                        f"(actual={actual} >= {threshold})"
                    )
                    # Keep slow gripper speed for precise correction control
                    if on_grasp_failed is not None:
                        on_grasp_failed()
                    if interrupt_event is not None:
                        interrupt_event.set()
                    return False, grasp_info

        # Set final gripper (last segment's value)
        if segments:
            self._set_gripper(segments[-1].gripper_pos)

        self._set_gripper_speed(255)
        print(f"[SkillExecutor] Skill '{skill_name}' execution complete.")
        return True, grasp_info
