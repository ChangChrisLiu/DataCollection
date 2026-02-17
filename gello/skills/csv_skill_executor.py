# gello/skills/csv_skill_executor.py
"""CSV-based skill executor for replaying recorded manipulation trajectories.

Loads waypoints from a CSV file (same format as joysticktst.py recordings).
The CSV is split into two sections:

  Relative section (rows 0..relative_count-1):
    Manipulation waypoints applied RELATIVE to the trigger TCP pose.
    This allows the same manipulation to work regardless of where
    the robot is positioned when the skill is triggered.

  Absolute section (remaining rows):
    Transfer waypoints in ABSOLUTE base-frame coordinates.
    Used to move the component to a fixed destination.

After the final waypoint, the gripper opens (part of the CSV) to release.

Interrupt support:
  Uses asynchronous moveL so the robot can be stopped mid-waypoint via
  speed_stop(). An interrupt_event (threading.Event) is polled at ~50Hz
  between waypoints and DURING each moveL motion.
"""

import csv
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gello.utils.transform_utils import (
    homogeneous_to_pose6d,
    pose6d_to_homogeneous,
)

# Thresholds for detecting that async moveL has reached its target
_POS_ARRIVED_M = 0.002     # 2mm position tolerance
_ROT_ARRIVED_RAD = 0.02    # ~1.1 deg rotation tolerance
_POLL_HZ = 50              # polling rate during async moveL
_MOVE_TIMEOUT_S = 30.0     # max time to wait for a single waypoint


@dataclass
class SkillWaypoint:
    """A single waypoint from the skill CSV."""

    joints: np.ndarray       # (6,) joint angles
    tcp_pose: np.ndarray     # (6,) [x,y,z,rx,ry,rz]
    gripper_pos: int         # 0-255


def load_skill_csv(csv_path: str) -> List[SkillWaypoint]:
    """Load waypoints from a skill CSV file.

    Expected columns: timestamp, j0-j5, tcp_x-tcp_rz, gripper_pos, skill_id, image_file
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
            waypoints.append(SkillWaypoint(joints=joints, tcp_pose=tcp, gripper_pos=gripper))
    return waypoints


class CSVSkillExecutor:
    """Execute skills from CSV files with relative/absolute waypoint handling.

    Uses asynchronous moveL for responsive interrupt support. Each moveL
    returns immediately and the executor polls TCP position until the
    robot arrives or an interrupt is detected.

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
        move_speed: float = 0.1,
        move_accel: float = 0.5,
    ):
        self._robot_client = robot_client
        self._obs_client = obs_client
        self._move_speed = move_speed
        self._move_accel = move_accel

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
            print(
                f"[SkillExecutor] Loaded '{name}' from {path}: "
                f"{len(waypoints)} waypoints "
                f"({rc} relative + {max(0, len(waypoints) - rc)} absolute)"
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
        """Async moveL with interrupt polling.

        Sends moveL(asynchronous=True), then polls TCP position at ~50Hz
        until the robot arrives or interrupt_event is set.

        Args:
            target_pose: (6,) target [x,y,z,rx,ry,rz].
            interrupt_event: If set, robot is stopped immediately.

        Returns:
            True if arrived, False if interrupted.
        """
        target = np.asarray(target_pose, dtype=np.float64)

        # Start async move
        self._robot_client.move_linear(
            pose=target,
            speed=self._move_speed,
            accel=self._move_accel,
            asynchronous=True,
        )

        # Poll until arrived or interrupted
        dt = 1.0 / _POLL_HZ
        t0 = time.time()
        while time.time() - t0 < _MOVE_TIMEOUT_S:
            # Check interrupt
            if interrupt_event is not None and interrupt_event.is_set():
                self._robot_client.speed_stop()
                time.sleep(0.05)
                return False

            # Check if robot has reached target
            actual = np.asarray(
                self._robot_client.get_tcp_pose_raw(), dtype=np.float64
            )
            pos_err = np.linalg.norm(actual[:3] - target[:3])
            rot_err = np.linalg.norm(actual[3:] - target[3:])
            if pos_err < _POS_ARRIVED_M and rot_err < _ROT_ARRIVED_RAD:
                return True

            time.sleep(dt)

        # Timeout — robot may be stuck or protective-stopped
        print(
            f"[SkillExecutor] WARNING: moveL timeout ({_MOVE_TIMEOUT_S}s). "
            f"Robot may not have reached target."
        )
        return True

    def execute(
        self,
        skill_name: str,
        trigger_tcp_raw: Optional[np.ndarray] = None,
        interrupt_event: Optional[threading.Event] = None,
        resume_absolute_only: bool = False,
    ) -> bool:
        """Execute a skill. Blocking — runs waypoints to completion or interruption.

        Uses async moveL internally so the robot can be stopped mid-motion
        when interrupt_event is set.

        Args:
            skill_name: Name of the skill (e.g. "cpu").
            trigger_tcp_raw: (6,) current TCP pose at trigger. Required unless
                resume_absolute_only=True.
            interrupt_event: If set by another thread, execution stops and the
                robot halts immediately (mid-waypoint).
            resume_absolute_only: If True, skip relative waypoints and execute
                only absolute (base-frame) waypoints.

        Returns:
            True if all waypoints completed, False if interrupted.
        """
        if skill_name not in self._skills:
            print(f"[SkillExecutor] Unknown skill: {skill_name}")
            return True

        waypoints = self._skills[skill_name]
        rel_count = self._relative_counts[skill_name]
        total = len(waypoints)

        if total == 0:
            print("[SkillExecutor] No waypoints to execute.")
            return True

        # Clear any stale interrupt from before this execution
        if interrupt_event is not None:
            interrupt_event.clear()

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
                f"{total} waypoints ({rel_count} relative + {total - rel_count} absolute)"
            )

        for i in range(start_idx, total):
            wp = waypoints[i]

            # Compute target TCP pose
            if i < rel_count and T_offset is not None:
                T_wp = pose6d_to_homogeneous(wp.tcp_pose)
                T_target = T_offset @ T_wp
                target_pose = homogeneous_to_pose6d(T_target)
            else:
                target_pose = wp.tcp_pose.copy()

            # Async moveL with interrupt polling
            try:
                arrived = self._move_and_wait(target_pose, interrupt_event)
            except Exception as e:
                print(f"[SkillExecutor] moveL failed at waypoint {i}: {e}")
                return False

            if not arrived:
                print(
                    f"\n[SkillExecutor] *** INTERRUPTED during waypoint "
                    f"{i + 1}/{total} ***"
                )
                return False

            # Set gripper position (robot is stopped, safe to use servoJ)
            try:
                joints = self._obs_client.get_joint_state() if self._obs_client else \
                    self._robot_client.get_joint_state()
                joints_cmd = joints.copy()
                joints_cmd[-1] = wp.gripper_pos / 255.0  # normalize to 0-1
                self._robot_client.command_joint_state(joints_cmd)
                time.sleep(0.05)  # brief settle time for gripper
            except Exception as e:
                print(f"[SkillExecutor] Gripper command failed at waypoint {i}: {e}")

            if (i + 1) % 5 == 0 or i == total - 1:
                print(
                    f"[SkillExecutor] Waypoint {i + 1}/{total} complete "
                    f"(gripper={wp.gripper_pos})"
                )

        print(f"[SkillExecutor] Skill '{skill_name}' execution complete.")
        return True
