# gello/agents/joystick_agent.py
"""Thrustmaster SOL-R2 HOSAS dual-stick agent for UR5e teleoperation.

This agent reads from a Thrustmaster SOL-R2 HOSAS (Hands-On-Stick-And-Stick)
dual flight controller and maps the inputs to Cartesian velocity commands
for the UR5e robot arm.

Hardware Mapping (Thrustmaster SOL-R2 HOSAS):
  Left Stick:
    Axis 0 (X) + Axis 1 (Y) → TCP translation X/Y (forward/back, left/right)
    Axis 3 (slider/mini-stick) → speed gain multiplier
    Mini-stick Y → gripper open/close
    Button 0 (trigger) → toggle data recording

  Right Stick:
    Axis 0 (X) + Axis 1 (Y) → TCP rotation Rx/Ry (roll/pitch)
    Axis 2 (twist) → TCP rotation Rz (yaw)
    Axis 3 (mini-stick Y) → TCP translation Z (up/down)
    Button 2 → skill: vertical reorient
    Button 3 → skill: go to home position

The agent converts Cartesian velocity to joint-space targets using MuJoCo IK,
similar to the SpacemouseAgent approach.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from gello.agents.agent import Agent


# ============================================================================
# Coordinate transforms (same as spacemouse_agent.py for UR5e)
# MuJoCo has a slightly different coordinate system than UR control box
# ============================================================================
mj2ur = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
ur2mj = np.linalg.inv(mj2ur)

# Controller-to-UR transform (adjust signs to match your physical mounting)
hosas2ur = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1],
])
ur2hosas = np.linalg.inv(hosas2ur)


def _apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]


# ============================================================================
# Default home position for UR5e (in joint radians)
# ============================================================================
HOME_JOINTS_DEG = [-90, -90, 90, -90, -90, 0]
HOME_JOINTS_RAD = [np.deg2rad(x) for x in HOME_JOINTS_DEG]


@dataclass
class HOSASConfig:
    """Configuration for Thrustmaster SOL-R2 HOSAS mapping."""
    # Speed limits
    max_speed_linear: float = 0.25    # m/s max TCP translation speed
    max_speed_angular: float = 0.50   # rad/s max TCP rotation speed

    # Deadzone for all axes
    deadzone: float = 0.05

    # Minimum gain when speed slider is at zero
    min_gain: float = 0.1

    # Gripper step per control cycle (0-1 scale, ~3/255 matches Gemini's GRIPPER_STEP=3)
    gripper_step: float = 0.012

    # Axis assignments for the right stick
    mini_stick_y_axis: int = 3  # mini-stick Y axis index (for Z translation)

    # IK rotation mode
    rotation_mode: str = "euler"  # "euler" or "rpy"


class JoystickAgent(Agent):
    """Thrustmaster SOL-R2 HOSAS dual-stick agent for UR5e (6-DOF + gripper).

    Uses Cartesian velocity input mapped through MuJoCo IK to produce
    joint-space commands, similar to SpacemouseAgent.
    """

    def __init__(
        self,
        robot_type: str = "ur5",
        config: Optional[HOSASConfig] = None,
        num_dofs: int = 7,
        left_index: int = 0,
        right_index: int = 1,
        verbose: bool = False,
    ) -> None:
        self.config = config or HOSASConfig()
        self.num_dofs = num_dofs
        self._verbose = verbose
        self._left_index = left_index
        self._right_index = right_index
        self._lock = threading.Lock()

        # Joystick state
        self._left_axes = np.zeros(6, dtype=float)
        self._right_axes = np.zeros(6, dtype=float)
        self._left_buttons = []
        self._right_buttons = []
        self._gripper_delta = 0.0
        self._skill_request = None  # "home", "reorient", or None

        # Initialize MuJoCo physics for IK (same as SpacemouseAgent)
        from dm_control import mjcf
        from gello.dm_control_tasks.arms.ur5e import UR5e

        if robot_type == "ur5":
            _robot = UR5e()
        else:
            raise ValueError(f"Unknown robot type for IK: {robot_type}")
        self.physics = mjcf.Physics.from_mjcf_model(_robot.mjcf_model)

        # Start background thread for joystick reading
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Background joystick reader
    # ------------------------------------------------------------------
    def _apply_deadzone(self, val: float) -> float:
        return 0.0 if abs(val) < self.config.deadzone else val

    def _get_gain(self, joy) -> float:
        """Read the base slider to compute a speed gain multiplier [min_gain, 1.0]."""
        try:
            val = joy.get_axis(3)  # Slider axis
            norm = (-val + 1) / 2.0
            return self.config.min_gain + (norm * (1.0 - self.config.min_gain))
        except Exception:
            return 1.0

    def _read_loop(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()

        while self._running:
            num_joys = pygame.joystick.get_count()
            if num_joys < 2:
                print(
                    f"[HOSAS] Need 2 joysticks for HOSAS, found {num_joys}. "
                    f"Waiting..."
                )
                pygame.joystick.quit()
                time.sleep(2.0)
                pygame.joystick.init()
                continue

            joy_left = pygame.joystick.Joystick(self._left_index)
            joy_right = pygame.joystick.Joystick(self._right_index)
            joy_left.init()
            joy_right.init()
            print(f"[HOSAS] Left:  {joy_left.get_name()}")
            print(f"[HOSAS] Right: {joy_right.get_name()}")

            try:
                while self._running:
                    pygame.event.pump()

                    # Read all axes
                    left_axes = np.zeros(max(6, joy_left.get_numaxes()))
                    for i in range(joy_left.get_numaxes()):
                        left_axes[i] = self._apply_deadzone(joy_left.get_axis(i))

                    right_axes = np.zeros(max(6, joy_right.get_numaxes()))
                    for i in range(joy_right.get_numaxes()):
                        right_axes[i] = self._apply_deadzone(joy_right.get_axis(i))

                    # Read buttons
                    left_buttons = [
                        joy_left.get_button(i)
                        for i in range(joy_left.get_numbuttons())
                    ]
                    right_buttons = [
                        joy_right.get_button(i)
                        for i in range(joy_right.get_numbuttons())
                    ]

                    # Gripper from left mini-stick Y
                    mini_y_idx = self.config.mini_stick_y_axis
                    grip_d = 0.0
                    if mini_y_idx < joy_left.get_numaxes():
                        val = self._apply_deadzone(joy_left.get_axis(mini_y_idx))
                        if val != 0:
                            # Mini-stick up (< 0) = close, down (> 0) = open
                            grip_d = -val * self.config.gripper_step

                    # Skill detection
                    skill = None
                    if len(right_buttons) > 2 and right_buttons[2]:
                        skill = "reorient"
                    elif len(right_buttons) > 3 and right_buttons[3]:
                        skill = "home"

                    # Compute gains
                    gain_left = self._get_gain(joy_left)
                    gain_right = self._get_gain(joy_right)

                    with self._lock:
                        self._left_axes = left_axes[:6].copy()
                        self._right_axes = right_axes[:6].copy()
                        self._left_buttons = left_buttons
                        self._right_buttons = right_buttons
                        self._gripper_delta = grip_d
                        self._skill_request = skill
                        self._gain_left = gain_left
                        self._gain_right = gain_right

                    time.sleep(0.005)  # ~200 Hz polling

            except Exception as e:
                print(f"[HOSAS] Device lost ({e}), will reconnect...")
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        import quaternion

        num_dof = 6  # arm joints (gripper is separate)
        current_qpos = obs["joint_positions"][:num_dof]
        current_gripper = obs["joint_positions"][-1]

        with self._lock:
            left_axes = self._left_axes.copy()
            right_axes = self._right_axes.copy()
            grip_d = self._gripper_delta
            skill = self._skill_request
            gain_l = getattr(self, "_gain_left", 1.0)
            gain_r = getattr(self, "_gain_right", 1.0)
            self._skill_request = None  # consume

        # Handle skills: return target joint positions directly
        if skill == "home":
            if self._verbose:
                print("[HOSAS] Skill: Home")
            home = np.array(HOME_JOINTS_RAD + [current_gripper])
            return home
        elif skill == "reorient":
            if self._verbose:
                print("[HOSAS] Skill: Reorient (vertical)")
            # Keep current joints, just return current (skill handled externally
            # or could be implemented via IK to force vertical EE)
            pass

        # -----------------------------------------------------------
        # Cartesian velocity from sticks
        # -----------------------------------------------------------
        # Left stick: translation X, Y
        tx = -left_axes[1] * self.config.max_speed_linear * gain_l
        ty = -left_axes[0] * self.config.max_speed_linear * gain_l

        # Right mini-stick: translation Z
        mini_y_idx = self.config.mini_stick_y_axis
        tz = 0.0
        if mini_y_idx < len(right_axes):
            tz = -right_axes[mini_y_idx] * self.config.max_speed_linear * gain_l

        # Right stick: rotation Rx, Ry; twist: Rz
        r = right_axes[0] * self.config.max_speed_angular * gain_r
        p = -right_axes[1] * self.config.max_speed_angular * gain_r
        y = 0.0
        if len(right_axes) > 2:
            y = -right_axes[2] * self.config.max_speed_angular * gain_r

        # -----------------------------------------------------------
        # Forward kinematics: get current EE pose in MuJoCo
        # -----------------------------------------------------------
        self.physics.data.qpos[:num_dof] = current_qpos
        self.physics.step()

        ee_rot = np.array(
            self.physics.named.data.site_xmat["attachment_site"]
        ).reshape(3, 3)
        ee_pos = np.array(self.physics.named.data.site_xpos["attachment_site"])

        # Transform from MuJoCo to UR coordinate space
        ee_rot = mj2ur[:3, :3] @ ee_rot
        ee_pos = _apply_transfer(mj2ur, ee_pos)

        # -----------------------------------------------------------
        # Apply velocity deltas
        # -----------------------------------------------------------
        # Translation delta
        trans_delta = _apply_transfer(hosas2ur, np.array([tx, ty, tz]))
        new_ee_pos = ee_pos + trans_delta

        # Rotation delta (decomposed per axis, same as SpacemouseAgent)
        scale = 1.0  # Already scaled above
        rot_x = np.eye(4)
        rot_x[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(np.array([-p, 0, 0]) * scale)
        )
        rot_y = np.eye(4)
        rot_y[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(np.array([0, r, 0]) * scale)
        )
        rot_z = np.eye(4)
        rot_z[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(np.array([0, 0, -y]) * scale)
        )
        rot_transform = hosas2ur @ rot_z @ rot_y @ rot_x @ ur2hosas

        if self.config.rotation_mode == "euler":
            new_ee_rot = rot_transform[:3, :3] @ ee_rot
        else:
            new_ee_rot = ee_rot @ rot_transform[:3, :3]

        # -----------------------------------------------------------
        # Inverse kinematics
        # -----------------------------------------------------------
        from dm_control.utils.inverse_kinematics import qpos_from_site_pose

        target_quat = quaternion.as_float_array(
            quaternion.from_rotation_matrix(ur2mj[:3, :3] @ new_ee_rot)
        )
        ik_result = qpos_from_site_pose(
            self.physics,
            "attachment_site",
            target_pos=_apply_transfer(ur2mj, new_ee_pos),
            target_quat=target_quat,
            tol=1e-14,
            max_steps=400,
        )
        self.physics.reset()

        if ik_result.success:
            new_qpos = ik_result.qpos[:num_dof]
        else:
            if self._verbose:
                print("[HOSAS] IK failed, holding position")
            new_qpos = current_qpos

        # -----------------------------------------------------------
        # Gripper
        # -----------------------------------------------------------
        new_gripper = np.clip(current_gripper + grip_d, 0.0, 1.0)

        return np.concatenate([new_qpos, [new_gripper]])

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
