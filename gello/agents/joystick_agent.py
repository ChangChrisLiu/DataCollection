# gello/agents/joystick_agent.py
"""Thrustmaster SOL-R2 HOSAS dual-stick agent for UR5e teleoperation.

This agent reads from a Thrustmaster SOL-R2 HOSAS (Hands-On-Stick-And-Stick)
dual flight controller and maps the inputs to Cartesian velocity commands
for the UR5e robot arm via RTDE speedL (UR handles IK internally).

Hardware Mapping (Thrustmaster SOL-R2 HOSAS):
  Left Stick:
    Axis 0 (X) + Axis 1 (Y) -> TCP translation X/Y
    Axis 3 (slider)          -> speed gain multiplier
    Mini-stick Y             -> gripper open/close
    Button 0 (trigger)       -> toggle data recording

  Right Stick:
    Axis 0 (X) + Axis 1 (Y) -> TCP rotation Rx/Ry (roll/pitch)
    Axis 2 (twist)           -> TCP rotation Rz (yaw)
    Axis 3 (mini-stick Y)   -> TCP translation Z (up/down)
    Button 2                 -> skill: vertical reorient
    Button 3                 -> skill: go to home position

Control mode:
  This agent outputs a dict with 'velocity' (6D Cartesian) and 'gripper_vel',
  which env.py routes to URRobot.command_cartesian_velocity() -> RTDE speedL().
  The UR controller handles IK internally at 500Hz.

  For skills (home, reorient), the agent outputs 'skill' commands that
  run_env.py handles via moveJ/moveL.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from gello.agents.agent import Agent

# Default home position for UR5e (in joint radians)
HOME_JOINTS_DEG = [-90, -90, 90, -90, -90, 0]
HOME_JOINTS_RAD = np.deg2rad(HOME_JOINTS_DEG)


@dataclass
class HOSASConfig:
    """Configuration for Thrustmaster SOL-R2 HOSAS mapping."""

    # Speed limits
    max_speed_linear: float = 0.25  # m/s max TCP translation speed
    max_speed_angular: float = 0.50  # rad/s max TCP rotation speed
    acceleration: float = 0.5  # TCP acceleration (m/s^2)
    watchdog_time: float = 0.1  # Safety watchdog for speedL (s)

    # Deadzone for all axes
    deadzone: float = 0.05

    # Minimum gain when speed slider is at zero
    min_gain: float = 0.1

    # Gripper step per control cycle (normalized 0-1, ~3/255)
    gripper_step: float = 0.012

    # Axis assignment for mini-stick Y (may vary by OS)
    mini_stick_y_axis: int = 3


class JoystickAgent(Agent):
    """Thrustmaster SOL-R2 HOSAS dual-stick agent for UR5e.

    Outputs Cartesian velocity commands (not joint positions).
    The act() method returns a dict:
      {
        'type': 'velocity',
        'velocity': np.ndarray(6),       # [vx, vy, vz, wx, wy, wz]
        'acceleration': float,
        'time': float,                   # watchdog
        'gripper_vel': float,            # gripper delta per step
      }
    or for skills:
      {
        'type': 'skill',
        'skill': 'home' | 'reorient',
      }
    """

    # Flag for env.py / run_env.py to detect velocity-mode agent
    control_mode = "cartesian_velocity"

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
        self._gain_left = 1.0
        self._gain_right = 1.0

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
        """Read the base slider to compute speed gain."""
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
                print(f"[HOSAS] Need 2 joysticks, found {num_joys}. " f"Waiting...")
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
                        joy_left.get_button(i) for i in range(joy_left.get_numbuttons())
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
                            # up (< 0) = close, down (> 0) = open
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
                        if skill is not None:
                            self._skill_request = skill
                        self._gain_left = gain_left
                        self._gain_right = gain_right

                    time.sleep(0.005)  # ~200 Hz polling

            except Exception as e:
                print(f"[HOSAS] Device lost ({e}), reconnecting...")
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return a velocity command dict.

        The control loop (run_env.py) checks the return type and
        routes velocity commands to
        robot.command_cartesian_velocity() -> speedL.
        """
        with self._lock:
            left_axes = self._left_axes.copy()
            right_axes = self._right_axes.copy()
            grip_d = self._gripper_delta
            skill = self._skill_request
            gain_l = self._gain_left
            gain_r = self._gain_right
            self._skill_request = None  # consume

        # Handle skills
        if skill is not None:
            if self._verbose:
                print(f"[HOSAS] Skill: {skill}")
            return {"type": "skill", "skill": skill}

        # Build 6D Cartesian velocity [vx, vy, vz, wx, wy, wz]
        cfg = self.config
        vel = np.zeros(6)

        # Left stick: translation X, Y
        vel[0] = -left_axes[1] * cfg.max_speed_linear * gain_l  # Fwd/Back
        vel[1] = -left_axes[0] * cfg.max_speed_linear * gain_l  # Left/Right

        # Right mini-stick: translation Z
        mini_y_idx = cfg.mini_stick_y_axis
        if mini_y_idx < len(right_axes):
            vel[2] = -right_axes[mini_y_idx] * cfg.max_speed_linear * gain_l  # Up/Down

        # Right stick: rotation
        vel[3] = right_axes[0] * cfg.max_speed_angular * gain_r  # Roll
        vel[4] = -right_axes[1] * cfg.max_speed_angular * gain_r  # Pitch
        if len(right_axes) > 2:
            vel[5] = -right_axes[2] * cfg.max_speed_angular * gain_r  # Yaw

        if self._verbose and np.any(np.abs(vel) > 0.01):
            print(
                f"[HOSAS] vel={vel}, grip_d={grip_d:.3f}, "
                f"gain_L={gain_l:.2f}, gain_R={gain_r:.2f}"
            )

        return {
            "type": "velocity",
            "velocity": vel,
            "acceleration": cfg.acceleration,
            "time": cfg.watchdog_time,
            "gripper_vel": grip_d,
        }

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
