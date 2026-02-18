# gello/agents/joystick_agent.py
"""Thrustmaster SOL-R2 HOSAS dual-stick agent for UR5e teleoperation.

Axis/button mappings match joysticktst.py exactly.

Hardware Mapping (Thrustmaster SOL-R2 HOSAS):
  Left Stick:
    Axis 0 (X)         -> TCP translation X
    Axis 1 (Y)         -> TCP translation Y
    Axis 2 (slider)    -> speed gain (0=up/fast, 1=down/slow)
    Axis 4 (mini-Y)    -> gripper open/close
    Button 16           -> interrupt/resume skill
    Button 25           -> start recording
    Button 34           -> home + stop recording
    Button 38           -> vertical reorient

  Right Stick:
    Axis 1 (Y)          -> TCP translation Z (push/pull)
    Axis 3 (mini-X)     -> TCP rotation Ry
    Axis 4 (mini-Y)     -> TCP rotation Rx
    Axis 5 (twist)      -> TCP rotation Rz
    Button 15            -> skill: CPU extraction
    Button 16            -> skill: RAM (reserved)
    Button 17            -> skill: Connector (reserved)
    Button 18            -> skill: (reserved)
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from gello.agents.agent import Action, Agent

# Home position for UR5e (in joint radians, matching joysticktst.py)
HOME_JOINTS_RAD = np.array(
    [-1.613282, -1.802157, -1.189979, -1.72829, 1.552156, 3.058179]
)
HOME_GRIPPER_POS = 3  # Nearly open (0-255 scale)

# ---------------------------------------------------------------------------
# Button constants (matching joysticktst.py)
# ---------------------------------------------------------------------------
L_BTN_REC_START = 25
L_BTN_HOME = 34
L_BTN_VERT = 38
L_BTN_INTERRUPT = 16  # Interrupt skill execution mid-waypoint

# Right stick skill buttons -> skill name
R_BTN_SKILL_MAP = {
    15: "cpu",
    16: "ram",
    17: "connector",
    18: "skill_4",
}

# ---------------------------------------------------------------------------
# Axis constants (matching joysticktst.py)
# ---------------------------------------------------------------------------
L_AXIS_X = 0
L_AXIS_Y = 1
L_AXIS_SLIDER = 2  # range [0=up, 1=down], read WITHOUT calibration
L_AXIS_MINI_Y = 4  # gripper

R_AXIS_Z = 1  # right stick push/pull -> TCP Z
R_AXIS_MINI_X = 3  # right mini-stick X -> TCP Ry
R_AXIS_MINI_Y = 4  # right mini-stick Y -> TCP Rx
R_AXIS_RZ = 5  # right twist -> TCP Rz


@dataclass
class HOSASConfig:
    """Configuration for Thrustmaster SOL-R2 HOSAS mapping."""

    # Speed limits (matching joysticktst.py)
    max_speed_linear: float = 0.05  # m/s max TCP translation speed
    max_speed_angular: float = 0.10  # rad/s max TCP rotation speed (Rx, Ry)
    max_speed_rz: float = 0.25  # rad/s max TCP Rz twist speed
    acceleration: float = 0.5
    watchdog_time: float = 0.1

    # Deadzone
    deadzone: float = 0.05

    # Minimum gain when slider is fully down
    min_gain: float = 0.1

    # Gripper step per cycle (normalized 0-1)
    # joysticktst.py uses GRIPPER_STEP=10 at ~200Hz; ZMQ pipeline runs at 30Hz
    # so scale up: 10/255 * (200/30) ≈ 0.26
    gripper_step: float = 0.26


class JoystickAgent(Agent):
    """Thrustmaster SOL-R2 HOSAS dual-stick agent for UR5e.

    Returns velocity commands or skill signals from act().
    Skill signals: "start_recording", "home", "reorient", "cpu", "ram", etc.
    """

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

        # Joystick state (updated by background thread)
        self._left_axes = np.zeros(8, dtype=float)
        self._right_axes = np.zeros(8, dtype=float)
        self._gripper_delta = 0.0
        self._gain = 1.0
        self._skill_request = None  # consumed by act()

        # Calibration offsets (set during init)
        self._left_center = np.zeros(8)
        self._right_center = np.zeros(8)

        # Previous button states for edge detection
        self._prev_left_btns = {}
        self._prev_right_btns = {}

        # Interrupt event — set by left btn 16, checked by skill executor
        self.interrupt_event = threading.Event()

        # Start background thread
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_deadzone(self, val: float) -> float:
        return 0.0 if abs(val) < self.config.deadzone else val

    def _read_axis(
        self, joy, axis_idx: int, is_left: bool, calibrate: bool = True
    ) -> float:
        """Read a joystick axis with optional calibration offset."""
        if axis_idx >= joy.get_numaxes():
            return 0.0
        raw = joy.get_axis(axis_idx)
        if calibrate:
            center = self._left_center if is_left else self._right_center
            if axis_idx < len(center):
                raw -= center[axis_idx]
        return raw

    def _rising_edge(self, buttons, prev_dict, btn_idx: int) -> bool:
        """Detect rising edge (button just pressed)."""
        now = buttons[btn_idx] if btn_idx < len(buttons) else False
        was = prev_dict.get(btn_idx, False)
        prev_dict[btn_idx] = now
        return now and not was

    def _calibrate(self, joy_left, joy_right) -> None:
        """Calibrate axis centers — sticks must be at rest."""
        import pygame

        print("[HOSAS] Calibrating axes -- keep hands off sticks...")
        time.sleep(1.0)
        num_samples = 50
        left_accum = np.zeros(max(8, joy_left.get_numaxes()))
        right_accum = np.zeros(max(8, joy_right.get_numaxes()))
        for _ in range(num_samples):
            pygame.event.pump()
            for ax in range(joy_left.get_numaxes()):
                left_accum[ax] += joy_left.get_axis(ax)
            for ax in range(joy_right.get_numaxes()):
                right_accum[ax] += joy_right.get_axis(ax)
            time.sleep(0.01)
        self._left_center = left_accum / num_samples
        self._right_center = right_accum / num_samples
        print("[HOSAS] Calibration complete.")

    def _get_gain(self, joy_left) -> float:
        """Read left slider (axis 2, no calibration) to compute speed gain."""
        val = self._read_axis(joy_left, L_AXIS_SLIDER, is_left=True, calibrate=False)
        return self.config.min_gain + ((1.0 - val) * (1.0 - self.config.min_gain))

    # ------------------------------------------------------------------
    # Background joystick reader
    # ------------------------------------------------------------------
    def _read_loop(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()

        while self._running:
            num_joys = pygame.joystick.get_count()
            if num_joys < 2:
                print(f"[HOSAS] Need 2 joysticks, found {num_joys}. Waiting...")
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

            self._calibrate(joy_left, joy_right)

            try:
                while self._running:
                    pygame.event.pump()

                    # --- Read axes (with calibration + deadzone) ---
                    n_left = max(8, joy_left.get_numaxes())
                    n_right = max(8, joy_right.get_numaxes())
                    left_axes = np.zeros(n_left)
                    right_axes = np.zeros(n_right)

                    for i in range(joy_left.get_numaxes()):
                        left_axes[i] = self._apply_deadzone(
                            self._read_axis(joy_left, i, is_left=True)
                        )
                    for i in range(joy_right.get_numaxes()):
                        right_axes[i] = self._apply_deadzone(
                            self._read_axis(joy_right, i, is_left=False)
                        )

                    # --- Read buttons ---
                    left_buttons = [
                        joy_left.get_button(i) for i in range(joy_left.get_numbuttons())
                    ]
                    right_buttons = [
                        joy_right.get_button(i)
                        for i in range(joy_right.get_numbuttons())
                    ]

                    # --- Gripper (left mini-stick Y, axis 4) ---
                    grip_d = 0.0
                    grip_raw = self._apply_deadzone(
                        self._read_axis(joy_left, L_AXIS_MINI_Y, is_left=True)
                    )
                    if grip_raw != 0:
                        grip_d = -grip_raw * self.config.gripper_step

                    # --- Edge-detect button presses ---
                    signal = None
                    if self._rising_edge(
                        left_buttons, self._prev_left_btns, L_BTN_REC_START
                    ):
                        signal = "start_recording"
                    elif self._rising_edge(
                        left_buttons, self._prev_left_btns, L_BTN_HOME
                    ):
                        signal = "home"
                    elif self._rising_edge(
                        left_buttons, self._prev_left_btns, L_BTN_VERT
                    ):
                        signal = "reorient"

                    # --- Interrupt button (left btn 16) ---
                    if self._rising_edge(
                        left_buttons, self._prev_left_btns, L_BTN_INTERRUPT
                    ):
                        self.interrupt_event.set()

                    if signal is None:
                        for btn_idx, skill_name in R_BTN_SKILL_MAP.items():
                            if self._rising_edge(
                                right_buttons, self._prev_right_btns, btn_idx
                            ):
                                signal = skill_name
                                break
                        # Still update edge state for other skill buttons
                        for btn_idx in R_BTN_SKILL_MAP:
                            if btn_idx not in self._prev_right_btns:
                                self._prev_right_btns[btn_idx] = (
                                    right_buttons[btn_idx]
                                    if btn_idx < len(right_buttons)
                                    else False
                                )

                    # --- Gain from left slider ---
                    gain = self._get_gain(joy_left)

                    # --- Publish state ---
                    with self._lock:
                        self._left_axes = left_axes.copy()
                        self._right_axes = right_axes.copy()
                        self._gripper_delta = grip_d
                        self._gain = gain
                        if signal is not None:
                            self._skill_request = signal

                    time.sleep(0.005)  # ~200 Hz polling

            except Exception as e:
                print(f"[HOSAS] Device lost ({e}), reconnecting...")
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> Action:
        """Return a velocity command dict, or a skill signal dict."""
        with self._lock:
            left_axes = self._left_axes.copy()
            right_axes = self._right_axes.copy()
            grip_d = self._gripper_delta
            skill = self._skill_request
            gain = self._gain
            self._skill_request = None  # consume

        # Handle skill/signal requests
        if skill is not None:
            if self._verbose:
                print(f"[HOSAS] Signal: {skill}")
            return {"type": "skill", "skill": skill}

        # Build 6D Cartesian velocity [vx, vy, vz, wx, wy, wz]
        # Mapping matches joysticktst.py exactly
        cfg = self.config
        vel = np.zeros(6)

        # Translation
        vel[0] = left_axes[L_AXIS_X] * cfg.max_speed_linear * gain
        vel[1] = -left_axes[L_AXIS_Y] * cfg.max_speed_linear * gain
        vel[2] = -right_axes[R_AXIS_Z] * cfg.max_speed_linear * gain

        # Rotation
        vel[3] = right_axes[R_AXIS_MINI_Y] * cfg.max_speed_angular * gain
        vel[4] = -right_axes[R_AXIS_MINI_X] * cfg.max_speed_angular * gain
        vel[5] = -right_axes[R_AXIS_RZ] * cfg.max_speed_rz * gain

        if self._verbose and np.any(np.abs(vel) > 0.01):
            print(f"[HOSAS] vel={vel}, grip_d={grip_d:.3f}, gain={gain:.2f}")

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
