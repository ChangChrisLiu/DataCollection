# gello/agents/joystick_agent.py
"""Joystick (gamepad) agent for UR5e teleoperation.

Uses pygame to read from a standard gamepad controller.
Left stick  = J0 (base), J1 (shoulder)
Right stick = J2 (elbow), J3 (wrist1)
Triggers    = J4 (wrist2), J5 (wrist3)
Buttons     = gripper open/close

All outputs are joint-velocity deltas applied on top of the current position.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from gello.agents.agent import Agent


@dataclass
class JoystickConfig:
    # Scale factors (rad/step) for each joint velocity
    joint_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    )
    gripper_speed: float = 0.1  # gripper open/close speed per step
    deadzone: float = 0.15  # stick deadzone


class JoystickAgent(Agent):
    """Gamepad-based agent for UR5e (6-DOF + gripper)."""

    def __init__(
        self,
        config: Optional[JoystickConfig] = None,
        device_index: int = 0,
        num_dofs: int = 7,  # 6 joints + 1 gripper for UR5e
    ) -> None:
        self.config = config or JoystickConfig()
        self.num_dofs = num_dofs
        self._device_index = device_index
        self._lock = threading.Lock()

        # Raw axis values from the gamepad (6 axes for 6 joints)
        self._axes = np.zeros(6, dtype=float)
        # Gripper command: 0.0 = fully open, 1.0 = fully closed
        self._gripper_delta = 0.0

        # Start background thread to read the gamepad
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------
    def _read_loop(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()

        while self._running:
            # Wait for a joystick to be connected
            while pygame.joystick.get_count() == 0 and self._running:
                print("Joystick agent: waiting for controller...")
                pygame.joystick.quit()
                time.sleep(2.0)
                pygame.joystick.init()

            if not self._running:
                break

            js = pygame.joystick.Joystick(self._device_index)
            js.init()
            print(f"Joystick agent: connected to '{js.get_name()}'")

            try:
                while self._running:
                    pygame.event.pump()

                    raw = np.zeros(6)
                    n_axes = js.get_numaxes()

                    # Map axes: left stick (0,1), right stick (2,3), triggers (4,5)
                    for i in range(min(6, n_axes)):
                        val = js.get_axis(i)
                        if abs(val) < self.config.deadzone:
                            val = 0.0
                        raw[i] = val

                    # Gripper from buttons: A=close, B=open (Xbox layout)
                    gripper_d = 0.0
                    n_buttons = js.get_numbuttons()
                    if n_buttons > 0 and js.get_button(0):  # A button
                        gripper_d = self.config.gripper_speed
                    if n_buttons > 1 and js.get_button(1):  # B button
                        gripper_d = -self.config.gripper_speed

                    with self._lock:
                        self._axes = raw
                        self._gripper_delta = gripper_d

                    time.sleep(0.005)  # ~200 Hz polling

            except Exception as e:
                print(f"Joystick agent: device lost ({e}), will reconnect...")
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        current_joints = obs["joint_positions"]  # length = num_dofs (7)

        with self._lock:
            axes = self._axes.copy()
            grip_d = self._gripper_delta

        # Compute joint deltas from axis values
        joint_deltas = axes * self.config.joint_scale

        # Build the full action = current + delta
        action = current_joints.copy()
        n_joints = min(6, len(current_joints) - 1)  # 6 arm joints
        action[:n_joints] += joint_deltas[:n_joints]

        # Gripper: last DOF, clamp to [0, 1]
        action[-1] = np.clip(current_joints[-1] + grip_d, 0.0, 1.0)

        return action

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
