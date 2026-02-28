# gello/agents/vla_agent.py
"""VLA model agent for UR5e robot inference.

All three backends use a server-client architecture so run_inference.py
can always run from the ``tele`` conda env:

  - OpenPI:       WebSocket client  → openpi serve_policy.py  (uv venv)
  - OpenVLA:      REST client       → openvla deploy.py       (conda vla)
  - OpenVLA-OFT:  REST client       → openvla-oft deploy.py   (conda oft)

Usage:
    adapter = OpenPIAdapter(host="127.0.0.1", port=8000)
    agent = VLAAgent(adapter, control_fps=10, prompt="Extract the CPU ...")
    agent.execute_step(obs, robot_client)
"""

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from gello.agents.safety import SafetyMonitor


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Resize RGB image to (size, size, 3) using PIL LANCZOS."""
    return np.array(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def _apply_eef_delta(action: np.ndarray, robot_client, safety: Optional[SafetyMonitor]):
    """Apply EEF delta action via moveL.  Shared by OpenVLA and OFT adapters.

    Args:
        action: (7,) [dx, dy, dz, droll, dpitch, dyaw, gripper_inverted]
            Position deltas in meters, rotation deltas in Euler RPY (rad).
            Gripper is inverted: 1.0=open, 0.0=close in model space.
        robot_client: ZMQClientRobot on the command port.
        safety: Optional SafetyMonitor.
    """
    current_tcp = np.array(robot_client.get_tcp_pose_raw())  # [x,y,z,rx,ry,rz] rotvec

    delta_pos = action[:3].copy()
    delta_rpy = action[3:6].copy()

    # Safety clamp
    if safety is not None:
        delta_pos, delta_rpy = safety.check_eef_action(delta_pos, delta_rpy)

    # Position: simple addition
    target_pos = current_tcp[:3] + delta_pos

    # Rotation: rotvec → RPY → add delta → RPY → rotvec
    # Single-step moveL is immune to near-pi rotvec flip (UR takes shortest path)
    current_rpy = Rotation.from_rotvec(current_tcp[3:]).as_euler("xyz")
    target_rpy = current_rpy + delta_rpy
    target_rotvec = Rotation.from_euler("xyz", target_rpy).as_rotvec()

    target_pose = np.concatenate([target_pos, target_rotvec])

    # Workspace bounds check
    if safety is not None and not safety.check_target_pose(target_pose):
        print("[SAFETY] Target pose outside workspace bounds — skipping")
        return

    robot_client.move_linear(target_pose, speed=0.25, accel=0.5, asynchronous=False)

    # Gripper — un-invert (model: 1=open, 0=close → robot: 0=open, 255=close)
    gripper_physical = 1.0 - action[6]
    gripper_int = int(np.clip(gripper_physical * 255, 0, 255))
    robot_client.set_gripper(gripper_int)


# ---------------------------------------------------------------------------
# VLA Agent
# ---------------------------------------------------------------------------


class VLAAgent:
    """Wraps a model adapter with action chunk buffering and stop detection.

    Stop detection uses a combined criterion: N consecutive actions in the
    chunk must have small delta (joint or EEF) from current state AND gripper
    past a threshold.  This is more robust than gripper-only detection,
    especially for CPU planner where gripper values are close to the stop
    threshold.
    """

    # Per-task stop thresholds (tuned on 50-episode sweep, planner v3)
    STOP_PARAMS = {
        "cpu": {"n_consec": 3, "delta_thresh": 4e-3, "grip_thresh": 0.97},
        "ram": {"n_consec": 5, "delta_thresh": 8e-3, "grip_thresh": 0.95},
        "default": {"n_consec": 3, "delta_thresh": 5e-3, "grip_thresh": 0.95},
    }

    def __init__(
        self,
        adapter,
        control_fps: int,
        prompt: str,
        task: str = "default",
        safety_monitor: Optional[SafetyMonitor] = None,
    ):
        self.adapter = adapter
        self.control_fps = control_fps
        self.prompt = prompt
        self.safety = safety_monitor
        self._action_queue: deque = deque()
        self._stop_detected = False
        self._stop_params = self.STOP_PARAMS.get(
            task, self.STOP_PARAMS["default"]
        )

    def _check_chunk_stop(
        self, actions: List[np.ndarray], current_state: np.ndarray
    ) -> bool:
        """Combined stop criterion on the full action chunk.

        Returns True if N consecutive actions satisfy:
          - delta (predicted - current state) < delta_thresh
          - gripper past grip_thresh (direction depends on adapter)
        """
        n_consec = self._stop_params["n_consec"]
        delta_thresh = self._stop_params["delta_thresh"]
        grip_thresh = self._stop_params["grip_thresh"]

        if len(actions) < n_consec:
            return False

        for i in range(len(actions) - n_consec + 1):
            all_ok = True
            for k in range(n_consec):
                a = actions[i + k]
                # Delta: joints[:6] for OpenPI, EEF[:6] for OpenVLA/OFT
                delta = float(np.linalg.norm(a[:6] - current_state[:6]))
                if delta > delta_thresh:
                    all_ok = False
                    break
                # Gripper check (adapter-aware direction)
                if not self.adapter.is_stop_gripper(a, grip_thresh):
                    all_ok = False
                    break
            if all_ok:
                return True
        return False

    def execute_step(self, obs: Dict[str, Any], robot_client) -> None:
        """Query model if queue empty, apply one action to robot."""
        if not self._action_queue:
            chunk = self.adapter.infer(obs, self.prompt)

            # Check combined stop criterion on the full chunk
            current_state = self.adapter.get_current_state(obs)
            if self._check_chunk_stop(chunk, current_state):
                self._stop_detected = True
                return

            self._action_queue.extend(chunk)

        action = self._action_queue.popleft()
        self.adapter.apply_action(action, obs, robot_client, self.safety)

    @property
    def stop_detected(self) -> bool:
        return self._stop_detected

    def reset(self) -> None:
        """Clear queue and stop flag for a new episode."""
        self._action_queue.clear()
        self._stop_detected = False


# ---------------------------------------------------------------------------
# OpenPI Adapter (WebSocket client → absolute joint actions)
# ---------------------------------------------------------------------------


class OpenPIAdapter:
    """WebSocket client to an OpenPI policy server.

    The server applies AbsoluteActions output transform, so the client
    receives ready-to-use absolute joint targets:
        actions[:, 0:6]  = absolute joint positions (rad)
        actions[:, 6]    = absolute gripper (0=open, 1=close, normalized)

    Server: cd /home/chris/openpi && uv run scripts/serve_policy.py ...
    Requires: pip install /home/chris/openpi/packages/openpi-client/
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 8000, open_loop_horizon: int = 10
    ):
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
        except ImportError:
            raise ImportError(
                "openpi-client is required for OpenPI inference.\n"
                "Install with: pip install /home/chris/openpi/packages/openpi-client/"
            )
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.open_loop_horizon = open_loop_horizon

    def infer(self, obs: Dict[str, Any], prompt: str) -> List[np.ndarray]:
        """Pack observation, query server, return list of actions."""
        joints = obs["joint_positions"][:6]
        gripper = obs["gripper_position"][0]  # normalized 0-1
        state = np.concatenate([joints, [gripper]]).astype(np.float32)

        base_img = _resize_rgb(obs["base_rgb"], 256)
        wrist_img = _resize_rgb(obs["wrist_rgb"], 256)

        request = {
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": prompt,
        }

        result = self.policy.infer(request)
        actions = result["actions"]  # (chunk_size, 7) absolute

        n = min(self.open_loop_horizon, len(actions))
        return [actions[i] for i in range(n)]

    def apply_action(
        self,
        action: np.ndarray,
        obs: Dict[str, Any],
        robot_client,
        safety: Optional[SafetyMonitor],
    ) -> None:
        """Apply absolute joint action via servoJ."""
        target = np.array(action, dtype=np.float64)

        if safety is not None:
            current_q = obs["joint_positions"][:6]
            delta = target[:6] - current_q
            safe_delta = safety.check_joint_action(delta, current_q)
            target[:6] = current_q + safe_delta

        target[6] = np.clip(target[6], 0.0, 1.0)
        robot_client.command_joint_state(target)

    def get_current_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return current joint positions for delta computation."""
        return np.array(obs["joint_positions"][:6], dtype=np.float32)

    def is_stop_gripper(self, action: np.ndarray, grip_thresh: float) -> bool:
        """OpenPI gripper: 0=open, 1=close. Stop at high values."""
        return float(action[6]) > grip_thresh


# ---------------------------------------------------------------------------
# OpenVLA Adapter (REST client → single-step EEF deltas)
# ---------------------------------------------------------------------------


class OpenVLAAdapter:
    """REST client to an OpenVLA deploy.py server.

    Action format: [dx, dy, dz, droll, dpitch, dyaw, gripper_inverted]
    Gripper inverted during training: model 1.0=open, 0.0=close.

    Server: conda activate vla && cd /home/chris/Sibo/openvla &&
            python vla-scripts/deploy.py --openvla_path <checkpoint> --port 8000
    Requires: pip install json-numpy requests
    """

    _json_patched = False

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        unnorm_key: str = "ur5e_vla_planner_10hz",
    ):
        try:
            import json_numpy
            import requests as _requests
        except ImportError:
            raise ImportError(
                "json-numpy and requests are required for OpenVLA inference.\n"
                "Install with: pip install json-numpy requests"
            )
        if not OpenVLAAdapter._json_patched:
            json_numpy.patch()
            OpenVLAAdapter._json_patched = True
        self._requests = _requests
        self.endpoint = f"http://{host}:{port}/act"
        self.unnorm_key = unnorm_key

    def infer(self, obs: Dict[str, Any], prompt: str) -> List[np.ndarray]:
        """Send image + prompt to server, get single action (7,)."""
        # OpenVLA server expects: {"image": ndarray, "instruction": str}
        base_img = _resize_rgb(obs["base_rgb"], 256)

        payload = {
            "image": base_img,
            "instruction": prompt,
            "unnorm_key": self.unnorm_key,
        }

        response = self._requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        action = np.array(response.json(), dtype=np.float64)
        return [action]

    def apply_action(
        self,
        action: np.ndarray,
        obs: Dict[str, Any],
        robot_client,
        safety: Optional[SafetyMonitor],
    ) -> None:
        """Apply EEF delta via moveL."""
        _apply_eef_delta(action, robot_client, safety)

    def get_current_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return current EEF pose for delta computation."""
        # For EEF delta models, actions are deltas — so "current state" is
        # zeros (delta from zero = the delta itself, checked against thresh).
        return np.zeros(6, dtype=np.float32)

    def is_stop_gripper(self, action: np.ndarray, grip_thresh: float) -> bool:
        """OpenVLA gripper is inverted: 1=open, 0=close. Stop at low values."""
        return float(action[6]) < (1.0 - grip_thresh)


# ---------------------------------------------------------------------------
# OpenVLA-OFT Adapter (REST client → multi-step EEF delta chunks)
# ---------------------------------------------------------------------------


class OpenVLAOFTAdapter:
    """REST client to an OpenVLA-OFT deploy.py server.

    Same EEF delta format as OpenVLA but with action chunking (8 steps).
    OFT server also accepts wrist images and proprioception.

    Server: conda activate oft && cd /home/chris/Sibo/openvla-oft &&
            python vla-scripts/deploy.py --pretrained_checkpoint <ckpt>
                --unnorm_key ur5e_vla_planner_10hz --use_l1_regression True
                --use_proprio True --num_images_in_input 2 --port 8777
    Requires: pip install json-numpy requests
    """

    _json_patched = False

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8777,
        unnorm_key: str = "ur5e_vla_planner_10hz",
    ):
        try:
            import json_numpy
            import requests as _requests
        except ImportError:
            raise ImportError(
                "json-numpy and requests are required for OpenVLA-OFT inference.\n"
                "Install with: pip install json-numpy requests"
            )
        if not OpenVLAOFTAdapter._json_patched:
            json_numpy.patch()
            OpenVLAOFTAdapter._json_patched = True
        self._requests = _requests
        self.endpoint = f"http://{host}:{port}/act"
        self.unnorm_key = unnorm_key

    def infer(self, obs: Dict[str, Any], prompt: str) -> List[np.ndarray]:
        """Send images + state + prompt to server, get action chunk."""
        # OFT server expects: full_image, wrist images, state, instruction
        base_img = _resize_rgb(obs["base_rgb"], 256)
        wrist_img = _resize_rgb(obs["wrist_rgb"], 256)

        joints = obs["joint_positions"][:6]
        gripper = obs["gripper_position"][0]
        state = np.concatenate([joints, [gripper]]).astype(np.float32)

        payload = {
            "full_image": base_img,
            "wrist_image_0": wrist_img,
            "state": state,
            "instruction": prompt,
        }

        response = self._requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        result = response.json()

        # Server returns list of actions (one per chunk step)
        if isinstance(result, list):
            return [np.array(a, dtype=np.float64) for a in result]
        else:
            return [np.array(result, dtype=np.float64)]

    def apply_action(
        self,
        action: np.ndarray,
        obs: Dict[str, Any],
        robot_client,
        safety: Optional[SafetyMonitor],
    ) -> None:
        """Apply EEF delta via moveL (same as OpenVLA)."""
        _apply_eef_delta(action, robot_client, safety)

    def get_current_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return zeros — actions are EEF deltas, so delta=action magnitude."""
        return np.zeros(6, dtype=np.float32)

    def is_stop_gripper(self, action: np.ndarray, grip_thresh: float) -> bool:
        """Same as OpenVLA — inverted gripper, stop at low values."""
        return float(action[6]) < (1.0 - grip_thresh)
