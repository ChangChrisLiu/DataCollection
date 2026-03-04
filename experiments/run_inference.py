#!/usr/bin/env python3
# experiments/run_inference.py
"""VLA inference pipeline for UR5e robot.

Deploys trained VLA models (OpenPI, OpenVLA, OpenVLA-OFT) on the UR5e
for performance validation.  Mirrors run_collection.py but replaces
human joystick input with model predictions.

All three backends use a server-client architecture so this script
always runs from the ``tele`` conda env:

  - OpenPI:       WebSocket client  → openpi serve_policy.py  (uv venv)
  - OpenVLA:      REST client       → openvla deploy.py       (conda vla)
  - OpenVLA-OFT:  REST client       → openvla-oft deploy.py   (conda oft)

Pipeline modes:
  planner:    Model approaches → CSV skill grasps → (if fail) correction model
  e2e:        Model handles entire trajectory
  correction: Correction-only (starts from current position → skill resume)

Dynamic home position:
  In ``correction`` mode or when ``--model-type openvla``, the robot's
  current joint positions are captured at startup as the session home.
  The robot stays where it is (no homing move) and returns to this
  captured position between episodes and on Ctrl+C.  Other modes use
  the default HOME_JOINTS_RAD / HOME_GRIPPER_POS constants.

Terminal architecture:
  T1: python experiments/launch_nodes.py --robot ur
  T2: python experiments/launch_camera_nodes.py --camera-settings configs/camera_settings.json
  T3: Start model server (see below)
  T4: python experiments/run_inference.py --mode <mode> --model-type <type> --task <task>

Example T4 commands:
  # Planner (OpenPI):
  python experiments/run_inference.py --mode planner --model-type openpi --task cpu

  # E2E (OpenPI):
  python experiments/run_inference.py --mode e2e --model-type openpi --task ram

  # Correction-only (OpenPI) — jog robot to a post-grasp-failure pose first:
  python experiments/run_inference.py --mode correction --model-type openpi --task cpu

  # Correction-only (OpenVLA):
  python experiments/run_inference.py --mode correction --model-type openvla \
      --task cpu --unnorm-key ur5e_vla_correction_10hz

  # Planner (OpenVLA) — dynamic home, lowers 0.1m before each episode:
  python experiments/run_inference.py --mode planner --model-type openvla \
      --task cpu --unnorm-key ur5e_vla_planner_10hz

Ctrl+C to e-stop the robot and save the current episode.
"""

import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import tyro
from PIL import Image
from scipy.spatial.transform import Rotation

from gello.agents.joystick_agent import HOME_GRIPPER_POS, HOME_JOINTS_RAD
from gello.agents.safety import SafetyMonitor
from gello.agents.vla_agent import (
    OpenPIAdapter,
    OpenVLAAdapter,
    OpenVLAOFTAdapter,
    VLAAgent,
)
from gello.data_utils.dataset_writer import DatasetWriter
from gello.data_utils.episode_buffer import EpisodeBuffer
from gello.skills.csv_skill_executor import CSVSkillExecutor
from gello.utils.transform_utils import quat_to_rotvec
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot

# ---------------------------------------------------------------------------
# OpenPI checkpoint registry
# ---------------------------------------------------------------------------

OPENPI_ROOT = "/home/chris/openpi"
OPENPI_BASE_CKPT_ROOT = "/home/chris/openpi_data/openpi-assets/checkpoints"

# Maps (base_model, target, fps) → (config_name, checkpoint_path_relative_to_checkpoints/)
# Config names are registered in openpi/src/openpi/training/config.py (no _v2 suffix).
# Checkpoint dirs include _v2 suffix (retrained on fixed gripper data).
OPENPI_CHECKPOINTS = {
    # --- Pi0.5-DROID fine-tuned (droid) ---
    ("droid", "planner", 10): (
        "pi05_droid_ur5e_planner_lora_10hz",
        "pi05_droid_ur5e_planner_lora_10hz_v2/49999",
    ),
    ("droid", "planner", 30): (
        "pi05_droid_ur5e_planner_lora_30hz",
        "pi05_droid_ur5e_planner_lora_30hz_v2/49999",
    ),
    ("droid", "e2e", 10): (
        "pi05_droid_ur5e_e2e_lora_10hz",
        "pi05_droid_ur5e_e2e_lora_10hz_v2/49999",
    ),
    ("droid", "e2e", 30): (
        "pi05_droid_ur5e_e2e_lora_30hz",
        "pi05_droid_ur5e_e2e_lora_30hz_v2/43000",
    ),
    ("droid", "correction", 10): (
        "pi05_droid_ur5e_correction_lora_10hz",
        "pi05_droid_ur5e_correction_lora_10hz/pi05_droid_ur5e_correction_lora_10hz_v2/49999",
    ),
    ("droid", "correction", 30): (
        "pi05_droid_ur5e_correction_lora_30hz",
        "pi05_droid_ur5e_correction_lora_30hz_v2/49999",
    ),
    # --- Pi0.5-base fine-tuned (base) ---
    ("base", "planner", 10): (
        "pi05_ur5e_planner_lora_10hz",
        "pi05_ur5e_planner_lora_10hz_v2/43000",
    ),
    ("base", "planner", 30): (
        "pi05_ur5e_planner_lora_30hz",
        "pi05_ur5e_planner_lora_30hz_v2/3000",
    ),
    ("base", "e2e", 10): (
        "pi05_ur5e_e2e_lora_10hz",
        "pi05_ur5e_e2e_lora_10hz_v2/49999",
    ),
    ("base", "e2e", 30): (
        "pi05_ur5e_e2e_lora_30hz",
        "pi05_ur5e_e2e_lora_30hz_v2/49999",
    ),
    ("base", "correction", 10): (
        "pi05_ur5e_correction_lora_10hz",
        "pi05_ur5e_correction_lora_10hz_v2/36000",
    ),
    ("base", "correction", 30): (
        "pi05_ur5e_correction_lora_30hz",
        "pi05_ur5e_correction_lora_30hz_v2/49999",
    ),
    # --- Pre-trained base models, zero-shot (no fine-tuning) ---
    # Uses non-LoRA configs (full weights) with base checkpoint + copied norm stats.
    ("droid_zeroshot", "planner", 10): ("pi05_droid_ur5e_planner_10hz", None),
    ("base_zeroshot", "planner", 10): ("pi05_ur5e_planner_10hz", None),
}

# Absolute paths for base model (zero-shot) checkpoints
OPENPI_ZEROSHOT_PATHS = {
    "droid_zeroshot": f"{OPENPI_BASE_CKPT_ROOT}/pi05_droid",
    "base_zeroshot": f"{OPENPI_BASE_CKPT_ROOT}/pi05_base",
}


def get_openpi_serve_cmd(base: str, target: str, fps: int, port: int) -> str:
    """Return the exact serve command for an OpenPI checkpoint."""
    key = (base, target, fps)
    if key not in OPENPI_CHECKPOINTS:
        return f"# ERROR: no checkpoint for ({base}, {target}, {fps}hz)"
    config_name, ckpt_path = OPENPI_CHECKPOINTS[key]
    if ckpt_path is None:
        # Zero-shot base model — use absolute path
        abs_path = OPENPI_ZEROSHOT_PATHS.get(base, "???")
        return (
            f"cd {OPENPI_ROOT} && uv run scripts/serve_policy.py --port {port} "
            f"policy:checkpoint \\\n"
            f"    --policy.config {config_name} \\\n"
            f"    --policy.dir {abs_path}"
        )
    return (
        f"cd {OPENPI_ROOT} && uv run scripts/serve_policy.py --port {port} "
        f"policy:checkpoint \\\n"
        f"    --policy.config {config_name} \\\n"
        f"    --policy.dir checkpoints/{ckpt_path}"
    )


def print_openpi_serve_commands(args) -> None:
    """Print the T3 serve commands needed for the current inference config."""
    base = args.openpi_base
    fps = args.fps
    mode = args.mode

    print("\n--- Expected T3 serve command(s) ---")
    if mode == "planner":
        cmd = get_openpi_serve_cmd(base, "planner", fps, args.server_port)
        print(f"# Planner server (port {args.server_port}):")
        print(cmd)
        if args.correction_server_port > 0:
            corr_port = args.correction_server_port
            cmd = get_openpi_serve_cmd(base, "correction", fps, corr_port)
            print(f"\n# Correction server (port {corr_port}):")
            print(cmd)
    elif mode == "e2e":
        cmd = get_openpi_serve_cmd(base, "e2e", fps, args.server_port)
        print(f"# E2E server (port {args.server_port}):")
        print(cmd)
    print("---\n")


# ---------------------------------------------------------------------------
# Recording helpers (inlined from run_collection.py to avoid import issues)
# ---------------------------------------------------------------------------


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Resize RGB image to (size, size, 3) using PIL LANCZOS."""
    return np.array(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def build_frame(
    obs: Dict[str, Any],
    camera_clients: Dict[str, ZMQClientCamera],
    image_size: int = 256,
    use_obs_cameras: bool = True,
) -> Dict[str, Any]:
    """Build a data frame from robot observations and camera images."""
    joints_full = obs["joint_positions"]
    joints_6 = list(joints_full[:6])

    ee = obs["ee_pos_quat"]
    pos = ee[:3]
    rotvec = quat_to_rotvec(ee[3:7])
    tcp_pose = list(np.concatenate([pos, rotvec]))

    gripper_norm = obs["gripper_position"][0]
    gripper_pos = int(round(gripper_norm * 255))

    frame_time = time.time()
    frame = {
        "timestamp": frame_time,
        "joint_positions": joints_6,
        "tcp_pose": tcp_pose,
        "gripper_pos": gripper_pos,
    }

    for name, cam in camera_clients.items():
        if use_obs_cameras and f"{name}_rgb" in obs:
            rgb = obs[f"{name}_rgb"]
            frame[f"{name}_timestamp"] = obs.get(f"{name}_timestamp", frame_time)
        else:
            ts, rgb, _depth = cam.read()
            frame[f"{name}_timestamp"] = ts
        frame[f"{name}_rgb"] = _resize_rgb(rgb, image_size)

    return frame


def build_frame_from_obs_client(
    obs_client: ZMQClientRobot,
    camera_clients: Dict[str, ZMQClientCamera],
    image_size: int = 256,
) -> Dict[str, Any]:
    """Build a data frame using the read-only obs client (for skill recording)."""
    robot_obs = obs_client.get_observations()
    return build_frame(robot_obs, camera_clients, image_size, use_obs_cameras=False)


def record_skill_thread(
    obs_client: ZMQClientRobot,
    camera_clients: Dict[str, ZMQClientCamera],
    buffer: EpisodeBuffer,
    stop_event: threading.Event,
    hz: int = 30,
    image_size: int = 256,
):
    """Background thread: records frames at ~hz during skill execution."""
    if hz <= 0:
        print("[RecordThread] Invalid hz <= 0, skipping.")
        return
    dt = 1.0 / hz
    count = 0
    while not stop_event.is_set():
        t0 = time.time()
        try:
            frame = build_frame_from_obs_client(obs_client, camera_clients, image_size)
            buffer.record_frame(frame)
            count += 1
        except Exception as e:
            print(f"[RecordThread] Error: {e}")

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print(f"[RecordThread] Stopped after {count} frames.")


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclass
class Args:
    # --- Model config ---
    model_type: str = "openpi"
    """Model backend: "openpi", "openvla", or "openvla_oft"."""

    # --- Task (drives both skill executor and language prompt) ---
    task: str = "cpu"
    """Task to perform: "cpu" or "ram". Sets the language prompt and skill CSV."""
    mode: str = "planner"
    """Pipeline mode: "planner" (approach + skill + correction), "e2e",
    or "correction" (correction-only, starts from current position)."""

    # --- Server connection ---
    server_host: str = "127.0.0.1"
    """Host for the model server (shared for all backends)."""
    server_port: int = 8000
    """Port for the model server. Defaults: OpenPI=8000, OpenVLA=8000, OFT=8777."""

    # --- Timing ---
    fps: int = 10
    """Inference control rate in Hz. Must match training FPS."""
    max_steps: int = 6000
    """Safety timeout in inference steps (6000 = 10 min at 10Hz). The robot runs until
    the model outputs a stable stop signal; this is only a hard ceiling."""

    # --- Robot (same defaults as run_collection.py) ---
    robot_port: int = 6001
    obs_port: int = 6002
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"

    # --- Skill config (same as run_collection.py) ---
    cpu_skill_csv: str = "CPU_Skills.csv"
    cpu_relative_count: int = 20
    ram_skill_csv: str = "RAM_Skills.csv"
    ram_relative_count: int = 5
    skill_move_speed: float = 0.1
    skill_move_accel: float = 0.04

    # --- Recording ---
    record: bool = True
    """Record inference episodes for evaluation."""
    record_hz: int = 30
    data_dir: str = "data/inference_episodes"
    image_size: int = 256

    # --- Safety ---
    disable_safety: bool = False

    # --- OpenPI specific ---
    openpi_base: str = "droid"
    """OpenPI base model: "droid" (Pi0.5-DROID), "base" (Pi0.5),
    "droid_zeroshot" (pre-trained, no fine-tuning), "base_zeroshot". Ignored for other backends."""

    open_loop_horizon: int = 10
    """How many of the action chunk to execute before re-querying."""

    # --- Correction model (planner mode) ---
    correction_server_port: int = 0
    """Port for correction model server (planner mode). 0 = no correction model."""

    correction_unnorm_key: str = ""
    """Normalization key for correction model (OpenVLA only). Empty = same as unnorm_key."""

    # --- Checkpoint tracking ---
    checkpoint_name: str = ""
    """Human-readable checkpoint ID for eval tracking. Auto-derived for OpenPI if empty."""

    correction_checkpoint_name: str = ""
    """Human-readable checkpoint ID for correction model. Auto-derived for OpenPI if empty."""

    # --- OpenVLA / OFT specific ---
    unnorm_key: str = "ur5e_vla_planner_10hz"
    """Normalization stats key for OpenVLA/OFT models."""


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------


def create_adapter(args: Args, port_override: int = 0, unnorm_key_override: str = ""):
    """Create a model adapter from CLI args.

    All three backends use server-client architecture — the adapter
    connects to a running model server over network.

    Args:
        args: CLI args.
        port_override: Override server port (used for correction model).
        unnorm_key_override: Override unnorm_key (used for correction model).
    """
    host = args.server_host
    port = port_override or args.server_port
    key = unnorm_key_override or args.unnorm_key

    if args.model_type == "openpi":
        return OpenPIAdapter(
            host=host,
            port=port,
            open_loop_horizon=args.open_loop_horizon,
        )
    elif args.model_type == "openvla":
        return OpenVLAAdapter(
            host=host,
            port=port,
            unnorm_key=key,
        )
    elif args.model_type == "openvla_oft":
        return OpenVLAOFTAdapter(
            host=host,
            port=port,
            unnorm_key=key,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


# ---------------------------------------------------------------------------
# Recording thread management
# ---------------------------------------------------------------------------


class RecordingThreadManager:
    """Manages the background recording thread for skill phases."""

    def __init__(self, obs_client, camera_clients, buffer, record_hz, image_size):
        self._obs_client = obs_client
        self._camera_clients = camera_clients
        self._buffer = buffer
        self._record_hz = record_hz
        self._image_size = image_size
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return  # Already running — avoid duplicate threads
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=record_skill_thread,
            args=(
                self._obs_client,
                self._camera_clients,
                self._buffer,
                self._stop_event,
                self._record_hz,
                self._image_size,
            ),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._stop_event = None
        self._thread = None


# ---------------------------------------------------------------------------
# Env helper (lightweight, no RobotEnv rate limiting)
# ---------------------------------------------------------------------------


def get_obs(robot_client, camera_clients) -> Dict[str, Any]:
    """Get observations from robot + cameras (no rate limiting)."""
    observations = {}

    for name, cam in camera_clients.items():
        timestamp, image, depth = cam.read()
        observations[f"{name}_timestamp"] = timestamp
        observations[f"{name}_rgb"] = image
        observations[f"{name}_depth"] = depth

    robot_obs = robot_client.get_observations()
    observations["joint_positions"] = robot_obs["joint_positions"]
    observations["joint_velocities"] = robot_obs["joint_velocities"]
    observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
    observations["gripper_position"] = robot_obs["gripper_position"]
    return observations


# ---------------------------------------------------------------------------
# Human evaluation label
# ---------------------------------------------------------------------------


def _prompt_human_label(episode_dir) -> None:
    """Ask the operator to label whether the episode was successful.

    Writes ``human_label`` (true/false) into the existing episode_meta.json.
    Ctrl+C during the prompt skips labeling (field remains absent).
    """
    import json
    from pathlib import Path

    meta_path = Path(episode_dir) / "episode_meta.json"
    if not meta_path.exists():
        return

    print()
    try:
        while True:
            ans = input("  Was this episode successful? (y/n): ").strip().lower()
            if ans in ("y", "yes"):
                label = True
                break
            elif ans in ("n", "no"):
                label = False
                break
            print("  Please enter y or n.")
    except (KeyboardInterrupt, EOFError):
        print("\n  Skipping label.")
        return

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        meta["human_label"] = label
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  Labeled: {'success' if label else 'failure'}")
    except Exception as e:
        print(f"  Failed to update metadata: {e}")


# ---------------------------------------------------------------------------
# Save and go home
# ---------------------------------------------------------------------------


def _full_stop(robot_client) -> None:
    """Stop all active control modes (servoJ, moveL, speedL).

    Calls all three stop methods so it's safe regardless of which control
    mode was last used.  Each is a no-op if that mode wasn't active.
    """
    try:
        robot_client.servo_stop()
    except Exception:
        pass
    try:
        robot_client.stop_linear()
    except Exception:
        pass
    try:
        robot_client.speed_stop()
    except Exception:
        pass


def save_and_go_home(
    robot_client,
    buffer: EpisodeBuffer,
    writer: DatasetWriter,
    rec_mgr: RecordingThreadManager,
    skill_name: str = "",
    outcome: str = "completed",
    grasp_info: Optional[Dict[str, Any]] = None,
    model_type: str = "",
    prompt: str = "",
    checkpoint_name: str = "",
    correction_checkpoint_name: str = "",
    mode: str = "",
    inference_fps: int = 0,
):
    """Stop recording, save episode, move home."""
    _full_stop(robot_client)
    rec_mgr.stop()

    episode_dir = None
    if buffer.phase != "idle":
        metadata = buffer.get_metadata()
        frames, segments = buffer.export()
        episode_meta = {
            **metadata,
            "skill_name": skill_name,
            "skill_outcome": outcome,
            "model_type": model_type,
            "prompt": prompt,
            "checkpoint_name": checkpoint_name,
            "correction_checkpoint_name": correction_checkpoint_name,
            "mode": mode,
            "inference_fps": inference_fps,
            **(grasp_info or {}),
        }
        episode_dir = writer.save_unified_episode(
            frames, segments, metadata=episode_meta
        )
        print(f"[INFERENCE] Recording saved ({len(frames)} frames).")

    home_j = (
        _session_home_joints if _session_home_joints is not None else HOME_JOINTS_RAD
    )
    home_g = (
        _session_home_gripper if _session_home_gripper is not None else HOME_GRIPPER_POS
    )

    robot_client.set_gripper_speed(255)
    robot_client.set_gripper(home_g)
    print("[INFERENCE] Moving to home position...")
    robot_client.move_joints(list(home_j), speed=0.5, accel=0.3)

    # Ask human to label the episode while robot homes
    if episode_dir is not None:
        _prompt_human_label(episode_dir)


# ---------------------------------------------------------------------------
# Non-blocking keystroke listener (for OFT manual stop)
# ---------------------------------------------------------------------------


class _KeyListener:
    """Background thread that sets a flag when the target key is pressed.

    Uses raw terminal mode so single keypresses are detected without Enter.
    Restores terminal settings on stop().
    """

    def __init__(self, key: str = "s"):
        self.key = key
        self.pressed = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._old_settings = None

    def start(self) -> None:
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self) -> None:
        while not self._stop.is_set():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch == self.key:
                    self.pressed.set()
                    return

    def stop(self) -> None:
        self._stop.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None


# ---------------------------------------------------------------------------
# Mode A: Planner + Correction
# ---------------------------------------------------------------------------


def run_planner_mode(
    args: Args,
    agent: VLAAgent,
    correction_agent: Optional[VLAAgent],
    robot_client,
    obs_client,
    camera_clients: Dict[str, ZMQClientCamera],
    skill_executor: CSVSkillExecutor,
    buffer: EpisodeBuffer,
    writer: DatasetWriter,
    rec_mgr: RecordingThreadManager,
    prompt: str = "",
) -> bool:
    """Planner inference → skill execution → optional correction → skill_resume.

    Returns True if a manual server swap to correction occurred (caller
    should prompt to swap back for the next episode).
    """

    # ---------------------------------------------------------------
    # Phase 1: Planner approach (replaces human teleop)
    # ---------------------------------------------------------------
    buffer.start()
    buffer.set_phase("teleop")

    last_wrist_ts = 0.0
    last_base_ts = 0.0
    obs = get_obs(robot_client, camera_clients)
    dt = 1.0 / args.fps

    # OFT manual stop: operator presses 's', then delta must also be small.
    oft_manual_stop = args.model_type == "openvla_oft" and args.mode == "planner"
    key_listener = None
    s_pressed = False
    manual_stop_triggered = False
    if oft_manual_stop:
        key_listener = _KeyListener(key="s")
        key_listener.start()
        print(
            f"\n[PLANNER] Starting approach ({args.max_steps} max steps @ {args.fps} Hz)"
        )
        print("[PLANNER] OFT mode — press 's' to signal stop (delta check follows)")
    else:
        print(
            f"\n[PLANNER] Starting approach ({args.max_steps} max steps @ {args.fps} Hz)"
        )

    for step in range(args.max_steps):
        t0 = time.time()

        agent.execute_step(obs, robot_client)

        # Record frame (camera-gated)
        if args.record:
            cur_w = obs.get("wrist_timestamp", 0.0)
            cur_b = obs.get("base_timestamp", 0.0)
            if (cur_w > 0 and cur_w > last_wrist_ts) or (
                cur_b > 0 and cur_b > last_base_ts
            ):
                frame = build_frame(obs, camera_clients, args.image_size)
                buffer.record_frame(frame)
                if cur_w > last_wrist_ts:
                    last_wrist_ts = cur_w
                if cur_b > last_base_ts:
                    last_base_ts = cur_b

        # --- Stop detection ---
        if agent.stop_detected:
            # Automatic stop (OpenPI / OpenVLA)
            print(f"\n[PLANNER] Stop signal at step {step} — triggering skill")
            break

        if oft_manual_stop:
            # Check if 's' was pressed
            if not s_pressed and key_listener.pressed.is_set():
                s_pressed = True
                print(
                    f"\n[PLANNER] 's' pressed at step {step}"
                    " — waiting for delta confirmation..."
                )

            # After 's' pressed, check EEF delta each chunk
            if s_pressed:
                current_state = agent.adapter.get_current_state(obs)
                if agent._check_chunk_stop(list(agent._action_queue), current_state):
                    manual_stop_triggered = True
                    print(
                        f"[PLANNER] Delta confirmed at step {step}"
                        " — triggering skill"
                    )
                    break

        obs = get_obs(robot_client, camera_clients)

        # Rate limit
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        # Status display (~1 Hz)
        if step > 0 and step % args.fps == 0:
            suffix = " [s pressed, waiting delta...]" if s_pressed else ""
            print(
                f"\r[PLANNER] step {step}/{args.max_steps} | "
                f"frames: {buffer.num_frames}{suffix}     ",
                end="",
                flush=True,
            )

    if key_listener is not None:
        key_listener.stop()

    stop_ok = agent.stop_detected or manual_stop_triggered
    if not stop_ok:
        print(f"\n[PLANNER] Max steps ({args.max_steps}) reached without stop signal")
        save_and_go_home(
            robot_client,
            buffer,
            writer,
            rec_mgr,
            skill_name=args.task,
            outcome="timeout",
            model_type=args.model_type,
            prompt=prompt,
            checkpoint_name=args.checkpoint_name,
            correction_checkpoint_name=args.correction_checkpoint_name,
            mode=args.mode,
            inference_fps=args.fps,
        )
        return False

    # ---------------------------------------------------------------
    # Phase 2: Skill execution
    # ---------------------------------------------------------------
    _full_stop(robot_client)
    time.sleep(0.2)

    # Reorient gripper to point straight down before skill execution.
    # Keeps XYZ position, forces tool-Z to [0,0,-1] while preserving yaw.
    current_tcp = robot_client.get_tcp_pose_raw()
    R_cur = Rotation.from_rotvec(current_tcp[3:]).as_matrix()
    tool_x = R_cur[:, 0].copy()
    tool_x[2] = 0.0
    tool_x /= np.linalg.norm(tool_x)
    new_z = np.array([0.0, 0.0, -1.0])
    new_y = np.cross(new_z, tool_x)
    new_y /= np.linalg.norm(new_y)
    R_vert = np.column_stack([tool_x, new_y, new_z])
    vertical_rotvec = Rotation.from_matrix(R_vert).as_rotvec()
    vertical_tcp = current_tcp.copy()
    vertical_tcp[3:] = vertical_rotvec
    print("[PLANNER] Reorienting gripper to vertical...")
    robot_client.move_linear(vertical_tcp, speed=0.05, accel=0.1, asynchronous=False)

    trigger_tcp = robot_client.get_tcp_pose_raw()
    buffer.set_phase("skill")

    if args.record:
        rec_mgr.start()

    interrupt_event = threading.Event()
    grasp_failed_flag = [False]
    grasp_info: Dict[str, Any] = {}

    def on_grasp_failed():
        grasp_failed_flag[0] = True
        rec_mgr.stop()
        print("\n[PIPELINE] Grasp failed — correction model needed")

    print(f"[PIPELINE] Executing skill '{args.task}'...")
    completed, grasp_info = skill_executor.execute(
        args.task,
        trigger_tcp_raw=trigger_tcp,
        interrupt_event=interrupt_event,
        on_grasp_failed=on_grasp_failed,
    )

    if completed:
        rec_mgr.stop()
        print("[PIPELINE] Skill completed successfully!")
        save_and_go_home(
            robot_client,
            buffer,
            writer,
            rec_mgr,
            skill_name=args.task,
            outcome="completed",
            grasp_info=grasp_info,
            model_type=args.model_type,
            prompt=prompt,
            checkpoint_name=args.checkpoint_name,
            correction_checkpoint_name=args.correction_checkpoint_name,
            mode=args.mode,
            inference_fps=args.fps,
        )
        return False

    if not grasp_failed_flag[0]:
        # Interrupted (e.g., drop detected) — abort
        rec_mgr.stop()
        print("[PIPELINE] Skill interrupted — saving episode")
        save_and_go_home(
            robot_client,
            buffer,
            writer,
            rec_mgr,
            skill_name=args.task,
            outcome="interrupted",
            grasp_info=grasp_info,
            model_type=args.model_type,
            prompt=prompt,
            checkpoint_name=args.checkpoint_name,
            correction_checkpoint_name=args.correction_checkpoint_name,
            mode=args.mode,
            inference_fps=args.fps,
        )
        return False

    # ---------------------------------------------------------------
    # Phase 3: Correction model (only reached if grasp failed)
    # ---------------------------------------------------------------
    if correction_agent is None and args.correction_server_port <= 0:
        print("[PIPELINE] No correction model configured — saving failed episode")
        save_and_go_home(
            robot_client,
            buffer,
            writer,
            rec_mgr,
            skill_name=args.task,
            outcome="grasp_failed_no_correction",
            grasp_info=grasp_info,
            model_type=args.model_type,
            prompt=prompt,
            checkpoint_name=args.checkpoint_name,
            correction_checkpoint_name=args.correction_checkpoint_name,
            mode=args.mode,
            inference_fps=args.fps,
        )
        return False

    # Manual server swap (correction_agent is None but port is configured)
    did_manual_swap = False
    if correction_agent is None:
        port = args.correction_server_port or args.server_port
        print("\n" + "=" * 60)
        print("  GRASP FAILED — SERVER SWAP REQUIRED")
        print("=" * 60)
        print("  In T3:")
        print("    1. Ctrl+C the planner server")
        print(f"    2. Start the correction server on port {port}:")
        if args.model_type == "openpi":
            corr_cmd = get_openpi_serve_cmd(
                args.openpi_base, "correction", args.fps, port
            )
            print()
            for line in corr_cmd.split("\n"):
                print(f"       {line}")
            print()
        print("    3. Wait for server to print 'Started server process'")
        print("  Then press Enter here in T4.")
        print("=" * 60)
        input("  Press Enter when correction server is ready...")

        correction_adapter = create_adapter(
            args,
            port_override=port,
            unnorm_key_override=args.correction_unnorm_key,
        )
        correction_agent = VLAAgent(
            correction_adapter,
            args.fps,
            prompt,
            task=args.task,
            safety_monitor=None if args.disable_safety else SafetyMonitor(),
        )
        did_manual_swap = True

    buffer.set_phase("correction")
    correction_agent.reset()
    obs = get_obs(robot_client, camera_clients)
    last_wrist_ts = 0.0
    last_base_ts = 0.0

    print(
        f"\n[CORRECTION] Starting correction ({args.max_steps} max steps @ {args.fps} Hz)"
    )

    for step in range(args.max_steps):
        t0 = time.time()

        correction_agent.execute_step(obs, robot_client)

        if args.record:
            cur_w = obs.get("wrist_timestamp", 0.0)
            cur_b = obs.get("base_timestamp", 0.0)
            if (cur_w > 0 and cur_w > last_wrist_ts) or (
                cur_b > 0 and cur_b > last_base_ts
            ):
                frame = build_frame(obs, camera_clients, args.image_size)
                buffer.record_frame(frame)
                if cur_w > last_wrist_ts:
                    last_wrist_ts = cur_w
                if cur_b > last_base_ts:
                    last_base_ts = cur_b

        if correction_agent.stop_detected:
            print(f"\n[CORRECTION] Stop signal at step {step} — resuming skill")
            break

        obs = get_obs(robot_client, camera_clients)

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        if step > 0 and step % args.fps == 0:
            print(
                f"\r[CORRECTION] step {step}/{args.max_steps} | "
                f"frames: {buffer.num_frames}     ",
                end="",
                flush=True,
            )

    if not correction_agent.stop_detected:
        print(f"\n[CORRECTION] Max steps reached without stop signal")
        save_and_go_home(
            robot_client,
            buffer,
            writer,
            rec_mgr,
            skill_name=args.task,
            outcome="correction_timeout",
            grasp_info=grasp_info,
            model_type=args.model_type,
            prompt=prompt,
            checkpoint_name=args.checkpoint_name,
            correction_checkpoint_name=args.correction_checkpoint_name,
            mode=args.mode,
            inference_fps=args.fps,
        )
        return False

    # ---------------------------------------------------------------
    # Phase 4: Skill resume (absolute waypoints only)
    # ---------------------------------------------------------------
    _full_stop(robot_client)
    time.sleep(0.1)

    buffer.set_phase("skill_resume")
    if args.record:
        rec_mgr.start()

    print("[PIPELINE] Resuming skill (absolute waypoints only)...")
    completed, resume_info = skill_executor.execute(
        args.task,
        interrupt_event=interrupt_event,
        resume_absolute_only=True,
    )
    grasp_info.update(resume_info)

    rec_mgr.stop()
    outcome = "completed_after_correction" if completed else "skill_resume_failed"
    print(f"[PIPELINE] Skill resume {'completed' if completed else 'failed'}.")
    save_and_go_home(
        robot_client,
        buffer,
        writer,
        rec_mgr,
        skill_name=args.task,
        outcome=outcome,
        grasp_info=grasp_info,
        model_type=args.model_type,
        prompt=prompt,
        checkpoint_name=args.checkpoint_name,
        correction_checkpoint_name=args.correction_checkpoint_name,
        mode=args.mode,
        inference_fps=args.fps,
    )
    return did_manual_swap


# ---------------------------------------------------------------------------
# Mode B: End-to-End
# ---------------------------------------------------------------------------


def run_e2e_mode(
    args: Args,
    agent: VLAAgent,
    robot_client,
    camera_clients: Dict[str, ZMQClientCamera],
    buffer: EpisodeBuffer,
    writer: DatasetWriter,
    rec_mgr: RecordingThreadManager,
    prompt: str = "",
):
    """E2E inference — model handles entire trajectory until timeout."""
    buffer.start()
    buffer.set_phase("teleop")  # Same phase label for consistency

    last_wrist_ts = 0.0
    last_base_ts = 0.0
    obs = get_obs(robot_client, camera_clients)
    dt = 1.0 / args.fps

    print(f"\n[E2E] Starting inference ({args.max_steps} max steps @ {args.fps} Hz)")
    print("[E2E] Press Ctrl+C to stop and save episode")

    for step in range(args.max_steps):
        t0 = time.time()

        agent.execute_step(obs, robot_client)

        if args.record:
            cur_w = obs.get("wrist_timestamp", 0.0)
            cur_b = obs.get("base_timestamp", 0.0)
            if (cur_w > 0 and cur_w > last_wrist_ts) or (
                cur_b > 0 and cur_b > last_base_ts
            ):
                frame = build_frame(obs, camera_clients, args.image_size)
                buffer.record_frame(frame)
                if cur_w > last_wrist_ts:
                    last_wrist_ts = cur_w
                if cur_b > last_base_ts:
                    last_base_ts = cur_b

        obs = get_obs(robot_client, camera_clients)

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        if step > 0 and step % args.fps == 0:
            print(
                f"\r[E2E] step {step}/{args.max_steps} | "
                f"frames: {buffer.num_frames}     ",
                end="",
                flush=True,
            )

    print(f"\n[E2E] Episode complete ({step + 1} steps)")
    save_and_go_home(
        robot_client,
        buffer,
        writer,
        rec_mgr,
        skill_name="e2e",
        outcome="complete",
        model_type=args.model_type,
        prompt=prompt,
        checkpoint_name=args.checkpoint_name,
        correction_checkpoint_name=args.correction_checkpoint_name,
        mode=args.mode,
        inference_fps=args.fps,
    )


# ---------------------------------------------------------------------------
# Mode C: Correction-Only
# ---------------------------------------------------------------------------


def run_correction_mode(
    args: Args,
    agent: VLAAgent,
    robot_client,
    obs_client,
    camera_clients: Dict[str, ZMQClientCamera],
    skill_executor: CSVSkillExecutor,
    buffer: EpisodeBuffer,
    writer: DatasetWriter,
    rec_mgr: RecordingThreadManager,
    prompt: str = "",
):
    """Correction-only mode — runs correction + skill_resume from current position.

    Skips the planner approach and first skill attempt. The robot starts
    wherever it is and runs the correction model until a stop signal fires,
    then executes the skill resume (absolute waypoints only).
    """
    buffer.start()
    buffer.set_phase("correction")

    last_wrist_ts = 0.0
    last_base_ts = 0.0
    obs = get_obs(robot_client, camera_clients)
    dt = 1.0 / args.fps

    print(
        f"\n[CORRECTION] Starting correction-only "
        f"({args.max_steps} max steps @ {args.fps} Hz)"
    )

    for step in range(args.max_steps):
        t0 = time.time()

        agent.execute_step(obs, robot_client)

        if args.record:
            cur_w = obs.get("wrist_timestamp", 0.0)
            cur_b = obs.get("base_timestamp", 0.0)
            if (cur_w > 0 and cur_w > last_wrist_ts) or (
                cur_b > 0 and cur_b > last_base_ts
            ):
                frame = build_frame(obs, camera_clients, args.image_size)
                buffer.record_frame(frame)
                if cur_w > last_wrist_ts:
                    last_wrist_ts = cur_w
                if cur_b > last_base_ts:
                    last_base_ts = cur_b

        if agent.stop_detected:
            print(
                f"\n[CORRECTION] Stop signal at step {step} — triggering skill resume"
            )
            break

        obs = get_obs(robot_client, camera_clients)

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        if step > 0 and step % args.fps == 0:
            print(
                f"\r[CORRECTION] step {step}/{args.max_steps} | "
                f"frames: {buffer.num_frames}     ",
                end="",
                flush=True,
            )

    if not agent.stop_detected:
        print("\n[CORRECTION] Max steps reached without stop signal")
        save_and_go_home(
            robot_client,
            buffer,
            writer,
            rec_mgr,
            skill_name=args.task,
            outcome="correction_timeout",
            model_type=args.model_type,
            prompt=prompt,
            checkpoint_name=args.checkpoint_name,
            correction_checkpoint_name=args.correction_checkpoint_name,
            mode=args.mode,
            inference_fps=args.fps,
        )
        return

    # ---------------------------------------------------------------
    # Skill resume (absolute waypoints only)
    # ---------------------------------------------------------------
    _full_stop(robot_client)
    time.sleep(0.2)

    # Reorient gripper to vertical before skill resume
    current_tcp = robot_client.get_tcp_pose_raw()
    R_cur = Rotation.from_rotvec(current_tcp[3:]).as_matrix()
    tool_x = R_cur[:, 0].copy()
    tool_x[2] = 0.0
    tool_x /= np.linalg.norm(tool_x)
    new_z = np.array([0.0, 0.0, -1.0])
    new_y = np.cross(new_z, tool_x)
    new_y /= np.linalg.norm(new_y)
    R_vert = np.column_stack([tool_x, new_y, new_z])
    vertical_rotvec = Rotation.from_matrix(R_vert).as_rotvec()
    vertical_tcp = current_tcp.copy()
    vertical_tcp[3:] = vertical_rotvec
    print("[CORRECTION] Reorienting gripper to vertical...")
    robot_client.move_linear(vertical_tcp, speed=0.05, accel=0.1, asynchronous=False)

    buffer.set_phase("skill_resume")
    if args.record:
        rec_mgr.start()

    interrupt_event = threading.Event()
    print("[PIPELINE] Resuming skill (absolute waypoints only)...")
    completed, resume_info = skill_executor.execute(
        args.task,
        interrupt_event=interrupt_event,
        resume_absolute_only=True,
    )

    rec_mgr.stop()
    outcome = "completed_correction_only" if completed else "skill_resume_failed"
    print(f"[PIPELINE] Skill resume {'completed' if completed else 'failed'}.")
    save_and_go_home(
        robot_client,
        buffer,
        writer,
        rec_mgr,
        skill_name=args.task,
        outcome=outcome,
        grasp_info=resume_info,
        model_type=args.model_type,
        prompt=prompt,
        checkpoint_name=args.checkpoint_name,
        correction_checkpoint_name=args.correction_checkpoint_name,
        mode=args.mode,
        inference_fps=args.fps,
    )


# ---------------------------------------------------------------------------
# Task instructions (must match training data from conversion_utils.py)
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = {
    "cpu": (
        "Extract the CPU from the Bracket by unlocking it first, "
        "then extract the CPU and place it inside the yellow square area, "
        "then back home."
    ),
    "ram": (
        "Extract the RAM from the slot and place it inside the blue square area, "
        "then back home."
    ),
}


# ---------------------------------------------------------------------------
# Dynamic home position (set at startup for correction mode / OpenVLA)
# ---------------------------------------------------------------------------

# When set, save_and_go_home() and Ctrl+C handler use these instead of
# the default HOME_JOINTS_RAD / HOME_GRIPPER_POS constants.
_session_home_joints: Optional[np.ndarray] = None
_session_home_gripper: Optional[int] = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args):
    # Resolve prompt from task — single source of truth
    prompt = TASK_INSTRUCTIONS.get(args.task)
    if prompt is None:
        raise ValueError(
            f"Unknown task: {args.task!r}. "
            f"Must be one of: {', '.join(TASK_INSTRUCTIONS.keys())}"
        )

    # Auto-derive checkpoint names for OpenPI if not explicitly set
    if args.model_type == "openpi":
        target_map = {"planner": "planner", "e2e": "e2e", "correction": "correction"}
        target = target_map.get(args.mode, args.mode)
        key = (args.openpi_base, target, args.fps)
        if not args.checkpoint_name and key in OPENPI_CHECKPOINTS:
            _, ckpt_path = OPENPI_CHECKPOINTS[key]
            args.checkpoint_name = ckpt_path.replace("/", "_")
        if (
            args.mode == "planner"
            and args.correction_server_port > 0
            and not args.correction_checkpoint_name
        ):
            corr_key = (args.openpi_base, "correction", args.fps)
            if corr_key in OPENPI_CHECKPOINTS:
                _, ckpt_path = OPENPI_CHECKPOINTS[corr_key]
                args.correction_checkpoint_name = ckpt_path.replace("/", "_")

    print("=" * 60)
    print("  VLA INFERENCE PIPELINE")
    print("=" * 60)
    print(f"  Model:    {args.model_type}")
    if args.model_type == "openpi":
        base_labels = {
            "droid": "Pi0.5-DROID (fine-tuned)",
            "base": "Pi0.5-base (fine-tuned)",
            "droid_zeroshot": "Pi0.5-DROID (zero-shot, no fine-tuning)",
            "base_zeroshot": "Pi0.5-base (zero-shot, no fine-tuning)",
        }
        print(f"  Base:     {base_labels.get(args.openpi_base, args.openpi_base)}")
    print(f"  Server:   {args.server_host}:{args.server_port}")
    print(f"  Task:     {args.task}")
    print(f"  Mode:     {args.mode}")
    print(f"  Prompt:   {prompt}")
    print(f"  FPS:      {args.fps} Hz")
    print(f"  Max steps: {args.max_steps}")
    if args.checkpoint_name:
        print(f"  Checkpoint: {args.checkpoint_name}")
    if args.mode == "planner":
        if args.correction_server_port > 0:
            print(f"  Correction: {args.server_host}:{args.correction_server_port}")
            if args.correction_checkpoint_name:
                print(f"  Corr ckpt:  {args.correction_checkpoint_name}")
            print("  Server swap: on grasp failure (manual)")
        else:
            print("  Correction: none")
    print(f"  Robot:    {args.hostname}:{args.robot_port}")
    print(f"  Record:   {args.record} -> {args.data_dir}")
    print(f"  Safety:   {'disabled' if args.disable_safety else 'enabled'}")
    print("=" * 60)

    # Print expected serve commands for OpenPI
    if args.model_type == "openpi":
        print_openpi_serve_commands(args)

    # ------------------------------------------------------------------
    # 1. Connect to robot + cameras
    # ------------------------------------------------------------------
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    obs_client = ZMQClientRobot(port=args.obs_port, host=args.hostname)

    camera_clients = {
        "wrist": ZMQClientCamera(
            port=args.wrist_camera_port,
            host=args.hostname,
            camera_name="wrist",
            dummy_shape_rgb=(720, 1280, 3),
            dummy_shape_depth=(720, 1280, 1),
        ),
        "base": ZMQClientCamera(
            port=args.base_camera_port,
            host=args.hostname,
            camera_name="base",
            dummy_shape_rgb=(720, 1280, 3),
            dummy_shape_depth=(720, 1280, 1),
        ),
    }

    # ------------------------------------------------------------------
    # 2. Create model adapter
    # ------------------------------------------------------------------
    print(f"\n[INIT] Creating {args.model_type} adapter...")
    adapter = create_adapter(args)
    print("[INIT] Adapter ready.")

    # ------------------------------------------------------------------
    # 3. Create safety monitor
    # ------------------------------------------------------------------
    safety = None if args.disable_safety else SafetyMonitor()

    # ------------------------------------------------------------------
    # 4. Create VLA agent
    # ------------------------------------------------------------------
    # OFT uses manual stop (operator presses 's') + delta confirmation,
    # so disable automatic stop detection for OFT planner mode.
    # Correction mode always uses automatic stop detection.
    auto_stop = args.mode not in ("e2e",) and args.model_type != "openvla_oft"
    agent = VLAAgent(
        adapter,
        args.fps,
        prompt,
        task=args.task,
        safety_monitor=safety,
        enable_stop_detection=auto_stop,
    )

    # ------------------------------------------------------------------
    # 5. Create correction agent (planner mode only)
    # ------------------------------------------------------------------
    correction_agent = None
    if args.mode == "planner" and args.correction_server_port > 0:
        if args.correction_server_port != args.server_port:
            # Dual-server mode: both planner + correction already running
            print("[INIT] Creating correction model adapter...")
            correction_adapter = create_adapter(
                args,
                port_override=args.correction_server_port,
                unnorm_key_override=args.correction_unnorm_key,
            )
            correction_agent = VLAAgent(
                correction_adapter,
                args.fps,
                prompt,
                task=args.task,
                safety_monitor=safety,
            )
            print("[INIT] Correction adapter ready.")
        else:
            # Same port — manual swap after grasp failure
            print("[INIT] Correction via manual server swap after grasp failure.")

    # ------------------------------------------------------------------
    # 6. Setup skill executor (planner + correction modes)
    # ------------------------------------------------------------------
    skill_executor = None
    if args.mode in ("planner", "correction"):
        skill_csvs = {
            "cpu": args.cpu_skill_csv,
            "ram": args.ram_skill_csv,
        }
        skill_rel_counts = {
            "cpu": args.cpu_relative_count,
            "ram": args.ram_relative_count,
        }
        skill_executor = CSVSkillExecutor(
            skill_csvs=skill_csvs,
            robot_client=robot_client,
            obs_client=obs_client,
            relative_counts=skill_rel_counts,
            grasp_thresholds={"cpu": 158, "ram": 226},
            move_speed=args.skill_move_speed,
            move_accel=args.skill_move_accel,
        )

    # ------------------------------------------------------------------
    # 7. Setup buffer + writer + recording manager
    # ------------------------------------------------------------------
    buffer = EpisodeBuffer()
    writer = DatasetWriter(data_dir=args.data_dir)
    rec_mgr = RecordingThreadManager(
        obs_client,
        camera_clients,
        buffer,
        args.record_hz,
        args.image_size,
    )

    # ------------------------------------------------------------------
    # 8. Verify connections + move to home position
    # ------------------------------------------------------------------
    print("\n[INIT] Verifying robot connection...")
    try:
        test_obs = robot_client.get_observations()
        print(f"[INIT] Robot OK — joints: {test_obs['joint_positions'][:3]}...")
    except Exception as e:
        print(
            f"[INIT] ERROR: Cannot reach robot server on {args.hostname}:{args.robot_port}"
        )
        print(f"[INIT] Make sure launch_nodes.py is running. Error: {e}")
        return

    global _session_home_joints, _session_home_gripper

    if args.mode == "correction" or args.model_type == "openvla":
        # Read current position as session home — robot stays where it is
        init_obs = obs_client.get_observations()
        _session_home_joints = np.array(init_obs["joint_positions"][:6])
        _session_home_gripper = int(round(init_obs["gripper_position"][0] * 255))
        print(
            f"[INIT] Dynamic home captured (joints[0:3]="
            f"{_session_home_joints[:3].round(3)}, gripper={_session_home_gripper})"
        )
        if args.mode == "correction":
            # Stay at current position — no homing move
            print("[INIT] Correction mode — skipping home move.")
        else:
            # OpenVLA non-correction: still move to current pos (no-op) to set gripper
            robot_client.set_gripper_speed(255)
            robot_client.set_gripper(_session_home_gripper)
    else:
        robot_client.set_gripper_speed(255)
        robot_client.set_gripper(HOME_GRIPPER_POS)
        print("[INIT] Moving to home position...")
        robot_client.move_joints(list(HOME_JOINTS_RAD), speed=0.5, accel=0.3)

    print("[INIT] Home reached. Starting inference...\n")

    # ------------------------------------------------------------------
    # 9. Run inference (loop for multiple episodes)
    # ------------------------------------------------------------------
    episode_num = 0
    try:
        while True:
            episode_num += 1
            print(f"\n{'=' * 60}")
            print(f"  EPISODE {episode_num}")
            print(f"{'=' * 60}")

            # Reset agent state for new episode
            agent.reset()
            if correction_agent is not None:
                correction_agent.reset()
            buffer = EpisodeBuffer()
            rec_mgr = RecordingThreadManager(
                obs_client,
                camera_clients,
                buffer,
                args.record_hz,
                args.image_size,
            )

            # OpenVLA: lower the end-effector 0.1m from home before each
            # episode so the model starts closer to the workspace.
            # Skip in correction mode — already at the correction start position.
            if args.model_type == "openvla" and args.mode != "correction":
                tcp = obs_client.get_tcp_pose_raw()
                tcp[2] -= 0.1  # lower Z by 0.1 m
                print("[INIT] OpenVLA pre-start: lowering EEF by 0.1 m...")
                robot_client.move_linear(tcp, speed=0.1, accel=0.3)

            if args.mode == "planner":
                swapped = run_planner_mode(
                    args,
                    agent,
                    correction_agent,
                    robot_client,
                    obs_client,
                    camera_clients,
                    skill_executor,
                    buffer,
                    writer,
                    rec_mgr,
                    prompt=prompt,
                )
                # Only prompt to swap back if a manual swap actually happened
                if swapped:
                    port = args.server_port
                    print("\n" + "=" * 60)
                    print("  SWAP BACK TO PLANNER FOR NEXT EPISODE")
                    print("=" * 60)
                    print("  In T3:")
                    print("    1. Ctrl+C the correction server (if running)")
                    print(f"    2. Start the planner server on port {port}:")
                    print("    3. Wait for 'Started server process'")
                    print("  Then press Enter here in T4.")
                    print("=" * 60)
                    input("  Press Enter when planner server is ready...")

            elif args.mode == "e2e":
                run_e2e_mode(
                    args,
                    agent,
                    robot_client,
                    camera_clients,
                    buffer,
                    writer,
                    rec_mgr,
                    prompt=prompt,
                )
            elif args.mode == "correction":
                run_correction_mode(
                    args,
                    agent,
                    robot_client,
                    obs_client,
                    camera_clients,
                    skill_executor,
                    buffer,
                    writer,
                    rec_mgr,
                    prompt=prompt,
                )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # After save_and_go_home, robot is at HOME. Loop for next episode.
            print(f"\n[INFERENCE] Episode {episode_num} complete. Starting next...")

    except KeyboardInterrupt:
        print("\n\n[E-STOP] Ctrl+C — stopping robot...")
        _full_stop(robot_client)
        rec_mgr.stop()

        episode_dir = None
        if buffer.phase != "idle":
            print("[E-STOP] Saving in-progress recording...")
            metadata = buffer.get_metadata()
            frames, segments = buffer.export()
            episode_meta = {
                **metadata,
                "skill_name": args.task if args.mode == "planner" else "e2e",
                "skill_outcome": "estop",
                "model_type": args.model_type,
                "prompt": prompt,
                "checkpoint_name": args.checkpoint_name,
                "correction_checkpoint_name": args.correction_checkpoint_name,
                "mode": args.mode,
                "inference_fps": args.fps,
            }
            episode_dir = writer.save_unified_episode(
                frames, segments, metadata=episode_meta
            )

        home_j = (
            _session_home_joints
            if _session_home_joints is not None
            else HOME_JOINTS_RAD
        )
        home_g = (
            _session_home_gripper
            if _session_home_gripper is not None
            else HOME_GRIPPER_POS
        )

        print("[E-STOP] Moving to home position...")
        try:
            robot_client.set_gripper_speed(255)
        except Exception as e:
            print(f"[E-STOP] set_gripper_speed failed: {e}")
        try:
            robot_client.set_gripper(home_g)
        except Exception as e:
            print(f"[E-STOP] set_gripper failed: {e}")
        try:
            robot_client.move_joints(list(home_j), speed=0.5, accel=0.3)
        except Exception as e:
            print(f"[E-STOP] move_joints failed: {e}")

        if episode_dir is not None:
            _prompt_human_label(episode_dir)

    print("Inference pipeline exited.")


if __name__ == "__main__":
    main(tyro.cli(Args))
