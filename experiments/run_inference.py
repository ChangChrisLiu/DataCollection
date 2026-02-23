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
  planner:  Model approaches → CSV skill grasps → (if fail) correction model
  e2e:      Model handles entire trajectory

Terminal architecture:
  T1: python experiments/launch_nodes.py --robot ur
  T2: python experiments/launch_camera_nodes.py --camera-settings camera_settings.json
  T3: Start model server (see README § Inference Pipeline for per-backend commands)
  T4: python experiments/run_inference.py --model-type openpi --prompt "pick up the cpu"

Ctrl+C to e-stop the robot and save the current episode.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import tyro
from PIL import Image

from gello.agents.joystick_agent import HOME_JOINTS_RAD
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
    prompt: str = ""
    """Language instruction. Auto-detected from --skill if empty (recommended)."""

    # --- Server connection ---
    server_host: str = "127.0.0.1"
    """Host for the model server (shared for all backends)."""
    server_port: int = 8000
    """Port for the model server. Defaults: OpenPI=8000, OpenVLA=8000, OFT=8777."""

    # --- Task mode ---
    mode: str = "planner"
    """Pipeline mode: "planner" (approach + skill + correction) or "e2e"."""
    skill: str = "cpu"
    """Which CSV skill for planner mode: "cpu" or "ram"."""

    # --- Timing ---
    fps: int = 10
    """Inference control rate in Hz. Must match training FPS."""
    max_steps: int = 300
    """Max inference steps before timeout."""

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
    open_loop_horizon: int = 10
    """How many of the action chunk to execute before re-querying."""

    # --- Correction model (planner mode) ---
    correction_server_port: int = 0
    """Port for correction model server (planner mode). 0 = no correction model."""

    # --- OpenVLA / OFT specific ---
    unnorm_key: str = "ur5e_vla_planner_10hz"
    """Normalization stats key for OpenVLA/OFT models."""


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------


def create_adapter(args: Args, port_override: int = 0):
    """Create a model adapter from CLI args.

    All three backends use server-client architecture — the adapter
    connects to a running model server over network.

    Args:
        args: CLI args.
        port_override: Override server port (used for correction model).
    """
    host = args.server_host
    port = port_override or args.server_port

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
            unnorm_key=args.unnorm_key,
        )
    elif args.model_type == "openvla_oft":
        return OpenVLAOFTAdapter(
            host=host,
            port=port,
            unnorm_key=args.unnorm_key,
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
# Save and go home
# ---------------------------------------------------------------------------


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
):
    """Stop recording, save episode, move home."""
    robot_client.speed_stop()
    rec_mgr.stop()

    if buffer.phase != "idle":
        metadata = buffer.get_metadata()
        frames, segments = buffer.export()
        episode_meta = {
            **metadata,
            "skill_name": skill_name,
            "skill_outcome": outcome,
            "model_type": model_type,
            "prompt": prompt,
            **(grasp_info or {}),
        }
        writer.save_unified_episode(frames, segments, metadata=episode_meta)
        print(f"[INFERENCE] Recording saved ({len(frames)} frames).")

    robot_client.set_gripper_speed(255)
    print("[INFERENCE] Moving to home position...")
    robot_client.move_joints(list(HOME_JOINTS_RAD), speed=0.5, accel=0.3)


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
):
    """Planner inference → skill execution → optional correction → skill_resume."""

    # ---------------------------------------------------------------
    # Phase 1: Planner approach (replaces human teleop)
    # ---------------------------------------------------------------
    buffer.start()
    buffer.set_phase("teleop")

    last_wrist_ts = 0.0
    last_base_ts = 0.0
    obs = get_obs(robot_client, camera_clients)
    dt = 1.0 / args.fps

    print(f"\n[PLANNER] Starting approach ({args.max_steps} max steps @ {args.fps} Hz)")

    for step in range(args.max_steps):
        t0 = time.time()

        agent.execute_step(obs, robot_client)

        # Record frame (camera-gated)
        if args.record:
            cur_w = obs.get("wrist_timestamp", 0.0)
            cur_b = obs.get("base_timestamp", 0.0)
            if (cur_w > 0 and cur_w > last_wrist_ts) or (cur_b > 0 and cur_b > last_base_ts):
                frame = build_frame(obs, camera_clients, args.image_size)
                buffer.record_frame(frame)
                if cur_w > last_wrist_ts:
                    last_wrist_ts = cur_w
                if cur_b > last_base_ts:
                    last_base_ts = cur_b

        if agent.stop_detected:
            print(f"\n[PLANNER] Stop signal at step {step} — triggering skill")
            break

        obs = get_obs(robot_client, camera_clients)

        # Rate limit
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        # Status display (~1 Hz)
        if step > 0 and step % args.fps == 0:
            print(
                f"\r[PLANNER] step {step}/{args.max_steps} | "
                f"frames: {buffer.num_frames}     ",
                end="", flush=True,
            )

    if not agent.stop_detected:
        print(f"\n[PLANNER] Max steps ({args.max_steps}) reached without stop signal")
        save_and_go_home(
            robot_client, buffer, writer, rec_mgr,
            skill_name=args.skill, outcome="timeout",
            model_type=args.model_type, prompt=args.prompt,
        )
        return

    # ---------------------------------------------------------------
    # Phase 2: Skill execution
    # ---------------------------------------------------------------
    robot_client.speed_stop()
    time.sleep(0.1)

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

    print(f"[PIPELINE] Executing skill '{args.skill}'...")
    completed, grasp_info = skill_executor.execute(
        args.skill,
        trigger_tcp_raw=trigger_tcp,
        interrupt_event=interrupt_event,
        on_grasp_failed=on_grasp_failed,
    )

    if completed:
        rec_mgr.stop()
        print("[PIPELINE] Skill completed successfully!")
        save_and_go_home(
            robot_client, buffer, writer, rec_mgr,
            skill_name=args.skill, outcome="completed",
            grasp_info=grasp_info,
            model_type=args.model_type, prompt=args.prompt,
        )
        return

    if not grasp_failed_flag[0]:
        # Interrupted (e.g., drop detected) — abort
        rec_mgr.stop()
        print("[PIPELINE] Skill interrupted — saving episode")
        save_and_go_home(
            robot_client, buffer, writer, rec_mgr,
            skill_name=args.skill, outcome="interrupted",
            grasp_info=grasp_info,
            model_type=args.model_type, prompt=args.prompt,
        )
        return

    # ---------------------------------------------------------------
    # Phase 3: Correction model
    # ---------------------------------------------------------------
    if correction_agent is None:
        print("[PIPELINE] No correction model configured — saving failed episode")
        save_and_go_home(
            robot_client, buffer, writer, rec_mgr,
            skill_name=args.skill, outcome="grasp_failed_no_correction",
            grasp_info=grasp_info,
            model_type=args.model_type, prompt=args.prompt,
        )
        return

    buffer.set_phase("correction")
    correction_agent.reset()
    obs = get_obs(robot_client, camera_clients)
    last_wrist_ts = 0.0
    last_base_ts = 0.0

    print(f"\n[CORRECTION] Starting correction ({args.max_steps} max steps @ {args.fps} Hz)")

    for step in range(args.max_steps):
        t0 = time.time()

        correction_agent.execute_step(obs, robot_client)

        if args.record:
            cur_w = obs.get("wrist_timestamp", 0.0)
            cur_b = obs.get("base_timestamp", 0.0)
            if (cur_w > 0 and cur_w > last_wrist_ts) or (cur_b > 0 and cur_b > last_base_ts):
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
                end="", flush=True,
            )

    if not correction_agent.stop_detected:
        print(f"\n[CORRECTION] Max steps reached without stop signal")
        save_and_go_home(
            robot_client, buffer, writer, rec_mgr,
            skill_name=args.skill, outcome="correction_timeout",
            grasp_info=grasp_info,
            model_type=args.model_type, prompt=args.prompt,
        )
        return

    # ---------------------------------------------------------------
    # Phase 4: Skill resume (absolute waypoints only)
    # ---------------------------------------------------------------
    robot_client.speed_stop()
    time.sleep(0.1)

    buffer.set_phase("skill_resume")
    if args.record:
        rec_mgr.start()

    print("[PIPELINE] Resuming skill (absolute waypoints only)...")
    completed, resume_info = skill_executor.execute(
        args.skill,
        interrupt_event=interrupt_event,
        resume_absolute_only=True,
    )
    grasp_info.update(resume_info)

    rec_mgr.stop()
    outcome = "completed_after_correction" if completed else "skill_resume_failed"
    print(f"[PIPELINE] Skill resume {'completed' if completed else 'failed'}.")
    save_and_go_home(
        robot_client, buffer, writer, rec_mgr,
        skill_name=args.skill, outcome=outcome,
        grasp_info=grasp_info,
        model_type=args.model_type, prompt=args.prompt,
    )


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
            if (cur_w > 0 and cur_w > last_wrist_ts) or (cur_b > 0 and cur_b > last_base_ts):
                frame = build_frame(obs, camera_clients, args.image_size)
                buffer.record_frame(frame)
                if cur_w > last_wrist_ts:
                    last_wrist_ts = cur_w
                if cur_b > last_base_ts:
                    last_base_ts = cur_b

        if agent.stop_detected:
            print(f"\n[E2E] Stop signal at step {step}")
            break

        obs = get_obs(robot_client, camera_clients)

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        if step > 0 and step % args.fps == 0:
            print(
                f"\r[E2E] step {step}/{args.max_steps} | "
                f"frames: {buffer.num_frames}     ",
                end="", flush=True,
            )

    outcome = "stop_signal" if agent.stop_detected else "timeout"
    print(f"\n[E2E] Episode complete ({step + 1} steps, {outcome})")
    save_and_go_home(
        robot_client, buffer, writer, rec_mgr,
        skill_name="e2e", outcome=outcome,
        model_type=args.model_type, prompt=args.prompt,
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
# Main
# ---------------------------------------------------------------------------


def main(args: Args):
    # Auto-detect prompt from skill if not explicitly provided
    if not args.prompt:
        args.prompt = TASK_INSTRUCTIONS.get(args.skill, args.skill)

    print("=" * 60)
    print("  VLA INFERENCE PIPELINE")
    print("=" * 60)
    print(f"  Model:    {args.model_type}")
    print(f"  Server:   {args.server_host}:{args.server_port}")
    print(f"  Mode:     {args.mode}")
    print(f"  Prompt:   {args.prompt}")
    print(f"  FPS:      {args.fps} Hz")
    print(f"  Max steps: {args.max_steps}")
    if args.mode == "planner":
        print(f"  Skill:    {args.skill}")
        if args.correction_server_port > 0:
            print(f"  Correction: {args.server_host}:{args.correction_server_port}")
        else:
            print("  Correction: none")
    print(f"  Robot:    {args.hostname}:{args.robot_port}")
    print(f"  Record:   {args.record} -> {args.data_dir}")
    print(f"  Safety:   {'disabled' if args.disable_safety else 'enabled'}")
    print("=" * 60)

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
    agent = VLAAgent(adapter, args.fps, args.prompt, safety)

    # ------------------------------------------------------------------
    # 5. Create correction agent (planner mode only)
    # ------------------------------------------------------------------
    correction_agent = None
    if args.mode == "planner" and args.correction_server_port > 0:
        print("[INIT] Creating correction model adapter...")
        correction_adapter = create_adapter(
            args,
            port_override=args.correction_server_port,
        )
        correction_agent = VLAAgent(
            correction_adapter, args.fps, args.prompt, safety,
        )
        print("[INIT] Correction adapter ready.")

    # ------------------------------------------------------------------
    # 6. Setup skill executor (planner mode only)
    # ------------------------------------------------------------------
    skill_executor = None
    if args.mode == "planner":
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
        obs_client, camera_clients, buffer, args.record_hz, args.image_size,
    )

    # ------------------------------------------------------------------
    # 8. Move to home position
    # ------------------------------------------------------------------
    print("\n[INIT] Moving to home position...")
    robot_client.move_joints(list(HOME_JOINTS_RAD), speed=0.5, accel=0.3)
    print("[INIT] Home reached. Starting inference...\n")

    # ------------------------------------------------------------------
    # 9. Run inference
    # ------------------------------------------------------------------
    try:
        if args.mode == "planner":
            run_planner_mode(
                args, agent, correction_agent,
                robot_client, obs_client, camera_clients,
                skill_executor, buffer, writer, rec_mgr,
            )
        elif args.mode == "e2e":
            run_e2e_mode(
                args, agent,
                robot_client, camera_clients,
                buffer, writer, rec_mgr,
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    except KeyboardInterrupt:
        print("\n\n[E-STOP] Ctrl+C — stopping robot...")
        try:
            robot_client.speed_stop()
        except Exception:
            pass
        rec_mgr.stop()

        if buffer.phase != "idle":
            print("[E-STOP] Saving in-progress recording...")
            metadata = buffer.get_metadata()
            frames, segments = buffer.export()
            episode_meta = {
                **metadata,
                "skill_name": args.skill if args.mode == "planner" else "e2e",
                "skill_outcome": "estop",
                "model_type": args.model_type,
                "prompt": args.prompt,
            }
            writer.save_unified_episode(frames, segments, metadata=episode_meta)

        print("[E-STOP] Moving to home position...")
        try:
            robot_client.set_gripper_speed(255)
            robot_client.move_joints(list(HOME_JOINTS_RAD), speed=0.5, accel=0.3)
        except Exception:
            pass

    print("Inference pipeline exited.")


if __name__ == "__main__":
    main(tyro.cli(Args))
