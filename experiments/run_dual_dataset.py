#!/usr/bin/env python3
# experiments/run_dual_dataset.py
"""VLA unified data collection pipeline (T4).

Records a single frame stream with 4 phase labels:
  teleop, skill, correction, skill_resume

Workflow per round:
  1. Press Start Recording (left btn 25) -> continuous recording begins
  2. Teleoperate with HOSAS joystick (phase: teleop, ~30Hz)
  3. Press Skill button (right btn 15=CPU, 16=RAM) -> phase: skill
     Skill executes with concurrent recording.
     a. If grasp verification fails -> recording paused, phase: correction
        Provide manual correction with joystick, recording resumes on input.
     b. Press skill button again -> phase: skill_resume (absolute WPs only)
  4. After skill completes, press Home (left btn 34) -> save episode + go home

Output per round:
  episode_MMDD_HHMMSS/
    frame_0000.pkl  (phase: teleop)
    ...
    frame_NNNN.pkl  (phase: skill_resume)
    episode_meta.json

Terminal architecture:
  T1: python experiments/launch_nodes.py --robot ur
  T2: python experiments/launch_camera_nodes.py --camera-settings camera_settings.json
  T4: python experiments/run_dual_dataset.py
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import tyro

from gello.data_utils.episode_buffer import EpisodeBuffer
from gello.data_utils.dataset_writer import DatasetWriter
from gello.env import RobotEnv
from gello.skills.csv_skill_executor import CSVSkillExecutor
from gello.utils.transform_utils import quat_to_rotvec
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot


@dataclass
class Args:
    robot_port: int = 6001
    obs_port: int = 6002
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    hz: int = 30
    data_dir: str = "data/vla_dataset"
    verbose: bool = False
    no_cameras: bool = False

    # Skill configuration
    cpu_skill_csv: str = "CPU_Skills.csv"
    cpu_relative_count: int = 20
    ram_skill_csv: str = "RAM_Skills.csv"
    ram_relative_count: int = 15
    skill_move_speed: float = 0.1
    skill_move_accel: float = 0.04


def build_frame(
    obs: Dict[str, Any],
    camera_clients: Dict[str, ZMQClientCamera],
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
            frame[f"{name}_rgb"] = obs[f"{name}_rgb"]
            frame[f"{name}_depth"] = obs[f"{name}_depth"]
            frame[f"{name}_timestamp"] = obs.get(f"{name}_timestamp", frame_time)
        else:
            ts, rgb, depth = cam.read()
            frame[f"{name}_rgb"] = rgb
            frame[f"{name}_depth"] = depth
            frame[f"{name}_timestamp"] = ts

    return frame


def build_frame_from_obs_client(
    obs_client: ZMQClientRobot,
    camera_clients: Dict[str, ZMQClientCamera],
) -> Dict[str, Any]:
    """Build a data frame using the read-only obs client (for skill recording)."""
    robot_obs = obs_client.get_observations()
    return build_frame(robot_obs, camera_clients, use_obs_cameras=False)


def record_skill_thread(
    obs_client: ZMQClientRobot,
    camera_clients: Dict[str, ZMQClientCamera],
    buffer: EpisodeBuffer,
    stop_event: threading.Event,
    hz: int = 30,
):
    """Background thread: records frames at ~30Hz during skill execution."""
    dt = 1.0 / hz
    count = 0
    while not stop_event.is_set():
        t0 = time.time()
        try:
            frame = build_frame_from_obs_client(obs_client, camera_clients)
            buffer.record_frame(frame)
            count += 1
        except Exception as e:
            print(f"[RecordThread] Error: {e}")

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print(f"[RecordThread] Stopped after {count} frames.")


def main(args: Args):
    print("=" * 60)
    print("  VLA UNIFIED DATA COLLECTION PIPELINE")
    print("=" * 60)
    print(f"  Robot:    {args.hostname}:{args.robot_port}")
    print(f"  Obs:      {args.hostname}:{args.obs_port}")
    print(f"  Rate:     {args.hz} Hz")
    print(f"  Data dir: {args.data_dir}")
    print(f"  CPU skill: {args.cpu_skill_csv} ({args.cpu_relative_count} rel)")
    print(f"  RAM skill: {args.ram_skill_csv} ({args.ram_relative_count} rel)")
    print("=" * 60)
    print(
        "  NOTE: Launch cameras with --camera-settings to lock exposure,\n"
        "        white balance, and gain for consistent data collection."
    )
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Connect to robot + cameras
    # ------------------------------------------------------------------
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    obs_client = ZMQClientRobot(port=args.obs_port, host=args.hostname)

    if args.no_cameras:
        camera_clients = {}
    else:
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

    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
    )

    # ------------------------------------------------------------------
    # 2. Create agent
    # ------------------------------------------------------------------
    from gello.agents.joystick_agent import JoystickAgent

    agent = JoystickAgent(
        robot_type="ur5",
        num_dofs=7,
        verbose=args.verbose,
    )

    # ------------------------------------------------------------------
    # 3. Setup skill executor
    # ------------------------------------------------------------------
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
        move_speed=args.skill_move_speed,
        move_accel=args.skill_move_accel,
    )

    # ------------------------------------------------------------------
    # 4. Setup buffer + writer
    # ------------------------------------------------------------------
    buffer = EpisodeBuffer()
    writer = DatasetWriter(data_dir=args.data_dir)

    # ------------------------------------------------------------------
    # 5. State tracking
    # ------------------------------------------------------------------
    skill_interrupted = False
    interrupted_skill_name: Optional[str] = None
    grasp_failed = False  # True when waiting for correction
    episode_grasp_info: Dict[str, Any] = {}
    active_skill_name: Optional[str] = None

    # Recording thread handles
    skill_stop_event: Optional[threading.Event] = None
    skill_rec_thread: Optional[threading.Thread] = None

    def _stop_recording_thread():
        nonlocal skill_stop_event, skill_rec_thread
        if skill_stop_event is not None:
            skill_stop_event.set()
        if skill_rec_thread is not None:
            skill_rec_thread.join(timeout=2.0)
        skill_stop_event = None
        skill_rec_thread = None

    def _start_recording_thread():
        nonlocal skill_stop_event, skill_rec_thread
        skill_stop_event = threading.Event()
        skill_rec_thread = threading.Thread(
            target=record_skill_thread,
            args=(obs_client, camera_clients, buffer, skill_stop_event, args.hz),
            daemon=True,
        )
        skill_rec_thread.start()

    def _on_grasp_failed():
        """Callback from executor when grasp verification fails."""
        nonlocal grasp_failed
        grasp_failed = True
        # Stop recording thread — recording paused until joystick input
        _stop_recording_thread()
        print(
            "\n[PIPELINE] *** GRASP FAILED — provide correction with joystick ***"
        )

    # ------------------------------------------------------------------
    # 6. Main loop
    # ------------------------------------------------------------------
    print("\n--- Controls ---")
    print("  Left Btn 25:  Start recording")
    print("  Left Btn 16:  Interrupt skill (manual correction)")
    print("  Left Btn 34:  Home + stop recording + save")
    print("  Left Btn 38:  Vertical reorient")
    print("  Right Btn 15: Trigger CPU skill (or resume after interrupt)")
    print("  Right Btn 16: Trigger RAM skill (or resume after interrupt)")
    print("  Ctrl+C:       Exit")
    print("----------------\n")

    obs = env.get_obs()
    loop_count = 0
    t_start = time.time()
    t_fps = time.time()

    last_wrist_ts = 0.0
    last_base_ts = 0.0

    try:
        while True:
            action = agent.act(obs)

            # ----------------------------------------------------------
            # Handle signals/skills
            # ----------------------------------------------------------
            if isinstance(action, dict) and action.get("type") == "skill":
                signal = action["skill"]

                # --- START RECORDING ---
                if signal == "start_recording":
                    if buffer.phase == "idle":
                        buffer.start()
                        last_wrist_ts = 0.0
                        last_base_ts = 0.0
                        episode_grasp_info = {}
                        active_skill_name = None
                        print("\n[PIPELINE] Recording started!")
                    else:
                        print("[PIPELINE] Already recording.")
                    obs = env.get_obs()
                    continue

                # --- HOME (stop recording + move home) ---
                elif signal == "home":
                    # Clean up any in-progress state
                    _stop_recording_thread()

                    if buffer.phase != "idle":
                        # Determine skill outcome
                        if active_skill_name:
                            has_correction = any(
                                f.get("phase") == "correction"
                                for f in buffer._frames
                            )
                            outcome = (
                                "completed_after_correction"
                                if has_correction
                                else "completed"
                            )
                        else:
                            outcome = "no_skill"

                        metadata = buffer.get_metadata()
                        frames, segments = buffer.export()

                        # Build episode metadata
                        episode_meta = {
                            **metadata,
                            "skill_name": active_skill_name or "",
                            "skill_outcome": outcome,
                            **episode_grasp_info,
                        }
                        episode_dir = writer.save_unified_episode(
                            frames, segments, metadata=episode_meta
                        )
                        writer.prompt_quality(episode_dir)
                        print("[PIPELINE] Recording saved.")

                    # Reset state
                    skill_interrupted = False
                    interrupted_skill_name = None
                    grasp_failed = False
                    episode_grasp_info = {}
                    active_skill_name = None

                    # Move home
                    print("[PIPELINE] Moving to home position...")
                    robot_client.speed_stop()
                    env._handle_skill("home")
                    obs = env.get_obs()
                    continue

                # --- REORIENT ---
                elif signal == "reorient":
                    robot_client.speed_stop()
                    env._handle_skill("reorient")
                    obs = env.get_obs()
                    continue

                # --- SKILL TRIGGER ---
                elif skill_executor.has_skill(signal):

                    # ====================================================
                    # RESUME after interrupt/grasp failure
                    # ====================================================
                    if skill_interrupted:
                        print(
                            f"\n[PIPELINE] Resuming interrupted skill "
                            f"'{interrupted_skill_name}' (absolute waypoints only)"
                        )

                        robot_client.speed_stop()
                        time.sleep(0.1)

                        # Phase: skill_resume
                        buffer.set_phase("skill_resume")
                        _start_recording_thread()

                        completed, grasp_info = skill_executor.execute(
                            interrupted_skill_name,
                            interrupt_event=agent.interrupt_event,
                            resume_absolute_only=True,
                        )

                        if completed:
                            _stop_recording_thread()
                            skill_interrupted = False
                            interrupted_skill_name = None
                            grasp_failed = False
                            print(
                                "[PIPELINE] Skill resumed and completed. "
                                "Press Home (btn 34) to save."
                            )
                        else:
                            _stop_recording_thread()
                            agent.interrupt_event.clear()
                            print(
                                "[PIPELINE] Skill interrupted again. "
                                "Continue manual teleop, press skill to retry."
                            )

                        obs = env.get_obs()
                        continue

                    # ====================================================
                    # NORMAL first-time skill trigger
                    # ====================================================
                    if buffer.phase != "teleop":
                        print(
                            f"[PIPELINE] Ignoring skill '{signal}' "
                            f"(phase={buffer.phase}, need 'teleop')"
                        )
                        obs = env.get_obs()
                        continue

                    print(f"\n[PIPELINE] Skill '{signal}' triggered!")
                    active_skill_name = signal

                    # Stop robot motion
                    robot_client.speed_stop()
                    time.sleep(0.1)

                    # Phase: skill
                    buffer.set_phase("skill")

                    # Capture trigger TCP for skill transform
                    trigger_tcp = robot_client.get_tcp_pose_raw()

                    # Start concurrent recording thread
                    _start_recording_thread()

                    # Execute skill (blocking, with interrupt + grasp verification)
                    print(f"[PIPELINE] Executing skill '{signal}'...")
                    completed, grasp_info = skill_executor.execute(
                        signal,
                        trigger_tcp_raw=trigger_tcp,
                        interrupt_event=agent.interrupt_event,
                        on_grasp_failed=_on_grasp_failed,
                    )
                    episode_grasp_info = grasp_info

                    if completed:
                        _stop_recording_thread()
                        print(
                            f"[PIPELINE] Skill '{signal}' complete. "
                            f"Press Home (btn 34) to save and go home."
                        )
                    elif grasp_failed:
                        # Grasp verification failed -> wait for correction
                        # Recording thread already stopped by _on_grasp_failed
                        skill_interrupted = True
                        interrupted_skill_name = signal
                        agent.interrupt_event.clear()
                        print(
                            "[PIPELINE] Waiting for joystick input to begin "
                            "correction recording..."
                        )
                    else:
                        # Manual interrupt (left btn 16)
                        _stop_recording_thread()
                        agent.interrupt_event.clear()
                        skill_interrupted = True
                        interrupted_skill_name = signal
                        buffer.set_phase("correction")
                        print(
                            f"\n[PIPELINE] Skill '{signal}' INTERRUPTED. "
                            f"Manual teleop active (phase: correction)."
                        )
                        print(
                            "[PIPELINE] Correct position manually, then press "
                            "skill button to resume (absolute only)."
                        )

                    obs = env.get_obs()
                    continue

                else:
                    print(f"[PIPELINE] Unknown signal: {signal}")
                    obs = env.get_obs()
                    continue

            # ----------------------------------------------------------
            # Normal velocity control + recording
            # ----------------------------------------------------------
            # Check if joystick has input (for correction detection)
            has_joystick_input = False
            if isinstance(action, np.ndarray):
                vel = action[:6] if len(action) > 6 else action
                has_joystick_input = np.max(np.abs(vel)) > 0.001

            # Handle grasp failure -> correction phase transition
            if grasp_failed and has_joystick_input:
                # Joystick input detected — begin correction recording
                grasp_failed = False
                buffer.set_phase("correction")
                print("\n[PIPELINE] Correction recording started!")

            # Record frames during teleop or correction
            if buffer.phase in ("teleop", "correction"):
                current_wrist_ts = obs.get("wrist_timestamp", 0.0)
                current_base_ts = obs.get("base_timestamp", 0.0)
                has_new = (
                    (current_wrist_ts > 0 and current_wrist_ts > last_wrist_ts)
                    or (current_base_ts > 0 and current_base_ts > last_base_ts)
                )

                if has_new:
                    frame = build_frame(obs, camera_clients)
                    buffer.record_frame(frame)
                    if current_wrist_ts > last_wrist_ts:
                        last_wrist_ts = current_wrist_ts
                    if current_base_ts > last_base_ts:
                        last_base_ts = current_base_ts

            # Step the environment (sends velocity to robot)
            obs = env.step(action)

            # ----------------------------------------------------------
            # Status display (~1Hz)
            # ----------------------------------------------------------
            loop_count += 1
            now = time.time()
            if now - t_fps >= 1.0:
                hz = loop_count / (now - t_fps)
                elapsed = now - t_start
                if grasp_failed:
                    phase_label = "GRASP_FAILED"
                elif skill_interrupted:
                    phase_label = "CORRECTION"
                else:
                    phase_label = buffer.phase.upper()
                n_frames = buffer.num_frames
                msg = (
                    f"\r[{phase_label}] T:{elapsed:.0f}s | Hz:{hz:.0f} | "
                    f"Frames:{n_frames}        "
                )
                print(msg, end="", flush=True)
                loop_count = 0
                t_fps = now

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...")

    # Cleanup
    try:
        robot_client.speed_stop()
    except Exception:
        pass

    _stop_recording_thread()

    if buffer.phase != "idle":
        print("[PIPELINE] Saving in-progress recording...")
        metadata = buffer.get_metadata()
        frames, segments = buffer.export()
        episode_meta = {
            **metadata,
            "skill_name": active_skill_name or "",
            "skill_outcome": "incomplete",
            **episode_grasp_info,
        }
        writer.save_unified_episode(frames, segments, metadata=episode_meta)

    agent.close()
    print("Pipeline exited.")


if __name__ == "__main__":
    main(tyro.cli(Args))
