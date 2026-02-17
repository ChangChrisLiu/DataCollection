#!/usr/bin/env python3
# experiments/run_dual_dataset.py
"""VLA dual-dataset collection pipeline (T4).

Workflow per round:
  1. Press Start Recording (left btn 25) -> continuous recording begins
  2. Teleoperate with HOSAS joystick (data saved at ~30Hz)
  3. Press Skill button (right btn 15=CPU, 16=RAM) -> 3 stop-signal frames
     inserted, then skill executes with concurrent recording
  3b. [Optional] Press Interrupt (left btn 16) -> skill stops, manual teleop
      resumes. Press skill button again -> resumes absolute waypoints only.
  4. After skill completes, press Home (left btn 34) -> recording stops,
     two datasets saved, robot moves home

Outputs per round:
  vla_planner/  - teleop + 3 stop frames (gripper=255)  [high-level planner]
  vla_executor/ - teleop + skill execution frames        [full executor]

Terminal architecture:
  T1: python experiments/launch_nodes.py --robot ur
  T2: python experiments/launch_camera_nodes.py --camera-settings camera_settings.json
  T4: python experiments/run_dual_dataset.py
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import tyro

from gello.data_utils.dual_dataset_buffer import DualDatasetBuffer
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
    cpu_relative_count: int = 19
    ram_skill_csv: str = "RAM_Skills.csv"
    ram_relative_count: int = 14
    skill_move_speed: float = 0.1
    skill_move_accel: float = 0.04


def build_frame(
    obs: Dict[str, Any],
    camera_clients: Dict[str, ZMQClientCamera],
) -> Dict[str, Any]:
    """Build a data frame from robot observations and camera images.

    Gripper position is read from the robot (not the joystick command).
    """
    # Joint positions (6 arm joints only)
    joints_full = obs["joint_positions"]
    joints_6 = list(joints_full[:6])

    # TCP pose as [x,y,z,rx,ry,rz] (rotation vector format)
    ee = obs["ee_pos_quat"]  # [x,y,z,qx,qy,qz,qw]
    pos = ee[:3]
    rotvec = quat_to_rotvec(ee[3:7])
    tcp_pose = list(np.concatenate([pos, rotvec]))

    # Gripper: from robot observation, convert 0-1 -> 0-255
    gripper_norm = obs["gripper_position"][0]
    gripper_pos = int(round(gripper_norm * 255))

    frame = {
        "timestamp": time.time(),
        "joint_positions": joints_6,
        "tcp_pose": tcp_pose,
        "gripper_pos": gripper_pos,
    }

    # Camera images
    for name, cam in camera_clients.items():
        ts, rgb, depth = cam.read()
        frame[f"{name}_rgb"] = rgb
        frame[f"{name}_depth"] = depth

    return frame


def build_frame_from_obs_client(
    obs_client: ZMQClientRobot,
    camera_clients: Dict[str, ZMQClientCamera],
) -> Dict[str, Any]:
    """Build a data frame using the read-only obs client (for skill recording)."""
    robot_obs = obs_client.get_observations()
    return build_frame(robot_obs, camera_clients)


def record_skill_thread(
    obs_client: ZMQClientRobot,
    camera_clients: Dict[str, ZMQClientCamera],
    buffer: DualDatasetBuffer,
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
            buffer.record_skill_frame(frame)
            count += 1
        except Exception as e:
            print(f"[RecordThread] Error: {e}")

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print(f"[RecordThread] Stopped after {count} frames.")


def main(args: Args):
    print("=" * 60)
    print("  VLA DUAL-DATASET COLLECTION PIPELINE")
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
    buffer = DualDatasetBuffer()
    writer = DatasetWriter(data_dir=args.data_dir)

    # ------------------------------------------------------------------
    # 5. Interrupt/resume state
    # ------------------------------------------------------------------
    skill_interrupted = False
    interrupted_skill_name = None
    # Recording thread handles (kept alive across interrupt/resume)
    skill_stop_event = None
    skill_rec_thread = None

    def _stop_recording_thread():
        """Stop the skill recording thread if it's running."""
        nonlocal skill_stop_event, skill_rec_thread
        if skill_stop_event is not None:
            skill_stop_event.set()
        if skill_rec_thread is not None:
            skill_rec_thread.join(timeout=2.0)
        skill_stop_event = None
        skill_rec_thread = None

    def _start_recording_thread():
        """Start a new skill recording thread."""
        nonlocal skill_stop_event, skill_rec_thread
        skill_stop_event = threading.Event()
        skill_rec_thread = threading.Thread(
            target=record_skill_thread,
            args=(obs_client, camera_clients, buffer, skill_stop_event, args.hz),
            daemon=True,
        )
        skill_rec_thread.start()

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

    # Track last saved camera timestamps to only record new frames
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
                        print("\n[PIPELINE] Recording started!")
                    else:
                        print("[PIPELINE] Already recording.")
                    obs = env.get_obs()
                    continue

                # --- HOME (stop recording + move home) ---
                elif signal == "home":
                    # Clean up interrupted state if active
                    if skill_interrupted:
                        _stop_recording_thread()
                        buffer.finish_skill()
                        skill_interrupted = False
                        interrupted_skill_name = None

                    if buffer.phase in ("teleop", "skill", "post_skill"):
                        # Stop and save (get metadata BEFORE export resets)
                        metadata = buffer.get_metadata()
                        ds_planner, ds_executor = buffer.export()
                        episode_dir = writer.save_dual_episode(
                            ds_planner, ds_executor, metadata=metadata
                        )
                        writer.prompt_quality(episode_dir)
                        print("[PIPELINE] Recording saved.")

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

                # --- SKILL TRIGGER (e.g. "cpu", "ram") ---
                elif skill_executor.has_skill(signal):

                    # ====================================================
                    # RESUME after interrupt: skip relative, absolute only
                    # ====================================================
                    if skill_interrupted:
                        print(
                            f"\n[PIPELINE] Resuming interrupted skill "
                            f"'{interrupted_skill_name}' (absolute waypoints only)"
                        )

                        # Stop robot motion from manual teleop
                        robot_client.speed_stop()
                        time.sleep(0.1)

                        # Start recording thread for the resume phase
                        _start_recording_thread()

                        # Execute absolute waypoints only (blocking)
                        completed = skill_executor.execute(
                            interrupted_skill_name,
                            interrupt_event=agent.interrupt_event,
                            resume_absolute_only=True,
                        )

                        if completed:
                            # Skill finished successfully
                            _stop_recording_thread()
                            buffer.finish_skill()
                            skill_interrupted = False
                            interrupted_skill_name = None
                            print(
                                "[PIPELINE] Skill resumed and completed. "
                                "Press Home (btn 34) to save."
                            )
                        else:
                            # Interrupted again during resume
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

                    # Stop robot motion
                    robot_client.speed_stop()
                    time.sleep(0.1)

                    # Insert 3 stop-signal frames (same pose, gripper=255)
                    buffer.insert_stop_signal(num_repeats=3)

                    # Capture trigger TCP for skill transform
                    trigger_tcp = robot_client.get_tcp_pose_raw()

                    # Start concurrent recording thread
                    _start_recording_thread()

                    # Execute skill (blocking, with interrupt support)
                    print(f"[PIPELINE] Executing skill '{signal}'...")
                    completed = skill_executor.execute(
                        signal,
                        trigger_tcp_raw=trigger_tcp,
                        interrupt_event=agent.interrupt_event,
                    )

                    if completed:
                        # Normal completion
                        _stop_recording_thread()
                        buffer.finish_skill()
                        print(
                            f"[PIPELINE] Skill '{signal}' complete. "
                            f"Press Home (btn 34) to save and go home."
                        )
                    else:
                        # Interrupted — stop recording, return to manual teleop
                        _stop_recording_thread()
                        agent.interrupt_event.clear()
                        skill_interrupted = True
                        interrupted_skill_name = signal
                        print(
                            f"\n[PIPELINE] Skill '{signal}' INTERRUPTED. "
                            f"Manual teleop active."
                        )
                        print(
                            "[PIPELINE] Correct position manually, then press "
                            "skill button to resume (absolute only)."
                        )
                        print(
                            "[PIPELINE] Or press Home (btn 34) to abandon and save."
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
            # Record teleop frames OR skill frames during interrupt
            if buffer.phase == "teleop" or skill_interrupted:
                current_wrist_ts = obs.get("wrist_timestamp", 0.0)
                current_base_ts = obs.get("base_timestamp", 0.0)
                has_new = (
                    (current_wrist_ts > 0 and current_wrist_ts > last_wrist_ts)
                    or (current_base_ts > 0 and current_base_ts > last_base_ts)
                )

                if has_new:
                    frame = build_frame(obs, camera_clients)
                    if skill_interrupted:
                        # During interrupt: manual corrections go into skill frames
                        buffer.record_skill_frame(frame)
                    else:
                        buffer.record_teleop_frame(frame)
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
                if skill_interrupted:
                    phase_label = "INTERRUPTED"
                else:
                    phase_label = buffer.phase.upper()
                n_teleop = buffer.num_teleop_frames
                n_skill = buffer.num_skill_frames
                msg = (
                    f"\r[{phase_label}] T:{elapsed:.0f}s | Hz:{hz:.0f} | "
                    f"Teleop:{n_teleop} Skill:{n_skill}        "
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

    if skill_interrupted:
        buffer.finish_skill()

    if buffer.phase != "idle":
        print("[PIPELINE] Saving in-progress recording...")
        metadata = buffer.get_metadata()
        ds_planner, ds_executor = buffer.export()
        writer.save_dual_episode(ds_planner, ds_executor, metadata=metadata)

    agent.close()
    print("Pipeline exited.")


if __name__ == "__main__":
    main(tyro.cli(Args))
