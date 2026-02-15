#!/usr/bin/env python3
# experiments/run_dual_dataset.py
"""Main orchestrator (T4) for the dual-dataset collection pipeline.

Implements the 4-phase pipeline:
  Phase 1 (Teleop):  HOSAS joystick velocity control at 30Hz
  Phase 2 (Trigger): Left Trigger -> stop motion, capture TCP pose
  Phase 3 (Skill):   SkillExecutor replays moveL trajectory
  Phase 4 (Done):    Export two datasets, prompt quality label

Terminal architecture:
  T1: python experiments/launch_nodes.py --robot ur (ports 6001 + 6002)
  T2: python experiments/launch_camera_nodes.py (ports 5000 + 5001)
  T4: python experiments/run_dual_dataset.py --skill-path skills/my_skill.pkl
"""

import time
from dataclasses import dataclass

import tyro

from gello.data_utils.dual_dataset_buffer import DualDatasetBuffer
from gello.data_utils.dataset_writer import DatasetWriter
from gello.data_utils.keyboard_interface import KBReset
from gello.env import RobotEnv
from gello.skills.skill_executor import SkillExecutor
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot


@dataclass
class DualDatasetArgs:
    robot_port: int = 6001
    obs_port: int = 6002  # Read-only server for obs during skill
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    hz: int = 30  # 30Hz to match camera rate
    skill_path: str = "skills/default_skill.pkl"
    data_dir: str = "data/dual_dataset"
    verbose: bool = False
    no_cameras: bool = False
    go_home_after_skill: bool = True
    move_speed: float = 0.1  # Skill moveL speed (m/s)
    move_accel: float = 0.5  # Skill moveL acceleration (m/s^2)


def main(args: DualDatasetArgs):
    print("=" * 60)
    print("  DUAL-DATASET COLLECTION PIPELINE")
    print("=" * 60)
    print(f"  Robot control port:  {args.robot_port}")
    print(f"  Obs polling port:    {args.obs_port}")
    print(f"  Control rate:        {args.hz} Hz")
    print(f"  Skill path:          {args.skill_path}")
    print(f"  Data directory:      {args.data_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Connect to robot (two clients: control + observation)
    # ------------------------------------------------------------------
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    obs_client = ZMQClientRobot(port=args.obs_port, host=args.hostname)

    # Camera clients
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
    # 2. Create agent (joystick HOSAS, runs in-process)
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
    skill_executor = SkillExecutor(
        skill_path=args.skill_path,
        robot_client=robot_client,
        obs_client=obs_client,
        move_speed=args.move_speed,
        move_accel=args.move_accel,
    )

    # ------------------------------------------------------------------
    # 4. Setup buffer and writer
    # ------------------------------------------------------------------
    buffer = DualDatasetBuffer()
    writer = DatasetWriter(data_dir=args.data_dir)
    kb = KBReset()

    # ------------------------------------------------------------------
    # 5. Main loop
    # ------------------------------------------------------------------
    print("\n--- Controls ---")
    print("  S: Start recording episode")
    print("  Q: Cancel/stop current episode")
    print("  Left Trigger (Button 0): Trigger skill execution")
    print("  Ctrl+C: Exit")
    print("----------------\n")

    recording = False
    start_time = time.time()
    loop_count = 0
    last_print_time = time.time()

    try:
        obs = env.get_obs()
    except Exception as e:
        print(f"Failed to get initial observation: {e}")
        return

    try:
        while True:
            # Check keyboard
            kb_state = kb.update()

            if kb_state == "start" and not recording:
                # Begin new episode
                buffer.start_teleop()
                recording = True
                print("\n[RECORDING] Episode started. Teleoperating...")

            elif kb_state == "normal" and recording:
                # Q pressed: cancel episode
                print("\n[CANCELLED] Episode cancelled.")
                recording = False
                buffer._phase = "idle"
                continue

            # Main teleop loop
            if recording and buffer.phase == "teleop":
                action = agent.act(obs)

                # Check for skill trigger
                if (
                    isinstance(action, dict)
                    and action.get("type") == "skill"
                    and action.get("skill") == "trigger"
                ):
                    # Phase 2: Trigger!
                    print("\n[TRIGGER] Skill trigger detected!")
                    robot_client.speed_stop()
                    time.sleep(0.1)  # Let robot decelerate

                    # Capture TCP pose at trigger moment
                    tcp_pose = robot_client.get_tcp_pose_raw()
                    buffer.trigger_skill(tcp_pose)

                    # Phase 3: Execute skill
                    print("[SKILL] Executing skill trajectory...")
                    for skill_obs, target_pose in skill_executor.execute(tcp_pose):
                        # Poll cameras too if obs_client provides robot-only data
                        if not args.no_cameras:
                            full_obs = env.get_obs()
                            if skill_obs is not None:
                                # Merge camera obs with robot obs from obs_client
                                for k, v in skill_obs.items():
                                    full_obs[k] = v
                            buffer.record_skill_frame(full_obs, target_pose)
                        else:
                            buffer.record_skill_frame(skill_obs, target_pose)

                    # Phase 4: Done
                    if args.go_home_after_skill:
                        print("[HOME] Moving to home position...")
                        env._handle_skill("home")

                    buffer.finish()

                    # Export datasets
                    metadata = buffer.get_episode_metadata()
                    ds1, ds2 = buffer.export_datasets()
                    episode_dir = writer.save_dual_episode(ds1, ds2, metadata=metadata)
                    writer.prompt_quality(episode_dir)

                    recording = False
                    obs = env.get_obs()
                    continue

                else:
                    # Normal teleop: record and step
                    buffer.record_teleop_frame(obs, action)
                    obs = env.step(action)

            else:
                # Not recording: still run control loop for live teleoperation
                action = agent.act(obs)

                # Handle non-trigger skills (home, reorient)
                if (
                    isinstance(action, dict)
                    and action.get("type") == "skill"
                    and action.get("skill") != "trigger"
                ):
                    env._handle_skill(action["skill"])
                    obs = env.get_obs()
                else:
                    obs = env.step(action)

            # Print Hz every second
            loop_count += 1
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                elapsed = current_time - start_time
                hz = loop_count / (current_time - last_print_time)
                status = "REC" if recording else "IDLE"
                frames = buffer.num_teleop_frames if recording else 0
                msg = (
                    f"\r[{status}] T:{elapsed:.1f}s | "
                    f"Hz:{hz:.1f} | Frames:{frames}        "
                )
                print(msg, end="", flush=True)
                last_print_time = current_time
                loop_count = 0

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...")

    # Cleanup
    try:
        robot_client.speed_stop()
    except Exception:
        pass

    agent.close()
    print("Dual-dataset pipeline exited.")


if __name__ == "__main__":
    main(tyro.cli(DualDatasetArgs))
