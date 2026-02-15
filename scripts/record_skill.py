#!/usr/bin/env python3
# scripts/record_skill.py
"""Record a skill trajectory via UR freedrive (teach) mode.

Usage:
    # With real robot (requires T1: launch_nodes.py running):
    python scripts/record_skill.py --name my_grasp --robot-port 6001

    # Mock mode (no robot needed, generates dummy trajectory):
    python scripts/record_skill.py --name test_skill --mock

The recorded skill is saved as relative (gripper-local) waypoints that
can be replayed from any starting TCP pose via the SkillExecutor.

Output format (skills/<name>.pkl):
    {
        "waypoints": [4x4 numpy arrays],   # relative SE(3) transforms
        "gripper_positions": [floats],      # gripper position at each frame
        "fps": 30,                          # recording rate
        "absolute_poses": [6D numpy arrays], # raw recorded poses (for debug)
    }
"""

import argparse
import os
import pickle
import time

import numpy as np

from gello.utils.transform_utils import compute_relative_waypoints


def record_with_robot(robot_client, fps: int = 30) -> dict:
    """Record a skill trajectory using freedrive mode.

    Args:
        robot_client: ZMQ client connected to robot server.
        fps: Recording frame rate (Hz).

    Returns:
        Dict with "absolute_poses" and "gripper_positions".
    """
    print("\n--- Skill Recording (Freedrive Mode) ---")
    print("The robot will enter freedrive mode.")
    print("Physically guide the robot through the desired motion.")
    print()

    # Enable freedrive
    robot_client.set_freedrive_mode(True)
    print("[FREEDRIVE ON] Robot is now free to move.")
    input("Press ENTER to start recording...")

    poses = []
    gripper_positions = []
    dt = 1.0 / fps

    print(f"Recording at {fps} Hz. Press ENTER to stop...")

    # Non-blocking input check via threading
    import threading

    stop_flag = threading.Event()

    def _wait_for_enter():
        input()
        stop_flag.set()

    t = threading.Thread(target=_wait_for_enter, daemon=True)
    t.start()

    frame_count = 0
    start_time = time.time()

    while not stop_flag.is_set():
        loop_start = time.time()

        tcp_pose = robot_client.get_tcp_pose_raw()  # [x,y,z,rx,ry,rz]
        joints = robot_client.get_joint_state()
        grip = float(joints[-1]) if len(joints) > 6 else 0.0

        poses.append(np.array(tcp_pose))
        gripper_positions.append(grip)
        frame_count += 1

        if frame_count % fps == 0:
            elapsed = time.time() - start_time
            print(
                f"\r  Recorded {frame_count} frames "
                f"({elapsed:.1f}s, {frame_count / elapsed:.1f} Hz)",
                end="",
                flush=True,
            )

        # Rate limiting
        elapsed_loop = time.time() - loop_start
        sleep_time = dt - elapsed_loop
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Disable freedrive
    robot_client.set_freedrive_mode(False)
    print(f"\n[FREEDRIVE OFF] Recorded {frame_count} frames.")

    return {
        "absolute_poses": poses,
        "gripper_positions": gripper_positions,
    }


def record_mock(fps: int = 30, duration: float = 3.0) -> dict:
    """Generate a mock skill trajectory for testing.

    Creates a simple linear motion along X with no rotation.
    """
    print(f"\n[MOCK] Generating {duration}s dummy trajectory at {fps} Hz...")
    n_frames = int(fps * duration)
    poses = []
    gripper_positions = []

    for i in range(n_frames):
        t = i / n_frames
        # Simple linear motion: 0.1m along X
        pose = np.array([0.3 + 0.1 * t, 0.0, 0.3, 0.0, 3.14, 0.0])
        poses.append(pose)
        gripper_positions.append(0.5)  # Half-open

    print(f"[MOCK] Generated {n_frames} frames.")
    return {
        "absolute_poses": poses,
        "gripper_positions": gripper_positions,
    }


def main():
    parser = argparse.ArgumentParser(description="Record a skill trajectory")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Skill name (saved to skills/<name>.pkl)",
    )
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--hostname", type=str, default="127.0.0.1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate dummy trajectory without robot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="skills",
        help="Output directory for skill files",
    )
    args = parser.parse_args()

    # Record
    if args.mock:
        raw_data = record_mock(fps=args.fps)
    else:
        from gello.zmq_core.robot_node import ZMQClientRobot

        print(f"Connecting to robot at {args.hostname}:{args.robot_port}...")
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        raw_data = record_with_robot(robot_client, fps=args.fps)

    # Convert to relative waypoints
    absolute_poses = raw_data["absolute_poses"]
    gripper_positions = raw_data["gripper_positions"]

    if len(absolute_poses) < 2:
        print("ERROR: Need at least 2 frames for a skill. Aborting.")
        return

    relative_waypoints = compute_relative_waypoints(absolute_poses)
    print(
        f"Converted {len(absolute_poses)} absolute poses -> "
        f"{len(relative_waypoints)} relative waypoints"
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.name}.pkl")

    skill_data = {
        "waypoints": relative_waypoints,
        "gripper_positions": gripper_positions[1:],  # Align with relative
        "fps": args.fps,
        "absolute_poses": absolute_poses,  # For debugging
    }

    with open(output_path, "wb") as f:
        pickle.dump(skill_data, f)

    print(f"\nSkill saved to: {output_path}")
    print(f"  Waypoints: {len(relative_waypoints)}")
    print(f"  Gripper frames: {len(skill_data['gripper_positions'])}")
    print(f"  FPS: {args.fps}")


if __name__ == "__main__":
    main()
