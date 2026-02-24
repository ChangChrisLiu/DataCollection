#!/usr/bin/env python3
"""Offline validation of OpenPI inference pipeline using collected episodes.

Loads real frames from collected .pkl episodes, packs them exactly like
run_inference.py does, sends to a running OpenPI server, and checks
that returned actions are sane (shape, range, stop signals near end).

Usage:
    1. Start the OpenPI planner server (--port must come BEFORE policy:checkpoint):
       cd ~/openpi && uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
           --policy.config pi05_droid_ur5e_planner_lora_10hz \
           --policy.dir checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/49999

    2. Run this script (from tele conda env):
       python scripts/validate_openpi_inference.py

    3. Optional flags:
       --host 127.0.0.1 --port 8000
       --cpu-episode data/vla_dataset/CPU_Extraction/episode_cpu_0218_140015
       --ram-episode data/vla_dataset/RAM_Extraction/episode_ram_0218_172359
       --frames-per-episode 10   (how many frames to test per episode)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Task prompts must match training data (same as run_inference.py)
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


def load_frame(path: str) -> dict:
    """Load a single .pkl frame."""
    with open(path, "rb") as f:
        return pickle.load(f)


def frame_to_obs(frame: dict) -> dict:
    """Convert a .pkl frame to the observation dict that get_obs() returns.

    The pkl frame has:
        joint_positions: list of 6 floats
        gripper_pos: int 0-255
        base_rgb: (256, 256, 3) uint8
        wrist_rgb: (256, 256, 3) uint8

    The adapter expects (via get_obs → robot.get_observations):
        joint_positions: np.array of 7 (6 joints + gripper normalized)
        gripper_position: np.array([gripper / 255.0])
        base_rgb: (256, 256, 3) uint8
        wrist_rgb: (256, 256, 3) uint8
    """
    joints_6 = np.array(frame["joint_positions"][:6], dtype=np.float64)
    gripper_norm = frame["gripper_pos"] / 255.0
    joints_7 = np.append(joints_6, gripper_norm)

    return {
        "joint_positions": joints_7,
        "gripper_position": np.array([gripper_norm]),
        "base_rgb": frame["base_rgb"],
        "wrist_rgb": frame["wrist_rgb"],
    }


def find_default_episode(data_dir: str, task: str) -> str:
    """Find the first episode directory for a task."""
    subdir = "CPU_Extraction" if task == "cpu" else "RAM_Extraction"
    base = Path(data_dir) / subdir
    if not base.exists():
        return ""
    episodes = sorted([d for d in base.iterdir() if d.is_dir()])
    return str(episodes[0]) if episodes else ""


def get_frame_paths(episode_dir: str) -> list:
    """Get sorted list of frame .pkl paths in an episode."""
    ep = Path(episode_dir)
    return sorted(ep.glob("frame_*.pkl"))


def validate_episode(policy, episode_dir: str, task: str, n_frames: int):
    """Send frames from one episode to the server and validate responses."""
    prompt = TASK_INSTRUCTIONS[task]
    frame_paths = get_frame_paths(episode_dir)

    if not frame_paths:
        print(f"  ERROR: No frames found in {episode_dir}")
        return False

    # Sample frames evenly across the episode (focus on teleop phase)
    total = len(frame_paths)
    indices = np.linspace(0, min(total - 1, total // 2), n_frames, dtype=int)

    print(f"  Episode: {Path(episode_dir).name}")
    print(f"  Task:    {task}")
    print(f"  Prompt:  {prompt[:60]}...")
    print(f"  Frames:  {total} total, testing {len(indices)} frames")
    print()

    all_ok = True
    for i, idx in enumerate(indices):
        frame = load_frame(str(frame_paths[idx]))
        obs = frame_to_obs(frame)
        phase = frame.get("phase", "unknown")

        # Pack exactly like OpenPIAdapter.infer()
        joints = obs["joint_positions"][:6]
        gripper = obs["gripper_position"][0]
        state = np.concatenate([joints, [gripper]]).astype(np.float32)

        request = {
            "observation/image": obs["base_rgb"],
            "observation/wrist_image": obs["wrist_rgb"],
            "observation/state": state,
            "prompt": prompt,
        }

        try:
            result = policy.infer(request)
        except Exception as e:
            print(f"  [{i}] frame {idx} ({phase}): INFER ERROR: {e}")
            all_ok = False
            continue

        actions = result.get("actions")
        if actions is None:
            print(f"  [{i}] frame {idx} ({phase}): ERROR: no 'actions' key in result")
            print(f"       keys: {list(result.keys())}")
            all_ok = False
            continue

        # Validate shape
        shape_ok = len(actions.shape) == 2 and actions.shape[1] == 7
        chunk_size = actions.shape[0]

        # Validate ranges
        joints_out = actions[:, :6]
        gripper_out = actions[:, 6]
        joints_in_range = np.all(np.abs(joints_out) < 2 * np.pi)
        gripper_in_range = np.all((gripper_out >= -0.1) & (gripper_out <= 1.1))

        # Check for stop signals
        stop_mask = gripper_out > 0.95
        n_stops = np.sum(stop_mask)

        # Compute delta from current state
        delta_j = actions[0, :6] - joints
        delta_norm = np.linalg.norm(delta_j)

        status = "OK" if (shape_ok and joints_in_range and gripper_in_range) else "FAIL"
        if not (shape_ok and joints_in_range and gripper_in_range):
            all_ok = False

        print(
            f"  [{i:2d}] frame {idx:4d} ({phase:12s}) | "
            f"shape {actions.shape} | "
            f"grip [{gripper_out.min():.3f}, {gripper_out.max():.3f}] | "
            f"delta_j0 {delta_norm:.4f} rad | "
            f"stops {n_stops}/{chunk_size} | "
            f"{status}"
        )

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Offline validation of OpenPI inference"
    )
    parser.add_argument("--host", default="127.0.0.1", help="OpenPI server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenPI server port")
    parser.add_argument(
        "--data-dir", default="data/vla_dataset", help="Root data directory"
    )
    parser.add_argument("--cpu-episode", default="", help="Specific CPU episode dir")
    parser.add_argument("--ram-episode", default="", help="Specific RAM episode dir")
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=10,
        help="Number of frames to test per episode",
    )
    args = parser.parse_args()

    # Connect to server
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError:
        print(
            "ERROR: openpi-client not installed.\n"
            "Install with: pip install /home/chris/openpi/packages/openpi-client/"
        )
        sys.exit(1)

    print("=" * 70)
    print("  OpenPI Offline Inference Validation")
    print("=" * 70)
    print(f"  Server: {args.host}:{args.port}")
    print()

    print("Connecting to OpenPI server...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print("Connected.\n")

    # Find episodes
    cpu_ep = args.cpu_episode or find_default_episode(args.data_dir, "cpu")
    ram_ep = args.ram_episode or find_default_episode(args.data_dir, "ram")

    results = {}

    if cpu_ep:
        print("-" * 70)
        print("CPU TASK")
        print("-" * 70)
        results["cpu"] = validate_episode(
            policy, cpu_ep, "cpu", args.frames_per_episode
        )
        print()

    if ram_ep:
        print("-" * 70)
        print("RAM TASK")
        print("-" * 70)
        results["ram"] = validate_episode(
            policy, ram_ep, "ram", args.frames_per_episode
        )
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for task, ok in results.items():
        print(f"  {task}: {'PASS' if ok else 'FAIL'}")

    if all(results.values()):
        print("\nAll validations passed. Pipeline is ready for live inference.")
    else:
        print("\nSome validations failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
