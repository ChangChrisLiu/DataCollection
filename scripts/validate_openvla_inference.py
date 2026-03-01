#!/usr/bin/env python3
"""Offline validation of OpenVLA inference pipeline using collected episodes.

Loads real frames from collected .pkl episodes, packs them exactly like
OpenVLAAdapter.infer() does, sends to a running OpenVLA server, and checks
that returned actions are sane (shape, range, stop signals near end).

Usage:
    1. Start the OpenVLA server:
       conda activate vla && cd ~/Sibo/openvla && \
       python vla-scripts/deploy.py --openvla_path openvla/openvla-7b --port 8000

    2. Run this script (from tele conda env):
       python scripts/validate_openvla_inference.py --port 8000 --unnorm-key bridge_orig

    3. Optional flags:
       --host 127.0.0.1 --port 8000
       --unnorm-key ur5e_vla_planner_10hz
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
    """Convert a .pkl frame to the observation dict that get_obs() returns."""
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


def validate_episode(
    endpoint: str, requests_mod, unnorm_key: str, episode_dir: str,
    task: str, n_frames: int,
):
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

        # Pack exactly like OpenVLAAdapter.infer()
        from PIL import Image

        base_rgb = np.array(
            Image.fromarray(obs["base_rgb"]).resize((256, 256), Image.LANCZOS)
        )

        payload = {
            "image": base_rgb,
            "instruction": prompt,
            "unnorm_key": unnorm_key,
        }

        try:
            response = requests_mod.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            # Server returns string "error" on failure (with 200 status)
            if isinstance(result, str):
                print(f"  [{i}] frame {idx} ({phase}): SERVER ERROR: {result}")
                all_ok = False
                continue
            action = np.array(result, dtype=np.float64)
        except Exception as e:
            print(f"  [{i}] frame {idx} ({phase}): INFER ERROR: {e}")
            all_ok = False
            continue

        # Validate shape — OpenVLA returns a single (7,) action
        shape_ok = action.shape == (7,)

        # Validate EEF delta ranges
        pos_deltas = action[:3]
        rot_deltas = action[3:6]
        gripper_val = action[6]

        pos_ok = np.all(np.abs(pos_deltas) < 0.1)  # < 10cm per step
        rot_ok = np.all(np.abs(rot_deltas) < 1.0)  # < 1 rad per step
        grip_ok = -0.1 <= gripper_val <= 1.1

        # Delta magnitude (EEF movement size)
        delta_norm = float(np.linalg.norm(action[:6]))

        # Stop signal: inverted gripper, close = low value
        is_stop = gripper_val < 0.05

        ok = shape_ok and pos_ok and rot_ok and grip_ok
        if not ok:
            all_ok = False

        status = "OK" if ok else "FAIL"
        stop_str = "STOP" if is_stop else "    "

        print(
            f"  [{i:2d}] frame {idx:4d} ({phase:12s}) | "
            f"shape {action.shape} | "
            f"pos [{pos_deltas.min():.4f}, {pos_deltas.max():.4f}] | "
            f"rot [{rot_deltas.min():.4f}, {rot_deltas.max():.4f}] | "
            f"grip {gripper_val:.3f} | "
            f"delta {delta_norm:.4f} | "
            f"{stop_str} | {status}"
        )

        if not shape_ok:
            print(f"       Expected shape (7,), got {action.shape}")
        if not pos_ok:
            print(f"       Position deltas out of range: {pos_deltas}")
        if not rot_ok:
            print(f"       Rotation deltas out of range: {rot_deltas}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Offline validation of OpenVLA inference"
    )
    parser.add_argument("--host", default="127.0.0.1", help="OpenVLA server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenVLA server port")
    parser.add_argument(
        "--unnorm-key",
        default="bridge_orig",
        help="Unnormalization key (default: bridge_orig for base model)",
    )
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

    # Import dependencies
    try:
        import json_numpy
        import requests
    except ImportError:
        print(
            "ERROR: json-numpy and requests are required.\n"
            "Install with: pip install json-numpy requests"
        )
        sys.exit(1)

    json_numpy.patch()

    endpoint = f"http://{args.host}:{args.port}/act"

    print("=" * 78)
    print("  OpenVLA Offline Inference Validation")
    print("=" * 78)
    print(f"  Server:     {args.host}:{args.port}")
    print(f"  Endpoint:   {endpoint}")
    print(f"  Unnorm key: {args.unnorm_key}")
    print()

    # Quick connectivity check
    print("Testing server connectivity...")
    try:
        test_payload = {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "instruction": "test",
            "unnorm_key": args.unnorm_key,
        }
        resp = requests.post(endpoint, json=test_payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, str):
            print(f"WARNING: Server returned error on test request: {result}")
            print("This may indicate an invalid unnorm_key. Continuing anyway...\n")
        else:
            action = np.array(result, dtype=np.float64)
            print(f"Server is responding. Test action shape: {action.shape}\n")
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {endpoint}")
        print("Is the OpenVLA server running?")
        sys.exit(1)
    except Exception as e:
        print(f"WARNING: Server test request returned: {e}")
        print("Continuing anyway...\n")

    # Find episodes
    cpu_ep = args.cpu_episode or find_default_episode(args.data_dir, "cpu")
    ram_ep = args.ram_episode or find_default_episode(args.data_dir, "ram")

    results = {}

    if cpu_ep:
        print("-" * 78)
        print("CPU TASK")
        print("-" * 78)
        results["cpu"] = validate_episode(
            endpoint, requests, args.unnorm_key, cpu_ep, "cpu",
            args.frames_per_episode,
        )
        print()

    if ram_ep:
        print("-" * 78)
        print("RAM TASK")
        print("-" * 78)
        results["ram"] = validate_episode(
            endpoint, requests, args.unnorm_key, ram_ep, "ram",
            args.frames_per_episode,
        )
        print()

    if not results:
        print("ERROR: No episodes found. Use --cpu-episode or --ram-episode.")
        sys.exit(1)

    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for task, ok in results.items():
        print(f"  {task}: {'PASS' if ok else 'FAIL'}")

    if all(results.values()):
        print("\nAll validations passed. Pipeline is ready for live inference.")
    else:
        print("\nSome validations failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
