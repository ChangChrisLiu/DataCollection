#!/usr/bin/env python3
"""Offline validation of OpenVLA-OFT inference pipeline using collected episodes.

Loads real frames from collected .pkl episodes, packs them exactly like
OpenVLAOFTAdapter.infer() does, sends to a running OpenVLA-OFT server, and
checks that returned action chunks are sane (shape, range, stop signals).

Usage:
    1. Start the OpenVLA-OFT server:
       conda activate oft && cd ~/Sibo/openvla-oft && \
       python vla-scripts/deploy.py \
           --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
           --unnorm_key bridge_orig --use_l1_regression True \
           --use_proprio True --num_images_in_input 2 --port 8777

    2. Run this script (from tele conda env):
       python scripts/validate_openvla_oft_inference.py --port 8777

    3. Optional flags:
       --host 127.0.0.1 --port 8777
       --cpu-episode data/vla_dataset/CPU_Extraction/episode_cpu_0218_140015
       --ram-episode data/vla_dataset/RAM_Extraction/episode_ram_0218_172359
       --frames-per-episode 10   (how many frames to test per episode)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

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

# Expected action chunk size (NUM_ACTIONS_CHUNK in OFT constants)
EXPECTED_CHUNK_SIZE = 8


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
    endpoint: str, requests_mod, episode_dir: str, task: str, n_frames: int,
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

        # Pack exactly like OpenVLAOFTAdapter.infer()
        from PIL import Image

        base_rgb = np.array(
            Image.fromarray(obs["base_rgb"]).resize((256, 256), Image.LANCZOS)
        )
        wrist_rgb = np.array(
            Image.fromarray(obs["wrist_rgb"]).resize((256, 256), Image.LANCZOS)
        )

        # State must be 8D EEF pose matching RLDS training format:
        # [x, y, z, roll, pitch, yaw, 0.0_pad, gripper]
        tcp = frame["tcp_pose"]  # [x,y,z,rx,ry,rz] rotation vector
        pos = np.array(tcp[:3])
        rpy = Rotation.from_rotvec(np.array(tcp[3:6])).as_euler("xyz")
        gripper = frame["gripper_pos"] / 255.0
        state = np.concatenate([pos, rpy, [0.0, gripper]]).astype(np.float32)

        payload = {
            "full_image": base_rgb,
            "wrist_image_0": wrist_rgb,
            "state": state,
            "instruction": prompt,
        }

        try:
            response = requests_mod.post(endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            # Server returns string "error" on failure (with 200 status)
            if isinstance(result, str):
                print(f"  [{i}] frame {idx} ({phase}): SERVER ERROR: {result}")
                print(f"       state dim={len(state)}, format=[x,y,z,r,p,y,0.0,grip]")
                all_ok = False
                continue
        except Exception as e:
            print(f"  [{i}] frame {idx} ({phase}): INFER ERROR: {e}")
            all_ok = False
            continue

        # Parse actions — server returns list of actions
        if isinstance(result, list):
            actions = [np.array(a, dtype=np.float64) for a in result]
        else:
            actions = [np.array(result, dtype=np.float64)]

        chunk_size = len(actions)

        # Validate chunk size
        chunk_ok = chunk_size == EXPECTED_CHUNK_SIZE

        # Validate each action in the chunk
        shapes_ok = True
        pos_ok = True
        rot_ok = True
        grip_ok = True
        n_stops = 0

        for a in actions:
            if a.shape != (7,):
                shapes_ok = False
            if np.any(np.abs(a[:3]) >= 0.1):
                pos_ok = False
            if np.any(np.abs(a[3:6]) >= 1.0):
                rot_ok = False
            if not (-0.1 <= a[6] <= 1.1):
                grip_ok = False
            # Stop signal: inverted gripper, close = low value
            if a[6] < 0.05:
                n_stops += 1

        # Aggregate stats across chunk
        all_actions = np.array([a for a in actions])
        grip_vals = all_actions[:, 6]
        delta_norms = [float(np.linalg.norm(a[:6])) for a in actions]
        mean_delta = np.mean(delta_norms)
        max_delta = np.max(delta_norms)

        ok = chunk_ok and shapes_ok and pos_ok and rot_ok and grip_ok
        if not ok:
            all_ok = False

        status = "OK" if ok else "FAIL"

        print(
            f"  [{i:2d}] frame {idx:4d} ({phase:12s}) | "
            f"chunk {chunk_size} | "
            f"grip [{grip_vals.min():.3f}, {grip_vals.max():.3f}] | "
            f"delta avg {mean_delta:.4f} max {max_delta:.4f} | "
            f"stops {n_stops}/{chunk_size} | "
            f"{status}"
        )

        if not chunk_ok:
            print(
                f"       Expected chunk size {EXPECTED_CHUNK_SIZE}, "
                f"got {chunk_size}"
            )
        if not shapes_ok:
            shapes = [a.shape for a in actions]
            print(f"       Bad action shapes: {shapes}")
        if not pos_ok:
            print(f"       Position deltas out of range (>0.1m)")
        if not rot_ok:
            print(f"       Rotation deltas out of range (>1.0 rad)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Offline validation of OpenVLA-OFT inference"
    )
    parser.add_argument("--host", default="127.0.0.1", help="OFT server host")
    parser.add_argument("--port", type=int, default=8777, help="OFT server port")
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
    print("  OpenVLA-OFT Offline Inference Validation")
    print("=" * 78)
    print(f"  Server:       {args.host}:{args.port}")
    print(f"  Endpoint:     {endpoint}")
    print(f"  Chunk size:   {EXPECTED_CHUNK_SIZE} (expected)")
    print(f"  State dim:    8 (EEF pose: x,y,z,r,p,y + 0.0 pad + gripper)")
    print()

    # Quick connectivity check
    print("Testing server connectivity...")
    try:
        test_state = np.zeros(8, dtype=np.float32)
        test_payload = {
            "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image_0": np.zeros((256, 256, 3), dtype=np.uint8),
            "state": test_state,
            "instruction": "test",
        }
        resp = requests.post(endpoint, json=test_payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, str):
            print(f"WARNING: Server returned error on test request: {result}")
            print("This may indicate a state dimension mismatch or other config issue.")
            print("Continuing anyway...\n")
        elif isinstance(result, list):
            actions = [np.array(a, dtype=np.float64) for a in result]
            print(
                f"Server is responding. Test chunk: {len(actions)} actions, "
                f"shape {actions[0].shape}\n"
            )
        else:
            action = np.array(result, dtype=np.float64)
            print(f"Server is responding. Test action shape: {action.shape}\n")
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {endpoint}")
        print("Is the OpenVLA-OFT server running?")
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
            endpoint, requests, cpu_ep, "cpu", args.frames_per_episode,
        )
        print()

    if ram_ep:
        print("-" * 78)
        print("RAM TASK")
        print("-" * 78)
        results["ram"] = validate_episode(
            endpoint, requests, ram_ep, "ram", args.frames_per_episode,
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
