#!/usr/bin/env python3
# scripts/convert_to_openvla_oft.py
"""Convert raw dual-dataset pickle episodes to OpenVLA-OFT RLDS format.

OpenVLA-OFT differences from base OpenVLA:
  - Images: 256x256 (model applies random/center crop to ~224x224)
  - Multi-image input: both base + wrist camera simultaneously
  - Proprioceptive state as model input (8-dim: 7 joints + gripper)
  - Continuous L1 regression (no discretization)
  - Action chunking: 10 steps at 30Hz = 0.33s lookahead
  - Normalization: bounds_q99 (1st/99th percentile -> [-1, 1])
  - No-op filtering: remove near-zero action frames

Usage:
    python scripts/convert_to_openvla_oft.py \\
        --input-dir data/dual_dataset \\
        --output-dir data/openvla_oft_rlds \\
        --dataset-type vla_full \\
        --chunk-size 10
"""

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# OpenVLA-OFT image resolution (model crops to ~224x224 during training)
OFT_IMG_SIZE = (256, 256)  # (H, W)

# Action chunk size (number of future action steps per prediction)
DEFAULT_CHUNK_SIZE = 10

# No-op threshold: discard frames where action L2 norm is below this
NOOP_THRESHOLD = 1e-4


def compute_delta_actions(frames: List[Dict]) -> List[np.ndarray]:
    """Compute delta EEF actions from consecutive TCP poses.

    Returns 7-dim: [dx, dy, dz, drx, dry, drz, gripper_continuous].
    """
    from scipy.spatial.transform import Rotation

    actions = []
    for i in range(len(frames) - 1):
        obs_curr = frames[i]["obs"]
        obs_next = frames[i + 1]["obs"]

        if obs_curr is None or obs_next is None:
            actions.append(np.zeros(7))
            continue

        pq_curr = obs_curr.get("ee_pos_quat", np.zeros(7))
        pq_next = obs_next.get("ee_pos_quat", np.zeros(7))

        dp = pq_next[:3] - pq_curr[:3]

        if np.allclose(pq_curr[3:], 0) or np.allclose(pq_next[3:], 0):
            dr = np.zeros(3)
        else:
            R_curr = Rotation.from_quat(pq_curr[3:7])
            R_next = Rotation.from_quat(pq_next[3:7])
            R_delta = R_curr.inv() * R_next
            dr = R_delta.as_rotvec()

        # Continuous gripper (not binarized for OFT)
        grip = obs_curr.get("gripper_position", np.array([0.0]))
        grip_val = float(grip[0]) if len(grip) > 0 else 0.0

        action = np.concatenate([dp, dr, [grip_val]])
        actions.append(action)

    if actions:
        actions.append(actions[-1].copy())
    else:
        actions.append(np.zeros(7))

    return actions


def filter_noops(
    frames: List[Dict],
    actions: List[np.ndarray],
    threshold: float = NOOP_THRESHOLD,
) -> tuple:
    """Remove near-zero action frames."""
    filtered_frames = []
    filtered_actions = []

    for frame, action in zip(frames, actions):
        # Check L2 norm of positional action (exclude gripper)
        if np.linalg.norm(action[:6]) > threshold:
            filtered_frames.append(frame)
            filtered_actions.append(action)

    removed = len(frames) - len(filtered_frames)
    if removed > 0:
        print(f"    No-op filter: removed {removed}/{len(frames)} frames")

    return filtered_frames, filtered_actions


def compute_bounds_q99(all_actions: List[np.ndarray]) -> Dict[str, Any]:
    """Compute bounds_q99 normalization stats.

    Maps actions to [-1, 1] using 1st and 99th percentiles (robust to outliers).
    """
    if not all_actions:
        return {}

    actions_arr = np.array(all_actions)
    q01 = np.percentile(actions_arr, 1, axis=0).tolist()
    q99 = np.percentile(actions_arr, 99, axis=0).tolist()
    mean = np.mean(actions_arr, axis=0).tolist()
    std = np.std(actions_arr, axis=0).tolist()

    return {
        "q01": q01,
        "q99": q99,
        "mean": mean,
        "std": std,
        "action_dim": actions_arr.shape[1],
        "normalization_type": "bounds_q99",
    }


def resize_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    if img is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    return cv2.resize(
        img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
    )


def convert_episode(
    episode_dir: Path,
    dataset_type: str,
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    filter_noop: bool = True,
) -> List[np.ndarray]:
    """Convert a single episode to OpenVLA-OFT format."""
    ds_dir = episode_dir / dataset_type
    if not ds_dir.exists():
        return []

    frame_files = sorted(glob.glob(str(ds_dir / "frame_*.pkl")))
    if not frame_files:
        return []

    frames = []
    for fpath in frame_files:
        with open(fpath, "rb") as f:
            frames.append(pickle.load(f))

    delta_actions = compute_delta_actions(frames)

    # No-op filtering
    if filter_noop:
        frames, delta_actions = filter_noops(frames, delta_actions)

    episode_name = episode_dir.name
    out_ep_dir = output_dir / episode_name
    out_ep_dir.mkdir(parents=True, exist_ok=True)

    for i, (frame, action) in enumerate(zip(frames, delta_actions)):
        obs = frame.get("obs", {})
        if obs is None:
            obs = {}

        # Resize to 256x256 (model crops during training)
        wrist_rgb = resize_image(obs.get("wrist_rgb"), OFT_IMG_SIZE)
        base_rgb = resize_image(obs.get("base_rgb"), OFT_IMG_SIZE)

        # Proprioceptive state: 8-dim (used as model input in OFT)
        joints = obs.get("joint_positions", np.zeros(7))
        grip = obs.get("gripper_position", np.array([0.0]))
        grip_val = float(grip[0]) if len(grip) > 0 else 0.0
        state = np.concatenate([joints[:7], [grip_val]])

        # Action chunk: current + next (chunk_size - 1) actions
        action_chunk = []
        for j in range(chunk_size):
            idx = min(i + j, len(delta_actions) - 1)
            action_chunk.append(delta_actions[idx])
        action_chunk = np.array(action_chunk, dtype=np.float32)  # (chunk_size, 7)

        converted = {
            "full_image": base_rgb,  # (256, 256, 3) uint8
            "wrist_image": wrist_rgb,  # (256, 256, 3) uint8
            "state": state.astype(np.float32),  # (8,)
            "action": action.astype(np.float32),  # (7,) single-step
            "action_chunk": action_chunk,  # (chunk_size, 7)
        }

        out_path = out_ep_dir / f"step_{i:04d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(converted, f)

    print(f"  Converted {episode_name}: {len(frames)} frames")
    return delta_actions


def main():
    parser = argparse.ArgumentParser(
        description="Convert dual-dataset episodes to OpenVLA-OFT format"
    )
    parser.add_argument("--input-dir", type=str, default="data/dual_dataset")
    parser.add_argument("--output-dir", type=str, default="data/openvla_oft_rlds")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="vla_full",
        choices=["vla_skill", "vla_full"],
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no-noop-filter", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    )

    if not episode_dirs:
        print(f"No episodes found in {input_dir}")
        return

    print(f"Found {len(episode_dirs)} episodes")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"No-op filter: {not args.no_noop_filter}")
    print()

    all_actions = []
    for ep_dir in episode_dirs:
        actions = convert_episode(
            ep_dir,
            args.dataset_type,
            output_dir,
            chunk_size=args.chunk_size,
            filter_noop=not args.no_noop_filter,
        )
        all_actions.extend(actions)

    if all_actions:
        stats = compute_bounds_q99(all_actions)
        stats_path = output_dir / "norm_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nNormalization stats (bounds_q99) saved to {stats_path}")

    print("\n--- OpenVLA-OFT Conversion Summary ---")
    print(f"  Episodes:          {len(episode_dirs)}")
    print(f"  Total frames:      {len(all_actions)}")
    print(f"  Image size:        {OFT_IMG_SIZE} (model crops to ~224x224)")
    print("  Action dim:        7 (delta EEF + continuous gripper)")
    print("  Proprio dim:       8 (7 joints + gripper)")
    print(f"  Chunk size:        {args.chunk_size}")
    print("  Normalization:     bounds_q99")
    print()
    print("Set in prismatic/vla/constants.py:")
    print("  ACTION_DIM = 7")
    print("  PROPRIO_DIM = 8")
    print(f"  NUM_OPEN_LOOP_STEPS = {args.chunk_size}")
    print('  ACTION_PROPRIO_NORMALIZATION_TYPE = "bounds_q99"')


if __name__ == "__main__":
    main()
