#!/usr/bin/env python3
# scripts/convert_to_openvla.py
"""Convert raw dual-dataset pickle episodes to OpenVLA RLDS format.

OpenVLA expects:
  - Images: 224x224 RGB
  - Actions: 7-dim EEF delta [dx, dy, dz, drx, dry, drz, gripper_binary]
  - Action encoding: EEF_POS (delta xyz + delta rotation vector)
  - State encoding: JOINT (7 joints + gripper = 8 dims)
  - Normalization: z-score (mean/std) or bounds (min/max) per action dim
  - Gripper: binarized (0/1), excluded from normalization
  - Format: RLDS/TFDS (TFRecord files)

Usage:
    python scripts/convert_to_openvla.py \\
        --input-dir data/dual_dataset \\
        --output-dir data/openvla_rlds \\
        --dataset-type vla_skill

This script generates:
  1. Converted pickle files with resized images and delta actions
  2. Normalization statistics (norm_stats.json)
  3. A summary of the conversion for manual RLDS registration
"""

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# OpenVLA image resolution
OPENVLA_IMG_SIZE = (224, 224)  # (H, W)


def compute_delta_actions(frames: List[Dict]) -> List[np.ndarray]:
    """Compute delta EEF actions from consecutive TCP poses.

    Args:
        frames: List of frame dicts with obs containing ee_pos_quat.

    Returns:
        List of 7-dim delta actions [dx,dy,dz,drx,dry,drz,gripper_binary].
        Length = len(frames) - 1 (last frame has no next target).
    """
    from scipy.spatial.transform import Rotation

    actions = []
    for i in range(len(frames) - 1):
        obs_curr = frames[i]["obs"]
        obs_next = frames[i + 1]["obs"]

        if obs_curr is None or obs_next is None:
            actions.append(np.zeros(7))
            continue

        # Get TCP poses (ee_pos_quat is [x,y,z,qx,qy,qz,qw])
        pq_curr = obs_curr.get("ee_pos_quat", np.zeros(7))
        pq_next = obs_next.get("ee_pos_quat", np.zeros(7))

        # Delta position
        dp = pq_next[:3] - pq_curr[:3]

        # Delta rotation (as rotation vector)
        if np.allclose(pq_curr[3:], 0) or np.allclose(pq_next[3:], 0):
            dr = np.zeros(3)
        else:
            R_curr = Rotation.from_quat(pq_curr[3:7])
            R_next = Rotation.from_quat(pq_next[3:7])
            R_delta = R_curr.inv() * R_next
            dr = R_delta.as_rotvec()

        # Gripper: binarize (>0.5 = closed = 1, else 0)
        grip = obs_curr.get("gripper_position", np.array([0.0]))
        grip_val = float(grip[0]) if len(grip) > 0 else 0.0
        grip_binary = 1.0 if grip_val > 0.5 else 0.0

        action = np.concatenate([dp, dr, [grip_binary]])
        actions.append(action)

    # Last frame: repeat the previous action (or zeros)
    if actions:
        actions.append(actions[-1].copy())
    else:
        actions.append(np.zeros(7))

    return actions


def resize_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize image to target (H, W)."""
    if img is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    return cv2.resize(
        img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
    )


def compute_norm_stats(all_actions: List[np.ndarray]) -> Dict[str, Any]:
    """Compute normalization statistics for actions.

    Returns z-score stats (mean, std) and bounds stats (min, max).
    Gripper dimension (last) is excluded from normalization.
    """
    if not all_actions:
        return {}

    actions_arr = np.array(all_actions)  # (N, 7)
    action_dim = actions_arr.shape[1]

    mean = np.mean(actions_arr, axis=0).tolist()
    std = np.std(actions_arr, axis=0).tolist()
    a_min = np.min(actions_arr, axis=0).tolist()
    a_max = np.max(actions_arr, axis=0).tolist()

    # Normalization mask: True = normalize, False = skip (gripper)
    norm_mask = [True] * (action_dim - 1) + [False]

    return {
        "mean": mean,
        "std": std,
        "min": a_min,
        "max": a_max,
        "action_normalization_mask": norm_mask,
        "action_dim": action_dim,
        "normalization_type": "normal",  # z-score
    }


def convert_episode(
    episode_dir: Path,
    dataset_type: str,
    output_dir: Path,
) -> List[np.ndarray]:
    """Convert a single episode to OpenVLA format.

    Returns list of delta actions for normalization stats computation.
    """
    ds_dir = episode_dir / dataset_type
    if not ds_dir.exists():
        return []

    # Load all frames
    frame_files = sorted(glob.glob(str(ds_dir / "frame_*.pkl")))
    if not frame_files:
        return []

    frames = []
    for fpath in frame_files:
        with open(fpath, "rb") as f:
            frames.append(pickle.load(f))

    # Compute delta actions
    delta_actions = compute_delta_actions(frames)

    # Create output directory
    episode_name = episode_dir.name
    out_ep_dir = output_dir / episode_name
    out_ep_dir.mkdir(parents=True, exist_ok=True)

    # Convert and save frames
    for i, (frame, action) in enumerate(zip(frames, delta_actions)):
        obs = frame.get("obs", {})
        if obs is None:
            obs = {}

        # Resize images to 224x224
        wrist_rgb = resize_image(obs.get("wrist_rgb"), OPENVLA_IMG_SIZE)
        base_rgb = resize_image(obs.get("base_rgb"), OPENVLA_IMG_SIZE)

        # Proprioceptive state: 7 joints + gripper = 8 dims
        joints = obs.get("joint_positions", np.zeros(7))
        grip = obs.get("gripper_position", np.array([0.0]))
        grip_val = float(grip[0]) if len(grip) > 0 else 0.0
        state = np.concatenate([joints[:7], [grip_val]])

        converted = {
            "image_primary": base_rgb,  # (224, 224, 3) uint8
            "image_wrist": wrist_rgb,  # (224, 224, 3) uint8
            "state": state.astype(np.float32),  # (8,)
            "action": action.astype(np.float32),  # (7,)
            "ee_pos_quat": obs.get("ee_pos_quat", np.zeros(7)).astype(np.float32),
        }

        out_path = out_ep_dir / f"step_{i:04d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(converted, f)

    print(f"  Converted {episode_name}: {len(frames)} frames")
    return delta_actions


def main():
    parser = argparse.ArgumentParser(
        description="Convert dual-dataset episodes to OpenVLA format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/dual_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/openvla_rlds",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="vla_skill",
        choices=["vla_skill", "vla_full"],
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all episodes
    episode_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    )

    if not episode_dirs:
        print(f"No episodes found in {input_dir}")
        return

    print(f"Found {len(episode_dirs)} episodes")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output: {output_dir}")
    print()

    # Convert all episodes
    all_actions = []
    for ep_dir in episode_dirs:
        actions = convert_episode(ep_dir, args.dataset_type, output_dir)
        all_actions.extend(actions)

    # Compute and save normalization stats
    if all_actions:
        stats = compute_norm_stats(all_actions)
        stats_path = output_dir / "norm_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nNormalization stats saved to {stats_path}")
        print(f"  Action mean: {stats['mean']}")
        print(f"  Action std:  {stats['std']}")

    # Summary
    print("\n--- OpenVLA Conversion Summary ---")
    print(f"  Episodes:     {len(episode_dirs)}")
    print(f"  Total frames: {len(all_actions)}")
    print(f"  Image size:   {OPENVLA_IMG_SIZE}")
    print("  Action dim:   7 (delta EEF + binary gripper)")
    print("  State dim:    8 (7 joints + gripper)")
    print()
    print("To register with OpenVLA, add entries to:")
    print("  1. prismatic/vla/datasets/rlds/oxe/configs.py")
    print("  2. prismatic/vla/datasets/rlds/oxe/transforms.py")
    print("  3. prismatic/vla/datasets/rlds/oxe/mixtures.py")


if __name__ == "__main__":
    main()
