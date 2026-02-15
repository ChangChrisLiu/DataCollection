#!/usr/bin/env python3
# scripts/convert_to_openpi.py
"""Convert raw dual-dataset pickle episodes to Pi-0.5 (OpenPI) LeRobot format.

Pi-0.5 expects:
  - Images: 256x256 (model applies resize_with_pad to 224x224 internally)
  - Multiple cameras: base (image) + wrist (wrist_image)
  - State: 8-dim float32 (7 joints + gripper)
  - Actions: 7-dim float32 (delta EEF or absolute joint)
  - Format: LeRobot (HuggingFace ecosystem)
  - Normalization: z-score or quantile (q01/q99) via compute_norm_stats.py
  - Action chunking: action_horizon=10 (10 steps at 30Hz)

Usage:
    python scripts/convert_to_openpi.py \\
        --input-dir data/dual_dataset \\
        --output-dir data/openpi_lerobot \\
        --dataset-type vla_full \\
        --action-horizon 10

This generates a pickle-based intermediate format. For full LeRobot
conversion, use the LeRobot API:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
"""

import argparse
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Pi-0.5 image resolution (stored; model resizes to 224x224 with padding)
PI_IMG_SIZE = (256, 256)  # (H, W)

# Default action horizon (number of future steps per prediction)
DEFAULT_ACTION_HORIZON = 10


def compute_delta_actions(frames: List[Dict]) -> List[np.ndarray]:
    """Compute delta EEF actions from consecutive TCP poses."""
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

        grip = obs_curr.get("gripper_position", np.array([0.0]))
        grip_val = float(grip[0]) if len(grip) > 0 else 0.0

        action = np.concatenate([dp, dr, [grip_val]])
        actions.append(action)

    if actions:
        actions.append(actions[-1].copy())
    else:
        actions.append(np.zeros(7))

    return actions


def resize_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    if img is None:
        return np.zeros((*target_size, 3), dtype=np.uint8)
    return cv2.resize(
        img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
    )


def compute_quantile_stats(all_actions: List[np.ndarray], all_states: List[np.ndarray]):
    """Compute normalization stats for Pi-0.5 (z-score + quantile).

    Pi-0.5 uses either z-score or quantile normalization.
    We compute both so the user can choose.
    """
    if not all_actions:
        return {}

    actions_arr = np.array(all_actions)
    states_arr = np.array(all_states) if all_states else np.zeros((1, 8))

    def _compute_stats(arr):
        return {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
        }

    return {
        "norm_stats": {
            "actions": _compute_stats(actions_arr),
            "state": _compute_stats(states_arr),
        },
        "action_dim": actions_arr.shape[1],
        "state_dim": states_arr.shape[1],
    }


def convert_episode(
    episode_dir: Path,
    dataset_type: str,
    output_dir: Path,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    task_description: str = "pick up the object",
) -> tuple:
    """Convert a single episode to Pi-0.5 (LeRobot) format.

    Returns (all_actions, all_states) for stats computation.
    """
    ds_dir = episode_dir / dataset_type
    if not ds_dir.exists():
        return [], []

    frame_files = sorted(glob.glob(str(ds_dir / "frame_*.pkl")))
    if not frame_files:
        return [], []

    frames = []
    for fpath in frame_files:
        with open(fpath, "rb") as f:
            frames.append(pickle.load(f))

    delta_actions = compute_delta_actions(frames)

    episode_name = episode_dir.name
    out_ep_dir = output_dir / episode_name
    out_ep_dir.mkdir(parents=True, exist_ok=True)

    all_actions = []
    all_states = []

    for i, (frame, action) in enumerate(zip(frames, delta_actions)):
        obs = frame.get("obs", {})
        if obs is None:
            obs = {}

        # Resize to 256x256
        base_rgb = resize_image(obs.get("base_rgb"), PI_IMG_SIZE)
        wrist_rgb = resize_image(obs.get("wrist_rgb"), PI_IMG_SIZE)

        # State: 8-dim (7 joints + gripper)
        joints = obs.get("joint_positions", np.zeros(7))
        grip = obs.get("gripper_position", np.array([0.0]))
        grip_val = float(grip[0]) if len(grip) > 0 else 0.0
        state = np.concatenate([joints[:7], [grip_val]]).astype(np.float32)

        all_actions.append(action)
        all_states.append(state)

        # LeRobot-style frame
        lerobot_frame = {
            "image": base_rgb,  # (256, 256, 3) uint8
            "wrist_image": wrist_rgb,  # (256, 256, 3) uint8
            "state": state,  # (8,) float32
            "actions": action.astype(np.float32),  # (7,) float32
            "task": task_description,
            "episode_index": 0,  # Will be reindexed during full conversion
            "frame_index": i,
            "timestamp": i / 30.0,  # 30Hz
        }

        out_path = out_ep_dir / f"frame_{i:04d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(lerobot_frame, f)

    # Save episode metadata
    meta = {
        "num_frames": len(frames),
        "action_horizon": action_horizon,
        "fps": 30,
        "task": task_description,
    }
    meta_path = out_ep_dir / "episode_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Converted {episode_name}: {len(frames)} frames")
    return all_actions, all_states


def generate_lerobot_script(output_dir: Path, num_episodes: int):
    """Generate a helper script for LeRobot dataset creation."""
    script = f'''#!/usr/bin/env python3
"""Helper script to create a LeRobot dataset from converted frames.

Run after convert_to_openpi.py to create a proper LeRobot dataset.
Requires: pip install lerobot

Usage:
    python {output_dir / "create_lerobot_dataset.py"}
"""
import glob
import pickle
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

output_dir = Path("{output_dir}")

dataset = LeRobotDataset.create(
    repo_id="datacollection_ewaste",
    robot_type="ur5e",
    fps=30,
    features={{
        "image": {{
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        }},
        "wrist_image": {{
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        }},
        "state": {{
            "dtype": "float32",
            "shape": (8,),
        }},
        "actions": {{
            "dtype": "float32",
            "shape": (7,),
        }},
    }},
    image_writer_threads=4,
)

# Load all episodes
episode_dirs = sorted(
    [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
)

for ep_dir in episode_dirs:
    frame_files = sorted(glob.glob(str(ep_dir / "frame_*.pkl")))
    for fpath in frame_files:
        with open(fpath, "rb") as f:
            frame = pickle.load(f)
        dataset.add_frame({{
            "image": frame["image"],
            "wrist_image": frame["wrist_image"],
            "state": frame["state"],
            "actions": frame["actions"],
            "task": frame.get("task", "pick up the object"),
        }})
    dataset.save_episode()
    print(f"Saved episode: {{ep_dir.name}}")

print(f"LeRobot dataset created: {{dataset.repo_id}}")
print(f"Then run: uv run scripts/compute_norm_stats.py --config-name your_config")
'''

    script_path = output_dir / "create_lerobot_dataset.py"
    with open(script_path, "w") as f:
        f.write(script)
    print(f"\nLeRobot creation script: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert dual-dataset episodes to Pi-0.5 (OpenPI) format"
    )
    parser.add_argument("--input-dir", type=str, default="data/dual_dataset")
    parser.add_argument("--output-dir", type=str, default="data/openpi_lerobot")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="vla_full",
        choices=["vla_skill", "vla_full"],
    )
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the object",
        help="Language task description for the dataset",
    )
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
    print(f"Action horizon: {args.action_horizon}")
    print(f"Task: {args.task}")
    print()

    all_actions = []
    all_states = []
    for ep_dir in episode_dirs:
        ep_actions, ep_states = convert_episode(
            ep_dir,
            args.dataset_type,
            output_dir,
            action_horizon=args.action_horizon,
            task_description=args.task,
        )
        all_actions.extend(ep_actions)
        all_states.extend(ep_states)

    # Save normalization stats
    if all_actions:
        stats = compute_quantile_stats(all_actions, all_states)
        stats_path = output_dir / "norm_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nNormalization stats saved to {stats_path}")

    # Generate LeRobot creation helper script
    generate_lerobot_script(output_dir, len(episode_dirs))

    print("\n--- Pi-0.5 (OpenPI) Conversion Summary ---")
    print(f"  Episodes:         {len(episode_dirs)}")
    print(f"  Total frames:     {len(all_actions)}")
    print(f"  Image size:       {PI_IMG_SIZE} (model pads to 224x224)")
    print("  Action dim:       7")
    print("  State dim:        8")
    print(f"  Action horizon:   {args.action_horizon}")
    print()
    print("Next steps:")
    print("  1. pip install lerobot")
    print(f"  2. python {output_dir / 'create_lerobot_dataset.py'}")
    print("  3. uv run scripts/compute_norm_stats.py --config-name your_config")
    print("  4. Define Inputs/Outputs/DataConfig/TrainConfig classes")


if __name__ == "__main__":
    main()
