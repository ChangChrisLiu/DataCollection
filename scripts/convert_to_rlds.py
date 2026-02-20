#!/usr/bin/env python3
# scripts/convert_to_rlds.py
"""Build RLDS TFRecord datasets for OpenVLA / OpenVLA-OFT.

Uses the TFDS Python API directly (no `tfds build` CLI) to avoid
the apache_beam dependency. Each target is a separate TFDS
GeneratorBasedBuilder following the kpertsch/rlds_dataset_builder template.

State/action format: EEF position + Euler RPY (not joints).
Language instruction auto-detected per episode (CPU vs RAM).
Supports downsampling via --fps (5/10/15/30).

Targets:
    e2e        - Full end-to-end trajectory (all phases)
    planner    - Teleop approach + stop signal
    correction - Recovery after grasp failure + stop signal + near-grasp expansion

Usage:
    # Build planner dataset at 10Hz
    python scripts/convert_to_rlds.py \\
        --target planner \\
        --data-path data/vla_dataset \\
        --fps 10

    # Build all targets at 30Hz (default)
    python scripts/convert_to_rlds.py \\
        --target all \\
        --data-path data/vla_dataset

After building:
    # Datasets are in ~/tensorflow_datasets/ur5e_vla_<target>/1.0.0/
    # Transfer to server:
    rsync -avz ~/tensorflow_datasets/ur5e_vla_planner/ server:~/tensorflow_datasets/ur5e_vla_planner/
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

TARGETS = {
    "e2e": ("ur5e_vla_e2e", "ur5e_vla_e2e_dataset_builder", "Ur5eVlaE2e"),
    "planner": ("ur5e_vla_planner", "ur5e_vla_planner_dataset_builder", "Ur5eVlaPlanner"),
    "correction": ("ur5e_vla_correction", "ur5e_vla_correction_dataset_builder", "Ur5eVlaCorrection"),
}

SCRIPTS_DIR = Path(__file__).resolve().parent


def build_target(target_key: str, data_path: str, image_size: int, fps: int):
    """Build one TFDS dataset target using the Python API."""
    dataset_name, module_name, class_name = TARGETS[target_key]
    builder_dir = SCRIPTS_DIR / dataset_name

    if not builder_dir.exists():
        print(f"ERROR: Builder directory not found: {builder_dir}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Building {dataset_name}")
    print(f"  Data path:  {data_path}")
    print(f"  FPS:        {fps}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Language:   auto-detected per episode (CPU/RAM)")
    print(f"{'=' * 60}\n")

    # Set environment variables for the builder
    os.environ["UR5E_DATA_PATH"] = str(Path(data_path).resolve())
    os.environ["UR5E_IMAGE_SIZE"] = str(image_size)
    os.environ["UR5E_FPS"] = str(fps)

    # Import the builder module dynamically
    sys.path.insert(0, str(builder_dir))
    sys.path.insert(0, str(SCRIPTS_DIR))
    module = importlib.import_module(module_name)
    builder_cls = getattr(module, class_name)

    # Build using Python API (avoids tfds CLI + apache_beam dependency)
    builder = builder_cls()
    builder.download_and_prepare()

    output_dir = Path(builder.data_dir)
    print(f"\n  Output: {output_dir}")

    # Quick verification
    ds = builder.as_dataset(split="train")
    n_episodes = 0
    total_steps = 0
    for traj in ds:
        n_episodes += 1
        total_steps += traj["episode_metadata"]["trajectory_length"].numpy()
    print(f"  Verified: {n_episodes} episodes, {total_steps} total steps")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build RLDS datasets for OpenVLA/OFT"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=list(TARGETS.keys()) + ["all"],
        help="Which dataset to build (e2e, planner, correction, or all)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/vla_dataset",
        help="Path to unified episode data directory",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        choices=[5, 10, 15, 30],
        help="Target FPS for downsampling (default: 30, no downsampling)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square image size (default 256 for OFT, 224 for base OpenVLA)",
    )
    args = parser.parse_args()

    if args.target == "all":
        targets = list(TARGETS.keys())
    else:
        targets = [args.target]

    outputs = []
    for t in targets:
        out = build_target(t, args.data_path, args.image_size, args.fps)
        outputs.append((t, out))

    # Summary
    print(f"\n{'=' * 60}")
    print("RLDS Build Summary")
    print(f"{'=' * 60}")
    for name, path in outputs:
        print(f"  {name:12s} -> {path}")
    print()
    print("To transfer to server:")
    for name, path in outputs:
        ds_name, _, _ = TARGETS[name]
        print(f"  rsync -avz {path}/ server:~/tensorflow_datasets/{ds_name}/")
    print()
    print("To verify:")
    for name, path in outputs:
        ds_name, _, _ = TARGETS[name]
        print(f"  python -c \"import tensorflow_datasets as tfds; b = tfds.builder('{ds_name}'); print(b.info)\"")


if __name__ == "__main__":
    main()
