#!/usr/bin/env python3
# scripts/convert_to_rlds.py
"""Build RLDS TFRecord datasets for OpenVLA / OpenVLA-OFT.

Wrapper script that sets environment variables and invokes `tfds build`
for the selected target dataset. Each target is a separate TFDS
GeneratorBasedBuilder following the kpertsch/rlds_dataset_builder template.

Targets:
    e2e        - Full end-to-end trajectory (all phases)
    planner    - Teleop approach + stop signal
    correction - Recovery after grasp failure + stop signal

Usage:
    # Build planner dataset (default 256x256 images)
    python scripts/convert_to_rlds.py \\
        --target planner \\
        --data-path data/vla_dataset \\
        --task "Pick up the CPU and place it in the socket"

    # Build all targets at once
    python scripts/convert_to_rlds.py \\
        --target all \\
        --data-path data/vla_dataset \\
        --task "Pick up the CPU and place it in the socket"

After building:
    # Datasets are in ~/tensorflow_datasets/ur5e_vla_<target>/1.0.0/
    # Transfer to server:
    rsync -avz ~/tensorflow_datasets/ur5e_vla_planner/ server:~/tensorflow_datasets/ur5e_vla_planner/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TARGETS = {
    "e2e": "ur5e_vla_e2e",
    "planner": "ur5e_vla_planner",
    "correction": "ur5e_vla_correction",
}

SCRIPTS_DIR = Path(__file__).resolve().parent


def build_target(target_key: str, data_path: str, task: str, image_size: int):
    """Build one TFDS dataset target."""
    dataset_name = TARGETS[target_key]
    builder_dir = SCRIPTS_DIR / dataset_name

    if not builder_dir.exists():
        print(f"ERROR: Builder directory not found: {builder_dir}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Building {dataset_name}")
    print(f"  Data path:  {data_path}")
    print(f"  Task:       {task}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"{'=' * 60}\n")

    env = os.environ.copy()
    env["UR5E_DATA_PATH"] = str(Path(data_path).resolve())
    env["UR5E_TASK"] = task
    env["UR5E_IMAGE_SIZE"] = str(image_size)

    cmd = ["tfds", "build", "--overwrite"]
    result = subprocess.run(cmd, cwd=str(builder_dir), env=env)

    if result.returncode != 0:
        print(f"ERROR: tfds build failed for {dataset_name}")
        sys.exit(1)

    output_dir = Path.home() / "tensorflow_datasets" / dataset_name
    print(f"\n  Output: {output_dir}")
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
        "--task",
        type=str,
        required=True,
        help="Language instruction for all episodes",
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
        out = build_target(t, args.data_path, args.task, args.image_size)
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
        ds_name = TARGETS[name]
        print(f"  rsync -avz ~/{path.relative_to(Path.home())}/ server:~/tensorflow_datasets/{ds_name}/")
    print()
    print("To verify:")
    for name, path in outputs:
        ds_name = TARGETS[name]
        print(f"  python -c \"import tensorflow_datasets as tfds; b = tfds.builder('{ds_name}'); print(b.info)\"")


if __name__ == "__main__":
    main()
