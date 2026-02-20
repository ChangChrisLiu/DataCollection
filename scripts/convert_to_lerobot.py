#!/usr/bin/env python3
# scripts/convert_to_lerobot.py
"""Convert .pkl episodes to LeRobot format for OpenPI (Pi-0 / Pi-0.5).

Supports three targets matching the RLDS conversion:
    e2e        - Full trajectory (all phases)
    planner    - Teleop approach + stop signal
    correction - Recovery after grasp failure + stop signal + near-grasp expansion

Language instruction auto-detected per episode (CPU vs RAM).
Supports downsampling via --fps (5/10/15/30).

IMPORTANT: Run this script from the OpenPI environment to ensure LeRobot format
compatibility (v2.1). The datacollection/tele conda env has LeRobot 0.4.3 (v3.0
format) which produces datasets that OpenPI cannot read.

Usage (from OpenPI environment):
    cd /path/to/DataCollection
    /path/to/openpi/.venv/bin/python scripts/convert_to_lerobot.py \\
        --target planner \\
        --data-dir data/vla_dataset \\
        --repo-id ChangChrisLiu/ur5e_planner \\
        --fps 10

OpenPI action format: ABSOLUTE next-step joint positions.
    action[t] = [joint_positions[t+1], gripper[t+1]/255]
    OpenPI's DeltaActions transform converts absolute -> delta during training.

Output schema (feature keys in LeRobot dataset):
    base_rgb      (256,256,3)   image    Base camera (OAK-D Pro, third-person)
    wrist_rgb     (256,256,3)   image    Wrist camera (RealSense D435i)
    state         (7,)          float32  [q0-q5, gripper/255]
    action        (7,)          float32  [q0_next..q5_next, gripper_next/255]
    task          string                 Language instruction (auto-detected)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversion_utils import (
    detect_skill_from_path,
    discover_episodes,
    downsample_frames,
    extract_near_grasp_frames,
    filter_frames_by_phase,
    get_task_instruction,
    get_trigger_params,
    load_episode_frames,
    load_episode_metadata,
    normalize_gripper,
    print_alignment_report,
    remove_noop_frames,
    synthesize_stop_signals,
    synthesize_trigger_signals,
    validate_episode_alignment,
)


# Target configurations
TARGETS = {
    "e2e": {
        "phases": {"teleop", "skill", "correction", "skill_resume"},
        "stop_signal": False,
        "amplify_trigger": False,
        "default_repo": "ChangChrisLiu/ur5e_e2e",
    },
    "planner": {
        "phases": {"teleop"},
        "stop_signal": True,
        "amplify_trigger": True,
        "default_repo": "ChangChrisLiu/ur5e_planner",
    },
    "correction": {
        "phases": {"correction"},
        "stop_signal": True,
        "amplify_trigger": True,
        "default_repo": "ChangChrisLiu/ur5e_correction",
    },
}


# Feature definitions — images stored as PNG (dtype="image") for OpenPI compatibility.
# OpenPI's LeRobot v2.1 loads dtype="image" as PIL images embedded in parquet.
# dtype="video" would require MP4 files and is NOT compatible with OpenPI's data loader.
FEATURES = {
    "base_rgb": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "wrist_rgb": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "gripper",
            ],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "joint_0",
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "gripper",
            ],
        },
    },
}


def convert_episode_frames(
    frames: list,
    dataset,
    task: str,
) -> None:
    """Add all frames of one episode to the LeRobot dataset."""
    n = len(frames)
    for i, frame in enumerate(frames):
        joints = np.array(frame["joint_positions"][:6], dtype=np.float32)
        gripper = np.float32(normalize_gripper(frame["gripper_pos"]))

        # State: current joint positions + gripper
        state = np.concatenate([joints, [gripper]])

        # Action: absolute next-step joint positions + next gripper
        phase = frame.get("phase", "")
        if phase in ("trigger_signal", "stop_signal"):
            # Trigger/stop: same position + gripper=1.0
            action = np.concatenate([joints, [np.float32(1.0)]])
        elif i < n - 1:
            next_frame = frames[i + 1]
            next_joints = np.array(
                next_frame["joint_positions"][:6], dtype=np.float32
            )
            next_gripper = np.float32(normalize_gripper(next_frame["gripper_pos"]))
            action = np.concatenate([next_joints, [next_gripper]])
        else:
            # Last frame: repeat current (zero delta)
            action = np.concatenate([joints, [gripper]])

        # Images already resized to 256x256 during data collection
        base_rgb = frame.get("base_rgb")
        wrist_rgb = frame.get("wrist_rgb")
        if base_rgb is None:
            base_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        if wrist_rgb is None:
            wrist_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

        dataset.add_frame(
            {
                "base_rgb": base_rgb,
                "wrist_rgb": wrist_rgb,
                "state": state,
                "action": action,
                "task": task,
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert .pkl episodes to LeRobot format for OpenPI"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="e2e",
        choices=list(TARGETS.keys()),
        help="Which dataset target (e2e, planner, correction)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root data directory (e.g. data/vla_dataset)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (default depends on target)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Language instruction override (default: auto-detect per episode from path)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        choices=[5, 10, 15, 30],
        help="Target FPS for downsampling (default: 30, no downsampling)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local output directory (default: ~/.cache/huggingface/lerobot/<repo-id>)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub after conversion",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Skip episodes with alignment warnings",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip alignment validation for speed",
    )
    parser.add_argument(
        "--keep-noops",
        action="store_true",
        help="Keep no-op frames (default: remove them)",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Threads for writing images (default: 4)",
    )
    args = parser.parse_args()

    target_cfg = TARGETS[args.target]
    repo_id = args.repo_id or f"{target_cfg['default_repo']}_{args.fps}hz"

    # Import lerobot — support both v2.1 (OpenPI) and v3.0 (datacollection env)
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # v2.1
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # v3.0

    # Discover episodes (unified format)
    episodes = discover_episodes(args.data_dir)
    if not episodes:
        print(f"No episodes found in {args.data_dir}")
        return

    n_tail, n_repeats = get_trigger_params(args.fps)

    print(f"Found {len(episodes)} episodes")
    print(f"Target:       {args.target}")
    print(f"Phase filter: {target_cfg['phases']}")
    print(f"Stop signal:  {target_cfg['stop_signal']}")
    print(f"Trigger amp:  {target_cfg['amplify_trigger']}")
    print(f"Repo ID:      {repo_id}")
    print(f"FPS:          {args.fps}")
    print(f"Trigger:      n_tail={n_tail}, n_repeats={n_repeats}")
    print(f"Task:         {args.task or 'auto-detect per episode'}")
    if args.target == "correction":
        print(f"Near-grasp:   enabled (correction expansion)")
    print()

    # Validate episodes
    if not args.skip_validation:
        print("Validating episodes...")
        valid_episodes = []
        for ep_path in episodes:
            frames = load_episode_frames(ep_path)
            report = validate_episode_alignment(
                frames, episode_path=str(ep_path)
            )
            print_alignment_report(report)
            if args.strict and not report.passed:
                print(f"  STRICT: skipping {ep_path.name}")
            else:
                valid_episodes.append(ep_path)
        episodes = valid_episodes
        print(f"\n{len(episodes)} episodes will be converted.\n")

    if not episodes:
        print("No episodes to convert.")
        return

    # Create LeRobot dataset — images stored as PNG (not video)
    create_kwargs = {
        "repo_id": repo_id,
        "fps": args.fps,
        "robot_type": "ur5e",
        "features": FEATURES,
        "use_videos": False,
        "image_writer_threads": args.image_writer_threads,
    }
    if args.root:
        create_kwargs["root"] = args.root

    dataset = LeRobotDataset.create(**create_kwargs)

    # Convert episodes
    total_frames = 0
    converted = 0
    skipped = 0
    near_grasp_episodes = 0

    for idx, ep_path in enumerate(episodes):
        frames = load_episode_frames(ep_path)

        # Detect skill and get per-episode language instruction
        try:
            skill_name = detect_skill_from_path(ep_path)
        except ValueError:
            print(f"  WARNING: Cannot detect skill from {ep_path.name}, skipping")
            skipped += 1
            continue
        task = args.task or get_task_instruction(skill_name)

        # Phase filter
        phase_frames = filter_frames_by_phase(frames, target_cfg["phases"])

        if phase_frames:
            # Downsample to target FPS
            phase_frames = downsample_frames(phase_frames, source_fps=30, target_fps=args.fps)

            # Remove no-op frames / trigger signal amplification
            if target_cfg["amplify_trigger"] and not args.keep_noops:
                phase_frames = synthesize_trigger_signals(
                    phase_frames, n_tail=n_tail, n_repeats=n_repeats
                )
            elif not args.keep_noops:
                phase_frames = remove_noop_frames(phase_frames)

            # Append stop signals
            if target_cfg["stop_signal"]:
                phase_frames = synthesize_stop_signals(phase_frames, num_repeats=3)

            if len(phase_frames) >= 2:
                convert_episode_frames(phase_frames, dataset, task)
                dataset.save_episode()
                total_frames += len(phase_frames)
                converted += 1
                print(
                    f"  Episode {idx}: {ep_path.name} [{skill_name}] -> {len(phase_frames)} frames"
                )
            else:
                skipped += 1
        else:
            skipped += 1

        # Near-grasp extraction for correction target
        if args.target == "correction":
            meta = load_episode_metadata(ep_path)
            outcome = meta.get("skill_outcome", "")
            has_correction = "correction" in meta.get("phase_counts", {})
            if outcome == "completed" and not has_correction:
                ng_frames = extract_near_grasp_frames(
                    frames, meta, skill_name, source_fps=30
                )
                if ng_frames:
                    ng_frames = downsample_frames(
                        ng_frames, source_fps=30, target_fps=args.fps
                    )
                    # No cleaning, no stop signals for near-grasp segments
                    if len(ng_frames) >= 2:
                        convert_episode_frames(ng_frames, dataset, task)
                        dataset.save_episode()
                        total_frames += len(ng_frames)
                        near_grasp_episodes += 1
                        print(
                            f"  Near-grasp {idx}: {ep_path.name} [{skill_name}] -> {len(ng_frames)} frames"
                        )

    # Finalize — required for LeRobot v3.0, not available in v2.1
    if hasattr(dataset, "finalize"):
        dataset.finalize()

    # Summary
    print(f"\n{'=' * 60}")
    print("LeRobot Conversion Summary")
    print(f"{'=' * 60}")
    print(f"  Target:             {args.target}")
    print(f"  Repo ID:            {repo_id}")
    print(f"  Episodes converted: {converted}/{len(episodes)}")
    print(f"  Episodes skipped:   {skipped}")
    if near_grasp_episodes:
        print(f"  Near-grasp added:   {near_grasp_episodes}")
    print(f"  Total frames:       {total_frames}")
    print(f"  FPS:                {args.fps}")
    print(f"  Trigger params:     n_tail={n_tail}, n_repeats={n_repeats}")
    print(f"  Image resolution:   256x256")
    print(f"  Image format:       PNG (embedded in parquet)")
    print(f"  State dim:          7 (6 joints + gripper)")
    print(f"  Action dim:         7 (6 joints + gripper, absolute next-step)")
    if args.root:
        print(f"  Local path:         {args.root}")
    print()

    if args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub(tags=["ur5e", "vla", "ewaste"])
        print(f"  Pushed to: https://huggingface.co/datasets/{repo_id}")
    else:
        print("To push to HuggingFace Hub later:")
        print(f"  huggingface-cli upload {repo_id} <local-path>")


if __name__ == "__main__":
    main()
