#!/usr/bin/env python3
"""Patch CPU pkl data: commanded gripper_pos 255 → 230 in teleop/correction phases.

Bug: _get_gripper_pos() recorded commanded gripper value (255 = fully close command)
instead of actual hardware reading (~230 = closed on air for CPU).

This makes CPU teleop gripper_pos identical to stop signals (both 1.0 normalized),
so the model can't distinguish "approaching with gripper closed" from "trigger skill."

Fix: Patch teleop and correction frames (joystick-controlled) where gripper_pos == 255
to 230 (actual hardware reading). Skill and skill_resume frames are left unchanged
since they use automated set_gripper() where commanded == actual.

Stop signals are safe: synthesized during conversion with hardcoded gripper_pos=255.
"""

import argparse
import glob
import os
import pickle
from collections import Counter


ACTUAL_CLOSED_ON_AIR = 230  # Robotiq 2F-85 fully closed on air
PHASES_TO_PATCH = {"teleop", "correction"}


def patch_episode(episode_dir: str, dry_run: bool = False) -> dict:
    """Patch a single episode directory. Returns stats."""
    stats = Counter()
    frames = sorted(glob.glob(os.path.join(episode_dir, "frame_*.pkl")))

    for frame_path in frames:
        with open(frame_path, "rb") as f:
            data = pickle.load(f)

        phase = data.get("phase", "")
        gripper = data.get("gripper_pos")
        stats["total"] += 1

        if phase in PHASES_TO_PATCH and gripper == 255:
            stats["patched"] += 1
            if not dry_run:
                data["gripper_pos"] = ACTUAL_CLOSED_ON_AIR
                with open(frame_path, "wb") as f:
                    pickle.dump(data, f)
        else:
            stats["skipped"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/vla_dataset/CPU_Extraction",
        help="Directory containing CPU episode folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count frames that would be patched without modifying files",
    )
    args = parser.parse_args()

    episodes = sorted(glob.glob(os.path.join(args.data_dir, "episode_*")))
    if not episodes:
        print(f"No episodes found in {args.data_dir}")
        return

    print(f"{'DRY RUN: ' if args.dry_run else ''}Patching {len(episodes)} episodes")
    print(f"  Phases: {PHASES_TO_PATCH}")
    print(f"  gripper_pos 255 → {ACTUAL_CLOSED_ON_AIR}")
    print()

    total_stats = Counter()
    for ep in episodes:
        stats = patch_episode(ep, dry_run=args.dry_run)
        total_stats += stats
        name = os.path.basename(ep)
        if stats["patched"] > 0:
            print(f"  {name}: {stats['patched']}/{stats['total']} frames patched")

    print()
    print(f"Total: {total_stats['patched']} frames patched, "
          f"{total_stats['skipped']} skipped, "
          f"{total_stats['total']} total")


if __name__ == "__main__":
    main()
