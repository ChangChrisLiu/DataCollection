#!/usr/bin/env python3
"""Aggregate evaluation results from inference episode metadata.

Parses episode_meta.json files written by run_inference.py and prints
a summary table grouped by (model_type, checkpoint_name, mode, task).

Usage:
    python scripts/eval_summary.py --data-dir data/inference_episodes
    python scripts/eval_summary.py --data-dir data/inference_episodes --csv results.csv
    python scripts/eval_summary.py --data-dir data/inference_episodes --model-type openpi
    python scripts/eval_summary.py --data-dir data/inference_episodes --task cpu
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SUCCESS_OUTCOMES = {"completed", "completed_after_correction", "stop_signal"}


def load_episodes(data_dir: str):
    """Find and load all episode_meta.json files under data_dir."""
    episodes = []
    for meta_path in sorted(Path(data_dir).rglob("episode_meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["_path"] = str(meta_path.parent)
            episodes.append(meta)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {meta_path}: {e}", file=sys.stderr)
    return episodes


def summarize(episodes):
    """Group episodes and compute success rates.

    Returns list of dicts with columns for the summary table.
    """
    groups = defaultdict(list)
    for ep in episodes:
        key = (
            ep.get("model_type", "unknown"),
            ep.get("checkpoint_name", ""),
            ep.get("mode", ""),
            ep.get("skill_name", "unknown"),
        )
        groups[key].append(ep)

    rows = []
    for (model, ckpt, mode, task), eps in sorted(groups.items()):
        n = len(eps)
        outcomes = defaultdict(int)
        for ep in eps:
            outcomes[ep.get("skill_outcome", "unknown")] += 1

        success = sum(outcomes[o] for o in SUCCESS_OUTCOMES)
        rate = (success / n * 100) if n > 0 else 0.0

        rows.append(
            {
                "model": model,
                "checkpoint": ckpt or "(none)",
                "mode": mode or "(none)",
                "task": task,
                "n": n,
                "success": success,
                "rate": rate,
                "timeout": outcomes.get("timeout", 0)
                + outcomes.get("correction_timeout", 0),
                "grasp_fail": outcomes.get("grasp_failed_no_correction", 0),
                "corrected": outcomes.get("completed_after_correction", 0),
                "estop": outcomes.get("estop", 0),
                "interrupted": outcomes.get("interrupted", 0),
            }
        )
    return rows


def print_table(rows):
    """Print a formatted ASCII table."""
    if not rows:
        print("No episodes found.")
        return

    headers = [
        ("Model", "model", 12),
        ("Checkpoint", "checkpoint", 24),
        ("Mode", "mode", 9),
        ("Task", "task", 6),
        ("N", "n", 4),
        ("OK", "success", 4),
        ("Rate", "rate", 7),
        ("Timeout", "timeout", 7),
        ("GraspFail", "grasp_fail", 9),
        ("Corrected", "corrected", 9),
        ("EStop", "estop", 5),
    ]

    # Header line
    header_str = " | ".join(h.ljust(w) for h, _, w in headers)
    print(header_str)
    print("-" * len(header_str))

    for row in rows:
        cells = []
        for _, key, width in headers:
            val = row[key]
            if key == "rate":
                cells.append(f"{val:.1f}%".rjust(width))
            elif isinstance(val, int):
                cells.append(str(val).rjust(width))
            else:
                cells.append(str(val).ljust(width))
        print(" | ".join(cells))


def write_csv(rows, path):
    """Write summary rows to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            row_copy["rate"] = f"{row_copy['rate']:.1f}"
            writer.writerow(row_copy)
    print(f"\nCSV saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate VLA evaluation results")
    parser.add_argument(
        "--data-dir",
        default="data/inference_episodes",
        help="Root directory containing episode folders",
    )
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    parser.add_argument(
        "--model-type", default="", help="Filter by model type (e.g. openpi)"
    )
    parser.add_argument("--task", default="", help="Filter by task (e.g. cpu, ram)")
    args = parser.parse_args()

    episodes = load_episodes(args.data_dir)
    print(f"Found {len(episodes)} episodes in {args.data_dir}\n")

    if args.model_type:
        episodes = [e for e in episodes if e.get("model_type") == args.model_type]
    if args.task:
        episodes = [e for e in episodes if e.get("skill_name") == args.task]

    rows = summarize(episodes)
    print_table(rows)

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
