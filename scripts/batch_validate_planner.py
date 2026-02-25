#!/usr/bin/env python3
"""Batch validation of OpenPI planner stop-signal accuracy across many episodes.

For each episode, tests frames at multiple points through the teleop phase:
  - Early teleop (10%, 25%)    → model should NOT send stop signal
  - Mid teleop (50%)           → model should NOT send stop signal
  - Late teleop (75%, 90%)     → model MAY start sending stop signals
  - Trigger point (last teleop frame) → model SHOULD send stop signal

Reports per-episode and aggregate stop-signal accuracy.

Usage:
    # Start OpenPI server first (in separate terminal):
    cd ~/openpi && uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
        --policy.config pi05_droid_ur5e_planner_lora_10hz \
        --policy.dir checkpoints/pi05_droid_ur5e_planner_lora_10hz/planner_v1/49999

    # Run batch validation:
    python scripts/batch_validate_planner.py
    python scripts/batch_validate_planner.py --n-episodes 20 --port 8000
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np

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

# Test points as fraction of teleop phase
TEST_POINTS = [
    ("early_10%", 0.10),
    ("early_25%", 0.25),
    ("mid_50%", 0.50),
    ("late_75%", 0.75),
    ("late_90%", 0.90),
    ("trigger", 1.00),  # last teleop frame
]


def load_frame(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def frame_to_obs(frame: dict) -> dict:
    joints_6 = np.array(frame["joint_positions"][:6], dtype=np.float64)
    gripper_norm = frame["gripper_pos"] / 255.0
    joints_7 = np.append(joints_6, gripper_norm)
    return {
        "joint_positions": joints_7,
        "gripper_position": np.array([gripper_norm]),
        "base_rgb": frame["base_rgb"],
        "wrist_rgb": frame["wrist_rgb"],
    }


def get_teleop_range(episode_dir: str):
    """Find the frame index range of the teleop phase."""
    frames = sorted(Path(episode_dir).glob("frame_*.pkl"))
    if not frames:
        return frames, -1, -1

    teleop_start = -1
    teleop_end = -1
    for i, fp in enumerate(frames):
        with open(fp, "rb") as f:
            d = pickle.load(f)
        phase = d.get("phase", "unknown")
        if phase == "teleop":
            if teleop_start == -1:
                teleop_start = i
            teleop_end = i

    return frames, teleop_start, teleop_end


def query_model(policy, frame_path, prompt):
    """Send one frame to the model and return stop signal info."""
    frame = load_frame(str(frame_path))
    obs = frame_to_obs(frame)

    joints = obs["joint_positions"][:6]
    gripper = obs["gripper_position"][0]
    state = np.concatenate([joints, [gripper]]).astype(np.float32)

    request = {
        "observation/image": obs["base_rgb"],
        "observation/wrist_image": obs["wrist_rgb"],
        "observation/state": state,
        "prompt": prompt,
    }

    result = policy.infer(request)
    actions = result["actions"]  # (chunk_size, 7)

    gripper_out = actions[:, 6]
    n_stops = int(np.sum(gripper_out > 0.95))
    chunk_size = len(actions)
    max_grip = float(gripper_out.max())
    mean_grip = float(gripper_out.mean())

    # Joint delta: how much the predicted action differs from current joints
    delta_j0 = float(np.linalg.norm(actions[0, :6] - joints))
    delta_j_mean = float(
        np.mean([np.linalg.norm(actions[i, :6] - joints) for i in range(chunk_size)])
    )

    # "has stop" = any action in the chunk has gripper > 0.95
    has_stop = n_stops > 0
    # "strong stop" = majority of chunk is stop
    strong_stop = n_stops > chunk_size // 2

    return {
        "has_stop": has_stop,
        "strong_stop": strong_stop,
        "n_stops": n_stops,
        "chunk_size": chunk_size,
        "max_grip": max_grip,
        "mean_grip": mean_grip,
        "delta_j0": delta_j0,
        "delta_j_mean": delta_j_mean,
    }


def find_episodes(data_dir, task, n_episodes, seed=42):
    """Find and randomly sample episodes for a task."""
    subdir = "CPU_Extraction" if task == "cpu" else "RAM_Extraction"
    base = Path(data_dir) / subdir
    if not base.exists():
        return []
    episodes = sorted([d for d in base.iterdir() if d.is_dir()])
    if len(episodes) <= n_episodes:
        return episodes
    rng = random.Random(seed)
    return sorted(rng.sample(episodes, n_episodes))


def validate_task(policy, episodes, task, verbose=True):
    """Run validation across all episodes for one task.

    Returns per-point aggregated stats.
    """
    prompt = TASK_INSTRUCTIONS[task]
    point_stats = {
        name: {"correct": 0, "total": 0, "stop_count": 0, "delta_j_sum": 0.0}
        for name, _ in TEST_POINTS
    }
    episode_results = []

    for ep_idx, ep_dir in enumerate(episodes):
        frames, teleop_start, teleop_end = get_teleop_range(str(ep_dir))

        if teleop_start < 0 or teleop_end < 0:
            if verbose:
                print(f"  [{ep_idx:2d}] {ep_dir.name}: SKIP (no teleop phase)")
            continue

        teleop_len = teleop_end - teleop_start + 1
        if teleop_len < 10:
            if verbose:
                print(
                    f"  [{ep_idx:2d}] {ep_dir.name}: SKIP (teleop too short: {teleop_len})"
                )
            continue

        ep_result = {"episode": ep_dir.name, "teleop_frames": teleop_len, "points": {}}

        point_summaries = []
        for name, frac in TEST_POINTS:
            if frac >= 1.0:
                idx = teleop_end
            else:
                idx = teleop_start + int(frac * (teleop_end - teleop_start))
            idx = min(idx, len(frames) - 1)

            try:
                info = query_model(policy, frames[idx], prompt)
            except Exception as e:
                point_summaries.append(f"{name}:ERR")
                continue

            ep_result["points"][name] = info

            # Expected behavior:
            #   early/mid (<=50%): should NOT stop
            #   late (75%, 90%): acceptable either way (transition zone)
            #   trigger (100%): SHOULD stop
            expected_no_stop = frac <= 0.50
            expected_stop = frac >= 1.0

            point_stats[name]["total"] += 1
            point_stats[name]["delta_j_sum"] += info["delta_j0"]
            if info["has_stop"]:
                point_stats[name]["stop_count"] += 1

            if expected_no_stop:
                if not info["has_stop"]:
                    point_stats[name]["correct"] += 1
            elif expected_stop:
                if info["has_stop"]:
                    point_stats[name]["correct"] += 1
            else:
                # Transition zone — count as correct either way
                point_stats[name]["correct"] += 1

            stop_str = (
                f"STOP({info['n_stops']}/{info['chunk_size']})"
                if info["has_stop"]
                else "no_stop"
            )
            dj = f"dj={info['delta_j0']:.4f}"
            point_summaries.append(f"{name}:{stop_str},{dj}")

        episode_results.append(ep_result)

        if verbose:
            summary = " | ".join(point_summaries)
            print(f"  [{ep_idx:2d}] {ep_dir.name} ({teleop_len:3d}f): {summary}")

    return point_stats, episode_results


def print_summary(task, point_stats):
    """Print aggregated accuracy table for a task."""
    print(
        f"\n  {'Point':<12} {'Stop%':>7} {'Correct%':>9} {'AvgDeltaJ':>10} {'Expected':>12}   n"
    )
    print(f"  {'-'*62}")
    for name, frac in TEST_POINTS:
        s = point_stats[name]
        if s["total"] == 0:
            continue
        stop_pct = s["stop_count"] / s["total"] * 100
        correct_pct = s["correct"] / s["total"] * 100
        avg_dj = s["delta_j_sum"] / s["total"]

        if frac <= 0.50:
            expected = "no stop"
        elif frac >= 1.0:
            expected = "STOP"
        else:
            expected = "either OK"

        print(
            f"  {name:<12} {stop_pct:6.1f}% {correct_pct:8.1f}%  {avg_dj:9.4f}  {expected:>12}   {s['total']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Batch validate OpenPI planner stop-signal accuracy"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="data/vla_dataset")
    parser.add_argument(
        "--n-episodes", type=int, default=20, help="Episodes per task to test"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--task",
        default="",
        choices=["", "cpu", "ram"],
        help="Test only one task (default: both)",
    )
    args = parser.parse_args()

    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError:
        print(
            "ERROR: openpi-client not installed.\n"
            "Install with: pip install /home/chris/openpi/packages/openpi-client/"
        )
        sys.exit(1)

    print("=" * 75)
    print("  OpenPI Planner — Batch Stop-Signal Validation")
    print("=" * 75)
    print(f"  Server: {args.host}:{args.port}")
    print(f"  Episodes per task: {args.n_episodes}")
    print(f"  Test points: {', '.join(n for n, _ in TEST_POINTS)}")
    print()

    print("Connecting to OpenPI server...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print("Connected.\n")

    tasks = [args.task] if args.task else ["cpu", "ram"]
    all_stats = {}

    for task in tasks:
        episodes = find_episodes(args.data_dir, task, args.n_episodes, args.seed)
        if not episodes:
            print(f"No {task} episodes found in {args.data_dir}")
            continue

        print("-" * 75)
        print(
            f"  {task.upper()} TASK — {len(episodes)} episodes, {len(TEST_POINTS)} points each"
        )
        print(f"  Prompt: {TASK_INSTRUCTIONS[task][:60]}...")
        print("-" * 75)

        point_stats, _ = validate_task(policy, episodes, task)
        all_stats[task] = point_stats
        print_summary(task, point_stats)
        print()

    # Grand summary
    print("=" * 75)
    print("  OVERALL SUMMARY")
    print("=" * 75)

    for task, stats in all_stats.items():
        trigger = stats.get("trigger", {})
        early_points = [
            s
            for (n, f), s in zip(TEST_POINTS, [stats[n] for n, _ in TEST_POINTS])
            if f <= 0.25
        ]
        late_point = trigger

        early_stop_rate = 0.0
        if early_points:
            total_early = sum(s["total"] for s in early_points)
            total_early_stops = sum(s["stop_count"] for s in early_points)
            if total_early > 0:
                early_stop_rate = total_early_stops / total_early * 100

        trigger_stop_rate = 0.0
        if late_point and late_point["total"] > 0:
            trigger_stop_rate = late_point["stop_count"] / late_point["total"] * 100

        print(f"\n  {task.upper()}:")
        print(f"    Early teleop false-stop rate: {early_stop_rate:.1f}%  (want ~0%)")
        print(
            f"    Trigger-point stop rate:      {trigger_stop_rate:.1f}%  (want ~100%)"
        )

    print()


if __name__ == "__main__":
    main()
