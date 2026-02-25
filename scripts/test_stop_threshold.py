#!/usr/bin/env python3
"""Test combined stop criterion (delta_j + gripper) across many episodes and thresholds.

Criterion: 3 consecutive actions in a chunk where:
  - delta_j (norm of predicted joints - current joints) < threshold
  - gripper > 0.95

Tests at 6 points through each episode's teleop phase (10%, 25%, 50%, 75%, 90%, trigger).
"""

import argparse
import glob
import os
import pickle
import random

import numpy as np
from openpi_client import websocket_client_policy as wcp

CPU_ROOT = "/home/chris/DataCollection/data/vla_dataset/CPU_Extraction"
RAM_ROOT = "/home/chris/DataCollection/data/vla_dataset/RAM_Extraction"

TASK_INSTRUCTIONS = {"cpu": "pick up the cpu", "ram": "pick up the ram stick"}

THRESHOLDS = [1.5e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3]
N_CONSEC = 3
GRIP_THRESH = 0.95
TEST_FRACS = [0.10, 0.25, 0.50, 0.75, 0.90, 1.0]


def load_episode_frames(ep_dir):
    pkls = sorted(glob.glob(os.path.join(ep_dir, "frame_*.pkl")))
    frames = []
    for p in pkls:
        with open(p, "rb") as f:
            frames.append(pickle.load(f))
    return frames


def get_teleop_frames(frames):
    return [f for f in frames if f.get("phase") == "teleop"]


def pack_obs(frame, prompt):
    """Pack frame into OpenPI server format (observation/* keys)."""
    img_base = frame["base_rgb"]
    img_wrist = frame["wrist_rgb"]
    joints = np.array(frame["joint_positions"][:6], dtype=np.float32)
    grip_val = frame["gripper_pos"]
    if isinstance(grip_val, (list, tuple)):
        grip_val = grip_val[0]
    grip = float(grip_val)
    # Normalize gripper to [0,1] if it's in [0,255] range
    if grip > 1.0:
        grip = grip / 255.0
    state = np.concatenate([joints, [np.float32(grip)]])
    return {
        "observation/image": img_base,
        "observation/wrist_image": img_wrist,
        "observation/state": state,
        "prompt": prompt,
    }


def check_stop_vs_state(actions, current_joints, threshold):
    """Check if N_CONSEC consecutive actions have delta from current state < threshold
    AND gripper > GRIP_THRESH."""
    chunk_size = actions.shape[0]
    if chunk_size < N_CONSEC:
        return False, 0.0, 0.0
    best_delta = float("inf")
    for i in range(chunk_size - N_CONSEC + 1):
        all_ok = True
        max_delta_in_run = 0.0
        for k in range(N_CONSEC):
            a = actions[i + k]
            delta = float(np.linalg.norm(a[:6] - current_joints))
            grip = float(a[6])
            max_delta_in_run = max(max_delta_in_run, delta)
            if grip < GRIP_THRESH or delta > threshold:
                all_ok = False
                break
        if all_ok:
            return True, max_delta_in_run, float(actions[i, 6])
        best_delta = min(best_delta, max_delta_in_run)
    return False, best_delta, float(actions[0, 6])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.set_printoptions(precision=4, suppress=True)

    cpu_dirs = sorted(glob.glob(os.path.join(CPU_ROOT, "episode_cpu_*")))
    ram_dirs = sorted(glob.glob(os.path.join(RAM_ROOT, "episode_ram_*")))
    random.shuffle(cpu_dirs)
    random.shuffle(ram_dirs)
    cpu_dirs = cpu_dirs[: args.n_episodes]
    ram_dirs = ram_dirs[: args.n_episodes]

    print(f"Testing {len(cpu_dirs)} CPU + {len(ram_dirs)} RAM episodes")
    print(f"Thresholds: {THRESHOLDS}")
    print(
        f"Criterion: {N_CONSEC} consec actions with delta_j_state < thresh AND grip > {GRIP_THRESH}"
    )
    print(f"Test points through teleop: {TEST_FRACS}")
    print()

    client = wcp.WebsocketClientPolicy("localhost", args.port)

    # Results structure
    results = {}
    # Also track raw delta_j values for analysis
    raw_deltas = {}

    for task, dirs in [("cpu", cpu_dirs), ("ram", ram_dirs)]:
        prompt = TASK_INSTRUCTIONS[task]
        results[task] = {}
        raw_deltas[task] = {f: [] for f in TEST_FRACS}
        for thresh in THRESHOLDS:
            results[task][thresh] = {
                f: {"correct": 0, "total": 0, "stop": 0} for f in TEST_FRACS
            }

        for ei, ep_dir in enumerate(dirs):
            frames = load_episode_frames(ep_dir)
            teleop = get_teleop_frames(frames)
            if len(teleop) < 20:
                print(
                    f"  SKIP {os.path.basename(ep_dir)}: only {len(teleop)} teleop frames"
                )
                continue

            for frac in TEST_FRACS:
                idx = min(int(frac * (len(teleop) - 1)), len(teleop) - 1)
                frame = teleop[idx]
                obs = pack_obs(frame, prompt)
                result = client.infer(obs)
                actions = np.array(result["actions"])
                current_joints = np.array(
                    frame["joint_positions"][:6], dtype=np.float32
                )

                # Compute min delta across the chunk for analysis
                deltas = [
                    float(np.linalg.norm(actions[i, :6] - current_joints))
                    for i in range(actions.shape[0])
                ]
                min_delta = min(deltas)
                raw_deltas[task][frac].append(min_delta)

                for thresh in THRESHOLDS:
                    is_stop, _, _ = check_stop_vs_state(
                        actions, current_joints, thresh
                    )
                    expected_stop = frac >= 1.0

                    results[task][thresh][frac]["total"] += 1
                    if is_stop:
                        results[task][thresh][frac]["stop"] += 1
                    if is_stop == expected_stop:
                        results[task][thresh][frac]["correct"] += 1

            if (ei + 1) % 10 == 0:
                print(f"  {task.upper()}: {ei+1}/{len(dirs)} episodes done")

        print(f"{task.upper()} complete.\n")

    # Print raw delta_j distribution first
    print("=" * 80)
    print("  RAW delta_j (min across chunk) DISTRIBUTION")
    print("=" * 80)
    for task in ["cpu", "ram"]:
        print(f"\n  {task.upper()}:")
        print(
            f"  {'Point':<10} {'Min':>10} {'P25':>10} {'Median':>10} {'P75':>10} {'Max':>10}   n"
        )
        print(f"  {'-' * 70}")
        for frac in TEST_FRACS:
            vals = raw_deltas[task][frac]
            if not vals:
                continue
            arr = np.array(vals)
            name = f"{int(frac*100)}%" if frac < 1.0 else "TRIGGER"
            print(
                f"  {name:<10} {arr.min():10.5f} {np.percentile(arr, 25):10.5f} "
                f"{np.median(arr):10.5f} {np.percentile(arr, 75):10.5f} {arr.max():10.5f}   {len(vals)}"
            )

    # Print stop detection results per threshold
    for task in ["cpu", "ram"]:
        print(f"\n{'=' * 80}")
        print(
            f"  {task.upper()} TASK — Stop Detection Rate "
            f"({N_CONSEC} consec, delta_j < thresh, grip > {GRIP_THRESH})"
        )
        print("=" * 80)

        header = f"  {'Threshold':<10}"
        for frac in TEST_FRACS:
            name = f"{int(frac*100)}%" if frac < 1.0 else "TRIGGER"
            header += f" {name:>8}"
        print(header)
        print(f"  {'-' * 68}")

        for thresh in THRESHOLDS:
            row = f"  {thresh:<10.1e}"
            for frac in TEST_FRACS:
                s = results[task][thresh][frac]
                if s["total"] > 0:
                    pct = s["stop"] / s["total"] * 100
                    row += f" {pct:7.1f}%"
                else:
                    row += f" {'N/A':>8}"
            print(row)

        print(f"  {'EXPECTED':<10}", end="")
        for frac in TEST_FRACS:
            exp = "  STOP" if frac >= 1.0 else " noStop"
            print(f" {exp:>8}", end="")
        print()

    # Summary: best threshold
    print(f"\n{'=' * 80}")
    print("  THRESHOLD SUMMARY (FalsePositiveRate vs TriggerDetectionRate)")
    print("=" * 80)
    print(
        f"  {'Thresh':<10} {'CPU FP%':>8} {'CPU TP%':>8} {'RAM FP%':>8} {'RAM TP%':>8}  Assessment"
    )
    print(f"  {'-' * 70}")
    for thresh in THRESHOLDS:
        scores = {}
        for task in ["cpu", "ram"]:
            fp = sum(
                results[task][thresh][f]["stop"] for f in TEST_FRACS if f < 1.0
            )
            fp_total = sum(
                results[task][thresh][f]["total"] for f in TEST_FRACS if f < 1.0
            )
            tp = results[task][thresh][1.0]["stop"]
            tp_total = results[task][thresh][1.0]["total"]
            scores[task] = {
                "fpr": fp / fp_total * 100 if fp_total > 0 else 0,
                "tpr": tp / tp_total * 100 if tp_total > 0 else 0,
            }

        # Assessment
        cpu_ok = scores["cpu"]["fpr"] < 5 and scores["cpu"]["tpr"] > 80
        ram_ok = scores["ram"]["fpr"] < 5 and scores["ram"]["tpr"] > 80
        if cpu_ok and ram_ok:
            assessment = "<<< GOOD >>>"
        elif cpu_ok or ram_ok:
            assessment = "partial"
        else:
            assessment = ""

        print(
            f"  {thresh:<10.1e} {scores['cpu']['fpr']:7.1f}% {scores['cpu']['tpr']:7.1f}% "
            f"{scores['ram']['fpr']:7.1f}% {scores['ram']['tpr']:7.1f}%  {assessment}"
        )


if __name__ == "__main__":
    main()
