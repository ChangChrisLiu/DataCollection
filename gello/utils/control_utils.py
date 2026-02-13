# gello/utils/control_utils.py
"""Shared utilities for robot control loops."""

import datetime
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from gello.agents.agent import Agent
from gello.env import RobotEnv
from gello.data_utils.keyboard_interface import KBReset


DEFAULT_MAX_JOINT_DELTA = 1.0


def angdiff(a, b):
    """Compute angular difference wrapped to [-pi, pi)."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    """Move robot to start position gradually."""
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(angdiff(start_pos, joints))
    id_max_joint_delta = np.argmax(abs_deltas)

    if abs_deltas[id_max_joint_delta] > DEFAULT_MAX_JOINT_DELTA:
        id_mask = abs_deltas > DEFAULT_MAX_JOINT_DELTA
        print("\nERROR: Target start position too far from current position!")
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids, abs_deltas[id_mask], start_pos[id_mask], joints[id_mask]
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} > {DEFAULT_MAX_JOINT_DELTA:.3f} , "
                f"leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return False

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(joints), (
        f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
    )

    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = angdiff(command_joints, current_joints)
        max_joint_delta_step = np.abs(delta).max()
        if max_joint_delta_step > max_delta:
            delta = delta / max_joint_delta_step * max_delta
        env.step(current_joints + delta)
    return True


class SaveInterface:
    """Keyboard-driven data saving interface.

    Saves at camera frame rate (~30Hz) only when new frames arrive.
    After stopping, prompts the user to mark the trajectory as Good or Failed.
    """

    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
    ):
        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        self.save_path: Optional[Path] = None
        self.last_saved_wrist_ts = 0.0
        self.last_saved_base_ts = 0.0
        self.frame_count = 0

        print("Save interface enabled. Use keyboard controls:")
        print("  S: Start recording (saves at ~30Hz on new camera frames)")
        print("  Q: Stop recording (then prompts Good/Not Good)")

    def _prompt_quality(self, finished_path: Path, finished_frame_count: int) -> None:
        """Ask user whether the recording was good or failed."""
        print(f"\nStopped recording. Saved {finished_frame_count} frames to {finished_path}")
        while True:
            user_input = input("  Was this demo successful? (g = Good / n = Not Good): ").strip().lower()
            if user_input == "n":
                try:
                    failed_path = finished_path.with_name(finished_path.name + "_Failed")
                    finished_path.rename(failed_path)
                    print(f"  Marked as failed: {failed_path.name}")
                except Exception as e:
                    print(f"  Failed to rename folder: {e}")
                break
            elif user_input == "g":
                print(f"  Data kept: {finished_path.name}")
                break
            else:
                print("  Invalid input. Please enter 'g' or 'n'.")

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """Update the save interface. Returns 'quit' to exit the control loop."""
        dt = datetime.datetime.now()
        state = self.kb_interface.update()

        if state == "start":
            if self.save_path is not None:
                print("\nWARNING: 'S' pressed while already recording. Previous data may be unlabeled.")
                self.save_path = None

            dt_time = datetime.datetime.now()
            self.save_path = (
                self.data_dir / self.agent_name / dt_time.strftime("%m%d_%H%M%S")
            )
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"\nRecording started -> {self.save_path}")
            self.last_saved_wrist_ts = 0.0
            self.last_saved_base_ts = 0.0
            self.frame_count = 0

        elif state == "save":
            if self.save_path is not None:
                current_wrist_ts = obs.get("wrist_timestamp", 0.0)
                current_base_ts = obs.get("base_timestamp", 0.0)
                has_new_wrist = current_wrist_ts > 0 and current_wrist_ts > self.last_saved_wrist_ts
                has_new_base = current_base_ts > 0 and current_base_ts > self.last_saved_base_ts
                has_new_frame = has_new_wrist or has_new_base

                if has_new_frame:
                    filename = f"frame_{self.frame_count:04d}_{dt.strftime('%Y%m%d_%H%M%S_%f')}.pkl"
                    filepath = self.save_path / filename
                    data_to_save = {"obs": obs, "action": action}
                    try:
                        with open(filepath, "wb") as f:
                            pickle.dump(data_to_save, f)
                        if has_new_wrist:
                            self.last_saved_wrist_ts = current_wrist_ts
                        if has_new_base:
                            self.last_saved_base_ts = current_base_ts
                        self.frame_count += 1
                    except Exception as e:
                        print(f"\nFailed to save frame: {filepath}, error: {e}")

        elif state == "normal":
            # Q was pressed — stop recording
            if self.save_path is not None:
                self._prompt_quality(self.save_path, self.frame_count)
                self.save_path = None

        else:
            raise ValueError(f"Invalid keyboard state: {state}")

        return None

    def finish(self) -> None:
        """Call on exit to handle any in-progress recording."""
        if self.save_path is not None:
            self._prompt_quality(self.save_path, self.frame_count)
            self.save_path = None


def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    """Run the main control loop."""
    # Start message
    try:
        from termcolor import colored
        start_msg = colored("\nStart", color="green", attrs=["bold"])
        colors_ok = True
    except ImportError:
        start_msg = "\nStart"
        colors_ok = False
    print(start_msg)

    start_time = time.time()
    last_print_time = time.time()
    loop_count = 0

    try:
        obs = env.get_obs()
    except Exception as e:
        print(f"\nFailed to get initial observation: {e}")
        return

    while True:
        try:
            action = agent.act(obs)

            if save_interface is not None:
                result = save_interface.update(obs, action)
                if result == "quit":
                    break

            obs = env.step(action)
            loop_count += 1

            # Print Hz every second
            current_time = time.time()
            if print_timing and (current_time - last_print_time >= 1.0):
                elapsed = current_time - start_time
                hz = loop_count / (current_time - last_print_time)
                msg = f"\rT:{elapsed:.1f}s | Hz:{hz:.1f}          "
                if colors_ok:
                    print(colored(msg, color="white", attrs=["bold"]), end="", flush=True)
                else:
                    print(msg, end="", flush=True)
                last_print_time = current_time
                loop_count = 0

        except KeyboardInterrupt:
            print("\n\nCtrl+C detected.")
            if save_interface is not None:
                save_interface.finish()
            break
        except Exception as e:
            print(f"\nControl loop error: {e}")
            import traceback
            traceback.print_exc()
            if save_interface is not None:
                save_interface.finish()
            break

    # Stop any velocity motion on exit (safe for joystick/speedL mode)
    try:
        if hasattr(env.robot(), "speed_stop"):
            env.robot().speed_stop()
    except Exception:
        pass

    print("Control loop ended.")
