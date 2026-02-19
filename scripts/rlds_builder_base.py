#!/usr/bin/env python3
# scripts/rlds_builder_base.py
"""Shared TFDS GeneratorBasedBuilder base for UR5e VLA RLDS datasets.

Each concrete builder (e2e, planner, correction) subclasses this and
only overrides PHASE_FILTER and APPEND_STOP_SIGNAL.

Environment variables (set by convert_to_rlds.py wrapper):
    UR5E_DATA_PATH     - Path to data/vla_dataset/
    UR5E_TASK          - Language instruction string
    UR5E_IMAGE_SIZE    - Square image dimension (default 256)

Build:
    cd scripts/ur5e_vla_planner && tfds build --overwrite
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversion_utils import (
    discover_episodes,
    filter_frames_by_phase,
    load_episode_frames,
    load_episode_metadata,
    normalize_gripper,
    remove_noop_frames,
    resize_rgb_pil,
    rotvec_to_rpy,
    synthesize_stop_signals,
)


class Ur5eVlaBuilderBase(tfds.core.GeneratorBasedBuilder):
    """Base TFDS builder for UR5e VLA datasets.

    Subclasses must set:
        PHASE_FILTER: set of phase names to include (e.g. {"teleop"})
        APPEND_STOP_SIGNAL: whether to append 3 stop-signal frames
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    # Override in subclasses
    PHASE_FILTER: Set[str] = set()
    APPEND_STOP_SIGNAL: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = None  # lazy load

    def _get_embed(self):
        if self._embed is None:
            # Force CPU for USE — avoids CUDA PTX errors on unsupported GPUs
            # (e.g. RTX 5090 compute 12.0 vs TF compiled for <=9.0)
            with tf.device("/cpu:0"):
                self._embed = hub.load(
                    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
                )
        return self._embed

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Base camera RGB (OAK-D Pro).",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Wrist camera RGB (RealSense D435i).",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc="[q0-q5, 0.0_pad, gripper_0to1]",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc="Delta joint positions [dq0-dq5].",
                    ),
                    "action_gripper": tfds.features.Tensor(
                        shape=(1,),
                        dtype=np.float32,
                        doc="Gripper position 0-1 (0=open, 1=close).",
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Discount factor (always 1.0).",
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Reward (1.0 on last step, 0.0 otherwise).",
                    ),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(
                        doc="Language task description.",
                    ),
                    "language_embedding": tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc="Universal Sentence Encoder embedding.",
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(
                        doc="Episode directory path.",
                    ),
                    "episode_id": tfds.features.Scalar(
                        dtype=np.int32,
                        doc="Episode index.",
                    ),
                    "success": tfds.features.Scalar(
                        dtype=np.bool_,
                        doc="Whether episode was successful.",
                    ),
                    "trajectory_length": tfds.features.Scalar(
                        dtype=np.int32,
                        doc="Number of steps in the trajectory.",
                    ),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        data_path = os.environ.get("UR5E_DATA_PATH", "data/vla_dataset")
        return {
            "train": self._generate_examples(data_path),
        }

    def _generate_examples(self, data_path: str):
        task = os.environ.get("UR5E_TASK", "Pick up the component and place it")
        image_size = int(os.environ.get("UR5E_IMAGE_SIZE", "256"))

        # Compute language embedding once (forced to CPU)
        embed_fn = self._get_embed()
        with tf.device("/cpu:0"):
            lang_embedding = embed_fn([task]).numpy()[0].astype(np.float32)

        episodes = discover_episodes(data_path)
        if not episodes:
            print(f"WARNING: No episodes found in {data_path}")
            return

        for ep_idx, ep_path in enumerate(episodes):
            meta = load_episode_metadata(ep_path)
            frames = load_episode_frames(ep_path)

            if not frames:
                continue

            # Phase filtering
            if self.PHASE_FILTER:
                frames = filter_frames_by_phase(frames, self.PHASE_FILTER)

            if not frames:
                continue

            # Remove no-op frames
            frames = remove_noop_frames(frames)

            # Append stop signals if configured
            if self.APPEND_STOP_SIGNAL:
                frames = synthesize_stop_signals(frames, num_repeats=3)

            if len(frames) < 2:
                continue

            # Determine success from metadata
            outcome = meta.get("skill_outcome", "completed")
            quality = meta.get("quality", "good")
            success = outcome in ("completed", "completed_after_correction") and quality != "bad"

            # Build steps
            steps = []
            for i, frame in enumerate(frames):
                is_first = i == 0
                is_last = i == len(frames) - 1

                # State: [q0-q5, 0.0, gripper_01]
                joints = frame["joint_positions"][:6]
                gripper_01 = normalize_gripper(frame["gripper_pos"])
                state = np.array(
                    list(joints) + [0.0, gripper_01],
                    dtype=np.float32,
                )

                # Action: delta joints [dq0-dq5]
                if not is_last:
                    next_f = frames[i + 1]
                    next_joints = next_f["joint_positions"][:6]
                    delta = np.array(next_joints, dtype=np.float64) - np.array(
                        joints, dtype=np.float64
                    )
                    action = delta.astype(np.float32)
                    next_gripper = normalize_gripper(next_f["gripper_pos"])
                else:
                    action = np.zeros(6, dtype=np.float32)
                    next_gripper = gripper_01

                # Action gripper (separate, as per existing UR5e RLDS schema)
                action_gripper = np.array([next_gripper], dtype=np.float32)

                # Images
                base_rgb = frame.get("base_rgb")
                wrist_rgb = frame.get("wrist_rgb")
                if base_rgb is not None:
                    base_rgb = resize_rgb_pil(base_rgb, (image_size, image_size))
                else:
                    base_rgb = np.zeros(
                        (image_size, image_size, 3), dtype=np.uint8
                    )
                if wrist_rgb is not None:
                    wrist_rgb = resize_rgb_pil(wrist_rgb, (image_size, image_size))
                else:
                    wrist_rgb = np.zeros(
                        (image_size, image_size, 3), dtype=np.uint8
                    )

                steps.append({
                    "observation": {
                        "image": base_rgb,
                        "wrist_image": wrist_rgb,
                        "state": state,
                    },
                    "action": action,
                    "action_gripper": action_gripper,
                    "discount": 1.0,
                    "reward": 1.0 if is_last else 0.0,
                    "is_first": is_first,
                    "is_last": is_last,
                    "is_terminal": is_last,
                    "language_instruction": task,
                    "language_embedding": lang_embedding,
                })

            episode = {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(ep_path),
                    "episode_id": np.int32(ep_idx),
                    "success": success,
                    "trajectory_length": np.int32(len(steps)),
                },
            }

            yield ep_idx, episode
