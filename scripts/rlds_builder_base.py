#!/usr/bin/env python3
# scripts/rlds_builder_base.py
"""Shared TFDS GeneratorBasedBuilder base for UR5e VLA RLDS datasets.

Each concrete builder (e2e, planner, correction) subclasses this and
only overrides PHASE_FILTER, APPEND_STOP_SIGNAL, and EXTRACT_NEAR_GRASP.

State/action format: EEF position + Euler RPY (not joints).
  state:  [x, y, z, roll, pitch, yaw, 0.0_pad, gripper_0to1]  (8D)
  action: [dx, dy, dz, droll, dpitch, dyaw]                    (6D)

Environment variables (set by convert_to_rlds.py wrapper):
    UR5E_DATA_PATH     - Path to data/vla_dataset/
    UR5E_IMAGE_SIZE    - Square image dimension (default 256)
    UR5E_FPS           - Target FPS for downsampling (default 30)

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
    compute_delta_eef_rpy,
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
    remove_noop_frames,
    resize_rgb_pil,
    rotvec_to_rpy,
    synthesize_stop_signals,
    synthesize_trigger_signals,
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
    AMPLIFY_TRIGGER: bool = False
    EXTRACT_NEAR_GRASP: bool = False

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
        image_size = int(os.environ.get("UR5E_IMAGE_SIZE", "256"))
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(image_size, image_size, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Base camera RGB (OAK-D Pro).",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(image_size, image_size, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Wrist camera RGB (RealSense D435i).",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc="[x, y, z, roll, pitch, yaw, 0.0_pad, gripper_0to1]",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc="Delta EEF position [dx, dy, dz, droll, dpitch, dyaw].",
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
        image_size = int(os.environ.get("UR5E_IMAGE_SIZE", "256"))
        fps = int(os.environ.get("UR5E_FPS", "30"))
        n_tail, n_repeats = get_trigger_params(fps)

        # Compute language embeddings per task (cache for the 2 unique strings)
        embed_fn = self._get_embed()
        embedding_cache = {}

        def get_embedding(task_str):
            if task_str not in embedding_cache:
                with tf.device("/cpu:0"):
                    embedding_cache[task_str] = (
                        embed_fn([task_str]).numpy()[0].astype(np.float32)
                    )
            return embedding_cache[task_str]

        episodes = discover_episodes(data_path)
        if not episodes:
            print(f"WARNING: No episodes found in {data_path}")
            return

        ep_idx = 0

        for ep_path in episodes:
            meta = load_episode_metadata(ep_path)
            frames = load_episode_frames(ep_path)

            if not frames:
                continue

            # Detect skill and get per-episode language instruction
            try:
                skill_name = detect_skill_from_path(ep_path)
            except ValueError:
                print(f"WARNING: Cannot detect skill from {ep_path}, skipping")
                continue
            task = get_task_instruction(skill_name)
            lang_embedding = get_embedding(task)

            # Phase filtering
            if self.PHASE_FILTER:
                phase_frames = filter_frames_by_phase(frames, self.PHASE_FILTER)
            else:
                phase_frames = list(frames)

            if phase_frames:
                # Downsample to target FPS
                phase_frames = downsample_frames(phase_frames, source_fps=30, target_fps=fps)

                # Trigger signal amplification (planner/correction only)
                if self.AMPLIFY_TRIGGER:
                    phase_frames = synthesize_trigger_signals(
                        phase_frames, n_tail=n_tail, n_repeats=n_repeats
                    )
                else:
                    phase_frames = remove_noop_frames(phase_frames)

                # Append stop signals if configured
                if self.APPEND_STOP_SIGNAL:
                    phase_frames = synthesize_stop_signals(phase_frames, num_repeats=3)

                if len(phase_frames) >= 2:
                    result = self._build_episode(
                        phase_frames, ep_path, meta, ep_idx,
                        task, lang_embedding, image_size,
                    )
                    if result is not None:
                        yield ep_idx, result
                        ep_idx += 1

            # Near-grasp extraction for correction data expansion
            if self.EXTRACT_NEAR_GRASP:
                outcome = meta.get("skill_outcome", "")
                has_correction = "correction" in meta.get("phase_counts", {})
                # Only extract from successful episodes without correction phase
                if outcome == "completed" and not has_correction:
                    ng_frames = extract_near_grasp_frames(
                        frames, meta, skill_name, source_fps=30
                    )
                    if ng_frames:
                        ng_frames = downsample_frames(
                            ng_frames, source_fps=30, target_fps=fps
                        )
                        # No cleaning, no stop signals for near-grasp segments
                        if len(ng_frames) >= 2:
                            result = self._build_episode(
                                ng_frames, ep_path, meta, ep_idx,
                                task, lang_embedding, image_size,
                            )
                            if result is not None:
                                yield ep_idx, result
                                ep_idx += 1

    def _build_episode(
        self,
        frames: List[Dict[str, Any]],
        ep_path: Path,
        meta: Dict[str, Any],
        ep_idx: int,
        task: str,
        lang_embedding: np.ndarray,
        image_size: int,
    ):
        """Build a single RLDS episode from processed frames."""
        # Determine success from metadata
        outcome = meta.get("skill_outcome", "completed")
        quality = meta.get("quality", "good")
        success = outcome in ("completed", "completed_after_correction") and quality != "bad"

        steps = []
        for i, frame in enumerate(frames):
            is_first = i == 0
            is_last = i == len(frames) - 1

            # State: EEF pose [x, y, z, roll, pitch, yaw, 0.0, gripper_01]
            tcp = frame["tcp_pose"]
            rpy = rotvec_to_rpy(np.array(tcp[3:6]))
            gripper_01 = normalize_gripper(frame["gripper_pos"])
            state = np.array(
                list(tcp[:3]) + list(rpy) + [0.0, gripper_01],
                dtype=np.float32,
            )

            # Action: delta EEF RPY [dx, dy, dz, droll, dpitch, dyaw]
            phase = frame.get("phase", "")
            if phase in ("trigger_signal", "stop_signal"):
                action = np.zeros(6, dtype=np.float32)
                next_gripper = 1.0
            elif not is_last:
                action = compute_delta_eef_rpy(
                    frame["tcp_pose"], frames[i + 1]["tcp_pose"]
                )
                next_gripper = normalize_gripper(frames[i + 1]["gripper_pos"])
            else:
                action = np.zeros(6, dtype=np.float32)
                next_gripper = gripper_01

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

        return {
            "steps": steps,
            "episode_metadata": {
                "file_path": str(ep_path),
                "episode_id": np.int32(ep_idx),
                "success": success,
                "trajectory_length": np.int32(len(steps)),
            },
        }
