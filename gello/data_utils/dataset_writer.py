# gello/data_utils/dataset_writer.py
"""Dataset persistence for VLA data collection.

Supports two save modes:

1. **Unified episodes** (new): Single frame stream with phase labels.
   Used by the unified recording pipeline with EpisodeBuffer.

       data/vla_dataset/
         episode_MMDD_HHMMSS/
           frame_0000.pkl
           frame_0001.pkl
           ...
           episode_meta.json

2. **Dual episodes** (legacy): Two sub-directories per episode.
   Used by the old DualDatasetBuffer pipeline.

       data/dual_dataset/
         episode_MMDD_HHMMSS/
           vla_planner/
             frame_0000_TIMESTAMP.pkl
             ...
           vla_executor/
             frame_0000_TIMESTAMP.pkl
             ...
"""

import datetime
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetWriter:
    """Write dual-dataset episodes to disk.

    Args:
        data_dir: Base directory for all episodes.
    """

    def __init__(self, data_dir: str = "data/dual_dataset"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_episode(
        self,
        frames: List[Dict[str, Any]],
        episode_dir: Path,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a list of frames as a dataset sub-directory.

        Args:
            frames: List of dicts with "obs" and "action" keys.
            episode_dir: Parent episode directory.
            dataset_name: Sub-directory name ("vla_skill" or "vla_full").
            metadata: Optional metadata to save as episode_meta.json.

        Returns:
            Path to the saved dataset directory.
        """
        ds_dir = episode_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for i, frame in enumerate(frames):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"frame_{i:04d}_{ts}.pkl"
            filepath = ds_dir / filename

            try:
                with open(filepath, "wb") as f:
                    pickle.dump(frame, f)
                saved_count += 1
            except Exception as e:
                print(f"[DatasetWriter] Failed to save {filepath}: {e}")

        # Save metadata
        if metadata is not None:
            meta_path = ds_dir / "episode_meta.json"
            meta_to_save = {
                **metadata,
                "num_frames_saved": saved_count,
                "dataset_name": dataset_name,
                "saved_at": datetime.datetime.now().isoformat(),
            }
            try:
                with open(meta_path, "w") as f:
                    json.dump(meta_to_save, f, indent=2, default=str)
            except Exception as e:
                print(f"[DatasetWriter] Failed to save metadata: {e}")

        print(
            f"[DatasetWriter] Saved {saved_count}/{len(frames)} " f"frames to {ds_dir}"
        )
        return ds_dir

    def save_dual_episode(
        self,
        ds_planner: List[Dict[str, Any]],
        ds_executor: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save both datasets for a single episode.

        Args:
            ds_planner: VLA planner frames (teleop + stop signals).
            ds_executor: VLA executor frames (teleop + skill execution).
            metadata: Episode metadata dict.

        Returns:
            Path to the episode directory.
        """
        dt_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        episode_dir = self.data_dir / f"episode_{dt_str}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[DatasetWriter] Saving episode to {episode_dir}")

        # Dataset A: VLA Planner (teleop + stop signals)
        meta1 = {**(metadata or {}), "type": "vla_planner"}
        self.save_episode(ds_planner, episode_dir, "vla_planner", meta1)

        # Dataset B: VLA Executor (teleop + skill execution)
        meta2 = {**(metadata or {}), "type": "vla_executor"}
        self.save_episode(ds_executor, episode_dir, "vla_executor", meta2)

        return episode_dir

    def save_unified_episode(
        self,
        frames: List[Dict[str, Any]],
        phase_segments: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a unified episode (single frame stream with phase labels).

        Args:
            frames: List of phase-labeled frame dicts from EpisodeBuffer.
            phase_segments: List of segment dicts from EpisodeBuffer.
            metadata: Additional metadata (skill_name, grasp info, etc.).

        Returns:
            Path to the episode directory.
        """
        dt_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        episode_dir = self.data_dir / f"episode_{dt_str}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[DatasetWriter] Saving unified episode to {episode_dir}")

        saved_count = 0
        for i, frame in enumerate(frames):
            filepath = episode_dir / f"frame_{i:04d}.pkl"
            try:
                with open(filepath, "wb") as f:
                    pickle.dump(frame, f)
                saved_count += 1
            except Exception as e:
                print(f"[DatasetWriter] Failed to save {filepath}: {e}")

        # Build episode_meta.json
        meta = {
            **(metadata or {}),
            "phase_segments": phase_segments,
            "num_frames": len(frames),
            "num_frames_saved": saved_count,
            "fps": 30,
            "saved_at": datetime.datetime.now().isoformat(),
        }
        meta_path = episode_dir / "episode_meta.json"
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception as e:
            print(f"[DatasetWriter] Failed to save metadata: {e}")

        print(
            f"[DatasetWriter] Saved {saved_count}/{len(frames)} frames "
            f"({len(phase_segments)} segments)"
        )
        return episode_dir

    def prompt_quality(self, episode_dir: Path) -> None:
        """Ask user whether the recording was successful."""
        print(f"\nEpisode saved to: {episode_dir}")
        while True:
            user_input = (
                input("  Was this demo successful? " "(g = Good / n = Not Good): ")
                .strip()
                .lower()
            )
            if user_input == "n":
                try:
                    failed_path = episode_dir.with_name(episode_dir.name + "_Failed")
                    episode_dir.rename(failed_path)
                    print(f"  Marked as failed: {failed_path.name}")
                except Exception as e:
                    print(f"  Failed to rename: {e}")
                break
            elif user_input == "g":
                print(f"  Data kept: {episode_dir.name}")
                break
            else:
                print("  Invalid input. Please enter 'g' or 'n'.")
