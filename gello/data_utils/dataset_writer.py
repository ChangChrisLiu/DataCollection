# gello/data_utils/dataset_writer.py
"""Dataset persistence for the dual-dataset pipeline.

Saves raw episode data as pickle files organized by episode and dataset type.
Each episode generates two sub-directories:
  - vla_skill/  (Dataset 1: VLA+Skill with hijacked gripper)
  - vla_full/   (Dataset 2: Full end-to-end VLA)

Directory structure:
    data/dual_dataset/
      episode_MMDD_HHMMSS/
        vla_skill/
          frame_0000_20240101_120000_000000.pkl
          ...
          episode_meta.json
        vla_full/
          frame_0000_20240101_120000_000000.pkl
          ...
          episode_meta.json
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
        dataset1: List[Dict[str, Any]],
        dataset2: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save both datasets for a single episode.

        Args:
            dataset1: VLA+Skill frames (hijacked gripper).
            dataset2: Full VLA frames (seamless teleop+skill).
            metadata: Episode metadata from DualDatasetBuffer.

        Returns:
            Path to the episode directory.
        """
        dt_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        episode_dir = self.data_dir / f"episode_{dt_str}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[DatasetWriter] Saving episode to {episode_dir}")

        # Dataset 1: VLA+Skill
        meta1 = {**(metadata or {}), "type": "vla_skill"}
        self.save_episode(dataset1, episode_dir, "vla_skill", meta1)

        # Dataset 2: Full VLA
        meta2 = {**(metadata or {}), "type": "vla_full"}
        self.save_episode(dataset2, episode_dir, "vla_full", meta2)

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
