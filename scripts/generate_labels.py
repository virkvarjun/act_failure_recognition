#!/usr/bin/env python3
"""
Generate SAFE/UNSAFE Labels

This script generates binary labels for all timesteps in success and failure episodes.
- Success episodes: all SAFE (0)
- Failure episodes: SAFE for t < t_failure - W, UNSAFE for t in [t_failure - W, t_failure - 1]

Output: labeled_dataset.parquet with columns [episode_id, frame_index, label, is_failure_episode]
"""

import json
import pandas as pd
from pathlib import Path

# Configuration
SUCCESS_DATASET = Path("/Users/arjunvirk/.cache/huggingface/lerobot/a23v/failure_recognition_success")
FAILURE_DATASET = Path("/Users/arjunvirk/.cache/huggingface/lerobot/a23v/failure_recognition_failure")
ANNOTATIONS_FILE = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/data/failure_annotations.json")
OUTPUT_DIR = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/data")

WARNING_WINDOW = 15  # W = 15 frames (1.5 sec at 10 Hz)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load parquet data from a lerobot dataset."""
    data_file = path / "data/chunk-000/file-000.parquet"
    return pd.read_parquet(data_file)


def load_annotations() -> dict:
    """Load failure annotations."""
    with open(ANNOTATIONS_FILE) as f:
        return json.load(f)


def generate_labels():
    """Generate SAFE/UNSAFE labels for all frames."""
    print("=" * 60)
    print("GENERATING SAFE/UNSAFE LABELS")
    print("=" * 60)
    
    # Load success dataset
    print("\n📂 Loading success dataset...")
    success_df = load_dataset(SUCCESS_DATASET)
    success_df["is_failure_episode"] = False
    success_df["label"] = 0  # All SAFE
    success_df["original_episode"] = success_df["episode_index"]
    print(f"   Loaded {len(success_df)} frames from {success_df['episode_index'].nunique()} episodes")
    
    # Load failure dataset
    print("\n📂 Loading failure dataset...")
    failure_df = load_dataset(FAILURE_DATASET)
    failure_df["is_failure_episode"] = True
    failure_df["original_episode"] = failure_df["episode_index"]
    
    # Shift episode indices to avoid collision
    max_success_ep = success_df["episode_index"].max()
    failure_df["episode_index"] = failure_df["episode_index"] + max_success_ep + 1
    print(f"   Loaded {len(failure_df)} frames from {failure_df['original_episode'].nunique()} episodes")
    
    # Load annotations and generate labels
    print("\n🏷️  Generating labels for failure episodes...")
    annotations = load_annotations()
    
    # Initialize all failure frames as SAFE
    failure_df["label"] = 0
    
    total_unsafe = 0
    for ep_str, ep_data in annotations["episodes"].items():
        ep_idx = int(ep_str)
        t_failure = ep_data["t_failure"]
        
        # Define UNSAFE window: [t_failure - W, t_failure - 1]
        unsafe_start = max(0, t_failure - WARNING_WINDOW)
        unsafe_end = t_failure - 1
        
        # Mark frames as UNSAFE
        mask = (
            (failure_df["original_episode"] == ep_idx) & 
            (failure_df["frame_index"] >= unsafe_start) & 
            (failure_df["frame_index"] <= unsafe_end)
        )
        failure_df.loc[mask, "label"] = 1
        
        num_unsafe = mask.sum()
        total_unsafe += num_unsafe
        print(f"   Episode {ep_idx}: t_failure={t_failure}, UNSAFE frames [{unsafe_start}, {unsafe_end}] = {num_unsafe}")
    
    # Combine datasets
    print("\n📊 Combining datasets...")
    combined_df = pd.concat([success_df, failure_df], ignore_index=True)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("LABEL SUMMARY")
    print("=" * 60)
    
    n_safe = (combined_df["label"] == 0).sum()
    n_unsafe = (combined_df["label"] == 1).sum()
    
    print(f"\nTotal frames: {len(combined_df)}")
    print(f"  SAFE (0):   {n_safe} ({100*n_safe/len(combined_df):.1f}%)")
    print(f"  UNSAFE (1): {n_unsafe} ({100*n_unsafe/len(combined_df):.1f}%)")
    print(f"\nClass imbalance ratio: {n_safe/n_unsafe:.1f}:1")
    print(f"Suggested class weight for UNSAFE: {n_safe/n_unsafe:.2f}")
    
    # Save to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "labeled_dataset.parquet"
    
    # Select relevant columns
    output_cols = [
        "episode_index", "frame_index", "timestamp",
        "observation.state", "action",
        "label", "is_failure_episode", "original_episode"
    ]
    
    # Filter to available columns
    available_cols = [c for c in output_cols if c in combined_df.columns]
    combined_df[available_cols].to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved labeled dataset to: {output_path}")
    print(f"   Columns: {available_cols}")
    
    return combined_df


if __name__ == "__main__":
    df = generate_labels()
