#!/usr/bin/env python3
"""
Create Windowed Dataset for Training

This script creates sliding window samples from the labeled dataset.
Each sample contains k=8 consecutive timesteps of features.
The label is taken from the final timestep of the window.

Features per timestep:
- observation.state (6 dims): joint positions
- action (6 dims): commanded actions
Total: 12 features × 8 timesteps = 96 input dimensions

Output: windowed_dataset.npz with train/val/test splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Configuration
LABELED_DATA = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/data/labeled_dataset.parquet")
OUTPUT_DIR = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/data")

WINDOW_SIZE = 8  # k = 8 timesteps
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def create_windows(df: pd.DataFrame, window_size: int = 8):
    """
    Create sliding window samples from episode data.
    
    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples,) labels
        episode_ids: (n_samples,) episode index for each sample
    """
    windows_X = []
    windows_y = []
    window_episodes = []
    
    # Process each episode separately (never cross episode boundaries)
    for episode_idx in df["episode_index"].unique():
        ep_data = df[df["episode_index"] == episode_idx].sort_values("frame_index")
        
        if len(ep_data) < window_size:
            print(f"  Skipping episode {episode_idx}: only {len(ep_data)} frames (need {window_size})")
            continue
        
        # Extract features
        states = np.stack(ep_data["observation.state"].values)  # (T, 6)
        actions = np.stack(ep_data["action"].values)  # (T, 6)
        labels = ep_data["label"].values  # (T,)
        
        # Concatenate state and action features
        features = np.concatenate([states, actions], axis=1)  # (T, 12)
        
        # Create sliding windows
        for t in range(window_size - 1, len(ep_data)):
            window_start = t - window_size + 1
            window_end = t + 1
            
            X_window = features[window_start:window_end]  # (window_size, 12)
            y_label = labels[t]  # Label at final timestep
            
            windows_X.append(X_window)
            windows_y.append(y_label)
            window_episodes.append(episode_idx)
    
    X = np.array(windows_X)  # (n_samples, window_size, n_features)
    y = np.array(windows_y)  # (n_samples,)
    episode_ids = np.array(window_episodes)  # (n_samples,)
    
    return X, y, episode_ids


def episode_stratified_split(X, y, episode_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data by episode (not by sample) to prevent leakage.
    Stratify by whether episode contains any UNSAFE labels.
    """
    np.random.seed(seed)
    
    # Get unique episodes and their characteristics
    unique_episodes = np.unique(episode_ids)
    
    # Determine if each episode has any UNSAFE labels
    episode_has_unsafe = {}
    for ep in unique_episodes:
        mask = episode_ids == ep
        episode_has_unsafe[ep] = np.any(y[mask] == 1)
    
    # Separate episodes by type
    unsafe_episodes = [ep for ep in unique_episodes if episode_has_unsafe[ep]]
    safe_episodes = [ep for ep in unique_episodes if not episode_has_unsafe[ep]]
    
    print(f"\n  Episodes with UNSAFE labels: {len(unsafe_episodes)}")
    print(f"  Episodes with only SAFE labels: {len(safe_episodes)}")
    
    # Shuffle
    np.random.shuffle(unsafe_episodes)
    np.random.shuffle(safe_episodes)
    
    # Split each group
    def split_list(lst, train_r, val_r):
        n = len(lst)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]
    
    unsafe_train, unsafe_val, unsafe_test = split_list(unsafe_episodes, train_ratio, val_ratio)
    safe_train, safe_val, safe_test = split_list(safe_episodes, train_ratio, val_ratio)
    
    # Combine
    train_episodes = set(unsafe_train + safe_train)
    val_episodes = set(unsafe_val + safe_val)
    test_episodes = set(unsafe_test + safe_test)
    
    # Create masks
    train_mask = np.array([ep in train_episodes for ep in episode_ids])
    val_mask = np.array([ep in val_episodes for ep in episode_ids])
    test_mask = np.array([ep in test_episodes for ep in episode_ids])
    
    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
        {"train": list(train_episodes), "val": list(val_episodes), "test": list(test_episodes)}
    )


def main():
    print("=" * 60)
    print("CREATING WINDOWED DATASET")
    print("=" * 60)
    print(f"\nWindow size: k = {WINDOW_SIZE} timesteps")
    print(f"Features: state (6) + action (6) = 12 per timestep")
    print(f"Input dimension: {WINDOW_SIZE} × 12 = {WINDOW_SIZE * 12}")
    
    # Load labeled data
    print("\n📂 Loading labeled dataset...")
    df = pd.read_parquet(LABELED_DATA)
    print(f"   Total frames: {len(df)}")
    print(f"   Episodes: {df['episode_index'].nunique()}")
    
    # Create windows
    print("\n🔧 Creating sliding windows...")
    X, y, episode_ids = create_windows(df, WINDOW_SIZE)
    
    print(f"\n   Total windows: {len(X)}")
    print(f"   Window shape: {X.shape}")  # (n_samples, 8, 12)
    print(f"   SAFE windows: {(y == 0).sum()}")
    print(f"   UNSAFE windows: {(y == 1).sum()}")
    
    # Split by episode
    print("\n📊 Splitting by episode (stratified)...")
    X_train, y_train, X_val, y_val, X_test, y_test, split_info = episode_stratified_split(
        X, y, episode_ids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    print(f"\n   Train: {len(X_train)} samples ({len(split_info['train'])} episodes)")
    print(f"   Val:   {len(X_val)} samples ({len(split_info['val'])} episodes)")
    print(f"   Test:  {len(X_test)} samples ({len(split_info['test'])} episodes)")
    
    # Class distribution per split
    print("\n   Class distribution:")
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        n_safe = (y_split == 0).sum()
        n_unsafe = (y_split == 1).sum()
        pct_unsafe = 100 * n_unsafe / len(y_split) if len(y_split) > 0 else 0
        print(f"     {name}: SAFE={n_safe}, UNSAFE={n_unsafe} ({pct_unsafe:.1f}%)")
    
    # Calculate class weight
    class_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\n   Class weight for UNSAFE: {class_weight:.2f}")
    
    # Flatten for MLP input
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"\n   Flattened shape: {X_train_flat.shape}")  # (n_samples, 96)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "windowed_dataset.npz"
    
    np.savez(
        output_path,
        X_train=X_train_flat,
        y_train=y_train,
        X_val=X_val_flat,
        y_val=y_val,
        X_test=X_test_flat,
        y_test=y_test,
        class_weight=np.array([class_weight]),
        window_size=np.array([WINDOW_SIZE]),
        n_features=np.array([12])
    )
    
    # Save split info
    split_path = OUTPUT_DIR / "split_info.json"
    with open(split_path, 'w') as f:
        json.dump({
            "train_episodes": [int(e) for e in split_info["train"]],
            "val_episodes": [int(e) for e in split_info["val"]],
            "test_episodes": [int(e) for e in split_info["test"]],
            "class_weight": float(class_weight),
            "window_size": WINDOW_SIZE,
            "n_features_per_step": 12
        }, f, indent=2)
    
    print(f"\n✅ Saved windowed dataset to: {output_path}")
    print(f"✅ Saved split info to: {split_path}")
    
    return X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test


if __name__ == "__main__":
    main()
