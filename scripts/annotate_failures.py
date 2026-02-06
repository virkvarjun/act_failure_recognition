#!/usr/bin/env python3
"""
Failure Annotation Tool

This script helps you annotate the failure timestep for each failure episode.
You will watch each episode video and enter the frame number where failure occurs.

Usage:
    python annotate_failures.py

Output:
    Creates failure_annotations.json with t_failure for each episode
"""

import json
from pathlib import Path
import subprocess
import sys

# Configuration
FAILURE_DATASET_PATH = Path("/Users/arjunvirk/.cache/huggingface/lerobot/a23v/failure_recognition_failure")
OUTPUT_FILE = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/data/failure_annotations.json")
WARNING_WINDOW = 15  # W = 15 steps (1.5 sec at 10 Hz)

# Episode info (pre-computed)
EPISODES = {
    0: {"frames": 196, "duration_s": 19.6},
    1: {"frames": 145, "duration_s": 14.5},
    2: {"frames": 146, "duration_s": 14.6},
    3: {"frames": 179, "duration_s": 17.9},
    4: {"frames": 151, "duration_s": 15.1},
    5: {"frames": 159, "duration_s": 15.9},
    6: {"frames": 139, "duration_s": 13.9},
    7: {"frames": 145, "duration_s": 14.5},
    8: {"frames": 133, "duration_s": 13.3},
    9: {"frames": 146, "duration_s": 14.6},
    10: {"frames": 127, "duration_s": 12.7},
    11: {"frames": 154, "duration_s": 15.4},
    12: {"frames": 206, "duration_s": 20.6},
    13: {"frames": 149, "duration_s": 14.9},
    14: {"frames": 136, "duration_s": 13.6},
}


def get_video_path(episode_idx: int) -> Path:
    """Get the video file path for an episode."""
    return FAILURE_DATASET_PATH / f"videos/observation.images.webcam/chunk-000/file-000.mp4"


def open_video(episode_idx: int):
    """Open video file in system viewer (macOS)."""
    video_path = get_video_path(episode_idx)
    if video_path.exists():
        # For lerobot datasets, videos are concatenated. Print frame range instead.
        start_frame = sum(EPISODES[i]["frames"] for i in range(episode_idx))
        end_frame = start_frame + EPISODES[episode_idx]["frames"] - 1
        print(f"\n📹 Episode {episode_idx} is in the video from frame {start_frame} to {end_frame}")
        print(f"   Duration: {EPISODES[episode_idx]['duration_s']:.1f}s ({EPISODES[episode_idx]['frames']} frames)")
        print(f"   At 10 FPS: starts at {start_frame/10:.1f}s, ends at {end_frame/10:.1f}s in the video")
    else:
        print(f"⚠️  Video not found: {video_path}")


def load_existing_annotations() -> dict:
    """Load existing annotations if any."""
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return {}


def save_annotations(annotations: dict):
    """Save annotations to file."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"✅ Saved annotations to {OUTPUT_FILE}")


def annotate_episode(episode_idx: int, existing_value: int = None) -> int:
    """Get failure timestep annotation for an episode."""
    info = EPISODES[episode_idx]
    
    print(f"\n{'='*50}")
    print(f"Episode {episode_idx}: {info['frames']} frames ({info['duration_s']:.1f}s)")
    print(f"Valid range: 0 to {info['frames'] - 1}")
    
    if existing_value is not None:
        print(f"Current annotation: t_failure = {existing_value}")
    
    open_video(episode_idx)
    
    print(f"\n💡 Enter the frame number where the FAILURE happens (drop/miss/collision)")
    print(f"   - Think about when the task becomes unrecoverable")
    print(f"   - UNSAFE labels will be applied to frames [{max(0, 't_failure - 15')}..t_failure-1]")
    
    while True:
        try:
            user_input = input(f"\nt_failure for episode {episode_idx} (or 's' to skip, 'q' to quit): ").strip()
            
            if user_input.lower() == 's':
                return existing_value  # Keep existing or None
            if user_input.lower() == 'q':
                return -1  # Signal to quit
            
            t_failure = int(user_input)
            
            if t_failure < 0 or t_failure >= info['frames']:
                print(f"❌ Invalid! Must be between 0 and {info['frames'] - 1}")
                continue
            
            # Show what will be labeled
            unsafe_start = max(0, t_failure - WARNING_WINDOW)
            unsafe_end = t_failure - 1
            num_unsafe = max(0, unsafe_end - unsafe_start + 1)
            
            print(f"\n📊 Labels for Episode {episode_idx}:")
            print(f"   SAFE:   frames [0, {unsafe_start - 1}] = {unsafe_start} frames")
            print(f"   UNSAFE: frames [{unsafe_start}, {unsafe_end}] = {num_unsafe} frames")
            print(f"   (frame {t_failure} is the failure point, excluded from training)")
            
            confirm = input("Confirm? (y/n): ").strip().lower()
            if confirm == 'y':
                return t_failure
            
        except ValueError:
            print("❌ Please enter a valid number")


def main():
    print("="*60)
    print("        FAILURE TIMESTEP ANNOTATION TOOL")
    print("="*60)
    print(f"\nDataset: {FAILURE_DATASET_PATH}")
    print(f"Episodes: 15 failure episodes")
    print(f"Warning window: W = {WARNING_WINDOW} frames (1.5s)")
    print("\nFor each episode, enter the frame where failure occurs.")
    print("The 15 frames BEFORE failure will be labeled UNSAFE.")
    
    annotations = load_existing_annotations()
    
    # Show summary of existing annotations
    if annotations:
        print(f"\n📋 Existing annotations: {len(annotations)}/15 episodes")
        for ep, t_f in sorted(annotations.items(), key=lambda x: int(x[0])):
            print(f"   Episode {ep}: t_failure = {t_f}")
    
    print("\n" + "-"*60)
    input("Press ENTER to start annotating...")
    
    for episode_idx in range(15):
        existing = annotations.get(str(episode_idx))
        result = annotate_episode(episode_idx, existing)
        
        if result == -1:  # Quit signal
            print("\n👋 Quitting. Progress saved.")
            save_annotations(annotations)
            return
        
        if result is not None:
            annotations[str(episode_idx)] = result
            save_annotations(annotations)
    
    print("\n" + "="*60)
    print("✅ ALL EPISODES ANNOTATED!")
    print("="*60)
    
    # Final summary
    print("\nFinal Annotations:")
    total_unsafe = 0
    for ep in range(15):
        t_f = annotations.get(str(ep))
        if t_f:
            unsafe_count = min(WARNING_WINDOW, t_f)
            total_unsafe += unsafe_count
            print(f"  Episode {ep:2d}: t_failure = {t_f:3d} → {unsafe_count} UNSAFE frames")
    
    print(f"\nTotal UNSAFE frames: {total_unsafe}")
    print(f"Annotations saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
