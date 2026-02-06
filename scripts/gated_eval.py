#!/usr/bin/env python3
"""
Gated Autonomous Evaluation Script

Runs a trained control policy (like ACT) but interrupts with a 
safety freeze if the failure prediction model triggers.
"""

import argparse
import logging
import time
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.configs import parser
from lerobot.robots import make_robot_from_config
from lerobot.policies.factory import make_policy
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.scripts.lerobot_record import RecordConfig

# Import our gater
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models.failure_gater import FailureGater

@parser.wrap()
def main(cfg: RecordConfig):
    # 1. Setup Robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # 2. Load Control Policy (ACT)
    # We assume the policy path is provided via command line as --policy.pretrained_path
    if not cfg.policy.pretrained_path:
        # Fallback to the one we are currently training if not specified
        cfg.policy.pretrained_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/lerobot/outputs/train/act_failure_success/checkpoints/last/pretrained_model")
        print(f"Using default policy path: {cfg.policy.pretrained_path}")

    # For safety, let's check if the path exists
    if not Path(cfg.policy.pretrained_path).exists():
        print(f"❌ Policy path not found: {cfg.policy.pretrained_path}")
        print("Please provide it via --policy.pretrained_path <PATH>")
        sys.exit(1)

    print(f"Loading policy from {cfg.policy.pretrained_path}...")
    policy = make_policy(cfg.policy, ds_meta=None)
    policy.eval()
    policy.to("mps" if torch.backends.mps.is_available() else "cpu")

    # 3. Load Safety Gater (Classifier)
    results_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/models/results.json")
    threshold = 0.29
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            threshold = data.get("optimal_threshold", 0.29)
            if threshold < 0.001:
                threshold = data.get("recall_at_10pct_fpr_threshold", 0.29)
            print(f"Loaded safety threshold: {threshold:.3f}")

    model_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/models/best_model.pt")
    gater = FailureGater(model_path=model_path, threshold=threshold)
    
    # 4. Initialize Loop
    listener, events = init_keyboard_listener()
    fps = cfg.dataset.fps
    freeze_until = 0
    is_frozen = False
    
    joint_keys = [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", 
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
    ]
    
    print("\n🚀 GATED AUTONOMOUS EVALUATION RUNNING")
    print("Policy and Safety Guard active. Press ESC or Q to quit.")
    
    try:
        while not events["stop_recording"]:
            start_loop_t = time.perf_counter()
            
            # A. Get Current State
            obs = robot.get_observation()
            
            # Assemble state vector for both Policy and Gater
            state_list = []
            for k in joint_keys:
                val = obs[k]
                if torch.is_tensor(val):
                    state_list.append(val.cpu().item())
                else:
                    state_list.append(float(val))
            state_np = np.array(state_list)
            
            # Prepare state tensor for policy (adding batch and temporal dims)
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).unsqueeze(0).to(policy.device)
            # Create observation dict for policy
            policy_obs = {"observation.state": state_tensor}
            # Add images if policy expects them (simplified: assuming no images for now or handled by make_policy)
            for k in obs:
                if k.startswith("observation.image"):
                    img = obs[k]
                    if not torch.is_tensor(img):
                        img = torch.from_numpy(img)
                    policy_obs[k] = img.float().unsqueeze(0).unsqueeze(0).to(policy.device)

            # B. Get Policy Action
            with torch.no_grad():
                policy_action = policy(policy_obs)
            
            # Extract action values for gating (assuming ACT returns a dict with 'action')
            # and it's [batch, chunk, dim]. We take first chunk step for simplicity.
            raw_action = policy_action["action"][0, 0] # [dim]
            action_np = raw_action.cpu().numpy().flatten()
            
            # Map back to dict for robot.send_action
            action_dict = {k: v for k, v in zip(joint_keys, action_np)}
            
            # C. Safety Gating Logic
            now = time.time()
            if now < freeze_until:
                if not is_frozen:
                    print("⚠️  SAFETY FREEZE TRIGGERED! ⏸️")
                    is_frozen = True
                freeze_action = {k: v for k, v in zip(joint_keys, state_np)}
                robot.send_action(freeze_action)
            else:
                if is_frozen:
                    print("✅ Safety reset. Resuming...")
                    is_frozen = False
                    gater.reset()
                
                # Check safety gate: current state + predicted action
                unsafe = gater.update(state_np, action_np)
                
                if unsafe:
                    print(f"🚨 UNSAFE state detected! counter={gater.alarm_counter}/{gater.hysteresis_count}")
                    freeze_until = now + 1.0 # Freeze for 1.0s safely
                    freeze_action = {k: v for k, v in zip(joint_keys, state_np)}
                    robot.send_action(freeze_action)
                else:
                    robot.send_action(action_dict)
            
            # Sync loop
            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))
            
    finally:
        print("\nStopping...")
        robot.disconnect()
        if listener:
            listener.stop()

if __name__ == "__main__":
    main()
