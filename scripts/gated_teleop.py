#!/usr/bin/env python3
"""
Gated Teleoperation Script

Runs the robot via teleoperation but interrupts with a freeze if 
the failure prediction model triggers an UNSAFE state.
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
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.scripts.lerobot_record import RecordConfig

# Import our gater
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models.failure_gater import FailureGater

@parser.wrap()
def main(cfg: RecordConfig):
    # Setup
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    
    # Load threshold from results if exists
    results_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/models/results.json")
    threshold = 0.29
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            threshold = data.get("optimal_threshold", 0.29)
            if threshold < 0.001: # Fallback if optimal is 0
                threshold = data.get("recall_at_10pct_fpr_threshold", 0.29)
            print(f"Loaded threshold: {threshold:.3f}")

    model_path = Path("/Users/arjunvirk/Desktop/Projects/lerobot/failure_policies_research/models/best_model.pt")
    gater = FailureGater(model_path=model_path, threshold=threshold)
    
    robot.connect()
    teleop.connect()
    listener, events = init_keyboard_listener()
    
    fps = cfg.dataset.fps
    freeze_until = 0
    is_frozen = False
    
    print("\n🚀 GATED TELEOP RUNNING")
    print("Failure prediction active. Watch out for safety freezes!")
    print("Press ESC or Q to quit.")
    
    try:
        while not events["stop_recording"]:
            start_loop_t = time.perf_counter()
            
            # Get data
            obs = robot.get_observation()
            
            # Robust state extraction
            state_key = "observation.state" if "observation.state" in obs else "state"
            if state_key not in obs:
                print(f"❌ Available keys in observation: {list(obs.keys())}")
                raise KeyError(f"Could not find state in robot observation. Available: {list(obs.keys())}")
            
            state = obs[state_key].cpu().numpy().flatten()
            
            # Are we currently in a freeze?
            now = time.time()
            if now < freeze_until:
                if not is_frozen:
                    print("⚠️  SAFETY FREEZE TRIGGERED! ⏸️")
                    is_frozen = True
                # Command current state to freeze
                robot.send_action(obs["observation.state"])
            else:
                if is_frozen:
                    print("✅ Safety reset. Resuming...")
                    is_frozen = False
                    gater.reset()
                
                # Normal teleop
                action = teleop.get_action()
                action_values = torch.cat([v for v in action.values()])
                
                # Check safety gate
                unsafe = gater.update(state, action_values.cpu().numpy())
                
                if unsafe:
                    freeze_until = now + 0.5 # Freeze for 0.5s
                    robot.send_action(obs["observation.state"]) # Immediate freeze
                else:
                    robot.send_action(action_values)
            
            # Sync loop
            dt_s = time.perf_counter() - start_loop_t
            precise_sleep(max(1 / fps - dt_s, 0.0))
            
    finally:
        print("\nStopping...")
        robot.disconnect()
        teleop.disconnect()
        if listener:
            listener.stop()

if __name__ == "__main__":
    main()
