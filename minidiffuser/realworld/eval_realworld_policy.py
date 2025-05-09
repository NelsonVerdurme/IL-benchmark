#!/usr/bin/env python3
from typing import Tuple, Dict, List

import os
import json
import jsonlines
import tap
import copy
from pathlib import Path
from filelock import FileLock
import time

import torch
import numpy as np
from scipy.special import softmax

import open3d as o3d
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R

from minidiffuser.train.utils.misc import set_random_seed
from minidiffuser.configs.default import get_config
from omegaconf import OmegaConf

from minidiffuser.train.train_diffusion_policy import MODEL_FACTORY
from minidiffuser.configs.rlbench.constants import get_robot_workspace
from minidiffuser.utils.robot_box import RobotBox
from minidiffuser.train.datasets.common import gen_seq_masks
from minidiffuser.evaluation.common import write_to_file
from minidiffuser.evaluation.eval_simple_policy import Actioner
from minidiffuser.realworld.realworld_env import RealworldEnv

class RealworldArguments(tap.Tap):
    exp_config: str
    device: str = 'cuda'  # cpu, cuda

    seed: int = 100  # seed for reproducibility
    taskvar: str = 'open_box+0'  # task+variation
    checkpoint: str = None

    # Real-world specific arguments
    collector_host: str = '127.0.0.1'
    collector_port: int = 5005
    executer_host: str = '127.0.0.1'
    executer_port: int = 5006
    
    wait_time: float = 0.5  # Wait time between action and next observation
    max_tries: int = 10  # Maximum number of steps per episode
    max_episodes: int = 5  # Maximum number of episodes to run
    
    save_obs_outs_dir: str = None  # Directory to save observations and actions

def main():
    args = RealworldArguments().parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create the actioner (policy)
    actioner = Actioner(args)
    
    # Create output directory for results
    pred_dir = os.path.join(actioner.config.output_dir, 'preds', f'seed{args.seed}', 'realworld')
    os.makedirs(pred_dir, exist_ok=True)
    
    # Setup result file
    outfile = os.path.join(pred_dir, 'results.jsonl')
    
    # Check if this evaluation has already been done
    existed_data = set()
    if os.path.exists(outfile):
        with jsonlines.open(outfile, 'r') as f:
            for item in f:
                existed_data.add((item['checkpoint'], '%s+%d'%(item['task'], item['variation'])))
    
    if (args.checkpoint, args.taskvar) in existed_data:
        print(f"Evaluation for {args.checkpoint} on {args.taskvar} already exists. Skipping.")
        return
    
    # Create real-world environment
    env = RealworldEnv(
        collector_host=args.collector_host,
        collector_port=args.collector_port,
        executer_host=args.executer_host,
        executer_port=args.executer_port,
        wait_time=args.wait_time
    )
    
    # Parse task and variation
    task_str, variation = args.taskvar.split('+')
    variation = int(variation)
    
    # Run the evaluation
    try:
        success_rate = env.evaluate(
            task_str=task_str,
            variation=variation,
            actioner=actioner,
            max_episodes=args.max_episodes,
            num_demos=args.max_episodes,  # Use max_episodes as num_demos
            log_dir=Path(pred_dir),
            max_tries=args.max_tries,
        )
        
        print(f"Testing Success Rate for {task_str}+{variation}: {success_rate:.4f}")
        
        # Save results
        write_to_file(
            outfile,
            {
                'checkpoint': args.checkpoint,
                'task': task_str, 
                'variation': variation,
                'num_demos': args.max_episodes, 
                'sr': success_rate,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure environment is disconnected
        if env.connected:
            env.disconnect()

if __name__ == '__main__':
    main()
