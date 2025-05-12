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

from minidiffuser.train.train_diffusion_realworld import MODEL_FACTORY
from minidiffuser.configs.rlbench.constants import get_robot_workspace
from minidiffuser.utils.robot_box import RobotBox
from minidiffuser.train.datasets.common import gen_seq_masks
from minidiffuser.evaluation.common import write_to_file
from minidiffuser.evaluation.eval_simple_policy import Actioner
from minidiffuser.realworld.realworld_env import RealworldEnv, visualize_pointcloud 

class RealworldArguments(tap.Tap):
    exp_config: str = "/home/huser/mini-diffuse-actor/experiments/logs/mini_close/logs/training_config.yaml"
    device: str = 'cuda'  # cpu, cuda

    seed: int = 100  # seed for reproducibility
    taskvar: str = 'close_box'  # task+variation
    checkpoint: str = "/home/huser/mini-diffuse-actor/experiments/logs/mini_close/ckpts/model_step_100000.pt"

    # Real-world specific arguments
    collector_host: str = '127.0.0.1'
    collector_port: int = 5007
    executer_host: str = '127.0.0.1'
    executer_port: int = 5006
    
    wait_time: float = 0.5  # Wait time between action and next observation
    max_tries: int = 10  # Maximum number of steps per episode
    max_episodes: int = 5  # Maximum number of episodes to run

class RealworldActioner(object):
    def __init__(self, args) -> None:
        self.args = args

        self.WORKSPACE = get_robot_workspace(real_robot=True)
        self.device = torch.device(args.device)

        config = OmegaConf.load(args.exp_config)
        self.config = config

        if args.checkpoint is not None:
            config.checkpoint = args.checkpoint
            
        print(config)

        model_class = MODEL_FACTORY[config.MODEL.model_class]
        self.model = model_class(config.MODEL)
        if config.checkpoint:
            checkpoint = torch.load(
                config.checkpoint, map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        OmegaConf.set_readonly(self.config, True)

        data_cfg = self.config.TRAIN_DATASET
        self.data_cfg = data_cfg
        self.instr_embeds = np.load(data_cfg.instr_embed_file, allow_pickle=True).item()
        if data_cfg.instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
        self.taskvar_instrs = json.load(open(data_cfg.taskvar_instr_file))

        self.TABLE_HEIGHT = self.WORKSPACE['TABLE_HEIGHT']
        
        # Set sample points parameters
        self.num_points = data_cfg.get('num_points', 4096)
        self.sample_points_by_distance = data_cfg.get('sample_points_by_distance', True)
        self.same_npoints_per_example = data_cfg.get('same_npoints_per_example', False)
        self.rm_pc_outliers = data_cfg.get('rm_pc_outliers', True)
        self.rm_pc_outliers_neighbors = data_cfg.get('rm_pc_outliers_neighbors', 25)
        self.xyz_shift = data_cfg.get('xyz_shift', 'center')
        self.xyz_norm = data_cfg.get('xyz_norm', True)
        self.use_height = data_cfg.get('use_height', False)
        self.rm_robot = data_cfg.get('rm_robot', 'none')
        self.rm_table = data_cfg.get('rm_table', True)

    
    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
        raise NotImplementedError("Robot box removal not implemented for real robot data")
    
    def _rm_pc_outliers(self, xyz, rgb=None):
        clf = LocalOutlierFactor(n_neighbors=self.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        return xyz, rgb
    
    def process_point_clouds(
        self, xyz, rgb, gt_sem=None, ee_pose=None, arm_links_info=None, taskvar=None
    ):
        # keep points in robot workspace
        xyz = xyz.reshape(-1, 3)
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                  (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                  (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        
        # Remove points below table if required
        if self.rm_table:
            in_mask = in_mask & (xyz[:, 2] > self.TABLE_HEIGHT)
            
        xyz = xyz[in_mask]
        rgb = rgb.reshape(-1, 3)[in_mask]
        if gt_sem is not None:
            gt_sem = gt_sem.reshape(-1)[in_mask]

        # downsampling - use the same voxel size as in training
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            self.config.MODEL.action_config.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )
        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]
        if gt_sem is not None:
            gt_sem = gt_sem[trace]

        # Remove robot points if requested, never used in our experiments
        if self.rm_robot.startswith('box'):
            mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.rm_robot)
            xyz = xyz[mask]
            rgb = rgb[mask]

        # Remove outliers if requested
        if self.rm_pc_outliers:
            xyz, rgb = self._rm_pc_outliers(xyz, rgb)
            
        # Apply statistical outlier removal for real robot data
        for _ in range(1):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
            xyz = xyz[outlier_masks]
            rgb = rgb[outlier_masks]

        # sampling points - match the training dataset logic exactly
        if len(xyz) > self.num_points:
            if self.sample_points_by_distance:
                dists = np.sqrt(np.sum((xyz - ee_pose[:3])**2, 1))
                probs = 1 / np.maximum(dists, 0.1)
                probs = np.maximum(softmax(probs), 1e-30) 
                probs = probs / sum(probs)
                point_idxs = np.random.choice(len(xyz), self.num_points, replace=False, p=probs)
            else:
                point_idxs = np.random.choice(len(xyz), self.num_points, replace=False)
        else:
            if self.same_npoints_per_example:
                point_idxs = np.random.choice(xyz.shape[0], self.num_points, replace=True)
            else:
                max_npoints = int(len(xyz) * np.random.uniform(0.95, 1))
                point_idxs = np.random.permutation(len(xyz))[:max_npoints]
                
        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        
        # # visualize point cloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd.colors = o3d.utility.Vector3dVector(rgb)
        # pcd.paint_uniform_color([0.5, 0.5, 0.5])
        # o3d.visualization.draw_geometries([pcd])
        
        height = xyz[:, -1] - self.TABLE_HEIGHT

        # normalize - match the dataset normalization approach
        if self.xyz_shift == 'none':
            centroid = np.zeros((3, ))
        elif self.xyz_shift == 'center':
            centroid = np.mean(xyz, 0)
        elif self.xyz_shift == 'gripper':
            centroid = copy.deepcopy(ee_pose[:3])
            
        if self.xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        xyz = (xyz - centroid) / radius
        height = height / radius
        ee_pose[:3] = (ee_pose[:3] - centroid) / radius
        
        # Normalize RGB to [-1, 1] as in dataset
        if rgb.max() > 1:
            rgb = rgb / 255.
        rgb = rgb * 2 - 1
        
        pc_ft = np.concatenate([xyz, rgb], 1)
        if self.use_height:
            pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

        return pc_ft, centroid, radius, ee_pose


    def preprocess_obs(self, taskvar, step_id, obs):
        rgb = np.stack(obs['rgb'], 0)  # (N, H, W, C)
        xyz = np.stack(obs['pc'], 0)  # (N, H, W, C)
        if 'gt_mask' in obs:
            gt_sem = np.stack(obs['gt_mask'], 0)  # (N, H, W) 
        else:
            gt_sem = None
        
        # select one instruction
        instr = self.taskvar_instrs[taskvar][0]
        instr_embed = self.instr_embeds[instr] if instr in self.instr_embeds else None
        
        # Handle case where instruction isn't found
        if instr_embed is None:
            # Try alternate lookup approaches
            alt_key = (taskvar, instr)
            if alt_key in self.instr_embeds:
                instr_embed = self.instr_embeds[alt_key]
            else:
                print(f"Warning: Could not find embedding for '{instr}', using zeros")
                # Get embedding size from first available embedding
                embed_size = next(iter(self.instr_embeds.values())).shape[-1]
                instr_embed = np.zeros((1, embed_size))
        
        # Ensure gripper has 8 dimensions (7 + gripper state)
        gripper_pose = copy.deepcopy(obs['gripper'])
        if gripper_pose.shape[-1] == 7:
            # Add gripper state (open=1.0 by default)
            gripper_state = np.array([1.0])
            gripper_pose = np.concatenate([gripper_pose, gripper_state])
            
        # Process point cloud - consistent with dataset
        pc_ft, pc_centroid, pc_radius, ee_pose = self.process_point_clouds(
            xyz, rgb, gt_sem=gt_sem, ee_pose=gripper_pose, 
            arm_links_info=obs['arm_links_info'], taskvar=taskvar
        )
        
        batch = {
            'pc_fts': torch.from_numpy(pc_ft).float(),
            'pc_centroids': pc_centroid,
            'pc_radius': pc_radius,
            'ee_poses': torch.from_numpy(ee_pose).float().unsqueeze(0),
            'step_ids': torch.LongTensor([step_id]),
            'txt_embeds': torch.from_numpy(instr_embed).float(),
            'txt_lens': [instr_embed.shape[0]],
            'npoints_in_batch': [pc_ft.shape[0]],
            'offset': torch.LongTensor([pc_ft.shape[0]]),
        }
        
        if self.config.MODEL.model_class == 'SimplePolicyPCT':
            batch['pc_fts'] = batch['pc_fts'].unsqueeze(0)
            batch['txt_masks'] = torch.from_numpy(
                gen_seq_masks(batch['txt_lens'])
            ).bool()
            batch['txt_embeds'] = batch['txt_embeds'].unsqueeze(0)
            
        return batch

    def predict(
        self, task_str=None, step_id=None, obs_state_dict=None
    ):
        taskvar = task_str
        batch = self.preprocess_obs(
            taskvar, step_id, obs_state_dict,
        )
        with torch.no_grad():
            actions = []
            for _ in range(getattr(self.args, 'num_ensembles', 1)):
                action = self.model(batch)[0].data.cpu()
                actions.append(action)
            if len(actions) > 1:
                avg_action = torch.stack(actions, 0).mean(0)
                pred_rot = torch.from_numpy(R.from_euler(
                    'xyz', np.mean([R.from_quat(x[3:-1]).as_euler('xyz') for x in actions], 0),
                ).as_quat())
                action = torch.cat([avg_action[:3], pred_rot, avg_action[-1:]], 0)
            else:
                action = actions[0]
        action[-1] = torch.sigmoid(action[-1]) > 0.5
        
        action = action.numpy()
        action[:3] = action[:3] * batch['pc_radius'] + batch['pc_centroids']
        # Ensure the action height is above the table
        action[2] = max(action[2], self.TABLE_HEIGHT+0.005)

        out = {
            'action': action
        }

        return out

def main():
    args = RealworldArguments().parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Create the actioner (policy)
    actioner = RealworldActioner(args)
    
    
    # Create real-world environment
    env = RealworldEnv(
        collector_host=args.collector_host,
        collector_port=args.collector_port,
        executer_host=args.executer_host,
        executer_port=args.executer_port,
        wait_time=args.wait_time
    )
    
    # Parse task and variation
    task_str = args.taskvar
    
    # Run the evaluation
    try:
        with torch.no_grad():
            obs = env.reset(visualize=True)
            step_id = 0
            while True:
                # Reset the environment
                
                # print observation dictionary elements shapes
                for k, v in obs.items():
                    if isinstance(v, list):
                        print(f"{k}: {len(v)}")
                    elif isinstance(v, np.ndarray):
                        print(f"{k}: {v.shape}")
                    else:
                        print(f"{k}: {v}")


                output = actioner.predict(
                        task_str=task_str,
                        step_id=step_id, obs_state_dict=obs)
                action = output["action"]
                step_id += 1
                
                
                if action is None:
                    print("No action predicted, some error happened!")
                    break
                
                print(f"Step {step_id}: Visualizing predicted action before execution...")
                
                visualize_pointcloud(
                    obs['pc'],  # point cloud (N, 3)
                    obs['rgb'],  # RGB (N, 3)
                    gripper_pose=obs['gripper'],  # gripper pose (N, 8)
                    action_pose=action,  # predicted gripper pose (position + quaternion)
                )
                input("Press Enter to execute this action, or Ctrl+C to abort...")
                
                next_obs, reward, done, info = env.step(action, visualize=True)
                obs = next_obs

    
        
        
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
