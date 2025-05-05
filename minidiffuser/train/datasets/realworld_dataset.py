import os
import numpy as np
import json
import copy
import random
import h5py
from scipy.special import softmax

import torch
from torch.utils.data import Dataset

from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R

from minidiffuser.train.datasets.common import (
    pad_tensors, gen_seq_masks, random_rotate_z
)
from minidiffuser.configs.rlbench.constants import (
    get_rlbench_labels, get_robot_workspace
)
from minidiffuser.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from minidiffuser.utils.robot_box import RobotBox
from minidiffuser.utils.action_position_utils import get_disc_gt_pos_prob
from minidiffuser.train.datasets.diffusion_policy_dataset import base_collate_fn, ptv3_collate_fn


class RealworldDataset(Dataset):
    def __init__(
            self, h5_data_path, instr_embed_file, taskvar_instr_file,
            num_points=10000, xyz_shift='center', xyz_norm=True, use_height=False,
            rot_type='quat', instr_embed_type='last', all_step_in_batch=True, 
            rm_table=True, rm_robot='none', include_last_step=False, augment_pc=False,
            sample_points_by_distance=False, same_npoints_per_example=False, 
            rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
            pos_type='cont', pos_bins=50, pos_bin_size=0.01, 
            pos_heatmap_type='plain', pos_heatmap_no_robot=False,
            aug_max_rot=45, real_robot=True, **kwargs
        ):

        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']
        assert pos_type in ['cont', 'disc']
        assert rot_type in ['quat', 'rot6d', 'euler', 'euler_delta', 'euler_disc']
        assert rm_robot in ['none', 'gt', 'box', 'box_keep_gripper']
        
        # Join script folder for task instructions
        if kwargs.get('project_root', None):
            taskvar_instr_file = os.path.join(kwargs['project_root'], taskvar_instr_file)
        else:
            taskvar_instr_file = os.path.join(os.path.dirname(__file__), taskvar_instr_file)
        
        # Load instruction embeddings and task instructions
        self.taskvar_instrs = json.load(open(taskvar_instr_file))
        self.instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}

        # Load real-world dataset from HDF5 file
        self.h5_file = h5py.File(h5_data_path, 'r')
        self.episodes = list(self.h5_file['episodes'].keys())
        
        # Create data IDs for all episodes and steps
        self.data_ids = []
        for ep_id in self.episodes:
            episode = self.h5_file['episodes'][ep_id]
            steps = sorted([k for k in episode.keys() if k.startswith('step_')], 
                          key=lambda x: int(x.split('_')[1]))
            
            if all_step_in_batch:
                self.data_ids.append((ep_id, steps))
            else:
                if include_last_step:
                    for step in steps:
                        self.data_ids.append((ep_id, step))
                else:
                    for step in steps[:-1]:  # Exclude last step
                        self.data_ids.append((ep_id, step))
        
        # If no taskvars specified, use a default one for real-world data
        self.default_taskvar = list(self.taskvar_instrs.keys())[0] if self.taskvar_instrs else "pick_up_cup+0"
        
        # Configuration parameters
        self.num_points = num_points
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm
        self.use_height = use_height
        self.pos_type = pos_type
        self.rot_type = rot_type
        self.rm_table = rm_table
        self.rm_robot = rm_robot
        self.all_step_in_batch = all_step_in_batch
        self.include_last_step = include_last_step
        self.augment_pc = augment_pc
        self.aug_max_rot = np.deg2rad(aug_max_rot)
        self.sample_points_by_distance = sample_points_by_distance
        self.rm_pc_outliers = rm_pc_outliers
        self.rm_pc_outliers_neighbors = rm_pc_outliers_neighbors
        self.same_npoints_per_example = same_npoints_per_example
        self.euler_resolution = euler_resolution
        self.pos_bins = pos_bins
        self.pos_bin_size = pos_bin_size
        self.pos_heatmap_type = pos_heatmap_type
        self.pos_heatmap_no_robot = pos_heatmap_no_robot
        self.real_robot = real_robot

        # Real robot workspace parameters
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()

    def __len__(self):
        return len(self.data_ids)
    
    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False
        robot_box = RobotBox(
            arm_links_info, keep_gripper=keep_gripper, 
            env_name='real' if self.real_robot else 'rlbench'
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        robot_point_ids = np.array(list(robot_point_ids))
        mask = np.ones((xyz.shape[0], ), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask
    
    def _rm_pc_outliers(self, xyz, rgb=None, return_idxs=False):
        clf = LocalOutlierFactor(n_neighbors=self.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        if return_idxs:
            return xyz, rgb, idxs
        else:
            return xyz, rgb
    
    def _rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot
    
    def _augment_pc(self, xyz, ee_pose, gt_action, gt_rot, aug_max_rot):
        # rotate around z-axis
        angle = np.random.uniform(-1, 1) * aug_max_rot
        xyz = random_rotate_z(xyz, angle=angle)
        ee_pose[:3] = random_rotate_z(ee_pose[:3], angle=angle)
        gt_action[:3] = random_rotate_z(gt_action[:3], angle=angle)
        ee_pose[3:-1] = self._rotate_gripper(ee_pose[3:-1], angle)
        gt_action[3:-1] = self._rotate_gripper(gt_action[3:-1], angle)
        if self.rot_type == 'quat':
            gt_rot = gt_action[3:-1]
        elif self.rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy() / 180.
        elif self.rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(gt_action[3:-1], self.euler_resolution)
        elif self.rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy()

        # add small noises (+-2mm)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz

        return xyz, ee_pose, gt_action, gt_rot
    
    def get_groundtruth_rotations(self, ee_poses):
        gt_rots = torch.from_numpy(ee_poses)   # quaternions
        if self.rot_type == 'euler':    # [-1, 1]
            gt_rots = self.rotation_transform.quaternion_to_euler(gt_rots[1:]) / 180.
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        elif self.rot_type == 'euler_disc': # 3D
            gt_rots = [quaternion_to_discrete_euler(x, self.euler_resolution) for x in gt_rots[1:]]
            gt_rots = torch.from_numpy(np.stack(gt_rots + gt_rots[-1:]))
        elif self.rot_type == 'euler_delta':
            gt_eulers = self.rotation_transform.quaternion_to_euler(gt_rots)
            gt_rots = (gt_eulers[1:] - gt_eulers[:-1]) % 360
            gt_rots[gt_rots > 180] -= 360
            gt_rots = gt_rots / 180.
            gt_rots = torch.cat([gt_rots, torch.zeros(1, 3)], 0)
        elif self.rot_type == 'rot6d':
            gt_rots = self.rotation_transform.quaternion_to_ortho6d(gt_rots)
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        else:
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        gt_rots = gt_rots.numpy()
        return gt_rots

    def __getitem__(self, idx):
        if self.all_step_in_batch:
            ep_id, steps = self.data_ids[idx]
        else:
            ep_id, step = self.data_ids[idx]
            steps = [step]

        # Select a task variation for instruction
        taskvar = self.default_taskvar

        episode = self.h5_file['episodes'][ep_id]
        
        outs = {
            'data_ids': [], 'pc_fts': [], 'step_ids': [],
            'pc_centroids': [], 'pc_radius': [], 'ee_poses': [], 
            'txt_embeds': [], 'gt_actions': [], 'gt_quaternion': []
        }
        if self.pos_type == 'disc':
            outs['disc_pos_probs'] = []

        # Prepare all step data for this episode
        all_gripper_poses = []
        for step_id in sorted([k for k in episode.keys() if k.startswith('step_')], 
                             key=lambda x: int(x.split('_')[1])):
            step = episode[step_id]
            gripper_pose = step['gripper'][...]
            gripper_state = np.array([1.0])  # Default open gripper
            all_gripper_poses.append(np.concatenate([gripper_pose, gripper_state]))
        
        all_gripper_poses = np.array(all_gripper_poses)
        gt_rots = self.get_groundtruth_rotations(all_gripper_poses[:, 3:7])
            
        for step_id in steps:
            t = int(step_id.split('_')[1]) if isinstance(step_id, str) else int(step_id)
            step = episode[f'step_{t}']
            
            # Get point cloud and gripper data
            xyz = step['xyz'][...]
            rgb = step['rgb'][...]
            
            # Get arm links info for robot box removal
            # For real robot data, we might not have detailed links info
            # Using a simplified structure compatible with RobotBox
            joint_states = step['joint_states'][...]
            gripper_pose = step['gripper'][...]
            gripper_state = np.array([1.0])  # Default open gripper
            
            # Simple bounding box info for robot arm
            # This is a placeholder - you'll need to adapt based on your robot model
            arm_links_info = ({
                'arm_link_0': np.array([-0.1, -0.1, -0.1, 0.1, 0.1, 0.5]),  # Example bbox
                'gripper': np.array([-0.05, -0.05, -0.05, 0.05, 0.05, 0.15])  # Example bbox
            }, {
                'arm_link_0': np.eye(4),  # Example pose matrix
                'gripper': np.eye(4)  # Example pose matrix
            })
            
            # Current end-effector pose
            ee_pose = np.concatenate([gripper_pose, gripper_state])
            
            # Target pose (next step or current if last step)
            if t < len(all_gripper_poses) - 1:
                gt_action = copy.deepcopy(all_gripper_poses[t+1])
            else:
                gt_action = copy.deepcopy(all_gripper_poses[t])
            
            gt_rot = gt_rots[t]

            # Randomly select one instruction
            instr = random.choice(self.taskvar_instrs[taskvar])
            instr_embed = self.instr_embeds[instr]

            # Remove background points (table, robot arm)
            if self.rm_table:
                mask = xyz[..., 2] > self.TABLE_HEIGHT
                xyz = xyz[mask]
                rgb = rgb[mask]
                
            if self.rm_robot.startswith('box'):
                mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.rm_robot)
                xyz = xyz[mask]
                rgb = rgb[mask]

            if self.rm_pc_outliers:
                xyz, rgb = self._rm_pc_outliers(xyz, rgb)

            # Sampling points
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
            height = xyz[:, -1] - self.TABLE_HEIGHT

            if self.pos_heatmap_no_robot:
                robot_box = RobotBox(
                    arm_links_info=arm_links_info,
                    env_name='real' if self.real_robot else 'rlbench'
                )
                robot_point_idxs = np.array(
                    list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1])
                )
            else:
                robot_point_idxs = None

            # Point cloud augmentation
            if self.augment_pc:
                xyz, ee_pose, gt_action, gt_rot = self._augment_pc(
                    xyz, ee_pose, gt_action, gt_rot, self.aug_max_rot
                )

            # Normalize point cloud
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

            gt_action[:3] = (gt_action[:3] - centroid) / radius
            ee_pose[:3] = (ee_pose[:3] - centroid) / radius
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)

            gt_action = np.concatenate([gt_action[:3], gt_rot, gt_action[-1:]], 0)

            # Normalize RGB to [-1, 1]
            if rgb.max() > 1:
                rgb = rgb / 255.
            rgb = rgb * 2 - 1
            
            pc_ft = np.concatenate([xyz, rgb], 1)
            if self.use_height:
                pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

            if self.pos_type == 'disc':
                # (npoints, 3, 100)
                disc_pos_prob = get_disc_gt_pos_prob(
                    xyz, gt_action[:3], pos_bins=self.pos_bins, 
                    pos_bin_size=self.pos_bin_size,
                    heatmap_type=self.pos_heatmap_type,
                    robot_point_idxs=robot_point_idxs
                )
                outs['disc_pos_probs'].append(torch.from_numpy(disc_pos_prob))
            
            outs['data_ids'].append(f'{ep_id}-{step_id}-t{t}')
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            outs['ee_poses'].append(torch.from_numpy(ee_pose).float())
            outs['gt_actions'].append(torch.from_numpy(gt_action).float())
            outs['gt_quaternion'].append(torch.from_numpy(all_gripper_poses[t, 3:7]).float())
            outs['step_ids'].append(t)
        
        return outs
    
    def close(self):
        """Close the HDF5 file"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
            
    def __del__(self):
        self.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_file', type=str, default='realworld_dataset/dataset.h5')
    parser.add_argument('--instr_embed_file', type=str, default='data/gembench/train_dataset/keysteps_bbox_pcd/instr_embeds_clip.npy')
    parser.add_argument('--taskvar_instr_file', type=str, default='assets/taskvars_instructions_new.json')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Create dataset
    dataset = RealworldDataset(
        h5_data_path=args.h5_file,
        instr_embed_file=args.instr_embed_file,
        taskvar_instr_file=args.taskvar_instr_file,
        num_points=4096, xyz_norm=True, xyz_shift='center',
        use_height=False, rot_type='euler_delta', 
        instr_embed_type='last', include_last_step=True,
        rm_robot='box_keep_gripper', rm_table=True,
        all_step_in_batch=True, same_npoints_per_example=False,
        sample_points_by_distance=True, augment_pc=False,
        rm_pc_outliers=True, real_robot=True
    )
    print(f'Total data samples: {len(dataset)}')
    
    # Test dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0, 
        collate_fn=ptv3_collate_fn
    )
    print(f'Total batches: {len(dataloader)}')
    
    # Examine one batch
    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f'{k}: {v.size()}')
        break
    
    dataset.close()
