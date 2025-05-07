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
# import open3d for visualization
import open3d as o3d
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
            self, data_dir, instr_embed_file, taskvar_instr_file, taskvar_file,
            num_points=10000, xyz_shift='center', xyz_norm=True, use_height=False,
            rot_type='quat', instr_embed_type='last', all_step_in_batch=True, 
            rm_table=True, rm_robot='none', include_last_step=False, augment_pc=False,
            sample_points_by_distance=False, same_npoints_per_example=False, 
            rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
            pos_type='cont', pos_bins=50, pos_bin_size=0.01, 
            pos_heatmap_type='plain', pos_heatmap_no_robot=False,
            aug_max_rot=45, real_robot=True, h5_filename='1cm.h5', **kwargs
        ):

        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']
        assert pos_type in ['cont', 'disc']
        assert rot_type in ['quat', 'rot6d', 'euler', 'euler_delta', 'euler_disc']
        assert rm_robot in ['none', 'gt', 'box', 'box_keep_gripper']
        
        # Join script folder for task instructions
        if kwargs.get('project_root', None):
            taskvar_instr_file = os.path.join(kwargs['project_root'], taskvar_instr_file)
            taskvar_file = os.path.join(kwargs['project_root'], taskvar_file)
        else:
            taskvar_instr_file = os.path.join(os.path.dirname(__file__), taskvar_instr_file)
            taskvar_file = os.path.join(os.path.dirname(__file__), taskvar_file)
        
        # Load instruction embeddings and task instructions
        self.taskvar_instrs = json.load(open(taskvar_instr_file))
        self.instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
        
        # Load taskvars from JSON file
        with open(taskvar_file, 'r') as f:
            self.taskvars = json.load(f)
        
        self.data_dir = data_dir
        self.h5_filename = h5_filename
        
        # Create data IDs for all episodes and steps across all taskvar subfolders
        self.episode_info = []
        
        for taskvar in self.taskvars:
            taskvar_dir = os.path.join(self.data_dir, taskvar)
            if not os.path.exists(taskvar_dir):
                print(f"Warning: Taskvar directory not found: {taskvar_dir}")
                continue
                
            h5_path = os.path.join(taskvar_dir, self.h5_filename)
            if not os.path.exists(h5_path):
                print(f"Warning: H5 file not found for taskvar '{taskvar}' at {h5_path}")
                continue
                
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    episodes = list(h5f['episodes'].keys()) if 'episodes' in h5f else list(h5f.keys())
                    
                    for ep_id in episodes:
                        episode = h5f['episodes'][ep_id] if 'episodes' in h5f else h5f[ep_id]
                        steps = sorted([k for k in episode.keys() if k.startswith('step_')], 
                                      key=lambda x: int(x.split('_')[1]))
                        
                        if all_step_in_batch:
                            self.episode_info.append({
                                'h5_path': h5_path,
                                'ep_id': ep_id,
                                'steps': steps,
                                'taskvar': taskvar
                            })
                        else:
                            if include_last_step:
                                for step in steps:
                                    self.episode_info.append({
                                        'h5_path': h5_path,
                                        'ep_id': ep_id,
                                        'steps': [step],
                                        'taskvar': taskvar
                                    })
                            else:
                                for step in steps[:-1]:  # Exclude last step
                                    self.episode_info.append({
                                        'h5_path': h5_path,
                                        'ep_id': ep_id,
                                        'steps': [step],
                                        'taskvar': taskvar
                                    })
            except Exception as e:
                print(f"Error loading H5 file {h5_path}: {e}")
        
        if len(self.episode_info) == 0:
            raise RuntimeError(f"No episodes found in {data_dir} with the provided taskvars")
            
        print(f"Loaded {len(self.episode_info)} episodes from {len(self.taskvars)} task variations")
        
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
        
        # Cache for open H5 files
        self.h5_cache = {}

    def __len__(self):
        return len(self.episode_info)
    
    def _get_h5_file(self, h5_path):
        """Get H5 file from cache or open a new one"""
        if h5_path not in self.h5_cache:
            self.h5_cache[h5_path] = h5py.File(h5_path, 'r')
        return self.h5_cache[h5_path]
    
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
        episode_data = self.episode_info[idx]
        h5_path = episode_data['h5_path']
        ep_id = episode_data['ep_id']
        steps = episode_data['steps']
        taskvar = episode_data['taskvar']
        
        h5f = self._get_h5_file(h5_path)
        episode = h5f['episodes'][ep_id] if 'episodes' in h5f else h5f[ep_id]
        
        outs = {
            'data_ids': [], 'pc_fts': [], 'step_ids': [],
            'pc_centroids': [], 'pc_radius': [], 'ee_poses': [], 
            'txt_embeds': [], 'gt_actions': [], 'gt_quaternion': []
        }
        if self.pos_type == 'disc':
            outs['disc_pos_probs'] = []

        # Prepare all step data for this episode
        all_gripper_poses = []
        for step_id in sorted([k for k in episode.keys() if k.startswith('step_')], key=lambda x: int(x.split('_')[1])):
            step = episode[step_id]
            gripper_pose = step['gripper'][...]
            # if it is not 7+1 dim, add gripper state
            if gripper_pose.shape[-1] == 7:
                print(f"Warning: Gripper pose shape is {gripper_pose.shape}, adding gripper state")
                gripper_state = np.array([1.0])  # Default open gripper
                all_gripper_poses.append(np.concatenate([gripper_pose, gripper_state]))
            else:
                all_gripper_poses.append(gripper_pose)
                
        
        all_gripper_poses = np.array(all_gripper_poses)
        gt_rots = self.get_groundtruth_rotations(all_gripper_poses[:, 3:7])
        
        # Randomly select instruction for this task
        if taskvar in self.taskvar_instrs:
            instr = random.choice(self.taskvar_instrs[taskvar])
            instr_embed_key = (taskvar, instr)
            
            # Look for the instruction embedding
            if instr_embed_key in self.instr_embeds:
                instr_embed = self.instr_embeds[instr_embed_key]
            else:
                # Try backup lookup approaches
                alt_key = instr  # Try just the instruction text as key
                if alt_key in self.instr_embeds:
                    instr_embed = self.instr_embeds[alt_key]
                else:
                    print(f"Warning: Could not find embedding for '{instr_embed_key}', using zeros")
                    # Get embedding size from first available embedding
                    embed_size = next(iter(self.instr_embeds.values())).shape[-1]
                    instr_embed = np.zeros((1, embed_size))
        else:
            print(f"Warning: No instructions found for taskvar '{taskvar}'")
            # Get embedding size from first available embedding
            embed_size = next(iter(self.instr_embeds.values())).shape[-1]
            instr_embed = np.zeros((1, embed_size))
            
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
            # if it is not 7+1 dim, add gripper state
            if gripper_pose.shape[-1] == 7:
                gripper_state = np.array([1.0])
                gripper_pose = np.concatenate([gripper_pose, gripper_state])
            else:
                gripper_pose = np.array(gripper_pose)
            
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
            ee_pose = gripper_pose
            
            # Target pose (next step or current if last step)
            if t < len(all_gripper_poses) - 1:
                gt_action = copy.deepcopy(all_gripper_poses[t+1])
            else:
                gt_action = copy.deepcopy(all_gripper_poses[t])
            
            gt_rot = gt_rots[t]

            # Remove background points (table, robot arm)
            if self.rm_table:
                mask = xyz[..., 2] > self.TABLE_HEIGHT
                xyz = xyz[mask]
                rgb = rgb[mask]
                
            # dont remove robot arm for real robot data    
            # if self.rm_robot.startswith('box'):
            #     mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.rm_robot)
            #     xyz = xyz[mask]
            #     rgb = rgb[mask]

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
            
            outs['data_ids'].append(f'{taskvar}/{ep_id}-{step_id}-t{t}')
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            outs['ee_poses'].append(torch.from_numpy(ee_pose).float())
            outs['gt_actions'].append(torch.from_numpy(gt_action).float())
            outs['gt_quaternion'].append(torch.from_numpy(all_gripper_poses[t, 3:7]).float())
            outs['step_ids'].append(t)
        
        return outs
    
    def close(self):
        """Close all open H5 files"""
        for h5f in self.h5_cache.values():
            h5f.close()
        self.h5_cache.clear()
            
    def __del__(self):
        self.close()


def create_sphere_at_pos(pos, radius=0.02, color=[0,0,1]):
    """Creates an Open3D sphere mesh at a given position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(pos)
    return sphere

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/huser/mini-diffuse-actor/realworld_dataset')
    parser.add_argument('--instr_embed_file', type=str, default='/home/huser/mini-diffuse-actor/realworld_dataset/instr_embeds_clip.npy')
    parser.add_argument('--taskvar_instr_file', type=str, default='/home/huser/mini-diffuse-actor/assets/taskvars_instructions_realworld.json')
    parser.add_argument('--taskvar_file', type=str, default='/home/huser/mini-diffuse-actor/assets/taskvars_realworld.json')
    parser.add_argument('--h5_filename', type=str, default='1cm.h5')
    parser.add_argument('--visualize_batch', action='store_true', help="Visualize a batch of data using Open3D")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Create dataset
    dataset = RealworldDataset(
        data_dir=args.data_dir,
        instr_embed_file=args.instr_embed_file,
        taskvar_instr_file=args.taskvar_instr_file,
        taskvar_file=args.taskvar_file,
        h5_filename=args.h5_filename,
        num_points=4096, xyz_norm=True, xyz_shift='center',
        use_height=False, rot_type='euler_delta', 
        instr_embed_type='last', include_last_step=False, # For single step processing
        rm_robot="none", rm_table=True,
        all_step_in_batch=False, # Process single steps per batch item
        same_npoints_per_example=False,
        sample_points_by_distance=True, augment_pc=False,
        rm_pc_outliers=True, real_robot=True
    )
    print(f'Total data samples: {len(dataset)}')
    
    # Test dataloader
    # Ensure batch_size is small if visualizing, e.g., 1 or 2, for clarity.
    # If all_step_in_batch is True, batch_size=1 is recommended for visualization.
    batch_size_for_loader = 1 if args.visualize_batch else 4
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_for_loader, shuffle=True, num_workers=0, # num_workers=0 for easier debugging with Open3D
        collate_fn=ptv3_collate_fn
    )
    print(f'Total batches: {len(dataloader)}')
    
    # Examine one batch
    for i_batch, batch in enumerate(dataloader):
        print(f"\n--- Batch {i_batch + 1} ---")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f'{k}: {v.size()}')
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], str): # data_ids
                 print(f'{k}: list of {len(v)} strings, e.g., {v[0]}')
            else:
                print(f'{k}: {type(v)}')

        if args.visualize_batch:
            print("\nVisualizing samples in the batch (close Open3D window to see the next sample)...")
            
            pc_fts_all = batch['pc_fts'].cpu().numpy()
            pc_btch_offsets = batch['offset'].cpu().numpy()
            
            ee_poses_all = batch['ee_poses'].cpu().numpy()
            gt_actions_all = batch['gt_actions'].cpu().numpy()
            
            data_ids = batch['data_ids']
            print(f"Data IDs: {data_ids}")

            num_samples_in_batch = ee_poses_all.shape[0]

            for i in range(num_samples_in_batch):
                end_idx = pc_btch_offsets[i]
                start_idx = pc_btch_offsets[i-1] if i-1 >= 0 else 0
                
                print(start_idx, end_idx)
                
                sample_pc_fts = pc_fts_all[start_idx:end_idx]
                print("Sample point cloud shape:", sample_pc_fts.shape)
                sample_xyz_normalized = sample_pc_fts[:, :3]
                sample_rgb_normalized = sample_pc_fts[:, 3:6] # Assuming RGB is next
                
                sample_ee_pose_pos_normalized = ee_poses_all[i, :3]
                sample_gt_action_pos_normalized = gt_actions_all[i, :3]

                print(f"\nVisualizing sample {i+1}/{num_samples_in_batch}, ID: {data_ids[i]} (Normalized Space)")
                print(f"  EE Pose (normalized): {sample_ee_pose_pos_normalized}")
                print(f"  GT Action Pos (normalized): {sample_gt_action_pos_normalized}")

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sample_xyz_normalized)
                pcd.colors = o3d.utility.Vector3dVector((sample_rgb_normalized + 1) / 2.0) # Convert RGB from [-1, 1] to [0, 1]
                
                ee_sphere = create_sphere_at_pos(sample_ee_pose_pos_normalized, radius=0.05, color=[0, 0, 1]) # Blue for EE
                gt_sphere = create_sphere_at_pos(sample_gt_action_pos_normalized, radius=0.05, color=[0, 1, 0]) # Green for GT
                
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])

                o3d.visualization.draw_geometries([pcd, ee_sphere, gt_sphere, coord_frame], 
                                                  window_name=f"Sample (Normalized): {data_ids[i]}")
        
        if i_batch >= 0: # Process only the first batch for this example
            break 
            
    dataset.close()
