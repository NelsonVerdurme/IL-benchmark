import argparse
import h5py
import numpy as np
import open3d as o3d

def create_lineset_from_bbox(bbox_min, bbox_max, color=[1, 0, 0]):
    """Creates an Open3D LineSet representing a bounding box."""
    points = [
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ]
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def visualize_point_cloud_with_bbox(xyz, rgb, bbox_min, bbox_max, title="Point Cloud with BBox"):
    """Visualizes point cloud with a bounding box."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        if rgb.max() > 1.0 and rgb.max() <= 255.0:
            rgb = rgb / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    bbox_lineset = create_lineset_from_bbox(bbox_min, bbox_max)
    
    print(f"Displaying: {title}. Close the window to continue...")
    o3d.visualization.draw_geometries([pcd, bbox_lineset])

def visualize_point_cloud(xyz, rgb, title="Point Cloud"):
    """Visualizes a single point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        if rgb.max() > 1.0 and rgb.max() <= 255.0: # Check if in 0-255 range
            rgb = rgb / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    print(f"Displaying: {title}. Close the window to continue...")
    o3d.visualization.draw_geometries([pcd])

def filter_points_by_bbox(xyz, rgb, bbox_min, bbox_max):
    """Filters points to keep only those inside the bounding box."""
    if xyz.shape[0] == 0:
        return xyz, rgb
    mask = (
        (xyz[:, 0] >= bbox_min[0]) & (xyz[:, 0] <= bbox_max[0]) &
        (xyz[:, 1] >= bbox_min[1]) & (xyz[:, 1] <= bbox_max[1]) &
        (xyz[:, 2] >= bbox_min[2]) & (xyz[:, 2] <= bbox_max[2])
    )
    filtered_xyz = xyz[mask]
    filtered_rgb = rgb[mask] if rgb is not None and rgb.shape[0] > 0 else None
    return filtered_xyz, filtered_rgb

def voxel_downsample(xyz, rgb, voxel_size):
    """Performs voxel downsampling on a point cloud."""
    if xyz.shape[0] == 0:
        return xyz, rgb
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None and rgb.shape[0] > 0:
        # Ensure rgb is normalized to 0-1 for Open3D colors
        if rgb.max() > 1.0 and rgb.max() <= 255.0:
            rgb_normalized = rgb / 255.0
        elif rgb.min() >= 0.0 and rgb.max() <= 1.0:
            rgb_normalized = rgb
        else: # If format is unexpected, don't set colors
            rgb_normalized = None 
            print("Warning: RGB data not in expected 0-1 or 0-255 range. Not setting colors for downsampling.")

        if rgb_normalized is not None:
             pcd.colors = o3d.utility.Vector3dVector(rgb_normalized)

    down_pcd = pcd.voxel_down_sample(voxel_size)
    
    down_xyz = np.asarray(down_pcd.points)
    down_rgb = None
    if down_pcd.has_colors():
        down_rgb = np.asarray(down_pcd.colors)
        # If original RGB was 0-255, consider converting back, or decide output format
        # For now, keep it as 0-1 as Open3D returns.
        
    return down_xyz, down_rgb

def main(args):
    """
    Loads point cloud data, visualizes with bbox, filters, downsamples, and visualizes again.
    Optionally saves processed data (including modified gripper) to a new HDF5 file if visualization is disabled.
    """
    bbox_min = np.array([args.workspace_bbox[0], args.workspace_bbox[2], args.workspace_bbox[4]])
    bbox_max = np.array([args.workspace_bbox[1], args.workspace_bbox[3], args.workspace_bbox[5]])

    if args.no_visualize and not args.output_h5_file:
        raise ValueError("--output_h5_file is required when --no_visualize is set.")

    try:
        with h5py.File(args.h5_file, 'r') as in_hf:
            input_episodes_group_name = None
            if 'episodes' in in_hf:
                input_episodes_group = in_hf['episodes']
                input_episodes_group_name = 'episodes'
            else:
                input_episodes_group = in_hf

            episode_keys = list(input_episodes_group.keys())
            print(f"Found {len(episode_keys)} episodes: {episode_keys}")

            out_hf = None
            output_episodes_group = None
            if args.no_visualize:
                out_hf = h5py.File(args.output_h5_file, 'w')
                if input_episodes_group_name:
                    output_episodes_group = out_hf.create_group(input_episodes_group_name)
                else:
                    output_episodes_group = out_hf
                print(f"Processing all steps and saving to {args.output_h5_file}")

            for ep_idx, ep_key in enumerate(episode_keys):
                print(f"\nProcessing Episode {ep_idx + 1}/{len(episode_keys)}: {ep_key}")
                episode = input_episodes_group[ep_key]
                
                current_out_ep_group = None
                if args.no_visualize and output_episodes_group is not None:
                    current_out_ep_group = output_episodes_group.create_group(ep_key)

                step_keys = sorted([k for k in episode.keys() if k.startswith('step_')],
                                   key=lambda x: int(x.split('_')[1]))
                
                if not step_keys:
                    print(f"  No steps found in episode {ep_key}")
                    continue

                steps_shown_this_episode = 0
                for step_idx, step_key in enumerate(step_keys):
                    if not args.no_visualize and steps_shown_this_episode >= args.max_steps:
                        print(f"  Reached max steps ({args.max_steps}) to show for episode {ep_key}. Moving to next episode.")
                        break
                    
                    print(f"  Processing Step {step_idx + 1}/{len(step_keys)}: {step_key}")
                    step_data = episode[step_key]
                    
                    print("    Data in current step:")
                    print(f"      Keys: {list(step_data.keys())}")
                    for key, value_item in step_data.items():
                        if key not in ['xyz', 'rgb']: # Already handled or will be
                            if isinstance(value_item, h5py.Dataset):
                                print(f"      {key}: HDF5 Dataset, shape {value_item.shape}, dtype {value_item.dtype}")
                                print(f"        Data: {value_item[...]}") # Print all data for small datasets
                            elif hasattr(value_item, 'shape') and hasattr(value_item, 'dtype'): # Numpy arrays
                                print(f"      {key}: shape {value_item.shape}, dtype {value_item.dtype}")
                            else:
                                print(f"      {key}: {type(value_item)}, value: {str(value_item)[:100]}") # Print first 100 chars for long values


                    if 'xyz' not in step_data:
                        print(f"    'xyz' data not found in {ep_key}/{step_key}. Skipping processing for this step.")
                        if args.no_visualize and current_out_ep_group is not None: # Save other data even if xyz is missing
                            out_step_group = current_out_ep_group.create_group(step_key)
                            for key_to_copy, data_to_copy in step_data.items():
                                if key_to_copy not in ['xyz', 'rgb']:
                                    try:
                                        out_step_group.create_dataset(key_to_copy, data=data_to_copy[...])
                                    except Exception as e_copy:
                                        print(f"      Could not copy key '{key_to_copy}' for {ep_key}/{step_key}: {e_copy}")
                        continue
                    
                    xyz_original = step_data['xyz'][...]
                    rgb_original = None
                    if 'rgb' in step_data:
                        rgb_original = step_data['rgb'][...]
                    else:
                        print(f"    'rgb' data not found in {ep_key}/{step_key}.")

                    if xyz_original.ndim != 2 or xyz_original.shape[1] != 3:
                        print(f"    Unexpected shape for 'xyz' data: {xyz_original.shape}. Skipping processing.")
                        continue
                    if rgb_original is not None and (rgb_original.ndim != 2 or rgb_original.shape[1] != 3 or rgb_original.shape[0] != xyz_original.shape[0]):
                        print(f"    Unexpected shape or mismatched points for 'rgb' data. RGB: {rgb_original.shape}, XYZ: {xyz_original.shape}.")
                        rgb_original = None # Process without color if problematic
                        
                    if not args.no_visualize:
                        print(f"    Original point cloud ({xyz_original.shape[0]} points)")
                        visualize_point_cloud_with_bbox(xyz_original, rgb_original, bbox_min, bbox_max,
                                                        title=f"{ep_key}/{step_key} - Original PC & BBox")

                    xyz_filtered, rgb_filtered = filter_points_by_bbox(xyz_original, rgb_original, bbox_min, bbox_max)
                    print(f"    Filtered point cloud ({xyz_filtered.shape[0]} points)")
                    
                    xyz_downsampled, rgb_downsampled = np.array([]), np.array([]) # Initialize
                    if xyz_filtered.shape[0] > 0:
                        xyz_downsampled, rgb_downsampled = voxel_downsample(xyz_filtered, rgb_filtered, args.voxel_size)
                        print(f"    Downsampled point cloud ({xyz_downsampled.shape[0]} points)")
                    else:
                        print("    No points left after filtering. Skipping downsampling.")
                        xyz_downsampled = xyz_filtered # empty
                        rgb_downsampled = rgb_filtered # None or empty

                    if not args.no_visualize:
                        if xyz_downsampled.shape[0] > 0:
                            visualize_point_cloud(xyz_downsampled, rgb_downsampled,
                                              title=f"{ep_key}/{step_key} - Processed (Filtered & Downsampled)")
                        else:
                            print("    Skipping visualization of processed cloud as it's empty.")
                    
                    # Process gripper data
                    gripper_to_save = None
                    if 'gripper' in step_data:
                        gripper_original = step_data['gripper'][...]
                        if 'joint_states' in step_data:
                            joint_states = step_data['joint_states'][...]
                            if gripper_original.shape == (7,) and joint_states.ndim == 1 and joint_states.size > 0:
                                gripper_8th_val = 1.0 if joint_states[-1] > 0.03 else 0.0
                                gripper_to_save = np.concatenate((gripper_original, [gripper_8th_val])).astype(gripper_original.dtype)
                                print(f"    Modified gripper: last element {gripper_8th_val} based on joint_states[-1]={joint_states[-1]}")
                            else:
                                print("    Warning: Gripper or joint_states data not in expected format for modification. Original gripper will be saved if applicable.")
                                gripper_to_save = gripper_original # Save original
                        else:
                            print("    Warning: joint_states data missing. Cannot modify gripper. Original gripper will be saved if applicable.")
                            gripper_to_save = gripper_original # Save original
                    else:
                        print("    Warning: Original gripper data not found.")

                    if args.no_visualize and current_out_ep_group is not None:
                        out_step_group = current_out_ep_group.create_group(step_key)
                        if xyz_downsampled.shape[0] > 0:
                            out_step_group.create_dataset('xyz', data=xyz_downsampled)
                            if rgb_downsampled is not None and rgb_downsampled.shape[0] > 0:
                                out_step_group.create_dataset('rgb', data=rgb_downsampled)
                        
                        if gripper_to_save is not None:
                            out_step_group.create_dataset('gripper', data=gripper_to_save)
                        
                        # Copy other original data
                        for key_to_copy, data_to_copy in step_data.items():
                            if key_to_copy not in ['xyz', 'rgb', 'gripper']: # Don't copy original raw point cloud
                                try:
                                    out_step_group.create_dataset(key_to_copy, data=data_to_copy[...])
                                except Exception as e_copy:
                                    print(f"      Could not copy key '{key_to_copy}' for {ep_key}/{step_key}: {e_copy}")
                    
                    steps_shown_this_episode += 1
                
                if not args.no_visualize and ep_idx < len(episode_keys) -1 :
                    cont = input("Show next episode? (y/n, default y): ").lower()
                    if cont == 'n':
                        print("Exiting visualization mode.")
                        break
            
            if out_hf:
                out_hf.close()
                print(f"Processed data saved to {args.output_h5_file}")
                        
    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {args.h5_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter, downsample, and visualize/save point clouds from HDF5.")
    parser.add_argument('--h5_file', type=str, default="/home/huser/mini-diffuse-actor/realworld_dataset/close_box/dataset.h5",
                        help="Path to the input HDF5 data file.")
    parser.add_argument('--workspace_bbox', type=float, nargs=6, 
                        default=[0.1, 0.9, -0.35, 0.5, -0.2, 0.7], # table height -0.04m
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),  
                        help="Workspace bounding box [xmin xmax ymin ymax zmin zmax].")
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help="Voxel size for downsampling (meters). Default: 0.01")
    parser.add_argument('--max_steps', type=int, default=3,
                        help="Maximum number of steps to visualize per episode (if visualization is on). Default: 3")
    parser.add_argument('--no_visualize', action='store_true',
                        help="Disable visualization and process all data, saving to output_h5_file.")
    parser.add_argument('--output_h5_file', type=str, default="/home/huser/mini-diffuse-actor/realworld_dataset/close_box/1cm.h5",
                        help="Path to save the processed HDF5 data (used if --no_visualize is active).")
    
    args = parser.parse_args()
    
    main(args)
