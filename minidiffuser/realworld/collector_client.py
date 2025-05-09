#!/usr/bin/env python3
import socket
import msgpack
import msgpack_numpy
import numpy as np
import open3d as o3d
import h5py
import os
from tqdm import tqdm

msgpack_numpy.patch()

def bytes_to_str(obj):
    if isinstance(obj, dict):
        return {k.decode() if isinstance(k, bytes) else k: bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [bytes_to_str(i) for i in obj]
    elif isinstance(obj, bytes):
        return obj.decode()
    else:
        return obj

def recv_msg(sock):
    header = sock.recv(4)
    if not header:
        return None
    msg_len = int.from_bytes(header, 'big')
    chunks = []
    while msg_len > 0:
        chunk = sock.recv(min(4096, msg_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        msg_len -= len(chunk)
    return msgpack.unpackb(b''.join(chunks), object_hook=msgpack_numpy.decode)

def ros_to_open3d_transform(pos, quat):
    # We apply a fixed rotation to align
    from scipy.spatial.transform import Rotation as R

    # First, original gripper rotation from ROS
    print(f"ROS gripper pose: {pos}, {quat}")
    rot_ros = R.from_quat(quat)

    # Then apply a fixed transformation: swap axes

    pos_fixed = np.array([pos[0], pos[1], pos[2]])  # Y -> X, Z -> Y, X -> Z

    return pos_fixed, rot_ros.as_matrix()

def visualize_xyzrgb_gripper(xyz, rgb, gripper_pose):
    if xyz.shape[0] == 0:
        print("[Client] Warning: Empty point cloud!")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.copy())
    rgb = rgb.copy().reshape(-1, 3)
    if rgb.max() > 1:
        rgb = rgb / 255.
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    pos_fixed, rot_fixed = ros_to_open3d_transform(gripper_pose[:3], gripper_pose[3:])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    frame.rotate(rot_fixed, center=(0, 0, 0))
    frame.translate(pos_fixed)





    o3d.visualization.draw_geometries([pcd, frame],)
    
def send_response(sock, data):
    packed = msgpack.packb(data, default=msgpack_numpy.encode)
    sock.sendall(len(packed).to_bytes(4, 'big') + packed)

def main():
    save_dir = 'realworld_dataset'
    os.makedirs(save_dir, exist_ok=True)
    hdf5_file = os.path.join(save_dir, 'dataset.h5')

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 5005))
    print("[Client] Connected to server.")

    episode_data = []
    counter = 0
    eps_counter = 0

    try:
        with h5py.File(hdf5_file, 'w') as f:
            f.create_group('episodes')

            while True:
                msg = recv_msg(sock)
                if msg is None:
                    break
                msg = bytes_to_str(msg)

                if msg['type'] == 'step':
                    print(f"[Client] Received Step {counter}")
                    obs = msg['data']
                    xyz = obs['xyz']
                    rgb = obs['rgb']
                    gripper = obs['gripper']
                    joint_states = obs['joint_states']

                    # Visualize
                    visualize_xyzrgb_gripper(xyz, rgb, gripper)

                    # Save into episode list
                    episode_data.append(obs)

                    # Reply ack
                    send_response(sock, {'status': 'continue'})

                    counter += 1

                elif msg['type'] == 'done':
                    print(f"[Client] Episode finished with {len(episode_data)} steps. Saving...")

                    # Save episode to HDF5
                    episode_group = f['episodes'].create_group(f'episode_{eps_counter}')
                    for step_idx, step_data in enumerate(episode_data):
                        step_group = episode_group.create_group(f'step_{step_idx}')
                        step_group.create_dataset('xyz', data=step_data['xyz'])
                        step_group.create_dataset('rgb', data=step_data['rgb'])
                        step_group.create_dataset('gripper', data=step_data['gripper'])
                        step_group.create_dataset('joint_states', data=step_data['joint_states'])

                    eps_counter += 1
                    print(f"[Client] Saved episode_{eps_counter}")
                    episode_data.clear()

                    # Reply ack
                    send_response(sock, {'status': 'ok'})

                else:
                    print("[Client] Unknown msg type", msg)

    except KeyboardInterrupt:
        print("[Client] Interrupted, closing...")

    sock.close()
    print("[Client] Closed connection.")
    print("[Client] Closed HDF5 file.")

if __name__ == '__main__':
    main()
