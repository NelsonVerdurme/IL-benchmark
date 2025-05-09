#!/usr/bin/env python3

import socket
import msgpack
import msgpack_numpy
import argparse
import time
import numpy as np
import os
import h5py

msgpack_numpy.patch()

def socket_send(sock, data):
    packed = msgpack.packb(data, default=msgpack_numpy.encode)
    sock.sendall(len(packed).to_bytes(4, 'big') + packed)

def load_episode_hdf5(hdf5_path):
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found at {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        episodes = f['episodes']
        first_episode_key = list(episodes.keys())[0]
        first_episode = episodes[first_episode_key]

        gripper_poses = []
        for step_key in first_episode:
            step_group = first_episode[step_key]
            gripper_poses.append(step_group['gripper'][...])

        return gripper_poses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/home/huser/mini-diffuse-actor/realworld_dataset/close_box/dataset.h5', help="Path to the HDF5 dataset file")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Server IP")
    parser.add_argument('--port', type=int, default=5006, help="Server port")
    args = parser.parse_args()

    gripper_poses = load_episode_hdf5(args.dataset_path)
    print(f"[Client] Loaded episode with {len(gripper_poses)} steps from HDF5 dataset.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[Client] Connected to server at {args.host}:{args.port}")

    for i, gripper_pose in enumerate(gripper_poses):
        msg = {'gripper': gripper_pose}
        socket_send(sock, msg)
        print(f"[Client] Sent step {i+1}/{len(gripper_poses)}")
        # after sending each step, press a key to continue
        input("Press Enter to continue...")
        time.sleep(2.0)

    print("[Client] Finished sending all steps.")
    sock.close()

if __name__ == '__main__':
    main()
