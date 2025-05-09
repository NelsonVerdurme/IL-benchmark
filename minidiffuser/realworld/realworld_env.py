import socket
import msgpack
import msgpack_numpy
import argparse
import time
import numpy as np
import os
import traceback
import h5py
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Any

msgpack_numpy.patch()

def socket_send(sock, data):
    packed = msgpack.packb(data, default=msgpack_numpy.encode)
    sock.sendall(len(packed).to_bytes(4, 'big') + packed)

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

def bytes_to_str(obj):
    if isinstance(obj, dict):
        return {k.decode() if isinstance(k, bytes) else k: bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [bytes_to_str(i) for i in obj]
    elif isinstance(obj, bytes):
        return obj.decode()
    else:
        return obj

def load_episode_hdf5(hdf5_path):
    """Load gripper poses from an HDF5 file."""
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

def visualize_pointcloud(xyz, rgb, gripper_pose=None):
    """Visualize the point cloud and gripper pose using Open3D."""
    if xyz.shape[0] == 0:
        print("[RealworldEnv] Warning: Empty point cloud!")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.copy().reshape(-1, 3))
    
    # Process RGB values
    rgb_vis = rgb.reshape(-1, 3)
    if rgb_vis.max() > 1:
        rgb_vis = rgb_vis / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rgb_vis)
    
    # Create visualization geometries
    geometries = [pcd]
    
    # Add gripper pose if provided
    if gripper_pose is not None:
        # Create coordinate frame for the gripper
        gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Apply rotation from quaternion
        if len(gripper_pose) >= 7:  # Check if we have quaternion data
            gripper_frame.rotate(
                o3d.geometry.get_rotation_matrix_from_quaternion(gripper_pose[3:7]),
                center=(0, 0, 0)
            )
        
        # Apply translation
        gripper_frame.translate(gripper_pose[:3])
        
        pos_fixed, rot_fixed = ros_to_open3d_transform(gripper_pose[:3], gripper_pose[3:])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        frame.rotate(rot_fixed, center=(0, 0, 0))
        frame.translate(pos_fixed)
        geometries.append(gripper_frame)
        geometries.append(frame)
    
    # Visualize
    o3d.visualization.draw_geometries(geometries)

class RealworldEnv:
    def __init__(
        self, 
        collector_host: str = '127.0.0.1', 
        collector_port: int = 5007,  # Updated default port to match the server
        executer_host: str = '127.0.0.1',
        executer_port: int = 5006,
        wait_time: float = 3.0,  # Increased default wait time for robot movement
        connection_retries: int = 3,
        connection_timeout: float = 5.0,
    ):
        """
        Initialize the realworld environment.
        
        Args:
            collector_host: Host address for the collector server (to get observations)
            collector_port: Port for the collector server
            executer_host: Host address for the executer server (to send actions)
            executer_port: Port for the executer server
            wait_time: Time to wait between sending action and receiving next observation
            connection_retries: Number of connection retry attempts
            connection_timeout: Timeout for connection attempts in seconds
        """
        self.collector_host = collector_host
        self.collector_port = collector_port
        self.executer_host = executer_host
        self.executer_port = executer_port
        self.wait_time = wait_time
        self.connection_retries = connection_retries
        self.connection_timeout = connection_timeout
        
        self.collector_sock = None
        self.executer_sock = None
        self.connected = False
        
        self.step_counter = 0
        self.episode_counter = 0
        
    def connect(self):
        """Connect to the collector and executer servers with retry logic."""
        for attempt in range(self.connection_retries):
            try:
                # Connect to executer (action server)
                print(f"[RealworldEnv] Attempting to connect to executer at {self.executer_host}:{self.executer_port}...")
                if self.executer_sock is None:
                    self.executer_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.executer_sock.settimeout(self.connection_timeout)
                    self.executer_sock.connect((self.executer_host, self.executer_port))
                    print(f"[RealworldEnv] Connected to executer at {self.executer_host}:{self.executer_port}")
                # Connect to collector (observation server)
                # sleep for wait_time to ensure executer is ready
                time.sleep(self.wait_time)
                print(f"[RealworldEnv] Attempting to connect to collector at {self.collector_host}:{self.collector_port}...")
                if self.collector_sock is None:
                    self.collector_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.collector_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Add this line
                    self.collector_sock.settimeout(self.connection_timeout)
                    self.collector_sock.connect((self.collector_host, self.collector_port))
                    print(f"[RealworldEnv] Connected to collector at {self.collector_host}:{self.collector_port}")
                

                
                # Reset timeouts to blocking mode for normal operation
                if self.collector_sock:
                    self.collector_sock.settimeout(10)  # Use a reasonable timeout instead of None
                if self.executer_sock:
                    self.executer_sock.settimeout(10)
                
                self.connected = True
                return True
            except socket.error as e:
                print(f"[RealworldEnv] Connection attempt {attempt+1}/{self.connection_retries} failed: {e}")
                # Close any sockets that might have been created
                if self.collector_sock:
                    self.collector_sock.close()
                    self.collector_sock = None
                if self.executer_sock:
                    self.executer_sock.close()
                    self.executer_sock = None
                time.sleep(self.connection_timeout)  # Wait before retrying
                
                if attempt < self.connection_retries - 1:
                    print(f"[RealworldEnv] Retrying in {self.connection_timeout} seconds...")
                else:
                    print("[RealworldEnv] All connection attempts failed")
                    return False
        
        return False
    
    def disconnect(self):
        """Disconnect from the servers with proper error handling."""
        try:
            if self.collector_sock:
                self.collector_sock.close()
                self.collector_sock = None
            if self.executer_sock:
                self.executer_sock.close()
                self.executer_sock = None
            self.connected = False
            print("[RealworldEnv] Disconnected from servers")
        except Exception as e:
            print(f"[RealworldEnv] Error during disconnect: {e}")
        finally:
            self.connected = False
            self.collector_sock = None
            self.executer_sock = None
    
    def get_observation(self, visualize=False) -> Dict[str, Any]:
        """
        Get an observation from the collector server.

        Args:
            visualize: Whether to visualize the received point cloud

        Returns:
            dict: Observation dictionary containing 'rgb', 'pc', 'gripper', etc.
        """
        if not self.connected:
            raise ConnectionError("Not connected to servers")
        
        try:
            print("[RealworldEnv] Waiting for observation from collector server", end="", flush=True)
            while True:
                try:
                    # Attempt to receive a message
                    msg = recv_msg(self.collector_sock)
                    if msg is not None:
                        print("\n[RealworldEnv] Received observation from collector server.")
                        break
                except socket.timeout:
                    # Print a dot to indicate waiting
                    print(".", end="", flush=True)
                    time.sleep(1)  # Wait for 1 second before retrying
            
            if msg is None:
                raise ConnectionError("Failed to receive message")
            
            # Send back confirmation
            confirmation = {'type': 'confirmation', 'status': 'received'}
            socket_send(self.collector_sock, confirmation)
            
            msg = bytes_to_str(msg)
            
            if msg['type'] == 'observation':  # Ensure the message type matches
                print(f"[RealworldEnv] Received observation for step {self.step_counter}")
                obs = msg['data']
                
                # Process observation to match RLBench format
                processed_obs = {
                    'rgb': [obs['rgb']],
                    'pc': [obs['xyz']],
                    'gripper': obs['gripper'],
                    'joint_states': obs['joint_states'],
                    'arm_links_info': obs.get('arm_links_info', None),
                }
                
                # Visualize if requested
                if visualize:
                    visualize_pointcloud(
                        obs['xyz'], 
                        obs['rgb'], 
                        obs['gripper']
                    )
                
                # Increment step counter
                self.step_counter += 1
                
                return processed_obs
            else:
                print(f"[RealworldEnv] Unexpected message type: {msg['type']}")
                return None
        except Exception as e:
            print(f"[RealworldEnv] Error getting observation: {e}")
            traceback.print_exc()
            return None
    
    def send_action(self, action: np.ndarray) -> bool:
        """
        Send an action to the executer server.
        
        Args:
            action: Action to send (7-dim vector [pos, quat, gripper])
            
        Returns:
            bool: Success status
        """
        if not self.connected:
            raise ConnectionError("Not connected to servers")
        
        try:
            msg = {'gripper': action}
            socket_send(self.executer_sock, msg)
            print(f"[RealworldEnv] Sent action: {action}")
            
            # Wait for the robot to complete the movement
            print(f"[RealworldEnv] Waiting {self.wait_time} seconds for robot movement...")
            time.sleep(self.wait_time)
            # Wait for confirmation
            confirmation = recv_msg(self.executer_sock)
            print(f"[RealworldEnv] Received confirmation: {confirmation}")
            # the dict are b'type' and b'status' contained in a dict
            confirmation = bytes_to_str(confirmation)
            # Check if confirmation is valid
            if confirmation and confirmation.get('type') == 'confirmation' and confirmation.get('status') == 'received':
                print("[Client] Server confirmed action receipt.")
            else:
                print("[Client] No confirmation received or invalid response.")
            
            return True
        except Exception as e:
            print(f"[RealworldEnv] Error sending action: {e}")
            traceback.print_exc()
            return False
    
    def reset(self, visualize=False) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.
        
        Args:
            visualize: Whether to visualize the received point cloud
            
        Returns:
            dict: Initial observation
        """
        if not self.connected:
            if not self.connect():
                raise ConnectionError("Failed to connect during reset")
        
        self.step_counter = 0
        self.episode_counter += 1
        print(f"[RealworldEnv] Starting episode {self.episode_counter}")
        
        # Get initial observation
        print("[RealworldEnv] Resetting environment by [doing nothing]...")
    
    def step(self, action: np.ndarray, visualize=False) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Execute an action and return the next observation.
        
        Args:
            action: Action to execute
            visualize: Whether to visualize the received point cloud
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Send action
        success = self.send_action(action)
        if not success:
            raise RuntimeError("Failed to send action")

        
        # Get next observation
        next_obs = self.get_observation(visualize=visualize)
        if next_obs is None:
            raise RuntimeError("Failed to get observation after action")
        
        # In real-world, we don't have automatic termination, so always False
        done = False
        
        # Reward is not applicable in real-world
        reward = 0.0
        
        # Additional info
        info = {
            'step_id': self.step_counter - 1,  # -1 because counter was incremented in get_observation
            'success': True  # Assume always successful for now
        }
        
        return next_obs, reward, done, info

def main():
    """
    Run a test case that loads actions from a file and shows the pointcloud without saving.
    The dummy policy simply executes the next action from the file regardless of the received state.
    """
    parser = argparse.ArgumentParser(description='RealworldEnv test case')
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/huser/mini-diffuse-actor/realworld_dataset/close_box/1cm.h5',
                        help='Path to the HDF5 dataset file with actions')
    parser.add_argument('--collector_host', type=str, default='127.0.0.1', 
                        help='Collector server host')
    parser.add_argument('--collector_port', type=int, default=5007, 
                        help='Collector server port')
    parser.add_argument('--executer_host', type=str, default='127.0.0.1', 
                        help='Executer server host')
    parser.add_argument('--executer_port', type=int, default=5006, 
                        help='Executer server port')
    parser.add_argument('--wait_time', type=float, default=0.5, 
                        help='Wait time between actions')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize point clouds')
    parser.add_argument('--max_steps', type=int, default=None, 
                        help='Maximum number of steps to run (default: all actions from file)')
    parser.add_argument('--interactive', action='store_true', 
                        help='Wait for user input between steps')
    
    args = parser.parse_args()
    
    # Load the actions from the dataset file
    try:
        gripper_poses = load_episode_hdf5(args.dataset_path)
        print(f"[RealworldEnv Test] Loaded {len(gripper_poses)} actions from dataset.")
        
        # Create and connect to the environment
        env = RealworldEnv(
            collector_host=args.collector_host,
            collector_port=args.collector_port,
            executer_host=args.executer_host,
            executer_port=args.executer_port,
            wait_time=args.wait_time
        )
        
        # Reset the environment and get initial observation
        print("[RealworldEnv Test] Resetting environment...")
        obs = env.reset(visualize=args.visualize)
        
        # Determine the number of steps to run
        num_steps = len(gripper_poses)
        if args.max_steps is not None:
            num_steps = min(num_steps, args.max_steps)
        
        # Run the test by executing each action in sequence
        print(f"[RealworldEnv Test] Running {num_steps} steps...")
        for step_idx, action in enumerate(gripper_poses[:num_steps]):
            print(f"[RealworldEnv Test] Step {step_idx+1}/{num_steps}")
            
            # Execute the action and get the next observation
            obs, reward, done, info = env.step(action, visualize=args.visualize)
            
            # Display step information
            print(f"[RealworldEnv Test] Action executed: {action}")
            print(f"[RealworldEnv Test] Reward: {reward}, Done: {done}, Info: {info}")
            
            # Wait for user input if in interactive mode
            if args.interactive:
                user_input = input("Press Enter to continue, 'q' to quit: ")
                if user_input.lower() == 'q':
                    break
        
        print("[RealworldEnv Test] Test completed successfully!")
    
    except FileNotFoundError as e:
        print(f"[RealworldEnv Test] Error: {e}")
    except ConnectionError as e:
        print(f"[RealworldEnv Test] Connection error: {e}")
    except KeyboardInterrupt:
        print("[RealworldEnv Test] Test interrupted by user")
    except Exception as e:
        print(f"[RealworldEnv Test] Error during test: {e}")
        traceback.print_exc()
    finally:
        # Ensure environment is disconnected
        if 'env' in locals() and env.connected:
            env.disconnect()

if __name__ == '__main__':
    main()
