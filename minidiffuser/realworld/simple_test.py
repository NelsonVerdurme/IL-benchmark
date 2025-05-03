#!/usr/bin/env python3
"""
Test script for saving and loading a dataset-like structure
using HDF5 (h5py) for variable-sized arrays and metadata.
"""
import os
import shutil
import h5py
import numpy as np

# --- Configuration ---
HDF5_FILE = 'test_hdf5_dataset.h5'

def save_dataset_to_hdf5(hdf5_path: str):
    """
    Saves a dataset-like structure into an HDF5 file.
    """
    print(f"Attempting to save dataset to: {hdf5_path}")

    # Create dataset-like structure
    dataset = {
        'xyz': [np.random.randn(np.random.randint(50, 100), 3).astype(np.float32) for _ in range(5)],
        'rgb': [np.random.randint(0, 255, (np.random.randint(50, 100), 3)).astype(np.uint8) for _ in range(5)],
        'actions': [np.random.randn(np.random.randint(5, 10), 7).astype(np.float32) for _ in range(5)],
        'description': 'This is a test dataset with variable-sized arrays'
    }

    # Save dataset to HDF5
    with h5py.File(hdf5_path, 'w') as f:
        for key, value in dataset.items():
            if isinstance(value, list):
                group = f.create_group(key)
                for i, item in enumerate(value):
                    group.create_dataset(str(i), data=item)
            else:
                f.attrs[key] = value
        print(f"Dataset successfully written to HDF5 at {hdf5_path}")

def load_dataset_from_hdf5(hdf5_path: str):
    """
    Loads a dataset-like structure from an HDF5 file.
    """
    print(f"\nAttempting to load dataset from: {hdf5_path}")
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found at {hdf5_path}")
        return None

    dataset = {'xyz': [], 'rgb': [], 'actions': [], 'description': None}
    with h5py.File(hdf5_path, 'r') as f:
        for key in f.keys():
            group = f[key]
            dataset[key] = [group[str(i)][...] for i in range(len(group))]
        for key in f.attrs.keys():
            dataset[key] = f.attrs[key]
        print("Dataset successfully loaded from HDF5.")
    return dataset

def load_and_verify_collector_dataset(hdf5_path: str):
    """
    Loads and verifies the dataset saved by the collector_client.
    """
    print(f"\nAttempting to load and verify dataset from: {hdf5_path}")
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found at {hdf5_path}")
        return None

    with h5py.File(hdf5_path, 'r') as f:
        print("\n--- Dataset Structure ---")
        for episode_key in f['episodes']:
            print(f"Episode: {episode_key}")
            episode_group = f['episodes'][episode_key]
            for step_key in episode_group:
                print(f"  Step: {step_key}")
                step_group = episode_group[step_key]
                for dataset_key in step_group:
                    print(f"    Dataset: {dataset_key}, Shape: {step_group[dataset_key].shape}")

if __name__ == '__main__':
    print("--- HDF5 Dataset Test ---")

    # --- Cleanup ---
    if os.path.exists(HDF5_FILE):
        print(f"Removing existing HDF5 file: {HDF5_FILE}")
        os.remove(HDF5_FILE)

    # --- Save ---
    save_dataset_to_hdf5(HDF5_FILE)

    # --- Load ---
    loaded_dataset = load_dataset_from_hdf5(HDF5_FILE)

    # --- Verification ---
    if loaded_dataset:
        print("\n--- Verification ---")
        print("Loaded Dataset:")
        print(f"XYZ: {len(loaded_dataset['xyz'])} arrays")
        print(f"RGB: {len(loaded_dataset['rgb'])} arrays")
        print(f"Actions: {len(loaded_dataset['actions'])} arrays")
        print(f"Description: {loaded_dataset['description']}")
    else:
        print("\nVerification skipped because dataset loading failed.")

    # --- Load and Verify Collector Dataset ---
    collector_hdf5_file = 'realworld_dataset/dataset.h5'
    load_and_verify_collector_dataset(collector_hdf5_file)

    print("\n--- Test Complete ---")