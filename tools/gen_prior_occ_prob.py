"""
Compute occupancy probability distribution from training dataset.

This script analyzes the training dataset to compute spatial occupancy statistics
that can be used as priors for position encoding in the model.

Usage:
    python tools/gen_prior_occ_prob.py --output data/nuscenes/occupancy_prior.pkl --subset train
"""

import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

occ_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]

def args_parser():
    parser = argparse.ArgumentParser(
        description='Compute occupancy probability distribution from training dataset'
    )
    parser.add_argument('--data_path', default='./data/nuscenes', help='Data file path')
    parser.add_argument(
        '--output',
        default='data/nuscenes/occupancy_prior.pkl',
        help='Output path for statistics'
    )
    parser.add_argument(
        '--grid-shape',
        type=int,
        nargs=3,
        default=[200, 200, 16],
        help='Grid dimensions (X Y Z)'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='train',
        choices=['train', 'val', 'trainval'],
        help='Dataset subset to use'
    )
    parser.add_argument(
        '--vis',
        action='store_true',
        help='Create visualizations of the statistics'
    )
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='work_dirs/occupancy_prior_viz',
        help='Directory for visualizations'
    )

    args = parser.parse_args()
    return args

def gen_occ_stats(args):
    """
    Compute occupancy probability distribution from dataset.

    Returns:
        dict: Dictionary containing occupancy statistics
    """
    grid_shape = tuple(args.grid_shape)
    subset = args.subset
    
    if subset == 'train':
        data_infos_path = os.path.join(args.data_path, 'nuscenes_infos_train_sweep.pkl')
        data_infos = pickle.load(open(data_infos_path, 'rb'))['infos']
    elif subset == 'val':
        data_infos_path = os.path.join(args.data_path, 'nuscenes_infos_val_sweep.pkl')
        data_infos = pickle.load(open(data_infos_path, 'rb'))['infos']
    else:
        raise ValueError(f"Unknown subset: {subset}")

    print(f"Dataset loaded: {len(data_infos)} samples")
    print(f"Grid shape: {grid_shape}")

    num_classes = len(occ_names)
    
    occ_count = np.zeros((*grid_shape, num_classes), dtype=np.float32)
    total_count = np.zeros(grid_shape, dtype=np.float32)
    occupied_samples = np.zeros(grid_shape, dtype=np.float32)

    print(f"\nProcessing {len(data_infos)} samples...")

    for idx in tqdm(range(len(data_infos)), desc="Computing occupancy statistics"):
        data_info = data_infos[idx]

        token = data_info['token']
        scene_name = data_info['scene_name']

        occ_root = os.path.join(args.data_path, 'gts')
        occ_file = os.path.join(occ_root, scene_name, token, 'labels.npz')

        if not os.path.exists(occ_file):
            print(f"\nWarning: GT file not found: {occ_file}")
            continue

        occ_data = np.load(occ_file)
        occ_labels = occ_data['semantics']
        mask_camera = occ_data['mask_camera']
        
        for class_id in range(num_classes):
            mask = (occ_labels == class_id)
            occ_count[..., class_id] += mask.astype(np.float32)
        
        total_count += mask_camera.astype(np.float32)
        occupied_mask = (occ_labels < 17) & mask_camera
        occupied_samples += occupied_mask.astype(np.float32)

    print("\nComputing statistics...")

    valid_mask = total_count > 0

    # Compute per-class occupancy probability for per voxel
    occ_prob = np.zeros_like(occ_count)
    for class_id in range(num_classes):
        occ_prob[valid_mask, class_id] = (
            occ_count[valid_mask, class_id] / total_count[valid_mask]
        )

    # Overall occupancy density (probability that voxel is occupied)
    occ_density = np.zeros(grid_shape, dtype=np.float32)
    occ_density[valid_mask] = (
        occupied_samples[valid_mask] / total_count[valid_mask]
    )
    
    # Compute entropy (uncertainty measure) : H = -sum(p * log(p))
    occ_prob_safe = np.clip(occ_prob, 1e-10, 1.0)
    entropy = -np.sum(
        occ_prob * np.log(occ_prob_safe),
        axis=-1
    )
    
    max_entropy = np.log(num_classes)
    entropy = entropy / max_entropy

    # Compile statistics
    statistics = {
        'occupancy_prob': occ_prob,       # (X, Y, Z, num_classes)
        'occupancy_density': occ_density, # (X, Y, Z)
        'entropy': entropy,               # (X, Y, Z)
        'total_count': total_count,       # (X, Y, Z)
        'grid_shape': grid_shape,
        'num_classes': num_classes,
    }

    print("\n" + "="*60)
    print("Occupancy Statistics Summary")
    print(f"\nOccupancy Density:")
    print(f"  Min:  {occ_density[valid_mask].min():.4f}")
    print(f"  Max:  {occ_density[valid_mask].max():.4f}")
    print(f"  Mean: {occ_density[valid_mask].mean():.4f}")
    print(f"  Std:  {occ_density[valid_mask].std():.4f}")
    print(f"\nEntropy (normalized):")
    print(f"  Min:  {entropy[valid_mask].min():.4f}")
    print(f"  Max:  {entropy[valid_mask].max():.4f}")
    print(f"  Mean: {entropy[valid_mask].mean():.4f}")
    print(f"  Std:  {entropy[valid_mask].std():.4f}")
    print("="*60)

    return statistics

def save_stats_result(statistics, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(statistics, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved occupancy statistics to: {output_path}")

def vis_stats(statistics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    occupancy_density = statistics['occupancy_density']
    entropy = statistics['entropy']
    grid_shape = statistics['grid_shape']

    num_z_slices = min(4, grid_shape[2])
    z_indices = np.linspace(0, grid_shape[2] - 1, num_z_slices, dtype=int)

    fig, axes = plt.subplots(2, num_z_slices, figsize=(16, 8))

    for i, z_idx in enumerate(z_indices):
        # Density
        ax = axes[0, i] if num_z_slices > 1 else axes[0]
        im1 = ax.imshow(occupancy_density[:, :, z_idx].T,
                        origin='lower', cmap='spring', vmin=0, vmax=1)
        ax.set_title(f'Occupancy Density (Z={z_idx})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im1, ax=ax)

        # Entropy
        ax = axes[1, i] if num_z_slices > 1 else axes[1]
        im2 = ax.imshow(entropy[:, :, z_idx].T,
                        origin='lower', cmap='cool', vmin=0, vmax=1)
        ax.set_title(f'Entropy (Z={z_idx})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im2, ax=ax)

    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'occ_stats.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {viz_path}")
    plt.close()


def main():
    args = args_parser()
    statistics = gen_occ_stats(args)
    save_stats_result(statistics, args.output)
    
    if args.vis:
        print("\nCreating visualizations...")
        vis_stats(statistics, args.viz_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
