#!/usr/bin/env python3
"""
Optimize camera extrinsics using depth map alignment.
Refines wrist camera extrinsics by minimizing difference between projected depth and ground truth depth.
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Import functions from the projection script
from project_pointcloud_to_first_person import (
    load_camera_data,
    create_pointcloud_from_depth,
    transform_points_to_camera2,
    project_points_to_depth_image
)


class ExtrinsicsOptimizer:
    """Optimize camera extrinsics using depth map alignment."""

    def __init__(self, cam1_dir, cam2_dir, device='cuda:0'):
        self.cam1_dir = cam1_dir
        self.cam2_dir = cam2_dir
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Load camera 2 intrinsics (constant across frames)
        cam2_path = Path(cam2_dir)
        cam2_id = cam2_path.name
        intrinsics_path = cam2_path / "intrinsics" / f"{cam2_id}.npy"
        self.K2 = torch.from_numpy(np.load(str(intrinsics_path)).astype(np.float32)).to(self.device)

        # Load camera 2 extrinsics (will be optimized)
        extrinsics_dir = cam2_path / "extrinsics"
        extrinsics_files = list(extrinsics_dir.glob("*.npy"))
        self.ext2_all = np.load(str(extrinsics_files[0]))

        print(f"Using device: {self.device}")
        print(f"Camera 2 intrinsics shape: {self.K2.shape}")

    def transform_points_torch(self, points_3d, T):
        """Transform 3D points using PyTorch tensor operations."""
        # Convert to homogeneous coordinates
        ones = torch.ones(len(points_3d), 1, device=self.device)
        points_homogeneous = torch.cat([points_3d, ones], dim=1)

        # Transform points
        points_transformed = (T @ points_homogeneous.T).T

        # Convert back to 3D coordinates
        points_3d_transformed = points_transformed[:, :3]

        return points_3d_transformed

    def project_points_torch(self, points_3d, image_shape):
        """Project 3D points to depth image using PyTorch operations."""
        H, W = image_shape

        # Filter points in front of camera
        valid_front = points_3d[:, 2] > 0
        points_front = points_3d[valid_front]

        if len(points_front) == 0:
            # Return empty depth map
            return torch.full((H, W), float('nan'), device=self.device)

        # Extract coordinates
        X = points_front[:, 0]
        Y = points_front[:, 1]
        Z = points_front[:, 2]

        # Project to pixel coordinates
        fx, fy = self.K2[0, 0], self.K2[1, 1]
        cx, cy = self.K2[0, 2], self.K2[1, 2]

        u_coords = fx * X / Z + cx
        v_coords = fy * Y / Z + cy

        # Round to nearest integer
        u_int = torch.round(u_coords).long()
        v_int = torch.round(v_coords).long()

        # Find valid projections within image bounds
        valid_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_valid = u_int[valid_bounds]
        v_valid = v_int[valid_bounds]
        depths_valid = Z[valid_bounds]

        if len(depths_valid) == 0:
            return torch.full((H, W), float('nan'), device=self.device)

        # Create depth image
        depth_image = torch.full((H, W), float('nan'), device=self.device)

        # Use scatter to update depth values (keep minimum depth)
        indices = v_valid * W + u_valid

        # For pixels with multiple points, keep the closest (minimum depth)
        current_depths = depth_image.flatten()[indices]

        # Create mask for valid updates
        nan_mask = torch.isnan(current_depths)
        update_mask = nan_mask | (depths_valid < current_depths)

        # Update depths
        depth_image.flatten()[indices[update_mask]] = depths_valid[update_mask]

        return depth_image

    def create_depth_from_extrinsics_torch(self, points_3d_cam1, T_1to2, image_shape):
        """Create depth map from extrinsics transform using PyTorch operations."""
        # Transform points to camera 2 coordinates
        points_3d_cam2 = self.transform_points_torch(points_3d_cam1, T_1to2)

        # Project to depth image
        depth_projected = self.project_points_torch(points_3d_cam2, image_shape)

        return depth_projected

    def depth_loss(self, depth_pred, depth_gt, valid_mask=None):
        """Compute depth map loss."""
        if valid_mask is None:
            # Create valid mask (both depths are valid)
            valid_mask = (~np.isnan(depth_pred)) & (~np.isnan(depth_gt)) & (depth_gt > 0)

        if not np.any(valid_mask):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Convert to tensors
        depth_pred_tensor = torch.from_numpy(depth_pred[valid_mask].astype(np.float32)).to(self.device)
        depth_gt_tensor = torch.from_numpy(depth_gt[valid_mask].astype(np.float32)).to(self.device)

        # Compute L1 loss
        loss = torch.mean(torch.abs(depth_pred_tensor - depth_gt_tensor))

        return loss

    def visualize_depth_maps(self, depth_pred, depth_gt, iteration, frame_idx, title_prefix="", show_plots=True):
        """Visualize predicted and ground truth depth maps."""
        if not show_plots:
            return

        # Convert tensors to numpy for visualization
        if torch.is_tensor(depth_pred):
            depth_pred_np = depth_pred.detach().cpu().numpy()
        else:
            depth_pred_np = depth_pred

        if torch.is_tensor(depth_gt):
            depth_gt_np = depth_gt.detach().cpu().numpy()
        else:
            depth_gt_np = depth_gt

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Predicted depth
        depth_pred_vis = depth_pred_np.copy()
        depth_pred_vis[np.isnan(depth_pred_vis)] = 0
        im1 = axes[0].imshow(depth_pred_vis, cmap='plasma', vmin=0, vmax=5)
        axes[0].set_title(f'{title_prefix}Predicted Depth\nFrame {frame_idx}, Iter {iteration}')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # Ground truth depth
        depth_gt_vis = depth_gt_np.copy()
        depth_gt_vis[np.isnan(depth_gt_vis)] = 0
        im2 = axes[1].imshow(depth_gt_vis, cmap='plasma', vmin=0, vmax=5)
        axes[1].set_title(f'{title_prefix}Ground Truth Depth\nFrame {frame_idx}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # Difference
        diff = depth_pred_vis - depth_gt_vis
        # Only show difference where both depths are valid
        valid_mask = (depth_pred_vis > 0) & (depth_gt_vis > 0)
        diff[~valid_mask] = np.nan

        im3 = axes[2].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
        axes[2].set_title(f'{title_prefix}Depth Difference\nFrame {frame_idx}, Iter {iteration}')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

        plt.tight_layout()

        # Save visualization to file instead of showing
        import os
        vis_dir = "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        filename = f"{vis_dir}/depth_optimization_frame_{frame_idx}_iter_{iteration}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {filename}")
        plt.close()

    def optimize_frame_extrinsics(self, frame_idx, points_3d_cam1, depth_gt, max_iterations=50, lr=1e-3, visualize=True, vis_freq=10):
        """
        Optimize extrinsics for a single frame.

        Args:
            frame_idx: Frame index
            points_3d_cam1: 3D points from camera 1 (numpy array)
            depth_gt: Ground truth depth map from camera 2 (numpy array)
            max_iterations: Maximum optimization iterations
            lr: Learning rate
            visualize: Whether to show depth map visualizations
            vis_freq: Frequency of visualization updates (every N iterations)

        Returns:
            optimized_ext2: Optimized extrinsics [R|t] 3x4 matrix
            loss_history: Loss values during optimization
        """
        # Get initial extrinsics
        ext2_initial = self.ext2_all[frame_idx].copy().astype(np.float32)

        # Convert to torch tensor for optimization
        ext2_tensor = torch.from_numpy(ext2_initial).to(self.device).requires_grad_(True)

        # Load camera 1 extrinsics for this frame
        cam1_depth_dir = os.path.join(self.cam1_dir, "depth_npy")
        _, _, _, ext1 = load_camera_data(self.cam1_dir, frame_idx, cam1_depth_dir)

        # Convert to torch
        ext1_tensor = torch.from_numpy(ext1.astype(np.float32)).to(self.device)

        # Convert points to torch
        points_3d_cam1_tensor = torch.from_numpy(points_3d_cam1.astype(np.float32)).to(self.device)

        # Convert depth_gt to torch
        depth_gt_tensor = torch.from_numpy(depth_gt.astype(np.float32)).to(self.device)

        # Create 4x4 transform matrices
        T1 = torch.eye(4, device=self.device)
        T1[:3, :] = ext1_tensor

        # Image shape
        image_shape = depth_gt.shape

        # Optimizer
        optimizer = optim.Adam([ext2_tensor], lr=lr)

        loss_history = []

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Create 4x4 transform matrix from current extrinsics
            T2 = torch.eye(4, device=self.device)
            T2[:3, :] = ext2_tensor

            # Compute relative transform T_1to2 = inv(T2) @ T1
            T_1to2 = torch.linalg.inv(T2) @ T1

            # Create depth map
            depth_pred = self.create_depth_from_extrinsics_torch(points_3d_cam1_tensor, T_1to2, image_shape)

            # Visualize depth maps periodically
            if visualize and (iteration % vis_freq == 0 or iteration == max_iterations - 1):
                self.visualize_depth_maps(depth_pred, depth_gt_tensor, iteration, frame_idx)

            # Compute loss
            loss = self.compute_depth_loss_torch(depth_pred, depth_gt_tensor)

            # Backpropagation
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_history.append(loss_value)


        # Get optimized extrinsics
        optimized_ext2 = ext2_tensor.detach().cpu().numpy()

        final_loss = loss_history[-1] if loss_history else 0

        return optimized_ext2, loss_history

    def compute_depth_loss_torch(self, depth_pred, depth_gt):
        """Compute depth loss using PyTorch tensors."""
        if depth_pred is None:
            # Return a large initial loss
            return torch.tensor(1000.0, device=self.device, requires_grad=True)

        # Create valid mask (both depths are valid and positive)
        valid_mask = (~torch.isnan(depth_pred)) & (~torch.isnan(depth_gt)) & (depth_gt > 0)

        if not torch.any(valid_mask):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Extract valid depths
        depth_pred_valid = depth_pred[valid_mask]
        depth_gt_valid = depth_gt[valid_mask]

        # Compute L1 loss
        loss = torch.mean(torch.abs(depth_pred_valid - depth_gt_valid))

        return loss

    def optimize_all_frames(self, output_dir, max_iterations=50, lr=1e-4, start_frame=None, end_frame=None, visualize=False, vis_freq=10):
        """
        Optimize extrinsics for all frames.

        Args:
            output_dir: Output directory for refined extrinsics
            max_iterations: Maximum iterations per frame
            lr: Learning rate
            start_frame: Starting frame index
            end_frame: Ending frame index
            visualize: Whether to show depth map visualizations
            vis_freq: Frequency of visualization updates
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get frame range
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(self.ext2_all) - 1

        print(f"Optimizing extrinsics for frames {start_frame} to {end_frame}")
        print(f"Output directory: {output_dir}")

        # Load camera 1 data (constant across frames)
        cam1_path = Path(self.cam1_dir)
        cam1_id = cam1_path.name
        intrinsics_path = cam1_path / "intrinsics" / f"{cam1_id}.npy"
        K1 = np.load(str(intrinsics_path))

        # Optimized extrinsics array
        optimized_ext2_all = np.zeros_like(self.ext2_all)

        total_frames = end_frame - start_frame + 1
        successful_frames = 0

        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Optimizing frames"):
            try:
                # Load camera 1 depth for this frame
                cam1_depth_dir = os.path.join(self.cam1_dir, "depth_npy")
                _, depth1, _, _ = load_camera_data(self.cam1_dir, frame_idx, cam1_depth_dir)

                # Create 3D point cloud from camera 1
                points_3d_cam1, _ = create_pointcloud_from_depth(depth1, K1, max_depth=10.0)

                # Load camera 2 ground truth depth
                depth2_path = Path(self.cam2_dir) / "depth_npy" / f"{frame_idx}.npz"
                depth2 = np.load(str(depth2_path))['depth']

                # Optimize extrinsics for this frame
                optimized_ext2, loss_history = self.optimize_frame_extrinsics(
                    frame_idx, points_3d_cam1, depth2, max_iterations, lr, visualize, vis_freq
                )

                optimized_ext2_all[frame_idx] = optimized_ext2
                successful_frames += 1

                # Save intermediate results every 10 frames
                if frame_idx % 10 == 0:
                    np.save(output_path / "wrist_left_extrinsics_3x4_refined.npy", optimized_ext2_all)

            except Exception as e:
                print(f"Error optimizing frame {frame_idx}: {e}")
                # Use original extrinsics if optimization fails
                optimized_ext2_all[frame_idx] = self.ext2_all[frame_idx]
                continue

        # Save final results
        np.save(output_path / "wrist_left_extrinsics_3x4_refined.npy", optimized_ext2_all)

        print(f"\nOptimization completed!")
        print(f"Total frames: {total_frames}")
        print(f"Successful optimizations: {successful_frames}")
        print(f"Output saved to: {output_path / 'wrist_left_extrinsics_3x4_refined.npy'}")


def create_argument_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimize camera extrinsics using depth map alignment"
    )
    parser.add_argument(
        "--cam1",
        default="datasets/samples/23897859",
        help="Third-person camera directory"
    )
    parser.add_argument(
        "--cam2",
        default="datasets/samples/17368348",
        help="First-person (wrist) camera directory"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/samples/17368348/extrinsics_refined",
        help="Output directory for refined extrinsics"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum optimization iterations per frame"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Starting frame index"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ending frame index"
    )
    parser.add_argument(
        "--cuda-device",
        default="cuda:0",
        help="CUDA device to use"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable depth map visualization during optimization"
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="Frequency of visualization updates (every N iterations)"
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    print("="*80)
    print("Camera Extrinsics Optimization using Depth Map Alignment")
    print("="*80)
    print(f"Third-person camera: {args.cam1}")
    print(f"First-person camera: {args.cam2}")
    print(f"Output directory: {args.output_dir}")

    # Create optimizer
    optimizer = ExtrinsicsOptimizer(args.cam1, args.cam2, args.cuda_device)

    # Run optimization
    optimizer.optimize_all_frames(
        args.output_dir,
        args.max_iterations,
        args.lr,
        args.start_frame,
        args.end_frame,
        args.visualize,
        args.vis_freq
    )

    print(f"\n{'='*80}")
    print("Optimization completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
