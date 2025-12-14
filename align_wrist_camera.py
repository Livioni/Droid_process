#!/usr/bin/env python3
"""
Align wrist camera using combined point clouds from two aligned third-person cameras.
Uses ICP (Iterative Closest Point) to optimize wrist camera extrinsics.

Usage:
    python align_wrist_camera.py \
        --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
        --cam-ext2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
        --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
        --output-dir datasets/samples/.../17368348/extrinsics_refined_icp

Output:
    - {wrist_camera_id}.npy: Aligned wrist camera extrinsics for all frames [N, 3, 4]
    - wrist_alignment_stats.png: Fitness and RMSE plots
"""

import numpy as np
import os,glob
import open3d as o3d
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import functions from the projection script
from project_pointcloud_to_first_person import (
    load_camera_data,
    create_pointcloud_from_depth,
    transform_points_to_camera2,
    compute_relative_pose
)


class WristCameraAligner:
    """Align wrist camera using combined point clouds from two third-person cameras."""

    def __init__(self, cam_ext1_dir, cam_ext2_dir, cam_wrist_dir):
        """
        Initialize the wrist camera aligner.
        
        Args:
            cam_ext1_dir: First third-person camera directory (reference)
            cam_ext2_dir: Second third-person camera directory
            cam_wrist_dir: Wrist camera directory
        """
        self.cam_ext1_dir = cam_ext1_dir
        self.cam_ext2_dir = cam_ext2_dir
        self.cam_wrist_dir = cam_wrist_dir

        # Load camera paths
        cam_ext1_path = Path(cam_ext1_dir)
        cam_ext2_path = Path(cam_ext2_dir)
        cam_wrist_path = Path(cam_wrist_dir)
        
        self.cam_ext1_id = cam_ext1_path.name
        self.cam_ext2_id = cam_ext2_path.name
        self.cam_wrist_id = cam_wrist_path.name

        # Load intrinsics
        intrinsics_path_ext1 = cam_ext1_path / "intrinsics" / f"{self.cam_ext1_id}_left.npy"
        intrinsics_path_ext2 = cam_ext2_path / "intrinsics" / f"{self.cam_ext2_id}_left.npy"
        intrinsics_path_wrist = cam_wrist_path / "intrinsics" / f"{self.cam_wrist_id}_left.npy"
        
        self.K_ext1 = np.load(str(intrinsics_path_ext1))
        self.K_ext2 = np.load(str(intrinsics_path_ext2))
        self.K_wrist = np.load(str(intrinsics_path_wrist))

        # Load extrinsics for external cameras
        extrinsics_dir_ext1 = cam_ext1_path / "extrinsics"
        extrinsics_files_ext1 = list(extrinsics_dir_ext1.glob("*_left.npy"))
        self.ext1_all = np.load(str(extrinsics_files_ext1[0]))
        
        # Load ext2 extrinsics (aligned or original)
        try:
            extrinsics_refined_dir = cam_ext2_path / "extrinsics_refined_icp" 
            extrinsics_file = glob.glob(os.path.join(extrinsics_refined_dir, '*.npy'))[0]
            self.ext2_all = np.load(extrinsics_file,allow_pickle=True)
        except:
            extrinsics_dir_ext2 = cam_ext2_path / "extrinsics"
            extrinsics_files_ext2 = list(extrinsics_dir_ext2.glob("*_left.npy"))
            self.ext2_all = np.load(str(extrinsics_files_ext2[0]))
        
        # Load wrist camera extrinsics (will be optimized)
        extrinsics_dir_wrist = cam_wrist_path / "extrinsics"
        extrinsics_files_wrist = list(extrinsics_dir_wrist.glob("*_left.npy"))
        self.ext_wrist_all = np.load(str(extrinsics_files_wrist[0]))

        print(f"External camera 1 ({self.cam_ext1_id}): {len(self.ext1_all)} frames")
        print(f"External camera 2 ({self.cam_ext2_id}): {len(self.ext2_all)} frames")
        print(f"Wrist camera ({self.cam_wrist_id}): {len(self.ext_wrist_all)} frames")

    def filter_projected_points_by_depth(self, points_3d_wrist, depth_wrist, K_wrist, 
                                        depth_threshold=0.05):
        """
        Filter projected points to keep only those visible in wrist camera.
        For each pixel, keep only the nearest point from the projected point cloud.
        
        Args:
            points_3d_wrist: Nx3 array of projected 3D points in wrist camera coordinates
            depth_wrist: HxW ground truth depth map from wrist camera
            K_wrist: 3x3 camera intrinsics for wrist camera
            depth_threshold: Threshold for considering a point as "nearest" (in meters)
            
        Returns:
            filtered_points: Mx3 array of filtered 3D points (M <= N)
        """
        H, W = depth_wrist.shape
        
        # Filter points in front of camera
        valid_front = points_3d_wrist[:, 2] > 0
        points_front = points_3d_wrist[valid_front]
        
        if len(points_front) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # Project to pixel coordinates
        fx, fy = K_wrist[0, 0], K_wrist[1, 1]
        cx, cy = K_wrist[0, 2], K_wrist[1, 2]
        
        X = points_front[:, 0]
        Y = points_front[:, 1]
        Z = points_front[:, 2]
        
        u_coords = fx * X / Z + cx
        v_coords = fy * Y / Z + cy
        
        # Round to nearest integer
        u_int = np.round(u_coords).astype(int)
        v_int = np.round(v_coords).astype(int)
        
        # Find valid projections within image bounds
        valid_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_valid = u_int[valid_bounds]
        v_valid = v_int[valid_bounds]
        points_valid = points_front[valid_bounds]
        depths_valid = Z[valid_bounds]
        
        if len(points_valid) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        # Create a depth image from projected points (keep minimum depth per pixel)
        projected_depth = np.full((H, W), np.inf, dtype=np.float32)
        point_indices = {}  # Map from pixel to list of point indices
        
        for i, (u, v, depth) in enumerate(zip(u_valid, v_valid, depths_valid)):
            pixel_key = (v, u)
            if pixel_key not in point_indices:
                point_indices[pixel_key] = []
            point_indices[pixel_key].append(i)
            
            # Keep minimum depth
            if depth < projected_depth[v, u]:
                projected_depth[v, u] = depth
        
        # Filter points: keep only those that are the nearest at their pixel
        # and are close to the ground truth depth
        filtered_indices = []
        
        for (v, u), indices in point_indices.items():
            # Get ground truth depth at this pixel
            gt_depth = depth_wrist[v, u]
            
            # Skip if no valid ground truth depth
            if gt_depth <= 0 or np.isnan(gt_depth):
                continue
            
            # Get the minimum projected depth at this pixel
            min_proj_depth = projected_depth[v, u]
            
            # Find points that are the nearest (within threshold of minimum depth)
            for idx in indices:
                point_depth = depths_valid[idx]
                
                # Keep point if it's the nearest and within threshold of GT depth
                if abs(point_depth - min_proj_depth) < depth_threshold:
                    # Additional filter: point should be close to GT depth
                    # This helps remove occluded points
                    if point_depth <= gt_depth + depth_threshold:
                        filtered_indices.append(idx)
        
        if len(filtered_indices) == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        filtered_points = points_valid[filtered_indices]
        
        return filtered_points

    def align_frame_icp(self, frame_idx, max_iterations=50, distance_threshold=0.05,
                       voxel_size=0.001, visualize=False):
        """
        Align wrist camera for a single frame using combined point clouds from both external cameras.
        
        Args:
            frame_idx: Frame index
            max_iterations: Maximum ICP iterations
            distance_threshold: Distance threshold for ICP correspondence (in meters)
            voxel_size: Voxel size for downsampling (in meters)
            visualize: Whether to visualize point clouds
            
        Returns:
            optimized_ext_wrist: Optimized wrist camera extrinsics [R|t] 3x4 matrix
            fitness: ICP fitness score
            rmse: ICP RMSE
        """
        print(f"\nAligning frame {frame_idx}...")
        
        # Get initial wrist camera extrinsics
        ext_wrist_initial = self.ext_wrist_all[frame_idx].copy()
        
        # Load depth maps
        cam_ext1_depth_dir = os.path.join(self.cam_ext1_dir, "depth_npy")
        cam_ext2_depth_dir = os.path.join(self.cam_ext2_dir, "depth_npy")
        cam_wrist_depth_dir = os.path.join(self.cam_wrist_dir, "depth_npy")
        
        _, depth_ext1, _, ext1 = load_camera_data(self.cam_ext1_dir, frame_idx, cam_ext1_depth_dir)
        _, depth_ext2, _, ext2 = load_camera_data(self.cam_ext2_dir, frame_idx, cam_ext2_depth_dir)
        _, depth_wrist, _, _ = load_camera_data(self.cam_wrist_dir, frame_idx, cam_wrist_depth_dir)
        
        # Use loaded ext2 if not using aligned version
        if frame_idx < len(self.ext2_all):
            ext2 = self.ext2_all[frame_idx]
        
        # Create point clouds from external cameras in their own coordinates
        points_3d_ext1, _ = create_pointcloud_from_depth(depth_ext1, self.K_ext1, max_depth=10.0)
        points_3d_ext2, _ = create_pointcloud_from_depth(depth_ext2, self.K_ext2, max_depth=10.0)
        
        print(f"  External camera 1 points: {len(points_3d_ext1)}")
        print(f"  External camera 2 points: {len(points_3d_ext2)}")
        
        if len(points_3d_ext1) < 100 or len(points_3d_ext2) < 100:
            print(f"  Insufficient external camera points, skipping frame {frame_idx}")
            return ext_wrist_initial, 0.0, float('inf')
        
        # Transform external camera points to world coordinates
        T_ext1 = np.eye(4)
        T_ext1[:3, :] = ext1
        
        T_ext2 = np.eye(4)
        T_ext2[:3, :] = ext2
        
        points_3d_world_ext1 = self._transform_points(points_3d_ext1, T_ext1)
        points_3d_world_ext2 = self._transform_points(points_3d_ext2, T_ext2)
        
        # Combine point clouds from both external cameras
        points_3d_world_combined = np.vstack([points_3d_world_ext1, points_3d_world_ext2])
        
        print(f"  Combined external points: {len(points_3d_world_combined)}")
        
        # Transform combined world points to wrist camera coordinates
        T_wrist_initial = np.eye(4)
        T_wrist_initial[:3, :] = ext_wrist_initial
        T_world_to_wrist = np.linalg.inv(T_wrist_initial)
        
        points_3d_wrist_projected = self._transform_points(points_3d_world_combined, T_world_to_wrist)
        
        # Filter projected points to keep only visible nearest points
        print(f"  Filtering projected points...")
        points_3d_wrist_filtered = self.filter_projected_points_by_depth(
            points_3d_wrist_projected, depth_wrist, self.K_wrist
        )
        
        print(f"  Filtered projected points: {len(points_3d_wrist_projected)} -> {len(points_3d_wrist_filtered)}")
        
        if len(points_3d_wrist_filtered) < 100:
            print(f"  Insufficient filtered points ({len(points_3d_wrist_filtered)}), skipping")
            return ext_wrist_initial, 0.0, float('inf')
        
        # Create point cloud from wrist camera depth (target)
        points_3d_wrist_gt, _ = create_pointcloud_from_depth(depth_wrist, self.K_wrist, max_depth=10.0)
        
        if len(points_3d_wrist_gt) < 100:
            print(f"  Insufficient wrist camera points ({len(points_3d_wrist_gt)}), skipping")
            return ext_wrist_initial, 0.0, float('inf')
        
        print(f"  Source points (filtered projected): {len(points_3d_wrist_filtered)}")
        print(f"  Target points (wrist camera): {len(points_3d_wrist_gt)}")
        
        # Convert to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(points_3d_wrist_filtered)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_3d_wrist_gt)
        
        # Downsample for efficiency
        source_pcd = source_pcd.voxel_down_sample(voxel_size)
        target_pcd = target_pcd.voxel_down_sample(voxel_size)
        
        print(f"  After downsampling - Source: {len(source_pcd.points)}, Target: {len(target_pcd.points)}")
        
        # Estimate normals for better ICP
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        
        # Run ICP - align source (projected external) to target (wrist)
        init_transform = np.eye(4)
        
        # Use point-to-plane ICP for better convergence
        reg_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, distance_threshold, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        
        print(f"  ICP fitness: {reg_result.fitness:.4f}, RMSE: {reg_result.inlier_rmse:.4f}")
        
        # The ICP transformation is in wrist camera coordinates
        T_icp = reg_result.transformation
        
        # Update wrist camera extrinsics
        # T_world_to_wrist_new = T_icp @ T_world_to_wrist
        # T_wrist_new = inv(T_icp @ inv(T_wrist_initial))
        T_wrist_new = np.linalg.inv(T_icp @ np.linalg.inv(T_wrist_initial))
        
        optimized_ext_wrist = T_wrist_new[:3, :]
        
        # Visualize if requested
        if visualize:
            self.visualize_point_clouds(source_pcd, target_pcd, T_icp, frame_idx)
        
        return optimized_ext_wrist, reg_result.fitness, reg_result.inlier_rmse

    def _transform_points(self, points, T):
        """Transform 3D points using a 4x4 transformation matrix."""
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        points_transformed = (T @ points_homogeneous.T).T
        return points_transformed[:, :3]

    def align_all_frames(self, output_dir, max_iterations=50, distance_threshold=0.05,
                        voxel_size=0.001, start_frame=None, end_frame=None, visualize=False):
        """
        Align wrist camera for all frames.
        
        Args:
            output_dir: Output directory for aligned extrinsics
            max_iterations: Maximum ICP iterations per frame
            distance_threshold: Distance threshold for ICP correspondence
            voxel_size: Voxel size for downsampling
            start_frame: Starting frame index
            end_frame: Ending frame index
            visualize: Whether to visualize point clouds
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get frame range
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(self.ext_wrist_all) - 1
        
        print(f"Aligning wrist camera for frames {start_frame} to {end_frame}")
        print(f"Output directory: {output_dir}")
        
        # Aligned extrinsics array
        aligned_ext_wrist_all = np.zeros_like(self.ext_wrist_all)
        
        # Statistics
        fitness_scores = []
        rmse_scores = []
        
        total_frames = end_frame - start_frame + 1
        successful_frames = 0
        
        for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Aligning frames"):
            try:
                # Align wrist camera for this frame
                optimized_ext_wrist, fitness, rmse = self.align_frame_icp(
                    frame_idx, max_iterations, distance_threshold, voxel_size, visualize
                )
                
                aligned_ext_wrist_all[frame_idx] = optimized_ext_wrist
                fitness_scores.append(fitness)
                rmse_scores.append(rmse)
                successful_frames += 1
                
                # Save intermediate results every 10 frames
                if frame_idx % 10 == 0:
                    np.save(output_path / f"{self.cam_wrist_id}.npy", aligned_ext_wrist_all)
                
            except Exception as e:
                print(f"Error aligning frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Use original extrinsics if alignment fails
                aligned_ext_wrist_all[frame_idx] = self.ext_wrist_all[frame_idx]
                continue
        
        # Save final results
        output_file = output_path / f"{self.cam_wrist_id}.npy"
        np.save(output_file, aligned_ext_wrist_all)
        
        # Print summary
        print(f"\nAlignment completed!")
        print(f"Total frames: {total_frames}")
        print(f"Successful alignments: {successful_frames}")
        if fitness_scores:
            print(f"Average fitness: {np.mean(fitness_scores):.4f}")
            print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
        print(f"Output saved to: {output_file}")
        print(f"Output shape: {aligned_ext_wrist_all.shape}")
        
        # Plot statistics
        if fitness_scores:
            self.plot_alignment_stats(fitness_scores, rmse_scores, output_path)

    def visualize_point_clouds(self, source_pcd, target_pcd, transform, frame_idx):
        """Visualize point clouds before and after ICP alignment."""
        try:
            # Create a copy for visualization
            source_transformed = o3d.geometry.PointCloud(source_pcd)
            source_transformed.transform(transform)
            
            # Color the point clouds
            source_pcd.paint_uniform_color([1, 0, 0])  # Red: original source (projected external)
            source_transformed.paint_uniform_color([0, 1, 0])  # Green: transformed source
            target_pcd.paint_uniform_color([0, 0, 1])  # Blue: target (wrist)
            
            # Save visualization
            vis_dir = Path("visualizations_wrist_alignment")
            vis_dir.mkdir(exist_ok=True)
            
            # Save point clouds as PLY files
            ply_path_source = vis_dir / f"wrist_frame_{frame_idx:06d}_external_original.ply"
            ply_path_transformed = vis_dir / f"wrist_frame_{frame_idx:06d}_external_aligned.ply"
            ply_path_target = vis_dir / f"wrist_frame_{frame_idx:06d}_wrist_target.ply"
            
            o3d.io.write_point_cloud(str(ply_path_source), source_pcd)
            o3d.io.write_point_cloud(str(ply_path_transformed), source_transformed)
            o3d.io.write_point_cloud(str(ply_path_target), target_pcd)
            
            print(f"  Saved point clouds: {ply_path_source.parent}")
                
        except Exception as e:
            print(f"  Warning: Visualization failed for frame {frame_idx}: {e}")

    def plot_alignment_stats(self, fitness_scores, rmse_scores, output_path):
        """Plot alignment statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Fitness scores
        axes[0].plot(fitness_scores)
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('ICP Fitness')
        axes[0].set_title('Wrist Camera Alignment - ICP Fitness Scores')
        axes[0].grid(True)
        
        # RMSE scores
        axes[1].plot(rmse_scores)
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('ICP RMSE (m)')
        axes[1].set_title('Wrist Camera Alignment - ICP RMSE Scores')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / "wrist_alignment_stats.png", dpi=150)
        print(f"Saved statistics plot: {output_path / 'wrist_alignment_stats.png'}")
        plt.close()


def create_argument_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Align wrist camera using combined point clouds from two third-person cameras"
    )
    parser.add_argument(
        "--cam-ext1",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/23897859",
        help="First third-person camera directory (reference)"
    )
    parser.add_argument(
        "--cam-ext2",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/27904255",
        help="Second third-person camera directory"
    )
    parser.add_argument(
        "--cam-wrist",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/17368348",
        help="Wrist camera directory"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined_icp",
        help="Output directory for aligned wrist camera extrinsics"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum ICP iterations per frame"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.05,
        help="Distance threshold for ICP correspondence (in meters)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.001,
        help="Voxel size for downsampling (in meters)"
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
        "--visualize",
        action="store_true",
        default=False,
        help="Enable point cloud visualization during alignment"
    )
    
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("="*80)
    print("Wrist Camera Alignment using Combined Third-Person Point Clouds")
    print("="*80)
    print(f"External camera 1 (reference): {args.cam_ext1}")
    print(f"External camera 2: {args.cam_ext2}")
    print(f"Wrist camera: {args.cam_wrist}")
    print(f"Output directory: {args.output_dir}")
    
    # Create aligner
    aligner = WristCameraAligner(
        args.cam_ext1, 
        args.cam_ext2, 
        args.cam_wrist,
    )
    
    # Align wrist camera for all frames
    aligner.align_all_frames(
        args.output_dir,
        args.max_iterations,
        args.distance_threshold,
        args.voxel_size,
        args.start_frame,
        args.end_frame,
        args.visualize
    )
    
    print(f"\n{'='*80}")
    print("Alignment completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
