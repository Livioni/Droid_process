#!/usr/bin/env python3
"""
Align wrist camera using point clouds from one or two aligned third-person cameras.
Uses ICP (Iterative Closest Point) to optimize wrist camera extrinsics.

Usage (with two external cameras):
    python align_wrist_camera.py \
        --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
        --cam-ext2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
        --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
        --output-dir datasets/samples/.../17368348/extrinsics_refined_icp

Usage (with one external camera):
    python align_wrist_camera.py \
        --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
        --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
        --output-dir datasets/samples/.../17368348/extrinsics_refined_icp

Output:
    - {wrist_camera_id}.npy: Aligned wrist camera extrinsics for all frames [N, 3, 4]
    - wrist_alignment_stats.png: Fitness and RMSE plots
"""

import numpy as np
import os
import open3d as o3d
from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import functions from the projection script
from project_pointcloud_to_first_person import (
    create_pointcloud_from_depth,
    transform_points_to_camera2,
    compute_relative_pose
)


def load_extrinsics_npys(camera_dir, camera_id=None, prefer_refined=True):
    """
    Load extrinsics array with priority (refined first by default).
    If prefer_refined is False, only search extrinsics/ npys.
    """
    camera_dir = Path(camera_dir)
    camera_id = camera_id or camera_dir.name

    refined_dir = camera_dir / "extrinsics_refined"
    raw_dir = camera_dir / "extrinsics"

    candidates = []

    if prefer_refined:
        candidates.extend([
            refined_dir / f"{camera_id}.npy",
            refined_dir / f"{camera_id}_left.npy",
        ])
        if refined_dir.exists():
            candidates.extend(sorted(refined_dir.glob("*.npy")))

    # Always try raw extrinsics
    candidates.append(raw_dir / f"{camera_id}_left.npy")
    if raw_dir.exists():
        candidates.extend(sorted(raw_dir.glob("*.npy")))

    for candidate in candidates:
        if candidate.exists():
            try:
                extrinsics_all = np.load(str(candidate), allow_pickle=True)
                print(f"Loaded extrinsics from {candidate}")
                return extrinsics_all
            except Exception as e:
                print(f"  Warning: failed to load extrinsics from {candidate}: {e}")
                continue

    raise FileNotFoundError(
        f"No extrinsics npy found in {refined_dir} or {raw_dir}"
    )

def load_camera_data(camera_dir, frame_idx, depth_dir=None, prefer_refined=True):
    """Load image, depth, intrinsics, and extrinsics for a camera."""
    camera_dir = Path(camera_dir)
    camera_id = camera_dir.name
    frame_idx = int(frame_idx)  # Ensure frame_idx is an integer
    

    # Load image (use left camera for stereo)
    image_path = camera_dir / "images" / "left" / f"{int(frame_idx):06d}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path))

    # Load depth - use specified depth directory if provided, otherwise use default
    if depth_dir is not None:
        depth_path = Path(depth_dir) / f"{int(frame_idx):06d}.npz"
    else:
        depth_path = camera_dir / "depth_npy" / f"{int(frame_idx):06d}.npz"

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    depth = np.load(str(depth_path))['depth']

    # Load intrinsics (use left camera for stereo)
    intrinsics_path = camera_dir / "intrinsics" / f"{camera_id}_left.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
    intrinsics = np.load(str(intrinsics_path))

    # Load extrinsics
    extrinsics_all = load_extrinsics_npys(camera_dir, camera_id, prefer_refined=prefer_refined)

    if frame_idx >= len(extrinsics_all):
        raise ValueError(f"Frame {frame_idx} out of range. Max frame: {len(extrinsics_all)-1}")

    extrinsics = extrinsics_all[frame_idx]  # Shape: (3, 4)

    return image, depth, intrinsics, extrinsics

class WristCameraAligner:
    """Align wrist camera using point clouds from one or two third-person cameras."""

    def __init__(self, cam_ext1_dir, cam_ext2_dir, cam_wrist_dir):
        """
        Initialize the wrist camera aligner.

        Args:
            cam_ext1_dir: First third-person camera directory (reference)
            cam_ext2_dir: Second third-person camera directory (optional)
            cam_wrist_dir: Wrist camera directory
        """
        self.cam_ext1_dir = cam_ext1_dir
        self.cam_ext2_dir = cam_ext2_dir
        self.cam_wrist_dir = cam_wrist_dir
        self.use_ext2 = cam_ext2_dir is not None

        # Load camera paths
        cam_ext1_path = Path(cam_ext1_dir)
        cam_wrist_path = Path(cam_wrist_dir)

        self.cam_ext1_id = cam_ext1_path.name
        self.cam_wrist_id = cam_wrist_path.name

        # Load intrinsics
        intrinsics_path_ext1 = cam_ext1_path / "intrinsics" / f"{self.cam_ext1_id}_left.npy"
        intrinsics_path_wrist = cam_wrist_path / "intrinsics" / f"{self.cam_wrist_id}_left.npy"

        self.K_ext1 = np.load(str(intrinsics_path_ext1))
        self.K_wrist = np.load(str(intrinsics_path_wrist))

        if self.use_ext2:
            cam_ext2_path = Path(cam_ext2_dir)
            self.cam_ext2_id = cam_ext2_path.name
            intrinsics_path_ext2 = cam_ext2_path / "intrinsics" / f"{self.cam_ext2_id}_left.npy"
            self.K_ext2 = np.load(str(intrinsics_path_ext2))

        # Load extrinsics (prefer refined, fallback to raw {camera}_left.npy)
        self.ext1_all = load_extrinsics_npys(cam_ext1_path, self.cam_ext1_id, prefer_refined=True)

        if self.use_ext2:
            self.ext2_all = load_extrinsics_npys(cam_ext2_path, self.cam_ext2_id, prefer_refined=True)
        
        # Load wrist camera extrinsics (will be optimized)
        self.ext_wrist_all = load_extrinsics_npys(cam_wrist_path, self.cam_wrist_id, prefer_refined=False)

        print(f"External camera 1 ({self.cam_ext1_id}): {len(self.ext1_all)} frames")
        if self.use_ext2:
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
                       voxel_size=0.001, visualize=False, icp_levels=3, voxel_factor=2.0,
                       use_multiscale=True, max_depth=10.0):
        """
        Align wrist camera for a single frame using combined point clouds from both external cameras.
        Uses a coarse-to-fine (multiscale) ICP strategy for better robustness by default.
        
        Args:
            frame_idx: Frame index
            max_iterations: Maximum ICP iterations (budget distributed across scales)
            distance_threshold: Base distance threshold for ICP correspondence (in meters)
            voxel_size: Finest voxel size for downsampling (in meters)
            visualize: Whether to visualize point clouds
            icp_levels: Number of multiscale levels (>=1)
            voxel_factor: Multiplicative factor between consecutive voxel sizes
            use_multiscale: Whether to run multiscale ICP (False -> single-scale)
            max_depth: Filter out depth values greater than this (in meters)
            
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
        cam_wrist_depth_dir = os.path.join(self.cam_wrist_dir, "depth_npy")

        _, depth_ext1, _, ext1 = load_camera_data(self.cam_ext1_dir, frame_idx, cam_ext1_depth_dir, prefer_refined=True)
        _, depth_wrist, _, _ = load_camera_data(self.cam_wrist_dir, frame_idx, cam_wrist_depth_dir, prefer_refined=False)

        depth_ext1 = self._clamp_depth(depth_ext1, max_depth)
        depth_wrist = self._clamp_depth(depth_wrist, max_depth)

        if self.use_ext2:
            cam_ext2_depth_dir = os.path.join(self.cam_ext2_dir, "depth_npy")
            _, depth_ext2, _, ext2 = load_camera_data(self.cam_ext2_dir, frame_idx, cam_ext2_depth_dir, prefer_refined=True)
            depth_ext2 = self._clamp_depth(depth_ext2, max_depth)
            # Use loaded ext2 if not using aligned version
            if frame_idx < len(self.ext2_all):
                ext2 = self.ext2_all[frame_idx]
        
        # Create point clouds from external cameras in their own coordinates
        points_3d_ext1, _ = create_pointcloud_from_depth(depth_ext1, self.K_ext1, max_depth=max_depth)

        print(f"  External camera 1 points: {len(points_3d_ext1)}")

        if self.use_ext2:
            points_3d_ext2, _ = create_pointcloud_from_depth(depth_ext2, self.K_ext2, max_depth=max_depth)
            print(f"  External camera 2 points: {len(points_3d_ext2)}")
            if len(points_3d_ext1) < 100 or len(points_3d_ext2) < 100:
                print(f"  Insufficient external camera points, skipping frame {frame_idx}")
                return ext_wrist_initial, 0.0, float('inf')
        else:
            if len(points_3d_ext1) < 100:
                print(f"  Insufficient external camera points, skipping frame {frame_idx}")
                return ext_wrist_initial, 0.0, float('inf')
        
        # Transform external camera points to world coordinates
        T_ext1 = np.eye(4)
        T_ext1[:3, :] = ext1

        points_3d_world_ext1 = self._transform_points(points_3d_ext1, T_ext1)

        if self.use_ext2:
            T_ext2 = np.eye(4)
            T_ext2[:3, :] = ext2
            points_3d_world_ext2 = self._transform_points(points_3d_ext2, T_ext2)
            # Combine point clouds from both external cameras
            points_3d_world_combined = np.vstack([points_3d_world_ext1, points_3d_world_ext2])
        else:
            # Use only first external camera
            points_3d_world_combined = points_3d_world_ext1

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
        points_3d_wrist_gt, _ = create_pointcloud_from_depth(depth_wrist, self.K_wrist, max_depth=max_depth)
        
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

        if use_multiscale:
            reg_result, T_icp = self._run_multiscale_icp(
                source_pcd,
                target_pcd,
                distance_threshold,
                voxel_size,
                max_iterations,
                icp_levels,
                voxel_factor,
            )

            if reg_result is None:
                print("  ICP failed (no valid multiscale levels)")
                return ext_wrist_initial, 0.0, float('inf')
        else:
            # Single-scale fallback: downsample once then run ICP
            source_pcd = source_pcd.voxel_down_sample(voxel_size)
            target_pcd = target_pcd.voxel_down_sample(voxel_size)

            print(f"  Single-scale downsampled - Source: {len(source_pcd.points)}, Target: {len(target_pcd.points)}")

            radius = max(voxel_size * 4.0, 0.01)
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30))
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30))

            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                distance_threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iterations,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )
            T_icp = reg_result.transformation
        
        print(f"  ICP fitness: {reg_result.fitness:.4f}, RMSE: {reg_result.inlier_rmse:.4f}")
        
        # Update wrist camera extrinsics
        # T_world_to_wrist_new = T_icp @ T_world_to_wrist
        # T_wrist_new = inv(T_icp @ inv(T_wrist_initial))
        T_wrist_new = np.linalg.inv(T_icp @ np.linalg.inv(T_wrist_initial))
        
        optimized_ext_wrist = T_wrist_new[:3, :]
        
        # Visualize if requested
        if visualize:
            if int(frame_idx) % 50 == 0:
                self.visualize_point_clouds(source_pcd, target_pcd, T_icp, frame_idx)
        
        return optimized_ext_wrist, reg_result.fitness, reg_result.inlier_rmse

    def _transform_points(self, points, T):
        """Transform 3D points using a 4x4 transformation matrix."""
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        points_transformed = (T @ points_homogeneous.T).T
        return points_transformed[:, :3]

    def _clamp_depth(self, depth, max_depth):
        """Zero out depth values greater than max_depth (keep shape)."""
        if max_depth is None:
            return depth
        depth_clamped = depth.copy()
        depth_clamped = np.where(np.isfinite(depth_clamped) & (depth_clamped <= max_depth), depth_clamped, 0)
        return depth_clamped

    def _build_multiscale_schedule(self, voxel_size, distance_threshold, max_iterations,
                                   icp_levels, voxel_factor):
        """Construct multiscale ICP schedule from finest voxel size and thresholds."""
        levels = max(1, int(icp_levels))
        factor = max(1.0, float(voxel_factor))
        # Coarse-to-fine voxel sizes (largest first)
        voxel_sizes = [voxel_size * (factor ** i) for i in reversed(range(levels))]
        distance_thresholds = [distance_threshold * (factor ** i) for i in reversed(range(levels))]
        
        # Distribute iteration budget across levels (ensure minimum iterations)
        base_iters = max(5, max_iterations // levels)
        max_iters = [base_iters] * (levels - 1)
        max_iters.append(max(max_iterations - base_iters * (levels - 1), base_iters))
        
        return list(zip(voxel_sizes, distance_thresholds, max_iters))

    def _run_multiscale_icp(self, source_pcd, target_pcd, distance_threshold, voxel_size,
                            max_iterations, icp_levels, voxel_factor):
        """Run coarse-to-fine ICP and return the final registration result and transform."""
        schedule = self._build_multiscale_schedule(
            voxel_size, distance_threshold, max_iterations, icp_levels, voxel_factor
        )
        
        current_transform = np.eye(4)
        reg_result = None
        
        for level_idx, (vs, dt, iters) in enumerate(schedule, start=1):
            src_lvl = source_pcd.voxel_down_sample(vs)
            tgt_lvl = target_pcd.voxel_down_sample(vs)
            
            if len(src_lvl.points) == 0 or len(tgt_lvl.points) == 0:
                print(f"  [Level {level_idx}] skipped (empty point cloud after voxel {vs:.4f})")
                continue
            
            # Estimate normals with a radius proportional to voxel size
            radius = max(vs * 4.0, 0.01)
            src_lvl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30))
            tgt_lvl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30))
            
            reg_result = o3d.pipelines.registration.registration_icp(
                src_lvl,
                tgt_lvl,
                dt,
                current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iters,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )
            
            current_transform = reg_result.transformation
            print(
                f"  [Level {level_idx}] voxel={vs:.4f}, thr={dt:.4f}, "
                f"iters={iters}, fitness={reg_result.fitness:.4f}, rmse={reg_result.inlier_rmse:.4f}"
            )
        
        return reg_result, current_transform

    def align_all_frames(self, output_dir, max_iterations=50, distance_threshold=0.05,
                        voxel_size=0.001, start_frame=None, end_frame=None, visualize=False,
                        icp_levels=3, voxel_factor=2.0, use_multiscale=True, max_depth=10.0):
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
            icp_levels: Number of multiscale levels (>=1)
            voxel_factor: Multiplicative factor between coarse-to-fine voxel sizes
            use_multiscale: Whether to run multiscale ICP (False -> single-scale)
            max_depth: Discard depth values greater than this (meters)
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
                    frame_idx, max_iterations, distance_threshold, voxel_size, visualize,
                    icp_levels, voxel_factor, use_multiscale, max_depth
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
        default=None,
        help="Second third-person camera directory (optional)"
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
        default=80,
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
        "--icp-levels",
        type=int,
        default=3,
        help="Number of coarse-to-fine ICP levels (>=1)"
    )
    parser.add_argument(
        "--voxel-factor",
        type=float,
        default=2.0,
        help="Multiplicative factor between coarse and fine voxel sizes"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=1.0,
        help="Ignore depth values greater than this (meters)"
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        default=False,
        help="Disable multiscale ICP and use single-scale ICP"
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
    if args.cam_ext2 is not None:
        print("Wrist Camera Alignment using Combined Third-Person Point Clouds")
    else:
        print("Wrist Camera Alignment using Single Third-Person Point Cloud")
    print("="*80)
    print(f"External camera 1 (reference): {args.cam_ext1}")
    if args.cam_ext2 is not None:
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
        args.visualize,
        args.icp_levels,
        args.voxel_factor,
        args.multiscale,
        args.max_depth
    )
    
    print(f"\n{'='*80}")
    print("Alignment completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
