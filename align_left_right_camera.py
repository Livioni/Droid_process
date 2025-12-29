#!/usr/bin/env python3
"""
Align two third-person cameras using ICP (Iterative Closest Point).
Since the cameras are static, we align using the first N frames and average the transformation.

Usage:
    python align_left_right_camera.py \
        --cam1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
        --cam2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
        --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp \
        --num-frames 10

Output:
    - {camera_id}.npy: Aligned extrinsics for all frames [N, 3, 4]
    - alignment_stats.png: Fitness and RMSE plots
"""

import numpy as np
import os
import open3d as o3d
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import functions from the projection script
from project_pointcloud_to_first_person import (
    load_camera_data,
    create_pointcloud_from_depth,
)


class StaticCameraAligner:
    """Align two static third-person cameras using ICP."""

    def __init__(self, cam1_dir, cam2_dir):
        """
        Initialize the aligner.
        
        Args:
            cam1_dir: Reference camera directory (camera 1)
            cam2_dir: Camera to be aligned (camera 2)
        """
        self.cam1_dir = cam1_dir
        self.cam2_dir = cam2_dir

        # Load camera intrinsics
        cam1_path = Path(cam1_dir)
        cam2_path = Path(cam2_dir)
        
        self.cam1_id = cam1_path.name
        self.cam2_id = cam2_path.name
        
        # Load intrinsics
        intrinsics_path1 = cam1_path / "intrinsics" / f"{self.cam1_id}_left.npy"
        intrinsics_path2 = cam2_path / "intrinsics" / f"{self.cam2_id}_left.npy"
        
        self.K1 = np.load(str(intrinsics_path1))
        self.K2 = np.load(str(intrinsics_path2))

        # Load camera extrinsics
        extrinsics_dir1 = cam1_path / "extrinsics"
        extrinsics_files1 = list(extrinsics_dir1.glob("*_left.npy"))
        self.ext1_all = np.load(str(extrinsics_files1[0]))
        
        extrinsics_dir2 = cam2_path / "extrinsics"
        try:
            extrinsics_files2 = list(extrinsics_dir2.glob("*_ma.npy"))
            self.ext2_all = np.load(str(extrinsics_files2[0]))
            print(f"Loaded extrinsics from {extrinsics_files2[0]}")
        except:
            extrinsics_files2 = list(extrinsics_dir2.glob("*_left.npy"))
            self.ext2_all = np.load(str(extrinsics_files2[0]))
            print(f"Loaded extrinsics from {extrinsics_files2[0]}")

        print(f"Camera 1 ({self.cam1_id}): {len(self.ext1_all)} frames")
        print(f"Camera 2 ({self.cam2_id}): {len(self.ext2_all)} frames")

    def align_single_frame(self, frame_idx, max_iterations=50, distance_threshold=0.05,
                          voxel_size=0.01, max_depth=10.0, visualize=False):
        """
        Align camera 2 to camera 1 for a single frame using ICP.
        
        Args:
            frame_idx: Frame index to align
            max_iterations: Maximum ICP iterations
            distance_threshold: Distance threshold for ICP correspondence (in meters)
            voxel_size: Voxel size for downsampling (in meters)
            max_depth: Only use points with depth <= max_depth meters when building point clouds
            visualize: Whether to visualize point clouds
            
        Returns:
            T_align: 4x4 transformation matrix to align camera 2 to camera 1
            fitness: ICP fitness score
            rmse: ICP RMSE
        """
        print(f"\nAligning frame {frame_idx}...")
        
        # Load depth maps
        cam1_depth_dir = os.path.join(self.cam1_dir, "depth_npy")
        cam2_depth_dir = os.path.join(self.cam2_dir, "depth_npy")
        
        _, depth1, _, ext1 = load_camera_data(self.cam1_dir, frame_idx, cam1_depth_dir)
        _, depth2, _, ext2 = load_camera_data(self.cam2_dir, frame_idx, cam2_depth_dir)
        
        # Create point clouds in camera coordinates (depth <= max_depth), then transform to world
        points_3d_cam1, _ = create_pointcloud_from_depth(depth1, self.K1, max_depth=float(max_depth))
        points_3d_cam2, _ = create_pointcloud_from_depth(depth2, self.K2, max_depth=float(max_depth))
        
        print(f"  Camera 1 points: {len(points_3d_cam1)}")
        print(f"  Camera 2 points: {len(points_3d_cam2)}")
        
        if len(points_3d_cam1) < 100 or len(points_3d_cam2) < 100:
            print(f"  Insufficient points, skipping frame {frame_idx}")
            return None, 0.0, float('inf')
        
        # Transform points to world coordinates
        T1 = np.eye(4)
        T1[:3, :] = ext1
        
        T2 = np.eye(4)
        T2[:3, :] = ext2
        
        # Transform camera coordinates to world coordinates
        points_3d_world1 = self._transform_points(points_3d_cam1, T1)
        points_3d_world2 = self._transform_points(points_3d_cam2, T2)
        
        # Convert to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(points_3d_world2)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_3d_world1)
        
        # Downsample for efficiency
        source_pcd = source_pcd.voxel_down_sample(voxel_size)
        target_pcd = target_pcd.voxel_down_sample(voxel_size)
        
        print(f"  After downsampling - Source: {len(source_pcd.points)}, Target: {len(target_pcd.points)}")
        
        # Estimate normals for better ICP
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        
        # Run ICP - align source (cam2) to target (cam1)
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
        
        # The transformation aligns camera 2 world points to camera 1 world points
        T_align = reg_result.transformation
        
        # Visualize if requested
        if visualize:
            self.visualize_point_clouds(source_pcd, target_pcd, T_align, frame_idx)
        
        return T_align, reg_result.fitness, reg_result.inlier_rmse

    def align_single_frame_robust(self, frame_idx, max_iterations=50, distance_threshold=0.05,
                                voxel_size=0.01, max_depth=10.0, visualize=False,
                                prev_transform=None):
        """
        Robust single frame alignment using multi-scale ICP and better initialization.

        Args:
            frame_idx: Frame index to align
            max_iterations: Maximum ICP iterations
            distance_threshold: Distance threshold for ICP correspondence (in meters)
            voxel_size: Voxel size for downsampling (in meters)
            max_depth: Only use points with depth <= max_depth meters when building point clouds
            visualize: Whether to visualize point clouds
            prev_transform: Previous frame's transformation for initialization

        Returns:
            T_align: 4x4 transformation matrix to align camera 2 to camera 1
            fitness: ICP fitness score
            rmse: ICP RMSE
        """
        print(f"\nAligning frame {frame_idx} (robust mode)...")

        # Load depth maps
        cam1_depth_dir = os.path.join(self.cam1_dir, "depth_npy")
        cam2_depth_dir = os.path.join(self.cam2_dir, "depth_npy")

        _, depth1, _, ext1 = load_camera_data(self.cam1_dir, frame_idx, cam1_depth_dir)
        _, depth2, _, ext2 = load_camera_data(self.cam2_dir, frame_idx, cam2_depth_dir)

        # Create point clouds in camera coordinates (depth <= max_depth)
        points_3d_cam1, _ = create_pointcloud_from_depth(depth1, self.K1, max_depth=float(max_depth))
        points_3d_cam2, _ = create_pointcloud_from_depth(depth2, self.K2, max_depth=float(max_depth))

        print(f"  Camera 1 points: {len(points_3d_cam1)}")
        print(f"  Camera 2 points: {len(points_3d_cam2)}")

        if len(points_3d_cam1) < 100 or len(points_3d_cam2) < 100:
            print(f"  Insufficient points, skipping frame {frame_idx}")
            return None, 0.0, float('inf')

        # Transform points to world coordinates
        T1 = np.eye(4)
        T1[:3, :] = ext1

        T2 = np.eye(4)
        T2[:3, :] = ext2

        points_3d_world1 = self._transform_points(points_3d_cam1, T1)
        points_3d_world2 = self._transform_points(points_3d_cam2, T2)

        # Convert to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(points_3d_world2)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_3d_world1)

        # Multi-scale ICP: coarse to fine registration
        voxel_sizes = [0.02, 0.01, 0.005]  # coarse to fine
        distance_thresholds = [0.1, 0.05, 0.02]  # corresponding thresholds

        current_transform = prev_transform if prev_transform is not None else np.eye(4)

        for i, (vox_size, dist_thresh) in enumerate(zip(voxel_sizes, distance_thresholds)):
            print(f"  Multi-scale ICP level {i+1}: voxel={vox_size}, threshold={dist_thresh}")

            # Downsample
            source_down = source_pcd.voxel_down_sample(vox_size)
            target_down = target_pcd.voxel_down_sample(vox_size)

            if len(source_down.points) < 50 or len(target_down.points) < 50:
                print(f"  Too few points after downsampling, using original voxel size")
                source_down = source_pcd.voxel_down_sample(voxel_size)
                target_down = target_pcd.voxel_down_sample(voxel_size)

            # Estimate normals
            source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=vox_size*2, max_nn=30))
            target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=vox_size*2, max_nn=30))

            # Apply current transform to source
            source_down.transform(current_transform)

            # Run ICP
            reg_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, dist_thresh, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iterations // len(voxel_sizes),
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )

            # Update transform
            current_transform = reg_result.transformation @ current_transform
            print(f"    Level {i+1}: fitness={reg_result.fitness:.4f}, rmse={reg_result.inlier_rmse:.4f}")

            # Early stop if fitness is very good
            if reg_result.fitness > 0.8 and reg_result.inlier_rmse < 0.02:
                break

        T_align = current_transform
        final_fitness = reg_result.fitness
        final_rmse = reg_result.inlier_rmse

        print(f"  Final ICP: fitness={final_fitness:.4f}, RMSE={final_rmse:.4f}")

        # Visualize if requested
        if visualize:
            self.visualize_point_clouds(source_pcd, target_pcd, T_align, frame_idx)

        return T_align, final_fitness, final_rmse

    def align_single_frame_ransac(self, frame_idx, max_iterations=50, distance_threshold=0.05,
                                voxel_size=0.01, max_depth=10.0, visualize=False):
        """
        RANSAC-based alignment using feature matching followed by ICP refinement.

        Args:
            frame_idx: Frame index to align
            max_iterations: Maximum ICP iterations for refinement
            distance_threshold: Distance threshold for ICP correspondence (in meters)
            voxel_size: Voxel size for downsampling (in meters)
            max_depth: Only use points with depth <= max_depth meters when building point clouds
            visualize: Whether to visualize point clouds

        Returns:
            T_align: 4x4 transformation matrix to align camera 2 to camera 1
            fitness: ICP fitness score
            rmse: ICP RMSE
        """
        print(f"\nAligning frame {frame_idx} (RANSAC-ICP mode)...")

        # Load depth maps
        cam1_depth_dir = os.path.join(self.cam1_dir, "depth_npy")
        cam2_depth_dir = os.path.join(self.cam2_dir, "depth_npy")

        _, depth1, _, ext1 = load_camera_data(self.cam1_dir, frame_idx, cam1_depth_dir)
        _, depth2, _, ext2 = load_camera_data(self.cam2_dir, frame_idx, cam2_depth_dir)

        # Create point clouds in camera coordinates (depth <= max_depth)
        points_3d_cam1, _ = create_pointcloud_from_depth(depth1, self.K1, max_depth=float(max_depth))
        points_3d_cam2, _ = create_pointcloud_from_depth(depth2, self.K2, max_depth=float(max_depth))

        print(f"  Camera 1 points: {len(points_3d_cam1)}")
        print(f"  Camera 2 points: {len(points_3d_cam2)}")

        if len(points_3d_cam1) < 100 or len(points_3d_cam2) < 100:
            print(f"  Insufficient points, skipping frame {frame_idx}")
            return None, 0.0, float('inf')

        # Transform points to world coordinates
        T1 = np.eye(4)
        T1[:3, :] = ext1

        T2 = np.eye(4)
        T2[:3, :] = ext2

        points_3d_world1 = self._transform_points(points_3d_cam1, T1)
        points_3d_world2 = self._transform_points(points_3d_cam2, T2)

        # Convert to Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(points_3d_world2)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(points_3d_world1)

        # Downsample for efficiency
        source_pcd = source_pcd.voxel_down_sample(voxel_size)
        target_pcd = target_pcd.voxel_down_sample(voxel_size)

        print(f"  After downsampling - Source: {len(source_pcd.points)}, Target: {len(target_pcd.points)}")

        # Estimate normals for feature computation
        radius_normal = voxel_size * 2
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))

        # Compute FPFH features
        radius_feature = voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        # RANSAC-based registration
        distance_threshold_ransac = voxel_size * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold_ransac,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold_ransac)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        print(f"  RANSAC: fitness={result_ransac.fitness:.4f}, inlier_rmse={result_ransac.inlier_rmse:.4f}")

        # Refine with ICP using point-to-plane
        distance_threshold_icp = distance_threshold_ransac * 0.5
        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, distance_threshold_icp, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )

        print(f"  ICP refinement: fitness={result_icp.fitness:.4f}, inlier_rmse={result_icp.inlier_rmse:.4f}")

        T_align = result_icp.transformation
        fitness = result_icp.fitness
        rmse = result_icp.inlier_rmse

        # Visualize if requested
        if visualize:
            self.visualize_point_clouds(source_pcd, target_pcd, T_align, frame_idx)

        return T_align, fitness, rmse

    def _transform_points(self, points, T):
        """Transform 3D points using a 4x4 transformation matrix."""
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        points_transformed = (T @ points_homogeneous.T).T
        return points_transformed[:, :3]

    def align_cameras_robust(self, num_frames=10, max_iterations=50, distance_threshold=0.05,
                            voxel_size=0.01, max_depth=10.0, visualize=False,
                            min_fitness=0.3, max_rmse=0.1, use_weighted_average=True):
        """
        Robust alignment using quality-based filtering and weighted averaging.

        Args:
            num_frames: Number of frames to use for alignment
            max_iterations: Maximum ICP iterations per frame
            distance_threshold: Distance threshold for ICP correspondence
            voxel_size: Voxel size for downsampling
            max_depth: Only use points with depth <= max_depth meters when building point clouds for ICP
            visualize: Whether to visualize point clouds
            min_fitness: Minimum fitness score to accept alignment
            max_rmse: Maximum RMSE to accept alignment
            use_weighted_average: Whether to use fitness-weighted average instead of simple average

        Returns:
            T_avg: Average transformation matrix
            transformations: List of transformation matrices for each frame
            fitness_scores: List of fitness scores
            rmse_scores: List of RMSE scores
        """
        print(f"Aligning cameras using first {num_frames} frames (robust mode)...")

        transformations = []
        fitness_scores = []
        rmse_scores = []

        for frame_idx in range(min(num_frames, len(self.ext1_all), len(self.ext2_all))):
            try:
                T_align, fitness, rmse = self.align_single_frame(
                    frame_idx, max_iterations, distance_threshold, voxel_size, max_depth, visualize
                )

                if T_align is not None and fitness > min_fitness and rmse < max_rmse:
                    transformations.append(T_align)
                    fitness_scores.append(fitness)
                    rmse_scores.append(rmse)
                    print(f"  Frame {frame_idx}: fitness={fitness:.4f}, rmse={rmse:.4f} ✓")
                else:
                    print(f"  Frame {frame_idx}: fitness={fitness:.4f}, rmse={rmse:.4f} ✗ (filtered)")

            except Exception as e:
                print(f"  Error aligning frame {frame_idx}: {e}")
                continue

        if len(transformations) == 0:
            raise ValueError("No successful alignments found!")

        print(f"\nSuccessfully aligned {len(transformations)}/{num_frames} frames")
        print(f"Average fitness: {np.mean(fitness_scores):.4f}")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

        # Use weighted average by fitness scores
        if use_weighted_average and len(transformations) > 1:
            T_avg = self._weighted_average_transformations(transformations, fitness_scores)
            print("Used fitness-weighted averaging")
        else:
            T_avg = self._average_transformations(transformations)

        return T_avg, transformations, fitness_scores, rmse_scores

    def align_cameras_multiscale(self, num_frames=10, max_iterations=50, max_depth=10.0,
                                visualize=False, min_fitness=0.3, max_rmse=0.1):
        """
        Multi-scale alignment with temporal consistency (uses previous frame for initialization).

        Args:
            num_frames: Number of frames to use for alignment
            max_iterations: Maximum ICP iterations per frame
            max_depth: Only use points with depth <= max_depth meters when building point clouds for ICP
            visualize: Whether to visualize point clouds
            min_fitness: Minimum fitness score to accept alignment
            max_rmse: Maximum RMSE to accept alignment

        Returns:
            T_avg: Average transformation matrix
            transformations: List of transformation matrices for each frame
            fitness_scores: List of fitness scores
            rmse_scores: List of RMSE scores
        """
        print(f"Aligning cameras using first {num_frames} frames (multi-scale with temporal consistency)...")

        transformations = []
        fitness_scores = []
        rmse_scores = []

        prev_transform = None

        for frame_idx in range(min(num_frames, len(self.ext1_all), len(self.ext2_all))):
            try:
                T_align, fitness, rmse = self.align_single_frame_robust(
                    frame_idx, max_iterations, visualize=visualize,
                    max_depth=max_depth, prev_transform=prev_transform
                )

                if T_align is not None and fitness > min_fitness and rmse < max_rmse:
                    transformations.append(T_align)
                    fitness_scores.append(fitness)
                    rmse_scores.append(rmse)
                    prev_transform = T_align  # Use this frame's result to initialize next frame
                    print(f"  Frame {frame_idx}: fitness={fitness:.4f}, rmse={rmse:.4f} ✓")
                else:
                    print(f"  Frame {frame_idx}: fitness={fitness:.4f}, rmse={rmse:.4f} ✗ (filtered)")
                    # Keep previous transform for next frame initialization
                    if prev_transform is None:
                        prev_transform = np.eye(4)  # fallback

            except Exception as e:
                print(f"  Error aligning frame {frame_idx}: {e}")
                continue

        if len(transformations) == 0:
            raise ValueError("No successful alignments found!")

        print(f"\nSuccessfully aligned {len(transformations)}/{num_frames} frames")
        print(f"Average fitness: {np.mean(fitness_scores):.4f}")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

        # Use weighted average by fitness scores
        if len(transformations) > 1:
            T_avg = self._weighted_average_transformations(transformations, fitness_scores)
            print("Used fitness-weighted averaging")
        else:
            T_avg = transformations[0]

        return T_avg, transformations, fitness_scores, rmse_scores

    def align_cameras_ransac(self, num_frames=10, max_iterations=50, distance_threshold=0.05,
                            voxel_size=0.01, max_depth=10.0, visualize=False,
                            min_fitness=0.2, max_rmse=0.1):
        """
        RANSAC-based alignment for multiple frames with quality filtering.

        Args:
            num_frames: Number of frames to use for alignment
            max_iterations: Maximum ICP iterations for refinement
            distance_threshold: Distance threshold for ICP correspondence
            voxel_size: Voxel size for downsampling
            max_depth: Only use points with depth <= max_depth meters when building point clouds for ICP
            visualize: Whether to visualize point clouds
            min_fitness: Minimum fitness score to accept alignment
            max_rmse: Maximum RMSE to accept alignment

        Returns:
            T_avg: Average transformation matrix
            transformations: List of transformation matrices for each frame
            fitness_scores: List of fitness scores
            rmse_scores: List of RMSE scores
        """
        print(f"Aligning cameras using first {num_frames} frames (RANSAC-ICP mode)...")

        transformations = []
        fitness_scores = []
        rmse_scores = []

        for frame_idx in range(min(num_frames, len(self.ext1_all), len(self.ext2_all))):
            try:
                T_align, fitness, rmse = self.align_single_frame_ransac(
                    frame_idx, max_iterations, distance_threshold, voxel_size, max_depth, visualize
                )

                if T_align is not None and fitness > min_fitness and rmse < max_rmse:
                    transformations.append(T_align)
                    fitness_scores.append(fitness)
                    rmse_scores.append(rmse)
                    print(f"  Frame {frame_idx}: fitness={fitness:.4f}, rmse={rmse:.4f} ✓")
                else:
                    print(f"  Frame {frame_idx}: fitness={fitness:.4f}, rmse={rmse:.4f} ✗ (filtered)")

            except Exception as e:
                print(f"  Error aligning frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(transformations) == 0:
            raise ValueError("No successful alignments found!")

        print(f"\nSuccessfully aligned {len(transformations)}/{num_frames} frames")
        print(f"Average fitness: {np.mean(fitness_scores):.4f}")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

        # Use weighted average by fitness scores
        if len(transformations) > 1:
            T_avg = self._weighted_average_transformations(transformations, fitness_scores)
            print("Used fitness-weighted averaging")
        else:
            T_avg = transformations[0]

        return T_avg, transformations, fitness_scores, rmse_scores

    def align_cameras(self, num_frames=10, max_iterations=50, distance_threshold=0.05,
                     voxel_size=0.01, max_depth=10.0, visualize=False):
        """
        Align camera 2 to camera 1 using the first N frames and averaging.

        Args:
            num_frames: Number of frames to use for alignment
            max_iterations: Maximum ICP iterations per frame
            distance_threshold: Distance threshold for ICP correspondence
            voxel_size: Voxel size for downsampling
            max_depth: Only use points with depth <= max_depth meters when building point clouds for ICP
            visualize: Whether to visualize point clouds

        Returns:
            T_avg: Average transformation matrix
            transformations: List of transformation matrices for each frame
            fitness_scores: List of fitness scores
            rmse_scores: List of RMSE scores
        """
        return self.align_cameras_robust(num_frames, max_iterations, distance_threshold,
                                       voxel_size, max_depth, visualize,
                                       min_fitness=0.5, max_rmse=float('inf'), use_weighted_average=False)

    def _average_transformations(self, transformations):
        """
        Average multiple transformation matrices.
        Uses simple averaging for rotation and translation.
        """
        # Extract rotations and translations
        rotations = [T[:3, :3] for T in transformations]
        translations = [T[:3, 3] for T in transformations]

        # Average rotation matrices (simple average, then orthogonalize)
        R_avg = np.mean(rotations, axis=0)

        # Orthogonalize using SVD
        U, _, Vt = np.linalg.svd(R_avg)
        R_avg = U @ Vt

        # Average translations
        t_avg = np.mean(translations, axis=0)

        # Construct averaged transformation
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg

        return T_avg

    def _weighted_average_transformations(self, transformations, weights):
        """
        Average multiple transformation matrices with weights.
        Uses fitness-weighted averaging for better robustness.
        """
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Extract rotations and translations
        rotations = [T[:3, :3] for T in transformations]
        translations = [T[:3, 3] for T in transformations]

        # Weighted average of rotation matrices
        R_avg = np.zeros((3, 3))
        for R, w in zip(rotations, weights):
            R_avg += w * R

        # Orthogonalize using SVD
        U, _, Vt = np.linalg.svd(R_avg)
        R_avg = U @ Vt

        # Weighted average of translations
        t_avg = np.zeros(3)
        for t, w in zip(translations, weights):
            t_avg += w * t

        # Construct averaged transformation
        T_avg = np.eye(4)
        T_avg[:3, :3] = R_avg
        T_avg[:3, 3] = t_avg

        return T_avg

    def apply_alignment_to_all_frames(self, T_align, output_dir):
        """
        Apply the alignment transformation to all frames of camera 2.
        
        Args:
            T_align: 4x4 transformation matrix to align camera 2 to camera 1
            output_dir: Output directory for aligned extrinsics
            
        Returns:
            aligned_ext2_all: Aligned extrinsics for all frames [N, 3, 4]
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nApplying alignment to all {len(self.ext2_all)} frames...")
        
        # Initialize aligned extrinsics array
        aligned_ext2_all = np.zeros_like(self.ext2_all)
        
        for frame_idx in tqdm(range(len(self.ext2_all)), desc="Applying alignment"):
            # Get original extrinsics
            ext2 = self.ext2_all[frame_idx]
            
            # Convert to 4x4 matrix
            T2 = np.eye(4)
            T2[:3, :] = ext2
            
            # Apply alignment: T2_aligned = T_align @ T2
            T2_aligned = T_align @ T2
            
            # Store aligned extrinsics
            aligned_ext2_all[frame_idx] = T2_aligned[:3, :]
        
        # Save aligned extrinsics
        output_file = output_path / f"{self.cam2_id}.npy"
        np.save(output_file, aligned_ext2_all)
        print(f"Saved aligned extrinsics to: {output_file}")
        print(f"Output shape: {aligned_ext2_all.shape}")
        
        return aligned_ext2_all

    def visualize_point_clouds(self, source_pcd, target_pcd, transform, frame_idx):
        """Visualize point clouds before and after ICP alignment."""
        try:
            # Create a copy for visualization
            source_transformed = o3d.geometry.PointCloud(source_pcd)
            source_transformed.transform(transform)
            
            # Color the point clouds
            source_pcd.paint_uniform_color([1, 0, 0])  # Red: original source (cam2)
            source_transformed.paint_uniform_color([0, 1, 0])  # Green: transformed source
            target_pcd.paint_uniform_color([0, 0, 1])  # Blue: target (cam1)
            
            # Save visualization
            vis_dir = Path("visualizations_alignment")
            vis_dir.mkdir(exist_ok=True)
            
            # Save point clouds as PLY files
            ply_path_source = vis_dir / f"align_frame_{frame_idx:06d}_cam2_original.ply"
            ply_path_transformed = vis_dir / f"align_frame_{frame_idx:06d}_cam2_aligned.ply"
            ply_path_target = vis_dir / f"align_frame_{frame_idx:06d}_cam1_reference.ply"
            
            o3d.io.write_point_cloud(str(ply_path_source), source_pcd)
            o3d.io.write_point_cloud(str(ply_path_transformed), source_transformed)
            o3d.io.write_point_cloud(str(ply_path_target), target_pcd)
            
            print(f"  Saved point clouds: {ply_path_source.parent}")
                
        except Exception as e:
            print(f"  Warning: Visualization failed for frame {frame_idx}: {e}")

    def plot_alignment_stats(self, fitness_scores, rmse_scores, output_dir):
        """Plot alignment statistics."""
        output_path = Path(output_dir)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Fitness scores
        axes[0].plot(fitness_scores, 'o-')
        axes[0].axhline(y=np.mean(fitness_scores), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(fitness_scores):.4f}')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('ICP Fitness')
        axes[0].set_title('ICP Fitness Scores')
        axes[0].legend()
        axes[0].grid(True)
        
        # RMSE scores
        axes[1].plot(rmse_scores, 'o-')
        axes[1].axhline(y=np.mean(rmse_scores), color='r', linestyle='--',
                       label=f'Mean: {np.mean(rmse_scores):.4f}')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('ICP RMSE (m)')
        axes[1].set_title('ICP RMSE Scores')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / "alignment_stats.png", dpi=150)
        print(f"Saved statistics plot: {output_path / 'alignment_stats.png'}")
        plt.close()


def create_argument_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Align two static third-person cameras using ICP"
    )
    parser.add_argument(
        "--cam1",
        default="datasets/samples/Fri_Jul__7_09:42:23_2023/22008760",
        help="Reference camera directory (camera 1)"
    )
    parser.add_argument(
        "--cam2",
        default="datasets/samples/Fri_Jul__7_09:42:23_2023/24400334",
        help="Camera to be aligned (camera 2)"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/samples/Fri_Jul__7_09:42:23_2023/24400334/extrinsics_refined",
        help="Output directory for aligned extrinsics"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to use for alignment"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum ICP iterations per frame"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.05,
        help="ICP correspondence distance threshold (in meters)"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=1.0,
        help="Only use points with depth <= max_depth meters when building point clouds for ICP"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.001,
        help="Voxel size for downsampling (in meters)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable point cloud visualization during alignment"
    )
    parser.add_argument(
        "--method",
        choices=["original", "robust", "multiscale", "ransac"],
        default="robust",
        help="Alignment method: original (basic), robust (quality filtering), multiscale (temporal consistency), ransac (feature-based)"
    )
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.4,
        help="Minimum fitness score to accept alignment (for robust/multiscale methods)"
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=0.1,
        help="Maximum RMSE to accept alignment (for robust/multiscale methods)"
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    print("="*80)
    print(f"Static Camera Alignment using ICP ({args.method} method)")
    print("="*80)
    print(f"Reference camera (cam1): {args.cam1}")
    print(f"Camera to align (cam2): {args.cam2}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of frames for alignment: {args.num_frames}")
    print(f"Method: {args.method}")

    # Create aligner
    aligner = StaticCameraAligner(args.cam1, args.cam2)

    # Choose alignment method
    if args.method == "original":
        T_avg, transformations, fitness_scores, rmse_scores = aligner.align_cameras(
            args.num_frames,
            args.max_iterations,
            args.distance_threshold,
            args.voxel_size,
            args.max_depth,
            args.visualize
        )
    elif args.method == "robust":
        T_avg, transformations, fitness_scores, rmse_scores = aligner.align_cameras_robust(
            args.num_frames,
            args.max_iterations,
            args.distance_threshold,
            args.voxel_size,
            args.max_depth,
            args.visualize,
            args.min_fitness,
            args.max_rmse,
            True  # use weighted average
        )
    elif args.method == "multiscale":
        T_avg, transformations, fitness_scores, rmse_scores = aligner.align_cameras_multiscale(
            args.num_frames,
            args.max_iterations,
            args.max_depth,
            args.visualize,
            args.min_fitness,
            args.max_rmse
        )
    elif args.method == "ransac":
        T_avg, transformations, fitness_scores, rmse_scores = aligner.align_cameras_ransac(
            args.num_frames,
            args.max_iterations,
            args.distance_threshold,
            args.voxel_size,
            args.max_depth,
            args.visualize,
            args.min_fitness,
            args.max_rmse
        )

    print("\nAverage transformation matrix:")
    print(T_avg)

    # Apply alignment to all frames
    aligned_ext2_all = aligner.apply_alignment_to_all_frames(T_avg, args.output_dir)

    # Plot statistics
    aligner.plot_alignment_stats(fitness_scores, rmse_scores, args.output_dir)

    print(f"\n{'='*80}")
    print("Alignment completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
