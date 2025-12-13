#!/usr/bin/env python3
"""
Project third-person point cloud to first-person view.
Reprojects depth information from camera 1 (third-person) to camera 2 (first-person) view.
"""

import numpy as np
import os
import cv2
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Standalone implementation - no external dependencies on MatchAnything
import numpy as np
import cv2
from pathlib import Path

# Optional CUDA acceleration
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CUDA acceleration available with {torch.cuda.device_count()} device(s)")
    else:
        print("CUDA not available, using CPU")
except ImportError:
    CUDA_AVAILABLE = False
    print("PyTorch not available, using CPU only")


def load_camera_data(camera_dir, frame_idx, depth_dir=None):
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

    # Load extrinsics (use left camera for stereo)
    extrinsics_dir = camera_dir / "extrinsics"
    extrinsics_path = extrinsics_dir / f"{camera_id}_left.npy"
    if not extrinsics_path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {extrinsics_path}")
    extrinsics_all = np.load(str(extrinsics_path))

    if frame_idx >= len(extrinsics_all):
        raise ValueError(f"Frame {frame_idx} out of range. Max frame: {len(extrinsics_all)-1}")

    extrinsics = extrinsics_all[frame_idx]  # Shape: (3, 4)

    return image, depth, intrinsics, extrinsics


def unproject_pixel_to_3d(u, v, depth, K):
    """
    Unproject a pixel to 3D point using depth and intrinsics.

    Args:
        u, v: Pixel coordinates
        depth: Depth value at (u, v)
        K: 3x3 intrinsics matrix

    Returns:
        X: 3D point in camera coordinates [x, y, z]
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return np.array([x, y, z])


def project_3d_to_pixel(X, K):
    """
    Project a 3D point to pixel coordinates.

    Args:
        X: 3D point [x, y, z]
        K: 3x3 intrinsics matrix

    Returns:
        u, v: Pixel coordinates
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * X[0] / X[2] + cx
    v = fy * X[1] / X[2] + cy

    return np.array([u, v])


def compute_relative_pose(T1_world, T2_world):
    """
    Compute relative pose from camera 1 to camera 2.

    Args:
        T1_world: 3x4 extrinsics [R|t] (camera-to-world transform)
        T2_world: 3x4 extrinsics [R|t] (camera-to-world transform)

    Returns:
        T_1to2: 4x4 transform from camera 1 to camera 2

    Note:
        Extrinsics [R|t] represent camera-to-world transform.
        To get cam1-to-cam2: T_1to2 = inv(T2_world) @ T1_world
        This transforms points from cam1 frame to cam2 frame.
    """
    # Convert extrinsics [R|t] to 4x4 matrices
    T1 = np.eye(4)
    T1[:3, :] = T1_world

    T2 = np.eye(4)
    T2[:3, :] = T2_world

    # Compute relative transform from camera 1 to camera 2
    # Since extrinsics are camera-to-world:
    # T_1to2 transforms points from cam1 coordinates to cam2 coordinates
    # T_1to2 = inv(T2_world) @ T1_world
    T_1to2 = np.linalg.inv(T2) @ T1

    return T_1to2


def create_pointcloud_from_depth(depth_map, intrinsics, max_depth=10.0):
    """
    Create 3D point cloud from depth map and camera intrinsics.

    Args:
        depth_map: HxW depth map
        intrinsics: 3x3 camera intrinsics matrix
        max_depth: Maximum valid depth value

    Returns:
        points_3d: Nx3 array of 3D points in camera coordinates
        valid_mask: HxW boolean mask of valid pixels
    """
    H, W = depth_map.shape

    # Create coordinate grids
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    u_coords = u_coords.astype(np.float32)
    v_coords = v_coords.astype(np.float32)

    # Vectorized unprojection
    depth_flat = depth_map.flatten()
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()

    # Create valid mask
    valid_mask = (depth_flat > 0) & (depth_flat <= max_depth)
    valid_mask = valid_mask.reshape(H, W)

    # Only process valid points
    valid_indices = np.where(valid_mask.flatten())[0]
    u_valid = u_flat[valid_indices]
    v_valid = v_flat[valid_indices]
    depth_valid = depth_flat[valid_indices]

    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), valid_mask

    # Vectorized 3D point calculation
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid

    points_3d = np.column_stack([x, y, z])

    return points_3d, valid_mask


def transform_points_to_camera2(points_3d_cam1, T_1to2):
    """
    Transform 3D points from camera 1 coordinates to camera 2 coordinates.

    Args:
        points_3d_cam1: Nx3 array of 3D points in camera 1 coordinates
        T_1to2: 4x4 transform matrix from camera 1 to camera 2

    Returns:
        points_3d_cam2: Nx3 array of 3D points in camera 2 coordinates
    """
    # Convert to homogeneous coordinates
    points_homogeneous = np.column_stack([points_3d_cam1, np.ones(len(points_3d_cam1))])

    # Transform points
    points_transformed = (T_1to2 @ points_homogeneous.T).T

    # Convert back to 3D coordinates
    points_3d_cam2 = points_transformed[:, :3]

    return points_3d_cam2


def project_points_to_depth_image(points_3d_cam2, intrinsics_cam2, image_shape):
    """
    Project 3D points to depth image in camera 2 view.

    Args:
        points_3d_cam2: Nx3 array of 3D points in camera 2 coordinates
        intrinsics_cam2: 3x3 camera intrinsics matrix for camera 2
        image_shape: (H, W) tuple for output image dimensions

    Returns:
        depth_image: HxW depth image
        valid_mask: HxW boolean mask of valid projections
        proj_stats: Dictionary with projection statistics
    """
    H, W = image_shape
    depth_image = np.full((H, W), np.nan, dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    if len(points_3d_cam2) == 0:
        proj_stats = {
            'total_points': 0,
            'behind_camera': 0,
            'out_of_bounds': 0,
            'valid_projections': 0
        }
        return depth_image, valid_mask, proj_stats

    # Vectorized processing
    X = points_3d_cam2[:, 0]  # x coordinates
    Y = points_3d_cam2[:, 1]  # y coordinates
    Z = points_3d_cam2[:, 2]  # z coordinates (depth)

    # Filter points in front of camera
    valid_front = Z > 0
    points_front = points_3d_cam2[valid_front]

    if len(points_front) == 0:
        proj_stats = {
            'total_points': len(points_3d_cam2),
            'behind_camera': len(points_3d_cam2),
            'out_of_bounds': 0,
            'valid_projections': 0
        }
        print(f"Projection analysis:")
        print(f"  Total points transformed: {len(points_3d_cam2)}")
        print(f"  Points behind camera: {len(points_3d_cam2)}")
        print(f"  Points out of image bounds: 0")
        print(f"  Valid projections: 0")
        return depth_image, valid_mask, proj_stats

    # Vectorized projection to pixel coordinates
    fx = intrinsics_cam2[0, 0]
    fy = intrinsics_cam2[1, 1]
    cx = intrinsics_cam2[0, 2]
    cy = intrinsics_cam2[1, 2]

    X_front = points_front[:, 0]
    Y_front = points_front[:, 1]
    Z_front = points_front[:, 2]

    u_coords = fx * X_front / Z_front + cx
    v_coords = fy * Y_front / Z_front + cy

    # Round to nearest integer and convert to int
    u_int = np.round(u_coords).astype(int)
    v_int = np.round(v_coords).astype(int)

    # Find valid projections within image bounds
    valid_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    u_valid = u_int[valid_bounds]
    v_valid = v_int[valid_bounds]
    depths_valid = Z_front[valid_bounds]

    # Count statistics
    behind_camera = len(points_3d_cam2) - len(points_front)
    out_of_bounds = len(points_front) - len(depths_valid)

    # Use advanced indexing to update depth image
    # For each pixel, keep the minimum depth (closest point)
    indices = v_valid * W + u_valid

    # Create arrays for the update
    flat_depth = depth_image.flatten()

    # For pixels that have multiple points, keep the closest (smallest depth)
    for idx, depth in zip(indices, depths_valid):
        current_depth = flat_depth[idx]
        if np.isnan(current_depth) or depth < current_depth:
            flat_depth[idx] = depth

    # Reshape back to image
    depth_image = flat_depth.reshape(H, W)

    # Update valid mask
    valid_mask_flat = valid_mask.flatten()
    valid_mask_flat[indices] = True
    valid_mask = valid_mask_flat.reshape(H, W)

    valid_projections = len(depths_valid)

    print(f"Projection analysis:")
    print(f"  Total points transformed: {len(points_3d_cam2)}")
    print(f"  Points behind camera: {behind_camera}")
    print(f"  Points out of image bounds: {out_of_bounds}")
    print(f"  Valid projections: {valid_projections}")

    # Return statistics as well
    proj_stats = {
        'total_points': len(points_3d_cam2),
        'behind_camera': behind_camera,
        'out_of_bounds': out_of_bounds,
        'valid_projections': valid_projections
    }

    return depth_image, valid_mask, proj_stats


def create_projected_depth_map(img1, depth1, K1, img2, depth2, K2, ext1, ext2, max_depth=10.0):
    """
    Create projected depth map by reprojecting camera 1's point cloud to camera 2's view.

    Args:
        img1, depth1, K1, ext1: Camera 1 data (third-person)
        img2, depth2, K2, ext2: Camera 2 data (first-person)
        max_depth: Maximum valid depth

    Returns:
        depth_projected: Projected depth map in camera 2's view
        valid_mask: Mask of valid depth values
        stats: Dictionary with projection statistics
        T_1to2: Relative transform matrix
    """
    print("Creating point cloud from camera 1 (third-person view)...")
    points_3d_cam1, valid_mask_cam1 = create_pointcloud_from_depth(depth1, K1, max_depth)
    print(f"Created {len(points_3d_cam1)} valid 3D points from camera 1")

    # Compute relative pose from camera 1 to camera 2
    print("Computing relative pose from camera 1 to camera 2...")
    T_1to2 = compute_relative_pose(ext1, ext2)
    print(f"T_1to2 transform:\n{T_1to2}")

    # Analyze camera poses
    print("Analyzing camera poses...")
    pos1 = ext1[:3, 3]
    pos2 = ext2[:3, 3]
    pos_diff = np.linalg.norm(pos1 - pos2)
    print(f"Camera 1 position: {pos1}")
    print(f"Camera 2 position: {pos2}")
    print(f"Position difference: {pos_diff:.3f} meters")

    # Transform points to camera 2 coordinates
    print("Transforming points to camera 2 coordinates...")
    points_3d_cam2 = transform_points_to_camera2(points_3d_cam1, T_1to2)
    print(f"Transformed {len(points_3d_cam2)} points to camera 2 coordinate system")

    # Analyze transformed points
    depths_cam2 = points_3d_cam2[:, 2]
    print(f"Depth statistics in camera 2 coordinates:")
    print(f"  Min depth: {np.min(depths_cam2):.3f}")
    print(f"  Max depth: {np.max(depths_cam2):.3f}")
    print(f"  Mean depth: {np.mean(depths_cam2):.3f}")
    print(f"  Points behind camera: {np.sum(depths_cam2 <= 0)}")
    print(f"  Points in front: {np.sum(depths_cam2 > 0)}")

    # Project to depth image
    print("Projecting points to camera 2 image plane...")
    image_shape = img2.shape[:2]  # (H, W)
    depth_projected, valid_mask_projected, proj_stats = project_points_to_depth_image(
        points_3d_cam2, K2, image_shape
    )

    # Compute statistics
    num_valid_original = np.sum(valid_mask_cam1)
    num_valid_projected = np.sum(valid_mask_projected)
    depth_range_original = [np.min(depth1[valid_mask_cam1]), np.max(depth1[valid_mask_cam1])]
    depth_range_projected = [np.nanmin(depth_projected), np.nanmax(depth_projected)]

    stats = {
        'num_points_original': len(points_3d_cam1),
        'num_valid_pixels_original': num_valid_original,
        'num_valid_pixels_projected': num_valid_projected,
        'depth_range_original': depth_range_original,
        'depth_range_projected': depth_range_projected,
        'projection_ratio': num_valid_projected / num_valid_original if num_valid_original > 0 else 0,
        'camera_positions': {'cam1': pos1, 'cam2': pos2, 'diff': pos_diff},
        'points_behind_camera': int(np.sum(depths_cam2 <= 0)),
        'points_in_front': int(np.sum(depths_cam2 > 0)),
        'projection_analysis': proj_stats
    }

    print("Projection statistics:")
    print(f"  Original valid pixels: {num_valid_original}")
    print(f"  Projected valid pixels: {num_valid_projected}")
    print(f"  Points behind camera 2: {stats['points_behind_camera']}")
    print(f"  Points in front of camera 2: {stats['points_in_front']}")

    return depth_projected, valid_mask_projected, stats


def save_depth_map(depth_map, output_path, valid_mask=None):
    """
    Save depth map to file, handling NaN values.

    Args:
        depth_map: HxW depth map (may contain NaN)
        output_path: Output file path
        valid_mask: Optional mask of valid pixels
    """
    # Replace NaN with 0 for saving
    depth_to_save = np.copy(depth_map)
    depth_to_save[np.isnan(depth_to_save)] = 0

    # Save as numpy array
    np.savez(output_path, depth=depth_to_save, valid_mask=valid_mask)
    print(f"Saved depth map to: {output_path}")


def create_argument_parser():
    """Create argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Project third-person point cloud to first-person view"
    )
    parser.add_argument(
        "--cam1",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/23897859",
        help="Camera 1 directory (third-person view)"
    )
    parser.add_argument(
        "--cam2",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/17368348",
        help="Camera 2 directory (first-person view)"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid depth"
    )
    parser.add_argument(
        "--output-depth",
        default="projected_depth.npz",
        help="Output path for projected depth map"
    )
    parser.add_argument(
        "--output-vis",
        default="visualizations/projection_visualization.png",
        help="Output path for visualization"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=False,
        help="Save results dictionary"
    )
    parser.add_argument(
        "--analyze-overlap",
        action="store_true",
        default=False,
        help="Analyze camera overlap and exit"
    )

    return parser


def create_visualization(img1, depth1, img2, depth2, depth_projected, valid_mask, output_path):
    """
    Create comprehensive visualization comparing original and projected depth maps.

    Args:
        img1, depth1: Camera 1 data (third-person)
        img2, depth2: Camera 2 data (first-person)
        depth_projected: Projected depth map
        valid_mask: Mask of valid projected pixels
        output_path: Output visualization path
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Camera 1 (third-person)
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Camera 1 Image\n(Third-Person View)')
    axes[0, 0].axis('off')

    # Camera 1 depth
    depth1_vis = np.copy(depth1)
    depth1_vis[depth1_vis <= 0] = np.nan  # Treat depth <= 0 as invalid (white)
    im1 = axes[0, 1].imshow(depth1_vis, cmap='plasma', vmin=0, vmax=5)
    axes[0, 1].set_title('Camera 1 Depth\n(Original)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # Point cloud visualization (Camera 1 view)
    axes[0, 2].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # Overlay depth information
    depth_overlay = cv2.applyColorMap(
        ((depth1_vis / 5.0) * 255).astype(np.uint8),
        cv2.COLORMAP_PLASMA
    )
    depth_overlay = cv2.cvtColor(depth_overlay, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), 0.7, depth_overlay, 0.3, 0)
    axes[0, 2].imshow(blended)
    axes[0, 2].set_title('Camera 1\n(Image + Depth Overlay)')
    axes[0, 2].axis('off')

    # Placeholder for row 1, col 3
    axes[0, 3].axis('off')

    # Row 2: Camera 2 (first-person)
    axes[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Camera 2 Image\n(First-Person View)')
    axes[1, 0].axis('off')

    # Camera 2 original depth
    depth2_vis = np.copy(depth2)
    depth2_vis[depth2_vis <= 0] = np.nan  # Treat depth <= 0 as invalid (white)
    im2 = axes[1, 1].imshow(depth2_vis, cmap='plasma', vmin=0, vmax=5)
    axes[1, 1].set_title('Camera 2 Depth\n(Ground Truth)')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)

    # Projected depth from camera 1
    depth_proj_vis = np.copy(depth_projected)
    depth_proj_vis[depth_proj_vis <= 0] = np.nan  # Treat depth <= 0 as invalid (white)
    im3 = axes[1, 2].imshow(depth_proj_vis, cmap='plasma', vmin=0, vmax=5)
    axes[1, 2].set_title('Projected Depth\n(from Camera 1)')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], shrink=0.8)

    # Comparison: projected vs ground truth
    axes[1, 3].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # Overlay projected depth
    depth_proj_overlay = cv2.applyColorMap(
        ((depth_proj_vis / 5.0) * 255).astype(np.uint8),
        cv2.COLORMAP_PLASMA
    )
    depth_proj_overlay = cv2.cvtColor(depth_proj_overlay, cv2.COLOR_BGR2RGB)
    blended_proj = cv2.addWeighted(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), 0.7, depth_proj_overlay, 0.3, 0)
    axes[1, 3].imshow(blended_proj)
    axes[1, 3].set_title('Camera 2\n(Image + Projected Depth)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    print("="*60)
    print("Point Cloud Projection: Third-Person to First-Person View")
    print("="*60)

    # Load data for both cameras
    print(f"\nLoading camera 1 (third-person): {args.cam1}")
    cam1_depth = os.path.join(args.cam1, "depth_npy")
    img1, depth1, K1, ext1 = load_camera_data(args.cam1, args.frame, cam1_depth)
    print(f"  Image shape: {img1.shape}")
    print(f"  Depth range: [{depth1.min():.3f}, {depth1.max():.3f}]")

    print(f"\nLoading camera 2 (first-person): {args.cam2}")
    cam2_depth = os.path.join(args.cam2, "depth_npy")
    img2, depth2, K2, ext2 = load_camera_data(args.cam2, args.frame, cam2_depth)
    print(f"  Image shape: {img2.shape}")
    print(f"  Depth range: [{depth2.min():.3f}, {depth2.max():.3f}]")

    # Create projected depth map
    print(f"\n{'='*60}")
    print("Projecting Point Cloud")
    print(f"{'='*60}")
    depth_projected, valid_mask, stats = create_projected_depth_map(
        img1, depth1, K1, img2, depth2, K2, ext1, ext2, args.max_depth
    )

    # Save projected depth map
    print(f"\n{'='*60}")
    print("Saving Results")
    print(f"{'='*60}")
    save_depth_map(depth_projected, args.output_depth, valid_mask)

    # Create visualization
    print(f"\n{'='*60}")
    print("Creating Visualization")
    print(f"{'='*60}")
    create_visualization(img1, depth1, img2, depth2, depth_projected, valid_mask, args.output_vis)

    # Save results dictionary
    if args.save_results:
        results = {
            'depth_projected': depth_projected,
            'valid_mask': valid_mask,
            'stats': stats,
            'cam1_intrinsics': K1,
            'cam2_intrinsics': K2,
            'cam1_extrinsics': ext1,
            'cam2_extrinsics': ext2,
            'frame_idx': args.frame
        }

        output_results = Path(args.output_depth).with_suffix('.results.npy')
        np.save(output_results, results)
        print(f"Saved complete results to: {output_results}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


def create_pointcloud_from_depth_cuda(depth_map, intrinsics, max_depth=10.0, device='cuda:0'):
    """
    CUDA-accelerated version of create_pointcloud_from_depth using PyTorch.

    Args:
        depth_map: HxW depth map (numpy array)
        intrinsics: 3x3 camera intrinsics matrix (numpy array)
        max_depth: Maximum valid depth value
        device: CUDA device string

    Returns:
        points_3d: Nx3 array of 3D points in camera coordinates
        valid_mask: HxW boolean mask of valid pixels
    """
    if not CUDA_AVAILABLE:
        print("CUDA not available, falling back to CPU version")
        return create_pointcloud_from_depth(depth_map, intrinsics, max_depth)

    H, W = depth_map.shape

    # Convert to torch tensors
    depth_tensor = torch.from_numpy(depth_map.astype(np.float32)).to(device)
    intrinsics_tensor = torch.from_numpy(intrinsics.astype(np.float32)).to(device)

    # Create coordinate grids
    u_coords, v_coords = torch.meshgrid(
        torch.arange(W, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        indexing='xy'
    )

    # Flatten
    depth_flat = depth_tensor.flatten()
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()

    # Create valid mask
    valid_mask = (depth_flat > 0) & (depth_flat <= max_depth)
    valid_mask_np = valid_mask.cpu().numpy().reshape(H, W)

    # Only process valid points
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), valid_mask_np

    u_valid = u_flat[valid_indices]
    v_valid = v_flat[valid_indices]
    depth_valid = depth_flat[valid_indices]

    # Vectorized 3D point calculation
    fx = intrinsics_tensor[0, 0]
    fy = intrinsics_tensor[1, 1]
    cx = intrinsics_tensor[0, 2]
    cy = intrinsics_tensor[1, 2]

    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid

    points_3d = torch.stack([x, y, z], dim=1).cpu().numpy()

    return points_3d, valid_mask_np


def project_points_to_depth_image_cuda(points_3d_cam2, intrinsics_cam2, image_shape, device='cuda:0'):
    """
    CUDA-accelerated version of project_points_to_depth_image using PyTorch.

    Args:
        points_3d_cam2: Nx3 array of 3D points in camera 2 coordinates (numpy array)
        intrinsics_cam2: 3x3 camera intrinsics matrix for camera 2 (numpy array)
        image_shape: (H, W) tuple for output image dimensions
        device: CUDA device string

    Returns:
        depth_image: HxW depth image
        valid_mask: HxW boolean mask of valid projections
        proj_stats: Dictionary with projection statistics
    """
    if not CUDA_AVAILABLE:
        print("CUDA not available, falling back to CPU version")
        return project_points_to_depth_image(points_3d_cam2, intrinsics_cam2, image_shape)

    H, W = image_shape
    depth_image = np.full((H, W), np.nan, dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)

    if len(points_3d_cam2) == 0:
        proj_stats = {
            'total_points': 0,
            'behind_camera': 0,
            'out_of_bounds': 0,
            'valid_projections': 0
        }
        return depth_image, valid_mask, proj_stats

    # Convert to torch tensors
    points_tensor = torch.from_numpy(points_3d_cam2.astype(np.float32)).to(device)
    intrinsics_tensor = torch.from_numpy(intrinsics_cam2.astype(np.float32)).to(device)

    # Vectorized processing
    X = points_tensor[:, 0]  # x coordinates
    Y = points_tensor[:, 1]  # y coordinates
    Z = points_tensor[:, 2]  # z coordinates (depth)

    # Filter points in front of camera
    valid_front = Z > 0
    points_front = points_tensor[valid_front]

    if len(points_front) == 0:
        proj_stats = {
            'total_points': len(points_3d_cam2),
            'behind_camera': len(points_3d_cam2),
            'out_of_bounds': 0,
            'valid_projections': 0
        }
        print(f"Projection analysis:")
        print(f"  Total points transformed: {len(points_3d_cam2)}")
        print(f"  Points behind camera: {len(points_3d_cam2)}")
        print(f"  Points out of image bounds: 0")
        print(f"  Valid projections: 0")
        return depth_image, valid_mask, proj_stats

    # Vectorized projection to pixel coordinates
    fx = intrinsics_tensor[0, 0]
    fy = intrinsics_tensor[1, 1]
    cx = intrinsics_tensor[0, 2]
    cy = intrinsics_tensor[1, 2]

    X_front = points_front[:, 0]
    Y_front = points_front[:, 1]
    Z_front = points_front[:, 2]

    u_coords = fx * X_front / Z_front + cx
    v_coords = fy * Y_front / Z_front + cy

    # Round to nearest integer
    u_int = torch.round(u_coords).long()
    v_int = torch.round(v_coords).long()

    # Find valid projections within image bounds
    valid_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    u_valid = u_int[valid_bounds]
    v_valid = v_int[valid_bounds]
    depths_valid = Z_front[valid_bounds]

    # Count statistics
    behind_camera = len(points_3d_cam2) - len(points_front)
    out_of_bounds = len(points_front) - len(depths_valid)

    if len(depths_valid) > 0:
        # Create depth image tensor on GPU
        depth_image_tensor = torch.full((H, W), float('nan'), dtype=torch.float32, device=device)

        # Use advanced indexing to update depth image
        indices = v_valid * W + u_valid

        # For each pixel, keep the minimum depth (closest point)
        # This is done efficiently using scatter_min
        depth_image_flat = depth_image_tensor.flatten()

        # Create a temporary tensor for scatter operation
        # We need to handle NaN values properly
        current_depths = depth_image_flat[indices]

        # Mask for positions that are NaN or should be updated
        update_mask = torch.isnan(current_depths) | (depths_valid < current_depths)

        # Update only the positions that should be updated
        depth_image_flat.scatter_(0, indices[update_mask], depths_valid[update_mask])

        # Copy back to CPU
        depth_image = depth_image_tensor.cpu().numpy()

        # Update valid mask
        valid_mask_flat = torch.zeros(H * W, dtype=torch.bool, device=device)
        valid_mask_flat[indices] = True
        valid_mask = valid_mask_flat.cpu().numpy().reshape(H, W)
    else:
        valid_projections = 0

    valid_projections = len(depths_valid)

    print(f"Projection analysis:")
    print(f"  Total points transformed: {len(points_3d_cam2)}")
    print(f"  Points behind camera: {behind_camera}")
    print(f"  Points out of image bounds: {out_of_bounds}")
    print(f"  Valid projections: {valid_projections}")

    # Return statistics as well
    proj_stats = {
        'total_points': len(points_3d_cam2),
        'behind_camera': behind_camera,
        'out_of_bounds': out_of_bounds,
        'valid_projections': valid_projections
    }

    return depth_image, valid_mask, proj_stats


def create_projected_depth_map_cuda(img1, depth1, K1, img2, depth2, K2, ext1, ext2, max_depth=10.0, device='cuda:0'):
    """
    CUDA-accelerated version of create_projected_depth_map.
    """
    print("Creating point cloud from camera 1 (third-person view) - CUDA...")
    points_3d_cam1, valid_mask_cam1 = create_pointcloud_from_depth_cuda(depth1, K1, max_depth, device)
    print(f"Created {len(points_3d_cam1)} valid 3D points from camera 1")

    # Compute relative pose from camera 1 to camera 2
    print("Computing relative pose from camera 1 to camera 2...")
    T_1to2 = compute_relative_pose(ext1, ext2)
    print(f"T_1to2 transform:\n{T_1to2}")

    # Analyze camera poses
    print("Analyzing camera poses...")
    pos1 = ext1[:3, 3]
    pos2 = ext2[:3, 3]
    pos_diff = np.linalg.norm(pos1 - pos2)
    print(f"Camera 1 position: {pos1}")
    print(f"Camera 2 position: {pos2}")
    print(f"Position difference: {pos_diff:.3f} meters")

    # Transform points to camera 2 coordinates
    print("Transforming points to camera 2 coordinates...")
    points_3d_cam2 = transform_points_to_camera2(points_3d_cam1, T_1to2)
    print(f"Transformed {len(points_3d_cam2)} points to camera 2 coordinate system")

    # Analyze transformed points
    depths_cam2 = points_3d_cam2[:, 2]
    print(f"Depth statistics in camera 2 coordinates:")
    print(f"  Min depth: {np.min(depths_cam2):.3f}")
    print(f"  Max depth: {np.max(depths_cam2):.3f}")
    print(f"  Mean depth: {np.mean(depths_cam2):.3f}")
    print(f"  Points behind camera: {np.sum(depths_cam2 <= 0)}")
    print(f"  Points in front: {np.sum(depths_cam2 > 0)}")

    # Project to depth image
    print("Projecting points to camera 2 image plane - CUDA...")
    image_shape = img2.shape[:2]  # (H, W)
    depth_projected, valid_mask_projected, proj_stats = project_points_to_depth_image_cuda(
        points_3d_cam2, K2, image_shape, device
    )

    # Compute statistics
    num_valid_original = np.sum(valid_mask_cam1)
    num_valid_projected = np.sum(valid_mask_projected)
    depth_range_original = [np.min(depth1[valid_mask_cam1]), np.max(depth1[valid_mask_cam1])]
    depth_range_projected = [np.nanmin(depth_projected), np.nanmax(depth_projected)]

    stats = {
        'num_points_original': len(points_3d_cam1),
        'num_valid_pixels_original': num_valid_original,
        'num_valid_pixels_projected': num_valid_projected,
        'depth_range_original': depth_range_original,
        'depth_range_projected': depth_range_projected,
        'projection_ratio': num_valid_projected / num_valid_original if num_valid_original > 0 else 0,
        'camera_positions': {'cam1': pos1, 'cam2': pos2, 'diff': pos_diff},
        'points_behind_camera': int(np.sum(depths_cam2 <= 0)),
        'points_in_front': int(np.sum(depths_cam2 > 0)),
        'projection_analysis': proj_stats
    }

    print("Projection statistics:")
    print(f"  Original valid pixels: {num_valid_original}")
    print(f"  Projected valid pixels: {num_valid_projected}")
    print(f"  Points behind camera 2: {stats['points_behind_camera']}")
    print(f"  Points in front of camera 2: {stats['points_in_front']}")

    return depth_projected, valid_mask_projected, stats


if __name__ == "__main__":
    main()
