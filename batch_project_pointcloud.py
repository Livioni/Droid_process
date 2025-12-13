#!/usr/bin/env python3
"""
Batch project point clouds from third-person camera to first-person (wrist) camera view.
Processes all frames in datasets/samples/23897859 and projects them to wrist camera 17368348.
"""

import numpy as np
import os
import cv2
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import functions from the main projection script
from project_pointcloud_to_first_person import (
    load_camera_data,
    create_projected_depth_map,
    create_projected_depth_map_cuda,
    CUDA_AVAILABLE
)


def get_frame_range(cam_dir):
    """Get the range of available frames in a camera directory."""
    image_dir = Path(cam_dir) / "images" / "left"
    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")

    # Get all .png files and extract frame numbers
    frame_files = list(image_dir.glob("*.png"))
    frame_indices = []

    for frame_file in frame_files:
        try:
            frame_idx = int(frame_file.stem)
            frame_indices.append(frame_idx)
        except ValueError:
            continue

    if not frame_indices:
        raise ValueError(f"No valid frame files found in {image_dir}")

    return min(frame_indices), max(frame_indices)


def create_visualization(img1, depth1, img2, depth2, depth_projected, frame_idx, output_dir, max_depth):
    """
    Create visualization showing all camera views and depth maps.

    Args:
        img1: Third-person camera image
        depth1: Third-person camera depth map
        img2: First-person camera image
        depth2: First-person camera depth map
        depth_projected: Projected depth map from third-person to first-person view
        frame_idx: Frame index
        output_dir: Output directory for visualization
        max_depth: Maximum depth value for consistent colorbar range
    """
    # Create output directory for visualizations
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Frame {frame_idx}: Camera Views and Depth Maps', fontsize=16)

    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Common colormap for depth maps
    cmap = cm.viridis
    vmin, vmax = 0, 5.0  # Fixed depth range 0-5m

    # Prepare depth maps (set depth <= 0 as invalid/white)
    depth1_vis = np.copy(depth1)
    depth1_vis[depth1_vis <= 0] = np.nan

    depth_projected_vis = np.copy(depth_projected)
    depth_projected_vis[depth_projected_vis <= 0] = np.nan

    depth2_vis = np.copy(depth2)
    depth2_vis[depth2_vis <= 0] = np.nan

    # Row 1: Third-person views
    axes[0, 0].imshow(img1_rgb)
    axes[0, 0].set_title('Third-Person Camera Image')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(depth1_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Third-Person Depth Map')
    axes[0, 1].axis('off')

    im2 = axes[0, 2].imshow(depth_projected_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Projected Depth (3rd→1st Person)')
    axes[0, 2].axis('off')

    # Row 2: First-person views
    axes[1, 0].imshow(img2_rgb)
    axes[1, 0].set_title('First-Person Camera Image')
    axes[1, 0].axis('off')

    im3 = axes[1, 1].imshow(depth2_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('First-Person Depth Map')
    axes[1, 1].axis('off')

    # Hide the last subplot
    axes[1, 2].axis('off')

    # Add colorbar for depth maps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Depth (meters)', rotation=270, labelpad=20)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Save the visualization
    output_path = vis_dir / f"{frame_idx}_visualization.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(output_path)


def batch_project_pointclouds(cam1_dir, cam2_dir, output_dir, max_depth=10.0, 
                              start_frame=None, end_frame=None, use_cuda=False, 
                              cuda_device='cuda:0', enable_visualization=False,
                              save_depth_map=False):
    """
    Batch process all frames to project point clouds from cam1 to cam2.

    Args:
        cam1_dir: Third-person camera directory
        cam2_dir: First-person (wrist) camera directory
        output_dir: Output directory for projected depth maps
        max_depth: Maximum valid depth value
        start_frame: Starting frame index (optional)
        end_frame: Ending frame index (optional)
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get frame range from cam1 (third-person camera)
    if start_frame is None or end_frame is None:
        min_frame, max_frame = get_frame_range(cam1_dir)
        if start_frame is None:
            start_frame = min_frame
        if end_frame is None:
            end_frame = max_frame

    print(f"Processing frames from {start_frame} to {end_frame}")
    print(f"Output directory: {output_dir}")
    print(f"CUDA acceleration: {'Enabled' if use_cuda and CUDA_AVAILABLE else 'Disabled'}")

    # Load camera 2 intrinsics and extrinsics (these are constant across frames)
    print("\nLoading wrist camera (camera 2) intrinsics and extrinsics...")
    cam2_path = Path(cam2_dir)
    cam2_id = cam2_path.name

    # Load camera 2 intrinsics
    intrinsics_path = cam2_path / "intrinsics" / f"{cam2_id}_left.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Camera 2 intrinsics not found: {intrinsics_path}")
    K2 = np.load(str(intrinsics_path))

    # Load camera 2 extrinsics (constant across frames)
    extrinsics_dir = cam2_path / "extrinsics"
    extrinsics_files = list(extrinsics_dir.glob("*_left.npy"))
    if not extrinsics_files:
        raise FileNotFoundError(f"No extrinsics found in: {extrinsics_dir}")
    extrinsics_path = extrinsics_files[0]
    ext2_all = np.load(str(extrinsics_path))

    # Get camera 2 image shape from first frame
    img2_path = cam2_path / "images"/ "left" / f"{start_frame:06d}.png"
    if not img2_path.exists():
        raise FileNotFoundError(f"Camera 2 image not found: {img2_path}")
    img2_sample = cv2.imread(str(img2_path))
    cam2_shape = img2_sample.shape[:2]  # (H, W)

    print(f"Camera 2 intrinsics shape: {K2.shape}")
    print(f"Camera 2 image shape: {cam2_shape}")
    print(f"Camera 2 extrinsics frames: {len(ext2_all)}")

    # Process each frame
    successful_frames = 0
    failed_frames = 0
    start_time = time.time()

    for frame_idx in tqdm(range(start_frame, end_frame + 1), desc="Processing frames"):
        # Load camera 1 data for this frame
        frame_idx = f"{frame_idx:06d}"
        cam1_depth_dir = os.path.join(cam1_dir, "depth_npy")
        img1, depth1, K1, ext1 = load_camera_data(cam1_dir, frame_idx, cam1_depth_dir)

        # Get camera 2 extrinsics for this frame
        if int(frame_idx) >= len(ext2_all):
            print(f"Warning: Frame {frame_idx} out of range for camera 2 extrinsics. Skipping.")
            failed_frames += 1
            continue

        ext2 = ext2_all[int(frame_idx)]

        # Load camera 2 image and depth for this frame
        img2_path = cam2_path / "images" / "left" / f"{int(frame_idx):06d}.png"
        if not img2_path.exists():
            print(f"Warning: Camera 2 image not found for frame {frame_idx}. Skipping.")
            failed_frames += 1
            continue

        img2 = cv2.imread(str(img2_path))

        depth2_path = cam2_path / "depth_npy" / f"{int(frame_idx):06d}.npz"
        if not depth2_path.exists():
            print(f"Warning: Camera 2 depth not found for frame {frame_idx}. Skipping.")
            failed_frames += 1
            continue

        depth2 = np.load(str(depth2_path))['depth']

        # Create projected depth map
        if use_cuda and CUDA_AVAILABLE:
            depth_projected, valid_mask, stats = create_projected_depth_map_cuda(
                img1, depth1, K1, img2, depth2, K2, ext1, ext2, max_depth, cuda_device
            )
        else:
            depth_projected, valid_mask, stats = create_projected_depth_map(
                img1, depth1, K1, img2, depth2, K2, ext1, ext2, max_depth
            )

        # Save projected depth map
        if save_depth_map:
            output_filename = f"{frame_idx}.npz"
            output_file_path = output_path / output_filename
            save_depth_map(depth_projected, str(output_file_path), valid_mask)

        # Create visualization if enabled
        if enable_visualization:
            vis_path = create_visualization(
                img1, depth1, img2, depth2, depth_projected,
                frame_idx, output_dir, max_depth
            )
            print(f"Visualization saved: {vis_path}")

        successful_frames += 1
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total frames processed: {end_frame - start_frame + 1}")
    print(f"Successful: {successful_frames}")
    print(f"Failed: {failed_frames}")
    print(f"Output directory: {output_dir}")


def create_argument_parser():
    """Create argument parser for the batch script."""
    parser = argparse.ArgumentParser(
        description="Batch project point clouds from third-person to first-person camera view"
    )
    parser.add_argument(
        "--cam1",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/23897859",
        help="Third-person camera directory"
    )
    parser.add_argument(
        "--cam2",
        default="datasets/samples/Sun_Jun_11_15:52:37_2023/17368348",
        help="First-person (wrist) camera directory"
    )
    parser.add_argument(
        "--output-dir",
        default="projected_depths_wrist",
        help="Output directory for projected depth maps"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid depth"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (optional)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=301,
        help="Ending frame index (optional)"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enable CUDA acceleration (if available)"
    )
    parser.add_argument(
        "--cuda-device",
        default="cuda:0",
        help="CUDA device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        default=True,
        help="Enable visualization of camera views and depth maps"
    )
    parser.add_argument(
        "--save-depth-map",
        action="store_true",
        default=False,
        help="Save projected depth map"
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    print("="*80)
    print("Batch Point Cloud Projection: Third-Person to First-Person View")
    print("="*80)
    print(f"Third-person camera: {args.cam1}")
    print(f"First-person camera: {args.cam2}")
    print(f"Output directory: {args.output_dir}")
    print(f"Visualization: {'Enabled' if args.visualization else 'Disabled'}")

    # Run batch processing
    batch_project_pointclouds(
        args.cam1,
        args.cam2,
        args.output_dir,
        args.max_depth,
        args.start_frame,
        args.end_frame,
        args.cuda,
        args.cuda_device,
        args.visualization,
        args.save_depth_map
    )

    print(f"\n{'='*80}")
    print("Batch processing completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
