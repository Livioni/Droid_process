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

# Import functions from the main projection script
from project_pointcloud_to_first_person import (
    load_camera_data,
    create_projected_depth_map,
    create_projected_depth_map_cuda,
    save_depth_map,
    CUDA_AVAILABLE
)


def get_frame_range(cam_dir):
    """Get the range of available frames in a camera directory."""
    image_dir = Path(cam_dir) / "images"
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


def batch_project_pointclouds(cam1_dir, cam2_dir, output_dir, max_depth=10.0, start_frame=None, end_frame=None, use_cuda=False, cuda_device='cuda:0'):
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
    intrinsics_path = cam2_path / "intrinsics" / f"{cam2_id}.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Camera 2 intrinsics not found: {intrinsics_path}")
    K2 = np.load(str(intrinsics_path))

    # Load camera 2 extrinsics (constant across frames)
    extrinsics_dir = cam2_path / "extrinsics"
    extrinsics_files = list(extrinsics_dir.glob("*.npy"))
    if not extrinsics_files:
        raise FileNotFoundError(f"No extrinsics found in: {extrinsics_dir}")
    extrinsics_path = extrinsics_files[0]
    ext2_all = np.load(str(extrinsics_path))

    # Get camera 2 image shape from first frame
    img2_path = cam2_path / "images" / f"{start_frame}.png"
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
        try:
            # Load camera 1 data for this frame
            cam1_depth_dir = os.path.join(cam1_dir, "depth_npy")
            img1, depth1, K1, ext1 = load_camera_data(cam1_dir, frame_idx, cam1_depth_dir)

            # Get camera 2 extrinsics for this frame
            if frame_idx >= len(ext2_all):
                print(f"Warning: Frame {frame_idx} out of range for camera 2 extrinsics. Skipping.")
                failed_frames += 1
                continue

            ext2 = ext2_all[frame_idx]

            # Load camera 2 image and depth for this frame
            img2_path = cam2_path / "images" / f"{frame_idx}.png"
            if not img2_path.exists():
                print(f"Warning: Camera 2 image not found for frame {frame_idx}. Skipping.")
                failed_frames += 1
                continue

            img2 = cv2.imread(str(img2_path))

            depth2_path = cam2_path / "depth_npy" / f"{frame_idx}.npz"
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
            output_filename = f"{frame_idx}.npz"
            output_file_path = output_path / output_filename
            save_depth_map(depth_projected, str(output_file_path), valid_mask)

            successful_frames += 1

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            failed_frames += 1
            continue

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total frames processed: {end_frame - start_frame + 1}")
    print(f"Successful: {successful_frames}")
    print(f"Failed: {failed_frames}")
    print(".2f")
    print(".2f")
    print(f"Output directory: {output_dir}")


def create_argument_parser():
    """Create argument parser for the batch script."""
    parser = argparse.ArgumentParser(
        description="Batch project point clouds from third-person to first-person camera view"
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
        default=None,
        help="Starting frame index (optional)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
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

    # Run batch processing
    batch_project_pointclouds(
        args.cam1,
        args.cam2,
        args.output_dir,
        args.max_depth,
        args.start_frame,
        args.end_frame,
        args.cuda,
        args.cuda_device
    )

    print(f"\n{'='*80}")
    print("Batch processing completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
