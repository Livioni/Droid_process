#!/usr/bin/env python3
"""
Temporal point cloud visualization for a single camera using viser.
Visualizes point clouds from 5 consecutive frames (current -2 to +2) with different colors.
"""

import numpy as np
import cv2
import viser
import argparse
import glob
import os
from pathlib import Path


def load_camera_data(camera_dir, frame_idx):
    frame_idx_clone = int(frame_idx)
    frame_idx = f"{frame_idx:06d}"
    """Load image, depth, intrinsics, and extrinsics for a camera."""
    camera_dir = Path(camera_dir)
    camera_id = camera_dir.name

    # Load image
    try:
        image_path = camera_dir / "images" / "left" / f"{frame_idx}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        image_path = camera_dir / "images" / f"{frame_idx}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load depth
    try:
        depth_backprojected_path = camera_dir / "depth_backproject" / f"{frame_idx}.npz"
        depth = np.load(str(depth_backprojected_path))["depth"]
        print(f"Loaded backprojected depth from {depth_backprojected_path}")
    except:
        depth_path = camera_dir / "depth_npy" / f"{frame_idx}.npz"
        depth = np.load(str(depth_path))["depth"]

    # Load intrinsics
    intrinsics_path = camera_dir / "intrinsics" / f"{camera_id}_left.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
    intrinsics = np.load(str(intrinsics_path))

    # Load extrinsics - find the extrinsics file
    try:
        extrinsics_refined_dir = camera_dir / "extrinsics_refined" 
        extrinsics_file = glob.glob(os.path.join(extrinsics_refined_dir, '*.npy'))[0]
        extrinsics = np.load(extrinsics_file,allow_pickle=True)[frame_idx_clone]
        print(f"Loaded refined extrinsics from {extrinsics_refined_dir}")
    except:
        extrinsics_dir = camera_dir / "extrinsics" / f"{camera_id}_left.npy"
        extrinsics = np.load(str(extrinsics_dir))[frame_idx_clone]
        print(f"Loaded extrinsics from {extrinsics_dir}")

    return image, depth, intrinsics, extrinsics


def depth_to_pointcloud(depth, intrinsics, image=None, max_depth=10.0):
    """
    Convert depth map to 3D point cloud in camera coordinates.

    Args:
        depth: (H, W) depth map
        intrinsics: (3, 3) camera intrinsics matrix
        image: (H, W, 3) RGB image (optional)
        max_depth: Maximum depth value to include

    Returns:
        points: (N, 3) 3D points in camera coordinates
        colors: (N, 3) RGB colors (if image provided)
    """
    H, W = depth.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()

    # Filter out invalid depths
    valid_mask = (depth_flat > 0) & (depth_flat < max_depth)
    u = u[valid_mask]
    v = v[valid_mask]
    depth_flat = depth_flat[valid_mask]

    # Unproject to 3D using intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (u - cx) * depth_flat / fx
    y = (v - cy) * depth_flat / fy
    z = depth_flat

    points = np.stack([x, y, z], axis=-1)  # (N, 3)

    # Get colors if image is provided
    colors = None
    if image is not None:
        colors = image[v, u]  # (N, 3)
        colors = colors.astype(np.float32) / 255.0  # Normalize to [0, 1]

    return points, colors


def transform_points(points, extrinsics):
    """
    Transform points from camera coordinates to world coordinates.

    Args:
        points: (N, 3) points in camera coordinates
        extrinsics: (3, 4) extrinsics matrix [R|t]

    Returns:
        transformed_points: (N, 3) points in world coordinates
    """
    R = extrinsics[:, :3]  # (3, 3)
    t = extrinsics[:, 3]   # (3,)

    # Transform: X_world = R * X_camera + t
    transformed = points @ R.T + t

    return transformed


def get_frame_color(frame_offset):
    """
    Get color for each frame based on its temporal offset.

    Args:
        frame_offset: Offset from center frame (-2 to +2)

    Returns:
        color: RGB color tuple
    """
    # Color scheme: Blue (past) -> Green (current) -> Red (future)
    colors = {
        -2: [0.0, 0.5, 1.0],  # Light blue
        -1: [0.0, 0.8, 0.8],  # Cyan
         0: [0.0, 1.0, 0.0],  # Green (current frame)
         1: [1.0, 0.8, 0.0],  # Orange
         2: [1.0, 0.0, 0.0],  # Red
    }
    return colors.get(frame_offset, [0.5, 0.5, 0.5])  # Gray fallback


def visualize_temporal_pointcloud(camera_dir, center_frame_idx, max_depth=10.0, downsample=1):
    """
    Visualize point clouds from 5 consecutive frames of a single camera using viser.

    Args:
        camera_dir: Camera directory path
        center_frame_idx: Center frame index (will show frames center-2 to center+2)
        max_depth: Maximum depth value to include
        downsample: Downsample factor for points (1 = no downsampling)
    """
    server = viser.ViserServer(port=8080)
    print(f"Viser server started at http://localhost:8080")

    camera_dir = Path(camera_dir)
    camera_id = camera_dir.name

    # Initialize point cloud data storage
    frame_data = {}
    frame_handles = {}
    visibility_checkboxes = {}

    # Frame offsets to visualize (-5, -5, 0, 3, 5)
    frame_offsets = [-5, -3, 0, 3, 5]

    all_points = []
    all_colors = []

    print(f"Loading frames {max(0, center_frame_idx-2)} to {center_frame_idx+2} for camera {camera_id}")

    for frame_offset in frame_offsets:
        frame_idx = center_frame_idx + frame_offset
        # Skip negative frame indices
        if frame_idx < 0:
            print(f"\nSkipping frame {frame_idx} (offset {frame_offset}) - negative frame index")
            continue

        frame_key = f"frame_{frame_idx}"
        print(f"\nProcessing frame {frame_idx} (offset {frame_offset})")

        try:
            # Load camera data
            image, depth, intrinsics, extrinsics = load_camera_data(camera_dir, frame_idx)
            print(f"  Image shape: {image.shape}")
            print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")

            # Convert depth to point cloud
            points, colors = depth_to_pointcloud(depth, intrinsics, image, max_depth)
            print(f"  Generated {len(points)} points")

            # Transform to world coordinates
            points_world = transform_points(points, extrinsics)

            # Downsample if needed
            if downsample > 1:
                indices = np.arange(0, len(points_world), downsample)
                points_world = points_world[indices]
                colors = colors[indices]
                print(f"  Downsampled to {len(points_world)} points")

            # Use original image colors without frame-specific tinting
            if colors is None:
                # Use gray color if no image colors available
                colors = np.ones((len(points_world), 3)) * 0.7

            all_points.append(points_world)
            all_colors.append(colors)

            # Store frame data
            frame_data[frame_key] = {
                'points': points_world,
                'colors': colors,
                'original_points': points_world.copy(),
                'original_colors': colors.copy(),
                'frame_idx': frame_idx,
                'frame_offset': frame_offset
            }

            # Add point cloud to viser
            pc_handle = server.scene.add_point_cloud(
                name=f"{camera_id}_frame_{frame_idx}",
                points=points_world,
                colors=colors,
                point_size=0.005,
            )
            frame_handles[frame_key] = pc_handle

            # Add camera frustum visualization for center frame only
            if frame_offset == 0:
                cam_pos = extrinsics[:, 3]
                cam_rot = extrinsics[:, :3]

                # Convert rotation matrix to quaternion (wxyz format)
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_matrix(cam_rot)
                quat_xyzw = rot.as_quat()  # Returns [x, y, z, w]
                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

                server.scene.add_frame(
                    name=f"camera_frame_{camera_id}",
                    wxyz=quat_wxyz,
                    position=cam_pos,
                    axes_length=0.1,
                    axes_radius=0.005,
                )
                print(f"  Camera position: {cam_pos}")

        except Exception as e:
            print(f"  Warning: Could not load frame {frame_idx}: {e}")
            continue

    # Combine all points for statistics
    if all_points:
        all_points_combined = np.concatenate(all_points, axis=0)
        print(f"\n{'='*60}")
        print(f"Total points across all frames: {len(all_points_combined)}")
        print(f"Point cloud bounds:")
        print(f"  X: [{all_points_combined[:, 0].min():.3f}, {all_points_combined[:, 0].max():.3f}]")
        print(f"  Y: [{all_points_combined[:, 1].min():.3f}, {all_points_combined[:, 1].max():.3f}]")
        print(f"  Z: [{all_points_combined[:, 2].min():.3f}, {all_points_combined[:, 2].max():.3f}]")
        print(f"{'='*60}")

        # Add interactive controls for point cloud visualization
        point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005)
        @point_size_slider.on_update
        def set_point_size(_):
            """Adjust the size of all points in the point clouds."""
            size = point_size_slider.value
            for handle in frame_handles.values():
                handle.point_size = size
            print(f"Point size updated to: {size}")

        # Add frame visibility toggles
        for frame_key, data in frame_data.items():
            frame_idx = data['frame_idx']
            frame_offset = data['frame_offset']
            frame_label = f"Frame {frame_idx}"
            if frame_offset == 0:
                frame_label += " (current)"
            checkbox = server.gui.add_checkbox(frame_label, initial_value=True)
            visibility_checkboxes[frame_key] = checkbox

        def create_visibility_callback(f_key):
            def toggle_visibility(_):
                frame_handles[f_key].visible = visibility_checkboxes[f_key].value
                data = frame_data[f_key]
                print(f"Frame {data['frame_idx']} visibility: {visibility_checkboxes[f_key].value}")
            return toggle_visibility

        for frame_key in frame_data.keys():
            visibility_checkboxes[frame_key].on_update(create_visibility_callback(frame_key))

        # Add frame navigation controls
        current_center_frame = server.gui.add_text("Current Center Frame", initial_value=str(center_frame_idx), disabled=True)

        def create_frame_navigation(offset):
            def navigate_frame(_):
                nonlocal center_frame_idx
                center_frame_idx += offset
                current_center_frame.value = str(center_frame_idx)
                print(f"Center frame changed to: {center_frame_idx}")
                # Note: In a real implementation, you would reload all frames here
                # For now, this just updates the display
            return navigate_frame

        prev_button = server.gui.add_button("Previous Frame (-1)")
        prev_button.on_click(create_frame_navigation(-1))

        next_button = server.gui.add_button("Next Frame (+1)")
        next_button.on_click(create_frame_navigation(1))

        reset_button = server.gui.add_button("Reset View")
        @reset_button.on_click
        def reset_view(_):
            """Reset all visualization parameters to defaults."""
            point_size_slider.value = 0.005

            # Reset point clouds
            for frame_key, data in frame_data.items():
                frame_handles[frame_key].point_size = 0.005
                frame_handles[frame_key].points = data['original_points']
                frame_handles[frame_key].colors = data['original_colors']
                frame_handles[frame_key].visible = True

            # Reset checkboxes
            for cb in visibility_checkboxes.values():
                cb.value = True

            print("Visualization reset to defaults")

        # Add legend for visualization info
        server.gui.add_text("Visualization Info:", initial_value="", disabled=True)
        server.gui.add_text("• Each frame shows its original RGB colors", initial_value="", disabled=True)
        server.gui.add_text("• Use visibility toggles to show/hide frames", initial_value="", disabled=True)

        # Add a ground plane for reference (optional)
        z_min = all_points_combined[:, 2].min()
        grid_size = 2.0
        grid_points = []
        for x in np.linspace(-grid_size, grid_size, 20):
            for y in np.linspace(-grid_size, grid_size, 20):
                grid_points.append([x, y, z_min])
        grid_points = np.array(grid_points)
        grid_colors = np.ones_like(grid_points) * 0.5  # Gray

        server.scene.add_point_cloud(
            name="ground_grid",
            points=grid_points,
            colors=grid_colors,
            point_size=0.001,
        )

    print(f"\nTemporal visualization ready! Open http://localhost:8080 in your browser.")
    print("Frames shown: center frame ±5 (5 frames total)")
    print("Press Ctrl+C to exit.")

    # Keep the server running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description="Visualize temporal point clouds for a single camera")
    parser.add_argument(
        "--camera",
        default="/opt/dlami/nvme/datasets/processed_droid/Fri_Aug_18_11:40:54_2023/18026681",
        help="Camera directory path"
    )
    parser.add_argument(
        "--center-frame",
        type=int,
        default=5,
        help="Center frame index (will show frames center-2 to center+2)"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=5.0,
        help="Maximum depth value to include (default: 5.0)"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Downsample factor for points (default: 1, no downsampling)"
    )

    args = parser.parse_args()

    visualize_temporal_pointcloud(
        args.camera,
        center_frame_idx=args.center_frame,
        max_depth=args.max_depth,
        downsample=args.downsample
    )


if __name__ == "__main__":
    main()