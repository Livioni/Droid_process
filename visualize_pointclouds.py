#!/usr/bin/env python3
"""
Multi-camera point cloud visualization using viser.
Visualizes point clouds from multiple cameras with their RGB colors.
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
        extrinsics_refined_dir = camera_dir / "extrinsics_refined_icp" 
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


def visualize_multi_camera_pointcloud(camera_dirs, frame_idx=0, max_depth=10.0, downsample=1):
    """
    Visualize point clouds from multiple cameras using viser.

    Args:
        camera_dirs: List of camera directory paths
        frame_idx: Frame index to visualize
        max_depth: Maximum depth value to include
        downsample: Downsample factor for points (1 = no downsampling)
    """
    server = viser.ViserServer(port=8080)
    print(f"Viser server started at http://localhost:8080")

    # Initialize point cloud data storage
    point_cloud_data = {}
    camera_frames = []
    
    # Camera colors for visualization
    camera_colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
    ]
    
    all_points = []
    all_colors = []
    
    for i, camera_dir in enumerate(camera_dirs):
        camera_dir = Path(camera_dir)
        camera_id = camera_dir.name
        print(f"\nProcessing camera {i+1}/{len(camera_dirs)}: {camera_id}")
        
        # Load camera data
        image, depth, intrinsics, extrinsics = load_camera_data(camera_dir, frame_idx)
        print(f"  Image shape: {image.shape}")
        print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"  Intrinsics:\n{intrinsics}")
        print(f"  Extrinsics:\n{extrinsics}")
        
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
        
        all_points.append(points_world)
        all_colors.append(colors)
        
        # Store point cloud data for interactive controls
        point_cloud_data[camera_id] = {
            'points': points_world,
            'colors': colors,
            'original_points': points_world.copy()
        }

        # Add point cloud to viser
        pc_handle = server.scene.add_point_cloud(
            name=f"camera_{camera_id}",
            points=points_world,
            colors=colors,
            point_size=0.005,
        )
        point_cloud_data[camera_id]['handle'] = pc_handle
        
        # Add camera frustum visualization
        # Create a coordinate frame at the camera position
        cam_pos = extrinsics[:, 3]
        cam_rot = extrinsics[:, :3]
        
        # Convert rotation matrix to quaternion (wxyz format)
        # Using a simple conversion (for visualization purposes)
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
        
    
    # Combine all points for statistics
    if all_points:
        all_points_combined = np.concatenate(all_points, axis=0)
        print(f"\n{'='*60}")
        print(f"Total points: {len(all_points_combined)}")
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
            for camera_data in point_cloud_data.values():
                camera_data['handle'].point_size = size
            print(f"Point size updated to: {size}")

        scale_slider = server.gui.add_slider("Scale Factor", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        @scale_slider.on_update
        def set_scale_factor(_):
            """Scale the entire point cloud."""
            scale = scale_slider.value
            for camera_data in point_cloud_data.values():
                # Scale points around their centroid
                centroid = np.mean(camera_data['original_points'], axis=0)
                scaled_points = centroid + scale * (camera_data['original_points'] - centroid)
                camera_data['handle'].points = scaled_points
            print(f"Scale factor updated to: {scale}")

        depth_slider = server.gui.add_slider("Max Depth", min=1.0, max=20.0, step=0.5, initial_value=max_depth)
        @depth_slider.on_update
        def set_max_depth(_):
            """Adjust maximum depth for point filtering."""
            # Note: This would require reprocessing depth data, so we show bounds instead
            depth = depth_slider.value
            print(f"Max depth filter: {depth} (re-run script to apply)")

        # Add camera visibility toggles
        visibility_checkboxes = {}
        for camera_id in point_cloud_data.keys():
            checkbox = server.gui.add_checkbox(f"Show {camera_id}", initial_value=True)
            visibility_checkboxes[camera_id] = checkbox

        def create_visibility_callback(cam_id):
            def toggle_visibility(_):
                point_cloud_data[cam_id]['handle'].visible = visibility_checkboxes[cam_id].value
                print(f"Camera {cam_id} visibility: {visibility_checkboxes[cam_id].value}")
            return toggle_visibility

        for camera_id in point_cloud_data.keys():
            visibility_checkboxes[camera_id].on_update(create_visibility_callback(camera_id))

        reset_button = server.gui.add_button("Reset View")
        @reset_button.on_click
        def reset_view(_):
            """Reset all visualization parameters to defaults."""
            # Reset sliders
            point_size_slider.value = 0.005
            scale_slider.value = 1.0
            depth_slider.value = max_depth

            # Reset point clouds
            for camera_data in point_cloud_data.values():
                camera_data['handle'].point_size = 0.005
                camera_data['handle'].points = camera_data['original_points']
                camera_data['handle'].visible = True

            # Reset checkboxes
            for cb in visibility_checkboxes.values():
                cb.value = True

            print("Visualization reset to defaults")

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
    
    print(f"\nVisualization ready! Open http://localhost:8080 in your browser.")
    print("Press Ctrl+C to exit.")
    
    # Keep the server running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-camera point clouds")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=[
            "datasets/samples/Sun_Jun_11_15:52:37_2023/17368348",
            "datasets/samples/Sun_Jun_11_15:52:37_2023/23897859",
            "datasets/samples/Sun_Jun_11_15:52:37_2023/27904255",
        ],
        help="List of camera directories"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize (default: 0)"
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=5.0,
        help="Maximum depth value to include (default: 10.0)"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Downsample factor for points (default: 1, no downsampling)"
    )
    
    args = parser.parse_args()
    
    visualize_multi_camera_pointcloud(
        args.cameras,
        frame_idx=args.frame,
        max_depth=args.max_depth,
        downsample=args.downsample
    )


if __name__ == "__main__":
    main()
