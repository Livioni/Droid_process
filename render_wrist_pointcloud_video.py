#!/usr/bin/env python3
"""
Generate wrist-camera point clouds per frame and render a third-person MP4.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_camera_data(camera_dir, frame_idx):
    frame_idx_clone = int(frame_idx)
    frame_idx = f"{frame_idx:06d}"
    camera_dir = Path(camera_dir)
    camera_id = camera_dir.name

    # Load image
    try:
        image_path = camera_dir / "images" / "left" / f"{frame_idx}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        image_path = camera_dir / "images" / f"{frame_idx}.png"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load depth
    try:
        depth_path = camera_dir / "depth_npy" / f"{frame_idx}_depth.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        print(f"Loaded depth from {depth_path}")
    except Exception:
        try:
            depth_backprojected_path = camera_dir / "depth_backproject" / f"{frame_idx}.npz"
            depth = np.load(str(depth_backprojected_path))["depth"]
            print(f"Loaded backprojected depth from {depth_backprojected_path}")
        except Exception:
            depth_path = camera_dir / "depth_npy" / f"{frame_idx}.npz"
            depth = np.load(str(depth_path))["depth"]

    # Load intrinsics
    intrinsics_path = camera_dir / "intrinsics" / f"{camera_id}_left.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
    intrinsics = np.load(str(intrinsics_path))

    # Load extrinsics
    if os.path.exists(camera_dir / "extrinsics_align" / f"{camera_id}.npy"):
        extrinsics_file = camera_dir / "extrinsics_align" / f"{camera_id}.npy"
        extrinsics = np.load(extrinsics_file, allow_pickle=True)[frame_idx_clone]
        print(f"Loaded refined extrinsics from {extrinsics_file}")
    elif os.path.exists(camera_dir / "extrinsics_refined" / f"{camera_id}.npy"):
        extrinsics_file = camera_dir / "extrinsics_refined" / f"{camera_id}.npy"
        extrinsics = np.load(str(extrinsics_file))[frame_idx_clone]
        print(f"Loaded extrinsics from {extrinsics_file}")
    if os.path.exists(camera_dir / "extrinsics" / f"{camera_id}_left.npy"):
        extrinsics_file = camera_dir / "extrinsics" / f"{camera_id}_left.npy"
        extrinsics = np.load(str(extrinsics_file))[frame_idx_clone]
        print(f"Loaded extrinsics from {extrinsics_file}")
    else:
        raise FileNotFoundError("Extrinsics not found.")

    return image, depth, intrinsics, extrinsics


def depth_to_pointcloud(depth, intrinsics, image=None, max_depth=10.0):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()

    valid_mask = (depth_flat > 0) & (depth_flat < max_depth)
    u = u[valid_mask]
    v = v[valid_mask]
    depth_flat = depth_flat[valid_mask]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (u - cx) * depth_flat / fx
    y = (v - cy) * depth_flat / fy
    z = depth_flat

    points = np.stack([x, y, z], axis=-1)

    colors = None
    if image is not None:
        colors = image[v, u]
        colors = colors.astype(np.float32) / 255.0

    return points, colors


def transform_points(points, extrinsics):
    R = extrinsics[:, :3]
    t = extrinsics[:, 3]
    return points @ R.T + t


def get_frame_indices(camera_dir):
    camera_dir = Path(camera_dir)
    image_dir = camera_dir / "images" / "left"
    if not image_dir.exists():
        image_dir = camera_dir / "images"
    image_paths = sorted(image_dir.glob("*.png"))
    indices = []
    for path in image_paths:
        stem = path.stem
        try:
            indices.append(int(stem))
        except ValueError:
            continue
    return sorted(indices)


def create_pointcloud(camera_dir, frame_idx, max_depth, downsample):
    image, depth, intrinsics, extrinsics = load_camera_data(camera_dir, frame_idx)
    points, colors = depth_to_pointcloud(depth, intrinsics, image, max_depth)
    points_world = transform_points(points, extrinsics)
    if downsample > 1:
        indices = np.arange(0, len(points_world), downsample)
        points_world = points_world[indices]
        colors = colors[indices]

    if colors is None:
        colors = np.ones((len(points_world), 3)) * 0.7

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def render_pointcloud_video(
    camera_dir,
    output_path,
    start_frame=None,
    end_frame=None,
    step=1,
    max_depth=5.0,
    downsample=10,
    width=1280,
    height=720,
    fps=30,
    point_size=2.0,
    fov=60.0,
    view_distance=2.0,
    view_height=1.0,
):
    frame_indices = get_frame_indices(camera_dir)
    if not frame_indices:
        raise RuntimeError(f"No frames found under {camera_dir}")

    if start_frame is None:
        start_frame = frame_indices[0]
    if end_frame is None:
        end_frame = frame_indices[-1]

    selected = [i for i in frame_indices if start_frame <= i <= end_frame and (i - start_frame) % step == 0]
    if not selected:
        raise RuntimeError("No frames selected. Check start/end/step.")

    first_pcd = create_pointcloud(camera_dir, selected[0], max_depth, downsample)
    bbox = first_pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_extent())
    if extent <= 0:
        extent = 1.0
    eye = center + np.array([0.0, -view_distance * extent, view_height * extent])
    up = np.array([0.0, 0.0, 1.0])

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.camera.set_projection(
        fov,
        width / float(height),
        0.01,
        1000.0,
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    renderer.scene.camera.look_at(center, eye, up)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = float(point_size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for idx in selected:
        pcd = create_pointcloud(camera_dir, idx, max_depth, downsample)
        if renderer.scene.has_geometry("pcd"):
            renderer.scene.remove_geometry("pcd")
        renderer.scene.add_geometry("pcd", pcd, material)
        renderer.scene.camera.look_at(center, eye, up)
        frame = renderer.render_to_image()
        frame_np = np.asarray(frame)
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 1) * 255
            frame_np = frame_np.astype(np.uint8)
        if frame_np.shape[-1] == 4:
            frame_np = frame_np[:, :, :3]
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
        print(f"Rendered frame {idx}")

    video.release()
    try:
        renderer.release()
    except AttributeError:
        pass
    print(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render wrist-camera point clouds to a third-person MP4 video"
    )
    parser.add_argument(
        "--camera",
        default="datasets/droid_wrist/Fri_Apr_21_17:15:10_2023/17368348",
        help="Camera directory path",
    )
    parser.add_argument(
        "--output",
        default="outputs/wrist_pointcloud_third_person.mp4",
        help="Output MP4 path",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="Start frame index")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index")
    parser.add_argument("--step", type=int, default=1, help="Frame step")
    parser.add_argument("--max-depth", type=float, default=5.0, help="Maximum depth in meters")
    parser.add_argument("--downsample", type=int, default=5, help="Downsample factor")
    parser.add_argument("--width", type=int, default=1280, help="Output video width")
    parser.add_argument("--height", type=int, default=720, help="Output video height")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size")
    parser.add_argument("--fov", type=float, default=60.0, help="Camera FOV (degrees)")
    parser.add_argument("--view-distance", type=float, default=0.2, help="View distance multiplier")
    parser.add_argument("--view-height", type=float, default=1.0, help="View height multiplier")

    args = parser.parse_args()
    output_path = os.path.join(args.camera, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    render_pointcloud_video(
        args.camera,
        output_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        step=args.step,
        max_depth=args.max_depth,
        downsample=args.downsample,
        width=args.width,
        height=args.height,
        fps=args.fps,
        point_size=args.point_size,
        fov=args.fov,
        view_distance=args.view_distance,
        view_height=args.view_height,
    )


if __name__ == "__main__":
    main()
