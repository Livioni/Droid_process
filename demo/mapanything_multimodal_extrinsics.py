#!/usr/bin/env python3
"""
Use MapAnything to estimate cam2world poses for all frames in a single camera directory.

This script processes all images in the camera directory and estimates poses for each frame.
Expected layout:
  {camera_dir}/images/left/XXXXXX.png
  {camera_dir}/depth_npy/XXXXXX.npz   (expects key: "depth", in meters)
  {camera_dir}/intrinsics/{camera_id}_left.npy   (3x3)

Example:
  python demo/mapanything_multimodal_extrinsics.py \
    --camera_dir datasets/droid_wrist/Fri_Apr_21_17:11:41_2023/17368348 \
    --output_dir datasets/droid_wrist/Fri_Apr_21_17:11:41_2023/17368348/poses_ma \
    --output_name poses.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def _try_import_mapanything():
    """
    Try importing mapanything. If it isn't installed in the current env,
    fall back to using the local submodule at repo_root/map-anything/.
    """
    try:
        from mapanything.models import MapAnything  # noqa: F401
        from mapanything.utils.image import preprocess_inputs  # noqa: F401
        return
    except Exception:
        # Add local "map-anything" to sys.path
        repo_root = Path(__file__).resolve().parents[1]
        local_pkg_root = repo_root / "map-anything"
        sys.path.insert(0, str(local_pkg_root))


_try_import_mapanything()

import torch  # noqa: E402
from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import preprocess_inputs  # noqa: E402


def _try_import_open3d():
    try:
        import open3d as o3d  # noqa: F401
        return True
    except Exception:
        return False


_OPEN3D_AVAILABLE = _try_import_open3d()


def _load_rgb_uint8(image_path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(str(image_path)).convert("RGB")
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {arr.shape} from {image_path}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _load_depth_meters(depth_path: Path) -> np.ndarray:
    """Load depth from PNG file (assuming depth in millimeters, convert to meters)."""
    from PIL import Image

    img = Image.open(str(depth_path))
    depth = np.array(img, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError(f"Expected depth (H,W), got {depth.shape} from {depth_path}")

    # Convert from millimeters to meters (assuming depth is in mm)
    depth = depth / 1000.0

    return depth


def _load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    K = np.load(str(intrinsics_path))
    if K.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3,3), got {K.shape} from {intrinsics_path}")
    if K.dtype != np.float32:
        K = K.astype(np.float32)
    return K


def _paths_for_frame(cam_dir: Path, frame_idx: int) -> Tuple[Path, Path, Path]:
    cam_id = cam_dir.name
    img_path = cam_dir / "images" / "left" / f"{frame_idx:06d}.png"
    depth_path = cam_dir / "depth_npy" / f"{frame_idx:06d}_depth.png"
    K_path = cam_dir / "intrinsics" / f"{cam_id}_left.npy"
    return img_path, depth_path, K_path


def _count_left_images(cam_dir: Path) -> int:
    left_dir = cam_dir / "images" / "left"
    if not left_dir.exists():
        raise FileNotFoundError(f"Missing folder: {left_dir}")
    return len(sorted(left_dir.glob("*.png")))


def _get_all_frame_indices(cam_dir: Path) -> list[int]:
    """Get all frame indices that have both image and depth files."""
    left_dir = cam_dir / "images" / "left"
    depth_dir = cam_dir / "depth_npy"

    if not left_dir.exists():
        raise FileNotFoundError(f"Missing folder: {left_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing folder: {depth_dir}")

    # Get frame indices from images
    image_frames = set()
    for img_path in left_dir.glob("*.png"):
        try:
            frame_idx = int(img_path.stem)
            image_frames.add(frame_idx)
        except ValueError:
            continue

    # Get frame indices from depth files
    depth_frames = set()
    for depth_path in depth_dir.glob("*_depth.png"):
        try:
            # Extract frame index from filename like "000050_depth.png"
            stem = depth_path.stem  # "000050_depth"
            frame_idx = int(stem.split('_')[0])  # "000050"
            depth_frames.add(frame_idx)
        except (ValueError, IndexError):
            continue

    # Return frames that have both image and depth
    valid_frames = sorted(image_frames & depth_frames)
    return valid_frames


def _depth_to_colored_points_cam(
    depth_m: np.ndarray,
    rgb_uint8: np.ndarray,
    K: np.ndarray,
    stride: int = 2,
    max_depth_m: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create colored 3D points in camera coordinates from depth + intrinsics.

    Returns:
        points_cam: (N,3) float32
        colors:     (N,3) float32 in [0,1]
    """
    if depth_m.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {depth_m.shape}")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"rgb must be (H,W,3), got {rgb_uint8.shape}")
    if depth_m.shape[:2] != rgb_uint8.shape[:2]:
        raise ValueError(
            f"depth and rgb resolution mismatch: depth={depth_m.shape}, rgb={rgb_uint8.shape}"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")

    H, W = depth_m.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[vv, uu].astype(np.float32)

    valid = z > 0
    if max_depth_m is not None:
        valid &= z <= float(max_depth_m)

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    colors = (rgb_uint8[vv.astype(np.int32), uu.astype(np.int32)].astype(np.float32) / 255.0).astype(
        np.float32, copy=False
    )
    return points_cam, colors


def _transform_points_cam2world(points_cam: np.ndarray, T_cam2world_4x4: np.ndarray) -> np.ndarray:
    """Apply cam2world (4x4) to Nx3 camera points -> Nx3 world points."""
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"points_cam must be (N,3), got {points_cam.shape}")
    if T_cam2world_4x4.shape != (4, 4):
        raise ValueError(f"T_cam2world must be (4,4), got {T_cam2world_4x4.shape}")
    R = T_cam2world_4x4[:3, :3].astype(np.float32, copy=False)
    t = T_cam2world_4x4[:3, 3].astype(np.float32, copy=False)
    return (points_cam @ R.T) + t[None, :]


def _write_ply_xyzrgb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save point cloud as PLY. Prefers Open3D if available; otherwise writes ASCII PLY.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points/colors length mismatch: {points.shape} vs {colors.shape}")

    if points.shape[0] == 0:
        raise ValueError("No points to write (0 points after filtering).")

    if _OPEN3D_AVAILABLE:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.clip(0.0, 1.0).astype(np.float64))
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
        return

    # Fallback: ASCII PLY with uint8 RGB
    rgb_u8 = (colors.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, rgb_u8):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _load_cam2world_extrinsics_for_frame(cam_dir: Path, frame_idx: int) -> np.ndarray:
    """
    Load the dataset-provided cam2world extrinsics for a given frame.
    Expected file: {cam_dir}/extrinsics/{camera_id}_left.npy with shape [N,3,4].
    """
    cam_id = cam_dir.name
    ext_path = cam_dir / "extrinsics" / f"{cam_id}_left.npy"
    if not ext_path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {ext_path}")
    ext_all = np.load(str(ext_path))
    if ext_all.ndim != 3 or ext_all.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsics shape [N,3,4], got {ext_all.shape} from {ext_path}")
    if frame_idx < 0 or frame_idx >= ext_all.shape[0]:
        raise IndexError(f"Frame {frame_idx} out of range for {ext_path} (N={ext_all.shape[0]})")
    ext = ext_all[frame_idx].astype(np.float32, copy=False)
    return ext


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate cam2world poses for all frames in a camera directory via MapAnything."
    )
    parser.add_argument(
        "--camera_dir",
        type=str,
        default="datasets/droid_wrist/Fri_Apr_21_17:15:10_2023/17368348",
        help="Camera directory containing images/left/, depth_npy/, and intrinsics/ folders.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/map-anything",
        help='HuggingFace model name (default: "facebook/map-anything").',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, e.g. "cuda" or "cpu". Default: auto.',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for poses (default: {camera_dir}/poses_ma).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="poses.npy",
        help="Output filename (default: poses.npy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="AMP dtype (default: bf16). Use fp32 if bf16 unsupported on your GPU/CPU.",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    camera_dir = Path(args.camera_dir)
    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")

    # Get all frame indices
    frame_indices = _get_all_frame_indices(camera_dir)
    if not frame_indices:
        raise ValueError(f"No valid image frames found in {camera_dir}/images/left/")

    print(f"Found {len(frame_indices)} frames: {frame_indices[0]} to {frame_indices[-1]}")

    out_dir = Path(args.output_dir) if args.output_dir else (camera_dir / "poses_ma")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --overwrite or change --output_name."
        )

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load reference frame (first frame)
    ref_frame = frame_indices[0]
    ref_img_path, ref_depth_path, ref_K_path = _paths_for_frame(camera_dir, ref_frame)

    if not ref_img_path.exists() or not ref_depth_path.exists() or not ref_K_path.exists():
        raise FileNotFoundError(f"Missing required files for reference frame {ref_frame}")

    ref_img = _load_rgb_uint8(ref_img_path)
    ref_depth = _load_depth_meters(ref_depth_path)
    ref_K = _load_intrinsics(ref_K_path)

    # Initialize poses array
    poses = []  # Will store [N, 3, 4] extrinsics for each frame

    # Set reference frame pose as identity (cam2world)
    ref_pose_3x4 = np.eye(3, 4, dtype=np.float32)
    poses.append(ref_pose_3x4)

    # Load all frames at once
    print(f"Loading {len(frame_indices)} frames...")
    all_images = []
    all_depths = []

    for frame_idx in frame_indices:
        img_path, depth_path, _ = _paths_for_frame(camera_dir, frame_idx)

        if not img_path.exists() or not depth_path.exists():
            print(f"Warning: Missing files for frame {frame_idx}, skipping")
            all_images.append(None)
            all_depths.append(None)
            continue

        img = _load_rgb_uint8(img_path)
        depth = _load_depth_meters(depth_path)
        all_images.append(img)
        all_depths.append(depth)

    # Prepare all views for MapAnything
    is_metric = torch.tensor([True])
    views = []

    for i, frame_idx in enumerate(frame_indices):
        if all_images[i] is None or all_depths[i] is None:
            continue

        view = {
            "img": all_images[i],
            "intrinsics": ref_K,  # Use the same intrinsics for all frames
            "depth_z": all_depths[i],
            "is_metric_scale": is_metric
        }
        views.append(view)

    if not views:
        raise ValueError("No valid frames to process")

    print(f"Processing {len(views)} views with MapAnything...")

    # Init model
    model = MapAnything.from_pretrained(args.model_name).to(device)
    model.eval()

    use_amp = device.startswith("cuda") and args.amp_dtype in ("bf16", "fp16")

    # Process all views at once
    processed_views = preprocess_inputs(views)

    with torch.no_grad():
        preds = model.infer(
            processed_views,
            memory_efficient_inference=True,
            use_amp=use_amp,
            amp_dtype=args.amp_dtype,
            apply_mask=False,
            mask_edges=False,
            apply_confidence_mask=False,
            confidence_percentile=10,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=False,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=False,
        )

    # Extract poses from predictions
    # Note: MapAnything returns poses in some world frame, we need to normalize them
    if len(preds) != len(views):
        raise RuntimeError(f"Expected {len(views)} predictions, got {len(preds)}")

    # Get all camera poses
    all_camera_poses = []
    for pred in preds:
        pose_4x4 = pred["camera_poses"][0]  # (4,4) pose
        all_camera_poses.append(pose_4x4)

    # Normalize poses so that the first pose becomes identity
    ref_pose = all_camera_poses[0]
    ref_pose_inv = torch.linalg.inv(ref_pose)

    for i, pose in enumerate(all_camera_poses):
        # Transform to reference frame
        normalized_pose = ref_pose_inv @ pose
        # Convert to 3x4 extrinsics (cam2world)
        ext_3x4 = normalized_pose[:3, :].detach().cpu().numpy().astype(np.float32)
        poses.append(ext_3x4)

    # Convert poses list to numpy array
    poses_array = np.array(poses, dtype=np.float32)  # Shape: [N, 3, 4]

    # Save poses
    np.save(str(out_path), poses_array)


    print("=" * 80)
    print("MapAnything pose estimation complete")
    print(f"camera_dir: {camera_dir}")
    print(f"frames:     {len(frame_indices)} ({frame_indices[0]} to {frame_indices[-1]})")
    print(f"device:     {device} (use_amp={use_amp}, amp_dtype={args.amp_dtype})")
    print(f"saved:      {out_path}")
    print(f"shape:      {poses_array.shape} (expected: [N,3,4], N={len(frame_indices)})")
    print("=" * 80)


if __name__ == "__main__":
    main()


