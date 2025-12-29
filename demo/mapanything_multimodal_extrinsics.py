#!/usr/bin/env python3
"""
Use MapAnything Multi-Modal Inference to estimate relative camera pose between two views,
then save the target view's cam2world extrinsics expanded to [N, 3, 4] for the whole sequence.

This script is tailored for the s2m2 "droid" dataset layout:
  {camera_dir}/images/left/000000.png
  {camera_dir}/depth_npy/000000.npz   (expects key: "depth", in meters)
  {camera_dir}/intrinsics/{camera_id}_left.npy   (3x3)

Example:
  python demo/mapanything_multimodal_extrinsics.py \
    --ref_cam /mnt/disk3.8-5/phs_github/s2m2/datasets/samples/Fri_Jul__7_09:42:23_2023/22008760 \
    --tgt_cam /mnt/disk3.8-5/phs_github/s2m2/datasets/samples/Fri_Jul__7_09:42:23_2023/24400334 \
    --frame 0 \
    --output_dir /mnt/disk3.8-5/phs_github/s2m2/datasets/samples/Fri_Jul__7_09:42:23_2023/24400334/extrinsics_refined \
    --output_name 24400334.npy
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


def _load_depth_meters(depth_npz_path: Path) -> np.ndarray:
    with np.load(str(depth_npz_path)) as data:
        if "depth" not in data:
            raise KeyError(
                f"{depth_npz_path} missing key 'depth'. Available keys: {list(data.keys())}"
            )
        depth = data["depth"]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth (H,W), got {depth.shape} from {depth_npz_path}")
    if depth.dtype == np.float16:
        depth = depth.astype(np.float32)
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
    depth_path = cam_dir / "depth_npy" / f"{frame_idx:06d}.npz"
    K_path = cam_dir / "intrinsics" / f"{cam_id}_left.npy"
    return img_path, depth_path, K_path


def _count_left_images(cam_dir: Path) -> int:
    left_dir = cam_dir / "images" / "left"
    if not left_dir.exists():
        raise FileNotFoundError(f"Missing folder: {left_dir}")
    return len(sorted(left_dir.glob("*.png")))


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
        description="Estimate target cam2world extrinsics via MapAnything multi-modal inference (2 views)."
    )
    parser.add_argument(
        "--ref_cam",
        type=str,
        required=True,
        help="Reference camera directory (world frame will be normalized to this view).",
    )
    parser.add_argument(
        "--tgt_cam",
        type=str,
        required=True,
        help="Target camera directory (we will save its cam2world extrinsics).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to use (default: 0).",
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
        help="Output directory (default: {tgt_cam}/extrinsics_refined).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename inside output_dir (default: {tgt_cam_id}_mapanything.npy).",
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
    parser.add_argument(
        "--save_ply",
        action="store_true",
        default=False,
        help="Also save a fused colored point cloud PLY (ref + tgt) in the original world frame.",
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        default=None,
        help="PLY output path (default: {output_dir}/{output_name with .ply}).",
    )
    parser.add_argument(
        "--ply_stride",
        type=int,
        default=3,
        help="Subsampling stride for point cloud generation (default: 3).",
    )
    parser.add_argument(
        "--ply_max_depth",
        type=float,
        default=10.0,
        help="Max depth (meters) for point cloud filtering (default: 10.0).",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ref_cam = Path(args.ref_cam)
    tgt_cam = Path(args.tgt_cam)
    if not ref_cam.exists():
        raise FileNotFoundError(f"ref_cam not found: {ref_cam}")
    if not tgt_cam.exists():
        raise FileNotFoundError(f"tgt_cam not found: {tgt_cam}")

    out_dir = Path(args.output_dir) if args.output_dir else (tgt_cam / "extrinsics")
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_id = tgt_cam.name
    out_name = args.output_name if args.output_name else f"{tgt_id}_ma.npy"
    out_path = out_dir / out_name
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --overwrite or change --output_name."
        )

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load inputs
    ref_img_path, ref_depth_path, ref_K_path = _paths_for_frame(ref_cam, args.frame)
    tgt_img_path, tgt_depth_path, tgt_K_path = _paths_for_frame(tgt_cam, args.frame)

    for p in [ref_img_path, ref_depth_path, ref_K_path, tgt_img_path, tgt_depth_path, tgt_K_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    ref_img = _load_rgb_uint8(ref_img_path)
    tgt_img = _load_rgb_uint8(tgt_img_path)
    ref_depth = _load_depth_meters(ref_depth_path)
    tgt_depth = _load_depth_meters(tgt_depth_path)
    ref_K = _load_intrinsics(ref_K_path)
    tgt_K = _load_intrinsics(tgt_K_path)

    # Prepare multi-modal views (depth is metric, so set is_metric_scale=True)
    # Note: preprocess_inputs will resize/crop images and will also resize depth & update intrinsics accordingly.
    is_metric = torch.tensor([True])
    views = [
        {"img": ref_img, "intrinsics": ref_K, "depth_z": ref_depth, "is_metric_scale": is_metric},
        {"img": tgt_img, "intrinsics": tgt_K, "depth_z": tgt_depth, "is_metric_scale": is_metric},
    ]
    processed_views = preprocess_inputs(views)

    # Init model
    model = MapAnything.from_pretrained(args.model_name).to(device)
    model.eval()

    use_amp = device.startswith("cuda") and args.amp_dtype in ("bf16", "fp16")

    with torch.no_grad():
        preds = model.infer(
            processed_views,
            memory_efficient_inference=False,
            use_amp=use_amp,
            amp_dtype=args.amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=10,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=False,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=False,
        )

    if len(preds) != 2:
        raise RuntimeError(f"Expected 2 predictions, got {len(preds)}")

    # Extract cam2world poses (OpenCV convention) and normalize so that ref pose becomes identity.
    pose_ref = preds[0]["camera_poses"][0]  # (4,4)
    pose_tgt = preds[1]["camera_poses"][0]  # (4,4)

    if pose_ref.shape != (4, 4) or pose_tgt.shape != (4, 4):
        raise RuntimeError(
            f"Unexpected pose shapes: ref={tuple(pose_ref.shape)}, tgt={tuple(pose_tgt.shape)}"
        )

    # Compute relative transform in the reference camera coordinate frame:
    # pose_rel = inv(pose_ref_pred) @ pose_tgt_pred
    # This makes pose_ref_rel = I, and pose_tgt_rel maps target-cam -> ref-cam.
    pose_ref_inv = torch.linalg.inv(pose_ref)
    pose_tgt_rel = pose_ref_inv @ pose_tgt

    # Lift relative pose into the dataset's original world frame using the dataset-provided
    # reference camera cam2world extrinsics at the same frame:
    #   T_tgt_world = T_ref_world_dataset @ T_tgt_rel
    ref_ext_3x4 = _load_cam2world_extrinsics_for_frame(ref_cam, args.frame)
    T_ref_world = np.eye(4, dtype=np.float32)
    T_ref_world[:3, :] = ref_ext_3x4

    T_tgt_rel = pose_tgt_rel.detach().cpu().numpy().astype(np.float32)
    T_tgt_world = T_ref_world @ T_tgt_rel
    ext_3x4 = T_tgt_world[:3, :].astype(np.float32, copy=False)

    # Save as [N, 3, 4] (camera-to-world extrinsics) expanded to the target sequence length.
    N = _count_left_images(tgt_cam)
    ext_all = np.repeat(ext_3x4[None, :, :], N, axis=0)

    np.save(str(out_path), ext_all)

    ply_path = None
    if args.save_ply:
        if args.ply_path is not None:
            ply_path = Path(args.ply_path)
        else:
            stem = out_path.name
            if stem.lower().endswith(".npy"):
                stem = stem[:-4]
            ply_path = out_dir / f"{stem}.ply"

        # Build fused point cloud using ORIGINAL (non-cropped) RGB/depth/K and ORIGINAL world extrinsics
        # Ref: dataset extrinsics; Tgt: predicted T_tgt_world
        T_ref_world_4x4 = T_ref_world
        T_tgt_world_4x4 = T_tgt_world

        pts_ref_cam, col_ref = _depth_to_colored_points_cam(
            ref_depth,
            ref_img,
            ref_K,
            stride=args.ply_stride,
            max_depth_m=args.ply_max_depth,
        )
        pts_tgt_cam, col_tgt = _depth_to_colored_points_cam(
            tgt_depth,
            tgt_img,
            tgt_K,
            stride=args.ply_stride,
            max_depth_m=args.ply_max_depth,
        )

        pts_ref_world = _transform_points_cam2world(pts_ref_cam, T_ref_world_4x4)
        pts_tgt_world = _transform_points_cam2world(pts_tgt_cam, T_tgt_world_4x4)

        pts_world = np.concatenate([pts_ref_world, pts_tgt_world], axis=0)
        cols = np.concatenate([col_ref, col_tgt], axis=0)

        _write_ply_xyzrgb(ply_path, pts_world, cols)

    print("=" * 80)
    print("MapAnything multi-modal pose estimation complete")
    print(f"ref_cam: {ref_cam}")
    print(f"tgt_cam: {tgt_cam}")
    print(f"frame:   {args.frame:06d}")
    print(f"device:  {device} (use_amp={use_amp}, amp_dtype={args.amp_dtype})")
    print(f"saved:   {out_path}")
    if ply_path is not None:
        print(f"saved:   {ply_path}")
    print(f"shape:   {ext_all.shape}  (expected: [N,3,4], N={N})")
    print("=" * 80)


if __name__ == "__main__":
    main()


