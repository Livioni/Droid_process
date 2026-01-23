#!/usr/bin/env python3
"""
Batch process stereo image pairs using FoundationStereo model to generate depth maps.
Compatible with demo/batch_stereo_depth_pytorch.py inputs/outputs:
- Input: dataset root with camera folders
- Output: save left-view depth to each camera's depth_npy folder
"""

import argparse
import logging
import os
import sys
import glob
from pathlib import Path

import numpy as np
import cv2
import torch
import imageio.v2 as imageio
from tqdm import tqdm
from omegaconf import OmegaConf


# Add project root and FoundationStereo to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
foundation_root = os.path.join(project_root, "FoundationStereo")
sys.path.insert(0, project_root)
sys.path.insert(0, foundation_root)

from core.utils.utils import InputPadder
from core.foundation_stereo import FoundationStereo


def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def discover_cameras(dataset_root, specified_cameras=None):
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"数据集目录不存在: {dataset_root}")

    cameras = []
    for camera_dir in sorted(dataset_path.iterdir()):
        if not camera_dir.is_dir():
            continue

        camera_id = camera_dir.name
        if specified_cameras and camera_id not in specified_cameras:
            continue

        required_dirs = ["images", "intrinsics", "extrinsics"]
        required_subdirs = ["left", "right"]

        missing_dirs = []
        for req_dir in required_dirs:
            dir_path = camera_dir / req_dir
            if not dir_path.exists():
                missing_dirs.append(req_dir)
                continue
            if req_dir == "images":
                for subdir in required_subdirs:
                    subdir_path = dir_path / subdir
                    if not subdir_path.exists():
                        missing_dirs.append(f"images/{subdir}")

        if missing_dirs:
            logging.warning("相机 %s 缺少必要目录: %s，跳过", camera_id, missing_dirs)
            continue

        intrinsic_left = camera_dir / "intrinsics" / f"{camera_id}_left.npy"
        intrinsic_right = camera_dir / "intrinsics" / f"{camera_id}_right.npy"
        extrinsic_left = camera_dir / "extrinsics" / f"{camera_id}_left.npy"
        extrinsic_right = camera_dir / "extrinsics" / f"{camera_id}_right.npy"

        missing_files = []
        if not intrinsic_left.exists():
            missing_files.append(f"intrinsics/{camera_id}_left.npy")
        if not intrinsic_right.exists():
            missing_files.append(f"intrinsics/{camera_id}_right.npy")
        if not extrinsic_left.exists():
            missing_files.append(f"extrinsics/{camera_id}_left.npy")
        if not extrinsic_right.exists():
            missing_files.append(f"extrinsics/{camera_id}_right.npy")

        if missing_files:
            logging.warning("相机 %s 缺少标定文件: %s，跳过", camera_id, missing_files)
            continue

        cameras.append(
            {
                "camera_id": camera_id,
                "camera_path": camera_dir,
                "left_images": camera_dir / "images" / "left",
                "right_images": camera_dir / "images" / "right",
                "intrinsic_left": intrinsic_left,
                "intrinsic_right": intrinsic_right,
                "extrinsic_left": extrinsic_left,
                "extrinsic_right": extrinsic_right,
                "depth_output": camera_dir,
            }
        )

    return cameras


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="批量处理立体图像对生成深度图 (FoundationStereo)"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets/samples",
        help="数据集根目录路径（包含所有相机文件夹）",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        help="指定要处理的相机ID列表（如果不指定则处理所有相机）",
    )

    parser.add_argument("--left_folder", type=str, help="左相机图像文件夹路径")
    parser.add_argument("--right_folder", type=str, help="右相机图像文件夹路径")
    parser.add_argument("--intrinsic_left", type=str, help="左相机内参npy文件路径")
    parser.add_argument("--intrinsic_right", type=str, help="右相机内参npy文件路径")
    parser.add_argument(
        "--extrinsic_left",
        type=str,
        help="左相机外参npy文件路径（每一帧的外参，第一维为帧索引）",
    )
    parser.add_argument(
        "--extrinsic_right",
        type=str,
        help="右相机外参npy文件路径（每一帧的外参，第一维为帧索引）",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.path.join(project_root, "weights", "FoundationStereo", "model_best_bp2.pth"),
        help="预训练模型路径",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--scale", default=1.0, type=float, help="图像缩放比例(<=1)")
    parser.add_argument(
        "--hiera",
        default=2,
        type=int,
        help="hierarchical inference (适用于>1K分辨率)",
    )
    parser.add_argument(
        "--valid_iters", type=int, default=64, help="前向迭代次数"
    )
    parser.add_argument(
        "--image_extension",
        type=str,
        default="png",
        help="图像文件扩展名 (png, jpg, etc.)",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="保存深度图可视化",
    )
    parser.add_argument(
        "--remove_invisible",
        type=int,
        default=1,
        help="移除左右视角不重叠区域（与run_demo.py一致）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="兼容模式输出目录（单相机处理）",
    )

    return parser


def load_camera_matrix(intrinsic_path):
    intrinsic_data = np.load(intrinsic_path, allow_pickle=True)
    if isinstance(intrinsic_data, np.ndarray) and intrinsic_data.dtype == object:
        intrinsic_data = intrinsic_data.item()
        if isinstance(intrinsic_data, dict):
            for key in ["K", "intrinsic", "camera_matrix", "intrinsics"]:
                if key in intrinsic_data:
                    intrinsic_matrix = intrinsic_data[key]
                    break
            else:
                raise ValueError(
                    f"无法从字典中找到内参矩阵，可用的键: {intrinsic_data.keys()}"
                )
        else:
            intrinsic_matrix = intrinsic_data
    else:
        intrinsic_matrix = intrinsic_data

    intrinsic_matrix = np.array(intrinsic_matrix).reshape(3, 3)
    fx = intrinsic_matrix[0, 0]
    return intrinsic_matrix, fx


def load_extrinsics_array(extrinsic_path):
    extrinsics = np.load(extrinsic_path, allow_pickle=True)
    return extrinsics


def compute_baseline_per_frame(extrinsic_left, extrinsic_right):
    if extrinsic_left.shape[0] == 3:
        t_left = extrinsic_left[:, 3]
    else:
        t_left = extrinsic_left[:3, 3]

    if extrinsic_right.shape[0] == 3:
        t_right = extrinsic_right[:, 3]
    else:
        t_right = extrinsic_right[:3, 3]

    baseline_m = float(np.linalg.norm(t_right - t_left))
    return baseline_m


def get_image_pairs(left_folder, right_folder, extension="png"):
    left_images = sorted(
        glob.glob(os.path.join(left_folder, f"*.{extension}")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )
    if len(left_images) == 0:
        raise ValueError(f"在 {left_folder} 中未找到 .{extension} 图像")

    image_pairs = []
    for left_path in left_images:
        filename = os.path.basename(left_path)
        right_path = os.path.join(right_folder, filename)
        if os.path.exists(right_path):
            image_pairs.append((left_path, right_path, filename))
        else:
            logging.warning("未找到对应的右图像: %s", right_path)

    logging.info("找到 %d 对立体图像", len(image_pairs))
    return image_pairs


def save_depth_visualization(depth_map, output_path):
    depth_vis = depth_map.copy()
    valid_mask = depth_vis > 0

    if np.any(valid_mask):
        depth_vis_clipped = np.clip(depth_vis, 0, 10)
        depth_vis_log = np.log(depth_vis_clipped + 1)
        min_val = np.min(depth_vis_log[valid_mask])
        max_val = np.max(depth_vis_log[valid_mask])
        depth_vis_norm = np.zeros_like(depth_vis_log, dtype=np.uint8)
        if max_val > min_val:
            depth_vis_norm[valid_mask] = (
                (depth_vis_log[valid_mask] - min_val) / (max_val - min_val)
            ) * 255
        depth_vis = depth_vis_norm.astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)

    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, depth_colored)


def load_model(args):
    if not os.path.exists(args.ckpt_dir):
        raise FileNotFoundError(f"模型权重不存在: {args.ckpt_dir}")

    cfg = OmegaConf.load(f"{os.path.dirname(args.ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)
    ckpt = torch.load(args.ckpt_dir, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model, args


def infer_disparity(model, args, img0, img1):
    scale = args.scale
    if scale <= 0 or scale > 1:
        raise ValueError("scale must be <=1 and >0")

    if scale != 1.0:
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)

    h, w = img0.shape[:2]
    img0_t = torch.as_tensor(img0).to(args.device).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).to(args.device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled="cuda" in str(args.device)):
        if not args.hiera:
            disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(
                img0_t, img1_t, iters=args.valid_iters, test_mode=True, small_ratio=0.5
            )
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(h, w)
    return disp


def process_single_camera(camera_info, args, model):
    camera_id = camera_info["camera_id"]
    logging.info("处理相机: %s", camera_id)

    depth_output_dir = camera_info["depth_output"]
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir = depth_output_dir / "depth_npy"
    depth_npy_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = None
    if args.save_visualization:
        vis_dir = depth_output_dir / "depth_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    _, fx_left = load_camera_matrix(str(camera_info["intrinsic_left"]))
    extrinsics_left = load_extrinsics_array(str(camera_info["extrinsic_left"]))
    extrinsics_right = load_extrinsics_array(str(camera_info["extrinsic_right"]))

    if extrinsics_left.shape[0] != extrinsics_right.shape[0]:
        raise ValueError(
            f"相机 {camera_id}: 左右外参帧数不匹配: "
            f"{extrinsics_left.shape[0]} vs {extrinsics_right.shape[0]}"
        )

    image_pairs = get_image_pairs(
        str(camera_info["left_images"]),
        str(camera_info["right_images"]),
        args.image_extension,
    )
    if len(image_pairs) == 0:
        logging.warning("相机 %s 未找到图像对，跳过", camera_id)
        return

    if len(image_pairs) > extrinsics_left.shape[0]:
        logging.warning(
            "相机 %s: 图像对数量(%d)超过外参数量(%d)，只处理前者",
            camera_id,
            len(image_pairs),
            extrinsics_left.shape[0],
        )
        image_pairs = image_pairs[: extrinsics_left.shape[0]]

    baselines = []

    for frame_idx, (left_path, right_path, filename) in enumerate(
        tqdm(image_pairs, desc=f"相机 {camera_id}")
    ):
        base_name = os.path.splitext(filename)[0]
        try:
            img0 = imageio.imread(left_path)
            img1 = imageio.imread(right_path)
            if img0.ndim == 2:
                img0 = np.stack([img0] * 3, axis=-1)
            if img1.ndim == 2:
                img1 = np.stack([img1] * 3, axis=-1)
            if img0.shape[-1] == 4:
                img0 = img0[:, :, :3]
            if img1.shape[-1] == 4:
                img1 = img1[:, :, :3]

            disp = infer_disparity(model, args, img0, img1)
            if args.remove_invisible:
                yy, xx = np.meshgrid(
                    np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij"
                )
                us_right = xx - disp
                disp[us_right < 0] = np.inf
            baseline_m = compute_baseline_per_frame(
                extrinsics_left[frame_idx], extrinsics_right[frame_idx]
            )
            baselines.append(baseline_m)

            fx = fx_left * args.scale
            disp_safe = disp.copy()
            disp_safe[disp_safe <= 0] = np.nan
            depth = fx * baseline_m / disp_safe
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )

            depth_path = depth_npy_dir / f"{base_name}.npz"
            np.savez_compressed(str(depth_path), depth=depth)

            if args.save_visualization:
                vis_path = vis_dir / f"{base_name}_depth.png"
                save_depth_visualization(depth, str(vis_path))
        except Exception as exc:
            logging.exception(
                "相机 %s: 处理 %s (frame %d) 出错: %s",
                camera_id,
                filename,
                frame_idx,
                str(exc),
            )
            continue

    if baselines:
        baselines = np.array(baselines, dtype=np.float64)
        logging.info(
            "相机 %s baseline(m): min=%.6f max=%.6f mean=%.6f std=%.6f",
            camera_id,
            baselines.min(),
            baselines.max(),
            baselines.mean(),
            baselines.std(),
        )
    logging.info("相机 %s 处理完成，输出: %s", camera_id, depth_npy_dir)


def main(args):
    set_logging()
    torch.autograd.set_grad_enabled(False)

    is_legacy_mode = bool(
        args.left_folder
        and args.right_folder
        and args.intrinsic_left
        and args.intrinsic_right
        and args.extrinsic_left
        and args.extrinsic_right
    )

    if is_legacy_mode:
        camera_info = {
            "camera_id": "single_camera",
            "camera_path": Path(args.left_folder).parent.parent,
            "left_images": Path(args.left_folder),
            "right_images": Path(args.right_folder),
            "intrinsic_left": Path(args.intrinsic_left),
            "intrinsic_right": Path(args.intrinsic_right),
            "extrinsic_left": Path(args.extrinsic_left),
            "extrinsic_right": Path(args.extrinsic_right),
            "depth_output": Path(args.output_dir),
        }
        cameras = [camera_info]
    else:
        cameras = discover_cameras(args.dataset_root, args.cameras)
        if not cameras:
            logging.error("未发现有效相机配置")
            return

    device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    logging.info("使用设备: %s", device)

    model, args = load_model(args)
    model = model.to(args.device)
    model.eval()

    for camera_info in cameras:
        process_single_camera(camera_info, args, model)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
