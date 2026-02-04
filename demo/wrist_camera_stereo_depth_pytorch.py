#!/usr/bin/env python3
"""
专门处理wrist camera的立体深度图生成
从场景文件夹中自动识别包含wrist_camera.txt的腕部相机文件夹
使用PyTorch模型从左右相机图像生成深度图
支持从左右相机外参动态计算每一帧的baseline和doffs
Process wrist camera stereo depth using PyTorch model to generate depth maps
"""

import argparse
import os
import sys
import glob
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
import torch._dynamo
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from s2m2.core.utils.model_utils import load_model, run_stereo_matching
from s2m2.core.utils.image_utils import read_images


def discover_cameras(dataset_root):
    """
    扫描数据集目录，发现包含wrist_camera.txt的腕部相机文件夹

    Args:
        dataset_root: 数据集根目录路径（场景文件夹）

    Returns:
        list: 相机信息字典列表，只包含找到的腕部相机（最多一个）
    """
    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        raise ValueError(f"数据集目录不存在: {dataset_root}")

    cameras = []

    # 扫描所有子目录
    for camera_dir in sorted(dataset_path.iterdir()):
        if not camera_dir.is_dir():
            continue

        camera_id = camera_dir.name

        # 检查是否为腕部相机（通过wrist_camera.txt文件标识）
        wrist_marker = camera_dir / 'wrist_camera.txt'
        if not wrist_marker.exists():
            continue

        print(f"发现腕部相机文件夹: {camera_id}")

        # 检查必要的子目录是否存在
        required_dirs = ['images', 'intrinsics', 'extrinsics']
        required_subdirs = ['left', 'right']

        missing_dirs = []
        for req_dir in required_dirs:
            dir_path = camera_dir / req_dir
            if not dir_path.exists():
                missing_dirs.append(req_dir)
                continue

            if req_dir == 'images':
                for subdir in required_subdirs:
                    subdir_path = dir_path / subdir
                    if not subdir_path.exists():
                        missing_dirs.append(f"images/{subdir}")

        if missing_dirs:
            print(f"警告: 相机 {camera_id} 缺少必要目录: {missing_dirs}，跳过")
            continue

        # 检查标定文件是否存在
        intrinsic_left = camera_dir / 'intrinsics' / f'{camera_id}_left.npy'
        intrinsic_right = camera_dir / 'intrinsics' / f'{camera_id}_right.npy'
        extrinsic_left = camera_dir / 'extrinsics' / f'{camera_id}_left.npy'
        extrinsic_right = camera_dir / 'extrinsics' / f'{camera_id}_right.npy'

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
            print(f"警告: 相机 {camera_id} 缺少标定文件: {missing_files}，跳过")
            continue

        cameras.append({
            'camera_id': camera_id,
            'camera_path': camera_dir,
            'left_images': camera_dir / 'images' / 'left',
            'right_images': camera_dir / 'images' / 'right',
            'intrinsic_left': intrinsic_left,
            'intrinsic_right': intrinsic_right,
            'extrinsic_left': extrinsic_left,
            'extrinsic_right': extrinsic_right,
            'depth_output': camera_dir  # 直接指向相机目录，不再是depths子文件夹
        })

        # 只返回找到的第一个腕部相机
        break

    return cameras


def get_args_parser():
    parser = argparse.ArgumentParser(description='处理wrist camera立体图像生成深度图 (PyTorch)')

    # 数据集参数
    parser.add_argument('--dataset_root', type=str,
                        default='datasets/droid/Fri_Jul__7_09:45:39_2023',
                        help='数据集场景目录路径（包含3个相机子文件夹，其中一个包含wrist_camera.txt）')
    
    # 模型参数
    parser.add_argument('--model_type', default='XL', type=str,
                        help='模型类型: S, M, L, XL')
    parser.add_argument('--num_refine', default=9, type=int,
                        help='局部迭代refinement次数')
    parser.add_argument('--torch_compile', action='store_true', 
                        help='使用torch.compile加速')
    parser.add_argument('--allow_negative', action='store_true', 
                        help='允许负视差（用于不完美的rectification）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备 (cuda:0, cuda:1, cpu)')
    parser.add_argument('--save_calibration', action='store_true',
                        help='保存每帧的标定参数')
    
    # 处理参数
    parser.add_argument('--confidence_threshold', type=float, default=0.99,
                        help='置信度阈值，小于此值的深度置0')
    parser.add_argument('--output_dir', type=str, default='output/batch_depth',
                        help='输出文件夹路径')
    parser.add_argument('--save_visualization', action='store_true',
                        help='保存深度图可视化')
    parser.add_argument('--save_disparity', action='store_true',
                        help='保存视差图')
    parser.add_argument('--image_extension', type=str, default='png',
                        help='图像文件扩展名 (png, jpg, etc.)')


    return parser


def load_camera_matrix(intrinsic_path):
    """
    加载单个相机内参矩阵
    
    Returns:
        tuple: (intrinsic_matrix, fx, fy, cx, cy)
    """
    intrinsic_data = np.load(intrinsic_path, allow_pickle=True)
    
    # 如果是字典格式
    if isinstance(intrinsic_data, np.ndarray) and intrinsic_data.dtype == object:
        intrinsic_data = intrinsic_data.item()
        if isinstance(intrinsic_data, dict):
            for key in ['K', 'intrinsic', 'camera_matrix', 'intrinsics']:
                if key in intrinsic_data:
                    intrinsic_matrix = intrinsic_data[key]
                    break
            else:
                raise ValueError(f"无法从字典中找到内参矩阵，可用的键: {intrinsic_data.keys()}")
        else:
            intrinsic_matrix = intrinsic_data
    else:
        intrinsic_matrix = intrinsic_data
    
    # 确保是3x3矩阵
    intrinsic_matrix = np.array(intrinsic_matrix).reshape(3, 3)
    
    # 提取参数
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    return intrinsic_matrix, fx, fy, cx, cy


def load_extrinsics_array(extrinsic_path):
    """
    加载外参数组（每一帧一个外参）
    
    Returns:
        np.ndarray: 外参数组，形状为 (N, 3, 4) 或 (N, 4, 4)
    """
    extrinsics = np.load(extrinsic_path, allow_pickle=True)
    
    print(f"  外参数组形状: {extrinsics.shape}")
    
    return extrinsics


def compute_baseline_and_doffs_per_frame(extrinsic_left, extrinsic_right, cx_left, cx_right):
    """
    从左右相机外参计算baseline和doffs
    
    Args:
        extrinsic_left: 左相机外参 (3x4 或 4x4)
        extrinsic_right: 右相机外参 (3x4 或 4x4)
        cx_left: 左相机主点x坐标
        cx_right: 右相机主点x坐标
    
    Returns:
        tuple: (baseline_mm, doffs)
    """
    # 提取平移向量
    if extrinsic_left.shape[0] == 3:
        t_left = extrinsic_left[:, 3]
    else:  # 4x4
        t_left = extrinsic_left[:3, 3]
    
    if extrinsic_right.shape[0] == 3:
        t_right = extrinsic_right[:, 3]
    else:  # 4x4
        t_right = extrinsic_right[:3, 3]
    
    # 计算相对平移（从左到右）
    relative_translation = t_right - t_left
    
    # 计算baseline（欧几里得距离，转换为mm）
    baseline_mm = np.linalg.norm(relative_translation) * 1000
    
    # 计算doffs
    doffs = cx_left - cx_right
    
    return baseline_mm, doffs


def get_image_pairs(left_folder, right_folder, extension='png'):
    """获取所有匹配的图像对"""
    left_images = sorted(glob.glob(os.path.join(left_folder, f'*.{extension}')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if len(left_images) == 0:
        raise ValueError(f"在 {left_folder} 中未找到 .{extension} 图像")
    
    image_pairs = []
    for left_path in left_images:
        filename = os.path.basename(left_path)
        right_path = os.path.join(right_folder, filename)
        
        if os.path.exists(right_path):
            image_pairs.append((left_path, right_path, filename))
        else:
            print(f"警告: 未找到对应的右图像: {right_path}")
    
    print(f"找到 {len(image_pairs)} 对立体图像")
    return image_pairs


def preprocess_images(left, right):
    """预处理图像"""
    img_height, img_width = left.shape[:2]
    
    # 确保尺寸是32的倍数
    img_height = (img_height // 32) * 32
    img_width = (img_width // 32) * 32
    
    left = left[:img_height, :img_width]
    right = right[:img_height, :img_width]
    
    return left, right


def calculate_depth_from_disparity(disparity, focal_length, baseline, doffs=0):
    """
    从视差图计算深度图
    depth = (focal_length * baseline) / (disparity + doffs)
    
    Args:
        disparity: 视差图 [H, W]
        focal_length: 焦距 (pixels)
        baseline: 基线距离 (mm)
        doffs: 视差偏移
    
    Returns:
        depth: 深度图 (mm) [H, W]
    """
    # 避免除零
    disparity_safe = np.where(disparity <= 0, 1e-6, disparity)
    
    # 计算深度
    depth = (focal_length * baseline) / (disparity_safe + doffs)
    
    # 将无效深度设为0
    depth = np.where(disparity <= 0, 0, depth)
    
    return depth


def save_depth_visualization(depth_map, output_path):
    """保存深度图可视化"""
    depth_vis = depth_map.copy()
    
    # 创建有效深度掩码
    valid_mask = depth_vis > 0
    
    if np.any(valid_mask):
        # 裁剪深度值以获得更好的可视化效果 (100mm to 10000mm)
        depth_vis_clipped = np.clip(depth_vis, 0, 10)
        
        # 应用对数缩放
        depth_vis_log = np.log(depth_vis_clipped + 1)
        
        # 归一化到0-255
        min_val = np.min(depth_vis_log[valid_mask])
        max_val = np.max(depth_vis_log[valid_mask])
        depth_vis_norm = np.zeros_like(depth_vis_log, dtype=np.uint8)
        if max_val > min_val:
            depth_vis_norm[valid_mask] = ((depth_vis_log[valid_mask] - min_val) / (max_val - min_val)) * 255
        depth_vis = depth_vis_norm.astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)
    
    # 应用colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    # 保存
    cv2.imwrite(output_path, depth_colored)


def save_disparity_visualization(disparity, output_path):
    """保存视差图可视化"""
    disp_vis = disparity.copy()
    valid_mask = disp_vis > 0

    if np.any(valid_mask):
        min_val = np.min(disp_vis[valid_mask])
        max_val = np.max(disp_vis[valid_mask])
        disp_norm = np.zeros_like(disp_vis, dtype=np.uint8)
        if max_val > min_val:
            disp_norm[valid_mask] = ((disp_vis[valid_mask] - min_val) / (max_val - min_val)) * 255
        disp_vis = disp_norm
    else:
        disp_vis = np.zeros_like(disp_vis, dtype=np.uint8)

    # 应用colormap
    disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

    # 保存
    cv2.imwrite(output_path, disp_colored)


def save_depth_png(depth_map, output_path):
    """
    保存深度图为PNG格式（16-bit，单位:mm）

    Args:
        depth_map: 深度图 [H, W]，单位为米
        output_path: 输出路径
    """
    # 将深度从米转换为毫米，然后保存为uint16
    depth_mm = depth_map * 1000.0  # 米 -> 毫米

    # 确保深度值在uint16范围内 (0-65535mm)
    # 假设最大深度为10米(10000mm)，超过的部分设为0
    depth_mm = np.clip(depth_mm, 0, 65535.0).astype(np.uint16)

    # 保存为16-bit PNG
    cv2.imwrite(output_path, depth_mm)


# =============================================================================
# 优化的深度图存储方法 / Optimized depth map storage methods
# =============================================================================



def process_single_camera(camera_info, args, device, model):
    """
    处理单个相机的立体深度图生成

    Args:
        camera_info: 相机信息字典
        args: 命令行参数
        device: 计算设备
        model: 已加载的PyTorch模型
    """
    camera_id = camera_info['camera_id']
    print(f"\n{'='*80}")
    print(f"处理相机: {camera_id}")
    print(f"{'='*80}")

    # 创建深度图输出目录
    depth_output_dir = camera_info['depth_output']
    depth_output_dir.mkdir(parents=True, exist_ok=True)

    # 其他子目录
    depth_npy_dir = depth_output_dir / 'depth_npy'
    depth_npy_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if args.save_visualization:
        vis_dir = depth_output_dir / 'depth_vis'
        vis_dir.mkdir(parents=True, exist_ok=True)

    disp_dir = None
    disp_vis_dir = None
    if args.save_disparity:
        disp_dir = depth_output_dir / 'disparity_npy'
        disp_vis_dir = depth_output_dir / 'disparity_vis'
        disp_dir.mkdir(parents=True, exist_ok=True)
        disp_vis_dir.mkdir(parents=True, exist_ok=True)

    calib_dir = None
    if args.save_calibration:
        calib_dir = depth_output_dir / 'calibration_per_frame'
        calib_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {depth_output_dir}")

    print("\n" + "-"*60)
    print("加载相机标定参数")
    print("-"*60)

    # 加载内参
    print(f"\n加载左相机内参: {camera_info['intrinsic_left']}")
    left_K, fx_left, fy_left, cx_left, cy_left = load_camera_matrix(str(camera_info['intrinsic_left']))
    print(f"  fx: {fx_left:.3f}, fy: {fy_left:.3f}")
    print(f"  cx: {cx_left:.3f}, cy: {cy_left:.3f}")

    print(f"\n加载右相机内参: {camera_info['intrinsic_right']}")
    right_K, fx_right, fy_right, cx_right, cy_right = load_camera_matrix(str(camera_info['intrinsic_right']))
    print(f"  fx: {fx_right:.3f}, fy: {fy_right:.3f}")
    print(f"  cx: {cx_right:.3f}, cy: {cy_right:.3f}")

    # 加载外参数组
    print(f"\n加载左相机外参数组: {camera_info['extrinsic_left']}")
    extrinsics_left = load_extrinsics_array(str(camera_info['extrinsic_left']))

    print(f"\n加载右相机外参数组: {camera_info['extrinsic_right']}")
    extrinsics_right = load_extrinsics_array(str(camera_info['extrinsic_right']))

    # 验证外参数组长度
    if extrinsics_left.shape[0] != extrinsics_right.shape[0]:
        raise ValueError(f"相机 {camera_id}: 左右外参数组长度不匹配: {extrinsics_left.shape[0]} vs {extrinsics_right.shape[0]}")

    num_extrinsics = extrinsics_left.shape[0]
    print(f"\n外参数组包含 {num_extrinsics} 帧")

    # 计算第一帧的baseline和doffs作为示例
    baseline_0, doffs_0 = compute_baseline_and_doffs_per_frame(
        extrinsics_left[0], extrinsics_right[0], cx_left, cx_right
    )
    print(f"\n第0帧参数示例:")
    print(f"  baseline: {baseline_0:.3f} mm")
    print(f"  doffs: {doffs_0:.3f}")

    # 获取图像对列表
    print("\n" + "-"*60)
    print("扫描图像文件")
    print("-"*60)
    image_pairs = get_image_pairs(str(camera_info['left_images']), str(camera_info['right_images']), args.image_extension)

    if len(image_pairs) == 0:
        print(f"相机 {camera_id}: 未找到匹配的图像对，跳过")
        return

    # 验证图像数量和外参数量
    if len(image_pairs) > num_extrinsics:
        print(f"相机 {camera_id}: 警告: 图像对数量({len(image_pairs)})超过外参数量({num_extrinsics})")
        print(f"将只处理前 {num_extrinsics} 对图像")
        image_pairs = image_pairs[:num_extrinsics]

    print("\n" + "-"*60)
    print(f"开始处理相机 {camera_id} 的 {len(image_pairs)} 对图像")
    print("-"*60)
    print(f"输出目录: {depth_output_dir}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print(f"焦距: {fx_left:.3f} pixels")
    print("-"*60 + "\n")

    # 统计信息
    total_valid_pixels = 0
    total_pixels = 0
    baselines = []
    doffs_list = []


    # 处理每对图像
    for frame_idx, (left_path, right_path, filename) in enumerate(tqdm(image_pairs, desc=f"相机 {camera_id}")):
        try:
            # 文件名处理
            base_name = os.path.splitext(filename)[0]

            # 计算当前帧的baseline和doffs
            baseline, doffs = compute_baseline_and_doffs_per_frame(
                extrinsics_left[frame_idx],
                extrinsics_right[frame_idx],
                cx_left,
                cx_right
            )

            baselines.append(baseline)
            doffs_list.append(doffs)

            # 读取图像
            left, right = read_images(left_path, right_path)

            # 预处理
            # left_processed, right_processed = preprocess_images(left, right)

            # 转换为torch tensor
            left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).to(device)
            right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).to(device)

            # 运行立体匹配
            with torch.no_grad():
                pred_disp, pred_occ, pred_conf, _, _ = run_stereo_matching(
                    model, left_torch, right_torch, device, N_repeat=1
                )

            # 转换为numpy
            pred_disp = pred_disp.cpu().numpy()
            pred_occ = pred_occ.cpu().numpy()
            pred_conf = pred_conf.cpu().numpy()

            # 计算深度图
            depth_map = calculate_depth_from_disparity(
                pred_disp, fx_left, baseline, doffs
            )

            # 应用置信度过滤
            depth_map[pred_conf < args.confidence_threshold] = 0

            # 限制最大深度到 10m：超过则置为 0（depth_map 当前单位为 mm）
            max_depth_mm = 10_000.0
            depth_map[depth_map > max_depth_mm] = 0

            # 转换为米单位用于统计和保存
            depth_map_meters = depth_map / 1000

            # 统计
            valid_pixels = np.sum(depth_map_meters > 0)
            total_valid_pixels += valid_pixels
            total_pixels += depth_map_meters.size

            # 保存深度图为uint16 PNG格式
            depth_png_path = depth_npy_dir / f'{base_name}_depth.png'
            save_depth_png(depth_map_meters, str(depth_png_path))

            # 保存深度图可视化
            if args.save_visualization:
                depth_vis_path = vis_dir / f'{base_name}_depth.png'
                save_depth_visualization(depth_map, str(depth_vis_path))

            # 保存视差图
            if args.save_disparity:
                disp_npy_path = disp_dir / f'{base_name}_disp.npy'
                np.save(str(disp_npy_path), pred_disp)

                disp_vis_path = disp_vis_dir / f'{base_name}_disp.png'
                save_disparity_visualization(pred_disp, str(disp_vis_path))

            if args.save_calibration:
                calib_params = {
                    'frame_idx': frame_idx,
                    'filename': filename,
                    'focal_length': fx_left,
                    'baseline': baseline,
                    'doffs': doffs,
                    'cx_left': cx_left,
                    'cx_right': cx_right
                }
                calib_path = calib_dir / f'{base_name}_calib.npy'
                np.save(str(calib_path), calib_params)

        except Exception as e:
            print(f"\n相机 {camera_id}: 处理 {filename} (frame {frame_idx}) 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 保存所有帧的baseline和doffs
    if args.save_calibration:
        all_calib_params = {
            'baselines': np.array(baselines),
            'doffs': np.array(doffs_list),
            'focal_length': fx_left,
            'cx_left': cx_left,
            'cx_right': cx_right
        }
        np.save(str(depth_output_dir / 'all_calibration_params.npy'), all_calib_params)


    # 输出相机处理统计信息
    print("\n" + "-"*60)
    print(f"相机 {camera_id} 处理完成!")
    print("-"*60)
    print(f"总图像对数: {len(image_pairs)}")
    print(f"有效深度像素比例: {100 * total_valid_pixels / total_pixels:.2f}%")
    print(f"\nBaseline 统计:")
    print(f"  最小值: {np.min(baselines):.3f} mm")
    print(f"  最大值: {np.max(baselines):.3f} mm")
    print(f"  平均值: {np.mean(baselines):.3f} mm")
    print(f"  标准差: {np.std(baselines):.3f} mm")
    print(f"\nDoffs 统计:")
    print(f"  最小值: {np.min(doffs_list):.3f}")
    print(f"  最大值: {np.max(doffs_list):.3f}")
    print(f"  平均值: {np.mean(doffs_list):.3f}")
    print(f"  标准差: {np.std(doffs_list):.3f}")
    print(f"\n结果保存在: {depth_output_dir}")
    print("-"*60)

    return {
        'camera_id': camera_id,
        'total_frames': len(image_pairs),
        'valid_pixels_ratio': 100 * total_valid_pixels / total_pixels,
        'baseline_stats': {
            'min': np.min(baselines), 'max': np.max(baselines),
            'mean': np.mean(baselines), 'std': np.std(baselines)
        },
        'doffs_stats': {
            'min': np.min(doffs_list), 'max': np.max(doffs_list),
            'mean': np.mean(doffs_list), 'std': np.std(doffs_list)
        }
    }


def main(args):
    # 扫描场景目录，查找腕部相机
    print(f"扫描场景目录: {args.dataset_root}")
    cameras = discover_cameras(args.dataset_root)

    if not cameras:
        print("错误: 未发现有效的腕部相机配置（包含wrist_camera.txt文件的文件夹）")
        return

    print(f"发现腕部相机: {cameras[0]['camera_id']}")

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 设置随机种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True

    print("\n" + "="*60)
    print("加载PyTorch模型")
    print("="*60)
    
    # 加载模型
    model = load_model(
        os.path.join(project_root, "weights/pretrain_weights"),
        args.model_type,
        not args.allow_negative,
        args.num_refine,
        device
    )

    if args.torch_compile:
        print("应用 torch.compile 优化...")
        model = torch.compile(model)

    print(f"✓ 模型加载完成: {args.model_type}")

    # 处理所有相机
    all_camera_stats = []

    for camera_info in cameras:
        try:
            camera_stats = process_single_camera(camera_info, args, device, model)
            if camera_stats:
                all_camera_stats.append(camera_stats)
        except Exception as e:
            print(f"处理相机 {camera_info['camera_id']} 时出现严重错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 输出总体统计信息
    if all_camera_stats:
        print("\n" + "="*80)
        print("所有相机处理完成!")
        print("="*80)
        print(f"成功处理的相机数量: {len(all_camera_stats)}")

        total_frames_all = sum(stats['total_frames'] for stats in all_camera_stats)
        avg_valid_pixels = np.mean([stats['valid_pixels_ratio'] for stats in all_camera_stats])

        print(f"总处理帧数: {total_frames_all}")
        print(f"平均有效深度像素比例: {avg_valid_pixels:.2f}%")

        # 显示每个相机的统计
        print(f"\n各相机统计:")
        for stats in all_camera_stats:
            print(f"  相机 {stats['camera_id']}: {stats['total_frames']} 帧, 有效像素 {stats['valid_pixels_ratio']:.1f}%")
            print(f"    Baseline: {stats['baseline_stats']['mean']:.1f}±{stats['baseline_stats']['std']:.1f} mm")
            print(f"    Doffs: {stats['doffs_stats']['mean']:.1f}±{stats['doffs_stats']['std']:.1f}")

        print(f"\n深度图保存为: uint16 PNG格式 (毫米单位)")
        print("="*80)
    else:
        print("错误: 没有相机被成功处理")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # 验证数据集目录存在
    if not os.path.exists(args.dataset_root):
        print(f"错误: 数据集场景目录不存在: {args.dataset_root}")
        sys.exit(1)

    print("=" * 80)
    print("Wrist Camera 立体深度图生成 (PyTorch)")
    print("=" * 80)
    print(f"场景目录: {args.dataset_root}")
    print(f"模型类型: {args.model_type}")
    print(f"设备: {args.device}")
    print("深度保存格式: uint16 PNG (毫米单位)")
    print(f"置信度阈值: {args.confidence_threshold}")
    if args.save_visualization:
        print("保存深度图可视化: 是")
    if args.save_disparity:
        print("保存视差图: 是")
    if args.save_calibration:
        print("保存标定参数: 是")
    if args.allow_negative:
        print("允许负视差: 是")
    if args.torch_compile:
        print("使用torch.compile: 是")
    print("=" * 80)

    main(args)
