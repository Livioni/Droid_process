#!/usr/bin/env python3
"""
批量处理立体图像对，使用PyTorch模型生成深度图
支持从左右相机外参动态计算每一帧的baseline和doffs
Batch process stereo image pairs using PyTorch model to generate depth maps
"""

import argparse
import os
import sys
import glob

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
import torch._dynamo
from tqdm import tqdm
import h5py
from scipy import sparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from s2m2.core.utils.model_utils import load_model, run_stereo_matching
from s2m2.core.utils.image_utils import read_images


def get_args_parser():
    parser = argparse.ArgumentParser(description='批量处理立体图像对生成深度图 (PyTorch)')
    
    # 必需参数
    parser.add_argument('--left_folder', type=str, 
                        default='datasets/l-r_images/17368348-stereo_left',
                        help='左相机图像文件夹路径')
    parser.add_argument('--right_folder', type=str, 
                        default='datasets/l-r_images/17368348-stereo_right',
                        help='右相机图像文件夹路径')
    
    # 标定文件参数
    parser.add_argument('--intrinsic_left', type=str, 
                        default='datasets/Sun_Jun_11_15:52:37_2023/intrinsics/17368348_left_intrinsic.npy',
                        help='左相机内参npy文件路径')
    parser.add_argument('--intrinsic_right', type=str, 
                        default='datasets/Sun_Jun_11_15:52:37_2023/intrinsics/17368348_right_intrinsic.npy',
                        help='右相机内参npy文件路径')
    parser.add_argument('--extrinsic_left', type=str, 
                        default='datasets/Sun_Jun_11_15:52:37_2023/extrinsics/wrist_left_extrinsics_3x4.npy',
                        help='左相机外参npy文件路径（每一帧的外参，第一维为帧索引）')
    parser.add_argument('--extrinsic_right', type=str, 
                        default='datasets/Sun_Jun_11_15:52:37_2023/extrinsics/wrist_right_extrinsics_3x4.npy',
                        help='右相机外参npy文件路径（每一帧的外参，第一维为帧索引）')
    
    # 模型参数
    parser.add_argument('--model_type', default='XL', type=str,
                        help='模型类型: S, M, L, XL')
    parser.add_argument('--num_refine', default=3, type=int,
                        help='局部迭代refinement次数')
    parser.add_argument('--torch_compile', action='store_true', 
                        help='使用torch.compile加速')
    parser.add_argument('--allow_negative', action='store_true', 
                        help='允许负视差（用于不完美的rectification）')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='计算设备 (cuda:0, cuda:1, cpu)')
    parser.add_argument('--save_calibration', action='store_true',
                        help='保存每帧的标定参数')
    
    # 处理参数
    parser.add_argument('--confidence_threshold', type=float, default=0,
                        help='置信度阈值，小于此值的深度置0')
    parser.add_argument('--output_dir', type=str, default='output/batch_depth',
                        help='输出文件夹路径')
    parser.add_argument('--save_visualization', action='store_true',
                        help='保存深度图可视化')
    parser.add_argument('--save_disparity', action='store_true',
                        help='保存视差图')
    parser.add_argument('--image_extension', type=str, default='png',
                        help='图像文件扩展名 (png, jpg, etc.)')

    # 深度图存储选项
    parser.add_argument('--depth_storage', type=str, default='npy',
                        choices=['npy', 'compressed', 'sparse', 'batch_hdf5'],
                        help='深度图存储方法: npy(原始), compressed(压缩), sparse(稀疏), batch_hdf5(批量HDF5)')
    parser.add_argument('--depth_dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'float64'],
                        help='深度图数据类型，float16最节省空间')

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


# =============================================================================
# 优化的深度图存储方法 / Optimized depth map storage methods
# =============================================================================

def save_depth_compressed(depth_map, output_path, dtype=np.float16):
    """
    保存压缩的深度图，使用更小的数据类型
    Save compressed depth map with smaller data type

    Args:
        depth_map: 深度图 [H, W]
        output_path: 输出路径
        dtype: 数据类型，默认float16节省空间
    """
    # 转换为更小的数据类型节省空间
    depth_compressed = depth_map.astype(dtype)
    np.savez_compressed(output_path, depth=depth_compressed, dtype=str(dtype))


def load_depth_compressed(input_path):
    """
    加载压缩的深度图
    Load compressed depth map

    Args:
        input_path: 输入路径

    Returns:
        depth_map: 深度图 [H, W]
    """
    data = np.load(input_path)
    depth_map = data['depth']

    # 如果保存时使用了float16，这里转换回float32用于计算
    if depth_map.dtype == np.float16:
        depth_map = depth_map.astype(np.float32)

    return depth_map


def save_depth_batch_hdf5(depth_maps, filenames, output_path):
    """
    将多个深度图批量保存到HDF5文件中
    Save multiple depth maps to HDF5 file in batch

    Args:
        depth_maps: 深度图列表
        filenames: 对应的文件名列表
        output_path: 输出HDF5文件路径
    """
    with h5py.File(output_path, 'w') as f:
        for depth_map, filename in zip(depth_maps, filenames):
            # 使用文件名作为数据集名称（去掉扩展名）
            base_name = os.path.splitext(filename)[0]
            # 保存为float16节省空间
            f.create_dataset(base_name, data=depth_map.astype(np.float16),
                           compression="gzip", compression_opts=6)


def load_depth_from_hdf5(hdf5_path, filename):
    """
    从HDF5文件中加载单个深度图
    Load single depth map from HDF5 file

    Args:
        hdf5_path: HDF5文件路径
        filename: 深度图文件名

    Returns:
        depth_map: 深度图 [H, W]
    """
    with h5py.File(hdf5_path, 'r') as f:
        base_name = os.path.splitext(filename)[0]
        depth_data = f[base_name][:]
        # 转换回float32用于计算
        return depth_data.astype(np.float32)


def save_depth_sparse(depth_map, output_path):
    """
    以稀疏格式保存深度图（适用于大量零值的情况）
    Save depth map in sparse format (suitable when many zeros)

    Args:
        depth_map: 深度图 [H, W]
        output_path: 输出路径
    """
    # 将深度图展平并找到非零元素
    flat_depth = depth_map.flatten()
    nonzero_indices = np.nonzero(flat_depth)[0]
    nonzero_values = flat_depth[nonzero_indices]

    # 保存稀疏数据
    sparse_data = {
        'shape': depth_map.shape,
        'indices': nonzero_indices,
        'values': nonzero_values.astype(np.float16),  # 节省空间
        'sparsity': len(nonzero_indices) / len(flat_depth)
    }

    np.savez_compressed(output_path, **sparse_data)


def load_depth_sparse(input_path):
    """
    加载稀疏格式的深度图
    Load depth map from sparse format

    Args:
        input_path: 输入路径

    Returns:
        depth_map: 深度图 [H, W]
    """
    data = np.load(input_path)

    # 重建深度图
    depth_map = np.zeros(data['shape'], dtype=np.float32)
    depth_map.flat[data['indices']] = data['values'].astype(np.float32)

    return depth_map


def load_depth_map(depth_dir, filename, storage_method='compressed'):
    """
    统一的深度图加载函数
    Unified depth map loading function

    Args:
        depth_dir: 深度图文件夹路径
        filename: 图像文件名
        storage_method: 存储方法 ('npy', 'compressed', 'sparse', 'batch_hdf5')

    Returns:
        depth_map: 加载的深度图
    """
    base_name = os.path.splitext(filename)[0]

    if storage_method == 'npy':
        depth_path = os.path.join(depth_dir, f'{base_name}.npy')
        return np.load(depth_path)

    elif storage_method == 'compressed':
        depth_path = os.path.join(depth_dir, f'{base_name}.npz')
        return load_depth_compressed(depth_path)

    elif storage_method == 'sparse':
        depth_path = os.path.join(depth_dir, f'{base_name}_sparse.npz')
        return load_depth_sparse(depth_path)

    elif storage_method == 'batch_hdf5':
        hdf5_path = os.path.join(depth_dir, 'depth_maps.h5')
        return load_depth_from_hdf5(hdf5_path, filename)

    else:
        raise ValueError(f"不支持的存储方法: {storage_method}")


def get_optimal_storage_method(depth_map, threshold=0.3):
    """
    根据深度图特征推荐最优存储方法
    Recommend optimal storage method based on depth map characteristics

    Args:
        depth_map: 深度图
        threshold: 稀疏度阈值，超过此值推荐稀疏存储

    Returns:
        method: 推荐的存储方法 ('compressed', 'sparse', 'batch_hdf5')
    """
    # 计算稀疏度（零值比例）
    sparsity = np.sum(depth_map == 0) / depth_map.size

    if sparsity > threshold:
        return 'sparse'
    else:
        return 'compressed'


def main(args):
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
    
    print("\n" + "="*60)
    print("加载标定参数")
    print("="*60)
    
    # 加载左相机内参
    print(f"\n加载左相机内参: {args.intrinsic_left}")
    left_K, fx_left, fy_left, cx_left, cy_left = load_camera_matrix(args.intrinsic_left)
    print(f"  fx: {fx_left:.3f}, fy: {fy_left:.3f}")
    print(f"  cx: {cx_left:.3f}, cy: {cy_left:.3f}")
    
    # 加载右相机内参
    print(f"\n加载右相机内参: {args.intrinsic_right}")
    right_K, fx_right, fy_right, cx_right, cy_right = load_camera_matrix(args.intrinsic_right)
    print(f"  fx: {fx_right:.3f}, fy: {fy_right:.3f}")
    print(f"  cx: {cx_right:.3f}, cy: {cy_right:.3f}")
    
    # 加载外参数组
    print(f"\n加载左相机外参数组: {args.extrinsic_left}")
    extrinsics_left = load_extrinsics_array(args.extrinsic_left)
    
    print(f"\n加载右相机外参数组: {args.extrinsic_right}")
    extrinsics_right = load_extrinsics_array(args.extrinsic_right)
    
    # 验证外参数组长度
    if extrinsics_left.shape[0] != extrinsics_right.shape[0]:
        raise ValueError(f"左右外参数组长度不匹配: {extrinsics_left.shape[0]} vs {extrinsics_right.shape[0]}")
    
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
    print("\n" + "="*60)
    print("扫描图像文件")
    print("="*60)
    image_pairs = get_image_pairs(args.left_folder, args.right_folder, args.image_extension)
    
    if len(image_pairs) == 0:
        print("错误: 未找到匹配的图像对")
        return
    
    # 验证图像数量和外参数量
    if len(image_pairs) > num_extrinsics:
        print(f"警告: 图像对数量({len(image_pairs)})超过外参数量({num_extrinsics})")
        print(f"将只处理前 {num_extrinsics} 对图像")
        image_pairs = image_pairs[:num_extrinsics]
    
    # 创建输出目录
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    depth_dir = os.path.join(output_dir, 'depth_npy')
    os.makedirs(depth_dir, exist_ok=True)
    
    if args.save_visualization:
        vis_dir = os.path.join(output_dir, 'depth_vis')
        os.makedirs(vis_dir, exist_ok=True)
    
    if args.save_disparity:
        disp_dir = os.path.join(output_dir, 'disparity_npy')
        disp_vis_dir = os.path.join(output_dir, 'disparity_vis')
        os.makedirs(disp_dir, exist_ok=True)
        os.makedirs(disp_vis_dir, exist_ok=True)
    
    # 保存每帧的baseline和doffs
    if args.save_calibration:
        calib_dir = os.path.join(output_dir, 'calibration_per_frame')
        os.makedirs(calib_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"开始处理 {len(image_pairs)} 对图像")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {args.confidence_threshold}")
    print(f"焦距: {fx_left:.3f} pixels")
    print("="*60 + "\n")
    
    # 统计信息
    total_valid_pixels = 0
    total_pixels = 0
    baselines = []
    doffs_list = []
    
    # 初始化批量存储容器（用于batch_hdf5模式）
    if args.depth_storage == 'batch_hdf5':
        batch_depth_maps = []
        batch_filenames = []

    # 处理每对图像
    for frame_idx, (left_path, right_path, filename) in enumerate(tqdm(image_pairs, desc="处理图像")):
        try:
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
            depth_map /= 1000
            
            # 统计
            valid_pixels = np.sum(depth_map > 0)
            total_valid_pixels += valid_pixels
            total_pixels += depth_map.size
            
            # 保存深度图 - 使用优化的存储方法
            base_name = os.path.splitext(filename)[0]

            if args.depth_storage == 'batch_hdf5':
                # 批量HDF5模式：收集数据，稍后批量保存
                batch_depth_maps.append(depth_map)
                batch_filenames.append(filename)
            else:
                # 其他存储模式：立即保存
                dtype = getattr(np, args.depth_dtype)

                if args.depth_storage == 'npy':
                    # 原始npy格式（保持兼容性）
                    depth_npy_path = os.path.join(depth_dir, f'{base_name}.npy')
                    np.save(depth_npy_path, depth_map.astype(dtype))

                elif args.depth_storage == 'compressed':
                    # 压缩格式
                    depth_npy_path = os.path.join(depth_dir, f'{base_name}.npz')
                    save_depth_compressed(depth_map, depth_npy_path, dtype)

                elif args.depth_storage == 'sparse':
                    # 稀疏格式
                    depth_npy_path = os.path.join(depth_dir, f'{base_name}_sparse.npz')
                    save_depth_sparse(depth_map, depth_npy_path)
            
            # 保存深度图可视化
            if args.save_visualization:
                depth_vis_path = os.path.join(vis_dir, f'{base_name}_depth.png')
                save_depth_visualization(depth_map, depth_vis_path)
            
            # 保存视差图
            if args.save_disparity:
                disp_npy_path = os.path.join(disp_dir, f'{base_name}_disp.npy')
                np.save(disp_npy_path, pred_disp)
                
                disp_vis_path = os.path.join(disp_vis_dir, f'{base_name}_disp.png')
                save_disparity_visualization(pred_disp, disp_vis_path)
            
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
                calib_path = os.path.join(calib_dir, f'{base_name}_calib.npy')
                np.save(calib_path, calib_params)

            
        except Exception as e:
            print(f"\n处理 {filename} (frame {frame_idx}) 时出错: {str(e)}")
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
        np.save(os.path.join(output_dir, 'all_calibration_params.npy'), all_calib_params)

    # 保存批量HDF5文件
    if args.depth_storage == 'batch_hdf5' and batch_depth_maps:
        hdf5_path = os.path.join(output_dir, 'depth_maps.h5')
        print(f"\n保存批量HDF5文件: {hdf5_path}")
        save_depth_batch_hdf5(batch_depth_maps, batch_filenames, hdf5_path)
        print(f"✓ 批量保存了 {len(batch_depth_maps)} 个深度图到HDF5文件")

    # 输出统计信息
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
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
    print(f"\n结果保存在: {output_dir}")
    print(f"深度图存储格式: {args.depth_storage} ({args.depth_dtype})")

    # 显示存储效率统计
    if args.depth_storage != 'batch_hdf5':
        try:
            # 计算单个文件的平均大小
            depth_files = []
            if args.depth_storage == 'npy':
                depth_files = glob.glob(os.path.join(depth_dir, "*.npy"))
            elif args.depth_storage == 'compressed':
                depth_files = glob.glob(os.path.join(depth_dir, "*.npz"))
            elif args.depth_storage == 'sparse':
                depth_files = glob.glob(os.path.join(depth_dir, "*_sparse.npz"))

            if depth_files:
                total_size = sum(os.path.getsize(f) for f in depth_files)
                avg_size = total_size / len(depth_files)
                print(f"平均文件大小: {avg_size/1024:.1f} KB")
                print(f"总存储大小: {total_size/1024/1024:.1f} MB")

                # 估算原始float32 npy的大小作为对比
                if args.depth_storage != 'npy':
                    estimated_original = avg_size * (32 / 16) if args.depth_dtype == 'float16' else avg_size
                    savings = (1 - avg_size/estimated_original) * 100
                    print(f"相比float32节省: {savings:.1f}%")
        except:
            pass

    print("="*60)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.left_folder):
        print(f"错误: 左图像文件夹不存在: {args.left_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.right_folder):
        print(f"错误: 右图像文件夹不存在: {args.right_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.intrinsic_left):
        print(f"错误: 左相机内参文件不存在: {args.intrinsic_left}")
        sys.exit(1)
    
    if not os.path.exists(args.intrinsic_right):
        print(f"错误: 右相机内参文件不存在: {args.intrinsic_right}")
        sys.exit(1)
    
    if not os.path.exists(args.extrinsic_left):
        print(f"错误: 左相机外参文件不存在: {args.extrinsic_left}")
        sys.exit(1)
    
    if not os.path.exists(args.extrinsic_right):
        print(f"错误: 右相机外参文件不存在: {args.extrinsic_right}")
        sys.exit(1)
    
    print("=" * 60)
    print("批量立体深度图生成 (PyTorch)")
    print("=" * 60)
    print(f"左图像文件夹: {args.left_folder}")
    print(f"右图像文件夹: {args.right_folder}")
    print(f"左相机内参: {args.intrinsic_left}")
    print(f"右相机内参: {args.intrinsic_right}")
    print(f"左相机外参: {args.extrinsic_left}")
    print(f"右相机外参: {args.extrinsic_right}")
    print(f"模型类型: {args.model_type}")
    print("=" * 60)
    
    main(args)
