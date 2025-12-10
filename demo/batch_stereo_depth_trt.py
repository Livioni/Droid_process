#!/usr/bin/env python3
"""
批量处理立体图像对，使用TensorRT engine生成深度图
支持从左右相机外参动态计算每一帧的baseline和doffs
Batch process stereo image pairs using TensorRT engine to generate depth maps
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
import tensorrt as trt
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from s2m2.core.utils.image_utils import read_images
from s2m2.core.utils.image_utils import image_pad, image_crop


def get_args_parser():
    parser = argparse.ArgumentParser(description='批量处理立体图像对生成深度图 (TensorRT)')

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

    # TensorRT模型参数
    parser.add_argument('--model_type', default='XL', type=str,
                        help='模型类型: S, M, L, XL')
    parser.add_argument('--img_width', type=int, default=1280,
                        help='输入图像宽度')
    parser.add_argument('--img_height', type=int, default=736,
                        help='输入图像高度')
    parser.add_argument('--precision', type=str, choices=['fp16', 'tf32', 'fp32'], default='fp32',
                        help='模型精度')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='CUDA设备 (cuda:0, cuda:1, cuda:2, etc.)')

    # 处理参数
    parser.add_argument('--confidence_threshold', type=float, default=0.1,
                        help='置信度阈值，小于此值的深度置0')
    parser.add_argument('--output_dir', type=str, default='output/batch_depth_trt',
                        help='输出文件夹路径')
    parser.add_argument('--save_visualization', action='store_true',
                        help='保存深度图可视化')
    parser.add_argument('--save_disparity', action='store_true',
                        help='保存视差图')
    parser.add_argument('--image_extension', type=str, default='png',
                        help='图像文件扩展名 (png, jpg, etc.)')
    parser.add_argument('--save_calibration', action='store_true',
                        help='保存每帧的标定参数')

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


def preprocess_images(left, right, target_height, target_width):
    """预处理图像"""
    # 调整图像尺寸
    if left.shape[1] >= target_width and left.shape[0] >= target_height:
        left = left[:target_height, :target_width]
        right = right[:target_height, :target_width]
    else:
        left = cv2.resize(left, dsize=(target_width, target_height))
        right = cv2.resize(right, dsize=(target_width, target_height))

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


def torch_gpu_alloc(shape, dtype=torch.float32, device="cuda"):
    """分配GPU内存"""
    t = torch.empty(shape, device=device, dtype=dtype).contiguous()
    return t


def main(args):
    # 设置CUDA设备
    if torch.cuda.is_available():
        # 解析设备ID
        if args.device.startswith('cuda:'):
            device_id = int(args.device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"CUDA设备 {args.device} 不存在，可用设备数量: {torch.cuda.device_count()}")
            torch.cuda.set_device(device_id)
            device = torch.device(args.device)
        else:
            device = torch.device(args.device)
    else:
        print("警告: CUDA不可用，使用CPU")
        device = torch.device('cpu')

    print(f'使用设备: {device}')

    # 设置随机种子
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    print("\n" + "="*60)
    print("加载TensorRT Engine")
    print("="*60)

    # 创建TensorRT runtime和logger
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    runtime = trt.Runtime(TRT_LOGGER)
    print(f"TensorRT版本: {trt.__version__}")

    # 加载engine文件
    trt_path = os.path.join(project_root, f"weights/trt_save/S2M2_{args.model_type}_{args.img_width}_{args.img_height}_{args.precision}.engine")
    print(f"加载Engine: {trt_path}")

    if not os.path.exists(trt_path):
        raise FileNotFoundError(f"Engine文件不存在: {trt_path}")

    with open(trt_path, 'rb') as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    print(f"✓ Engine加载完成: {args.model_type} ({args.precision})")

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
    print(f"输入尺寸: {args.img_height}x{args.img_width}")
    print("="*60 + "\n")

    # 预分配GPU内存用于输出
    out1 = torch_gpu_alloc((1, 1, args.img_height, args.img_width), device=device)
    out2 = torch_gpu_alloc((1, 1, args.img_height, args.img_width), device=device)
    out3 = torch_gpu_alloc((1, 1, args.img_height, args.img_width), device=device)

    # 统计信息
    total_valid_pixels = 0
    total_pixels = 0
    baselines = []
    doffs_list = []

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

            # 预处理图像
            left_processed, right_processed = preprocess_images(left, right, args.img_height, args.img_width)

            # 转换为torch tensor
            input_left = (torch.from_numpy(left_processed).permute(-1, 0, 1).unsqueeze(0)).to(device, dtype=torch.float32).contiguous()
            input_right = (torch.from_numpy(right_processed).permute(-1, 0, 1).unsqueeze(0)).to(device, dtype=torch.float32).contiguous()

            # img_height, img_width = 736, 1280
    
            # 设置bindings
            bindings = [
                input_left.data_ptr(), # input 0
                input_right.data_ptr(), # input 1
                out1.data_ptr(), # output 0
                out2.data_ptr(), # output 1
                out3.data_ptr() # output 2
            ]

            # 执行推理
            context.execute_v2(bindings=bindings)

            # # 转换为numpy并裁剪回原始尺寸
            # pred_disp = image_crop(out1.contiguous().cpu().numpy(), (img_height, img_width)).squeeze().astype(np.float32)
            # pred_occ = image_crop(out2.contiguous().cpu().numpy(), (img_height, img_width)).squeeze().astype(np.float32)
            # pred_conf = image_crop(out3.contiguous().cpu().numpy(), (img_height, img_width)).squeeze().astype(np.float32)
            pred_disp = np.squeeze(out1.contiguous().cpu().numpy()).astype(np.float32)
            pred_occ = np.squeeze(out2.contiguous().cpu().numpy()).astype(np.float32)
            pred_conf = np.squeeze(out3.contiguous().cpu().numpy()).astype(np.float32)

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

            # 保存深度图npy
            base_name = os.path.splitext(filename)[0]
            depth_npy_path = os.path.join(depth_dir, f'{base_name}.npy')
            depth_map = depth_map.astype(np.float32)
            np.save(depth_npy_path, depth_map)

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
    all_calib_params = {
        'baselines': np.array(baselines),
        'doffs': np.array(doffs_list),
        'focal_length': fx_left,
        'cx_left': cx_left,
        'cx_right': cx_right
    }
    np.save(os.path.join(output_dir, 'all_calibration_params.npy'), all_calib_params)

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
    print("批量立体深度图生成 (TensorRT)")
    print("=" * 60)
    print(f"左图像文件夹: {args.left_folder}")
    print(f"右图像文件夹: {args.right_folder}")
    print(f"左相机内参: {args.intrinsic_left}")
    print(f"右相机内参: {args.intrinsic_right}")
    print(f"左相机外参: {args.extrinsic_left}")
    print(f"右相机外参: {args.extrinsic_right}")
    print(f"模型类型: {args.model_type}")
    print(f"输入尺寸: {args.img_height}x{args.img_width}")
    print(f"精度: {args.precision}")
    print(f"CUDA设备: {args.device}")
    print("=" * 60)

    main(args)