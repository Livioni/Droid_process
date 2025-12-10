#!/usr/bin/env python3
"""
深度图加载示例 / Depth Map Loading Example

演示如何加载不同存储格式的深度图
Demonstrates how to load depth maps in different storage formats
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from batch_stereo_depth_pytorch import (
    load_depth_compressed,
    load_depth_sparse,
    load_depth_from_hdf5,
    load_depth_map
)


def load_and_display_depth_example():
    """
    加载和显示深度图的示例
    Example of loading and displaying depth maps
    """

    # 示例参数 - 请根据实际情况修改
    output_dir = "output/batch_depth"
    depth_dir = os.path.join(output_dir, "depth")
    sample_filename = "000000.png"  # 示例文件名

    print("深度图加载示例 / Depth Map Loading Example")
    print("="*50)

    # 方法1: 压缩格式 (Compressed format)
    print("\n1. 加载压缩格式深度图 / Loading compressed depth map:")
    try:
        depth_compressed = load_depth_map(depth_dir, sample_filename, 'compressed')
        print(f"   ✓ 成功加载，形状: {depth_compressed.shape}, 类型: {depth_compressed.dtype}")
        print(f"   深度范围: {depth_compressed[depth_compressed > 0].min():.3f} - {depth_compressed.max():.3f} mm")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")

    # 方法2: 稀疏格式 (Sparse format)
    print("\n2. 加载稀疏格式深度图 / Loading sparse depth map:")
    try:
        depth_sparse = load_depth_map(depth_dir, sample_filename, 'sparse')
        print(f"   ✓ 成功加载，形状: {depth_sparse.shape}, 类型: {depth_sparse.dtype}")
        sparsity = np.sum(depth_sparse == 0) / depth_sparse.size
        print(f"   稀疏度: {sparsity:.1%}")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")

    # 方法3: HDF5批量格式 (HDF5 batch format)
    print("\n3. 从HDF5批量文件加载深度图 / Loading from HDF5 batch file:")
    try:
        depth_hdf5 = load_depth_map(depth_dir, sample_filename, 'batch_hdf5')
        print(f"   ✓ 成功加载，形状: {depth_hdf5.shape}, 类型: {depth_hdf5.dtype}")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")

    # 方法4: 原始NPY格式 (Original NPY format)
    print("\n4. 加载原始NPY格式深度图 / Loading original NPY depth map:")
    try:
        depth_npy = load_depth_map(depth_dir, sample_filename, 'npy')
        print(f"   ✓ 成功加载，形状: {depth_npy.shape}, 类型: {depth_npy.dtype}")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")


def compare_storage_sizes(output_dir="output/batch_depth"):
    """
    比较不同存储格式的文件大小
    Compare file sizes of different storage formats
    """
    depth_dir = os.path.join(output_dir, "depth")

    print("\n存储大小对比 / Storage Size Comparison")
    print("="*50)

    formats = {
        'npy': '*.npy',
        'compressed': '*.npz',
        'sparse': '*_sparse.npz',
        'batch_hdf5': 'depth_maps.h5'
    }

    for fmt, pattern in formats.items():
        try:
            if fmt == 'batch_hdf5':
                filepath = os.path.join(depth_dir, pattern)
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / 1024 / 1024
                    print(f"{fmt:12}: {size_mb:.2f} MB")
            else:
                import glob
                files = glob.glob(os.path.join(depth_dir, pattern))
                if files:
                    total_size = sum(os.path.getsize(f) for f in files)
                    avg_size_kb = total_size / len(files) / 1024
                    total_mb = total_size / 1024 / 1024
                    print(f"{fmt:12}: {avg_size_kb:.1f} KB/文件, 总计 {total_mb:.2f} MB")
        except Exception as e:
            print(f"{fmt:12}: 计算失败 / Calculation failed")


def batch_load_example(output_dir="output/batch_depth"):
    """
    批量加载示例
    Batch loading example
    """
    depth_dir = os.path.join(output_dir, "depth")

    print("\n批量加载示例 / Batch Loading Example")
    print("="*50)

    # 假设有一些文件名列表
    sample_files = ["000000.png", "000001.png", "000002.png"]  # 示例文件列表

    for fmt in ['compressed', 'sparse', 'batch_hdf5']:
        print(f"\n{fmt.upper()} 格式批量加载 / {fmt.upper()} batch loading:")
        loaded_count = 0

        for filename in sample_files:
            try:
                depth = load_depth_map(depth_dir, filename, fmt)
                loaded_count += 1
                print(f"   ✓ {filename}: {depth.shape}")
            except Exception as e:
                print(f"   ✗ {filename}: {e}")

        print(f"   成功加载 {loaded_count}/{len(sample_files)} 个文件")


if __name__ == "__main__":
    # 检查输出目录是否存在
    output_dir = "output/batch_depth"
    if not os.path.exists(output_dir):
        print(f"错误: 输出目录不存在 {output_dir}")
        print("请先运行 batch_stereo_depth_pytorch.py 生成深度图")
        sys.exit(1)

    # 运行示例
    load_and_display_depth_example()
    compare_storage_sizes()
    batch_load_example()

    print("\n" + "="*50)
    print("示例完成! / Example completed!")
    print("="*50)