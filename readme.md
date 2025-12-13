# Droid 数据处理
## 环境准备

```bash
git clone https://github.com/Livioni/s2m2.git
conda env create -n droid -f environment.yml
conda activate droid
pip install -e .

# 安装pyzed follow the instructions in https://github.com/stereolabs/zed-python-api
# to verify pyzed
python -c "import pyzed.sl as sl"
```

Download S2M2 Pre-trained Models
Create a directory for weights and download the desired models from the links below.

```bash
mkdir weights
mkdir weights/pretrain_weights
```


| Model | Download | Model Size |
| :---: | :---: | :--: |
| **S** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH128NTR1.pth) | 26.5M | 
| **M** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH192NTR2.pth) | 80.4M | 
| **L** | [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH256NTR3.pth) | 181M | 
| **XL**| [Download](https://huggingface.co/minimok/s2m2/resolve/main/CH384NTR3.pth) | 406M | 


## 提取元数据

<div align="center">
  <img src="assets/droid_setup.png" alt="setup" width="800"/>
</div>


droid的每个场景的目录：

```bash
datasets/Sun_Jun_11_15:52:37_2023
├── metadata_ILIAD+j807b3f8+2023-06-11-15h-52m-37s.json
├── recordings
│   ├── MP4
│   │   ├── 17368348.mp4
│   │   ├── 17368348-stereo.mp4 #双目
│   │   ├── 23897859.mp4
│   │   ├── 23897859-stereo.mp4
│   │   ├── 27904255.mp4
│   │   └── 27904255-stereo.mp4
│   └── SVO
│       ├── 17368348.svo #内参
│       ├── 23897859.svo
│       └── 27904255.svo
├── trajectory.h5 #外参
└── trajectory_im128.h5
```

```bash
#提取所需数据
python process_camera_data.py --input_dir datasets/Sun_Jun_11_15:52:37_2023 --output_dir datasets/samples

#提取完成
tree -d datasets/samples

datasets/samples
└── Sun_Jun_11_15:52:37_2023
    ├── 17368348
    │   ├── extrinsics
    │   ├── images
    │   │   ├── left
    │   │   └── right
    │   └── intrinsics
    ├── 23897859
    │   ├── extrinsics
    │   ├── images
    │   │   ├── left
    │   │   └── right
    │   └── intrinsics
    └── 27904255
        ├── extrinsics
        ├── images
        │   ├── left
        │   └── right
        └── intrinsics
```

**相机类型标识文件：**
- `wrist_camera.txt` - 腕部相机
- `ext1.txt` - 外部相机1
- `ext2.txt` - 外部相机2


## 使用S2M2模型获得深度图


### 批量处理所有相机

使用 `process_camera_data.py` 提取数据后，可以使用修改后的批量脚本自动处理所有相机：

```bash
# 处理所有相机
python demo/batch_stereo_depth_pytorch.py \
  --dataset_root datasets/samples/Sun_Jun_11_15:52:37_2023 \
  --model_type XL \
  --confidence_threshold 0.2 \
  --depth_storage compressed \
  --depth_dtype float16 \
  --save_visualization

#confidence_threshold 这里得看看，0-1，越高保留的点越少

# 或者只处理指定相机
python demo/batch_stereo_depth_pytorch.py \
  --dataset_root datasets/samples/Sun_Jun_11_15:52:37_2023 \
  --cameras 17368348 23897859 \
  --model_type L \
  --device cuda:0
```

处理完成后，每个相机文件夹下会自动创建 `depths/` 子文件夹，包含：
```
datasets/samples/{camera_id}/depths/
├── depth_npy/          # 深度图文件
├── depth_vis/          # 深度图可视化（如果启用）
├── disparity_npy/      # 视差图（如果启用）
├── disparity_vis/      # 视差图可视化（如果启用）
└── calibration_per_frame/  # 每帧标定参数（如果启用）
```

### 参数说明

**多相机模式参数：**
- `--dataset_root`: 数据集根目录（默认: `datasets/samples`）
- `--cameras`: 指定要处理的相机ID列表（可选，不指定则处理所有相机）

**模型和处理参数：**
- `--model_type`: 模型类型 S/M/L/XL（默认: XL）
- `--device`: 计算设备 cuda:0/cuda:1/cpu（默认: cuda:0）
- `--depth_storage`: 深度图存储格式 npy/compressed/sparse/batch_hdf5（默认: compressed）
- `--depth_dtype`: 深度图数据类型 float16/float32/float64（默认: float16）
- `--confidence_threshold`: 置信度阈值（默认: 0.1）
- `--save_visualization`: 保存深度图可视化
- `--save_disparity`: 保存视差图
- `--save_calibration`: 保存每帧标定参数

**性能优化参数：**
- `--torch_compile`: 使用torch.compile加速
- `--allow_negative`: 允许负视差（用于不完美的rectification）


## 初步可视化

### 同一时刻可视化
使用visualize_pointclouds.py初步可视化该场景是否有问题。

```bash
#修改    
'''
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
'''

python visualize_pointclouds.py --frame 100 #可视化第100帧时刻三个相机的点云
```

### 时序可视化

可视化腕部相机时序上 ±5帧的点云图看下是否对齐。

```bash
python visualize_temporal_pointclouds.py
```



## 优化wrist camera pose (to be fixed)

- 以第三人称相机（参考相机）为基准，利用其深度图创建3D点云
- 将这些3D点通过腕部相机的当前外参投影到腕部相机坐标系
- 比较投影得到的深度图与腕部相机实际深度图的差异
- 使用梯度下降优化腕部相机的外参，使投影深度与实际深度尽可能接近


<div align="center">
  <img src="assets/projection_visualization.png" alt="projection" width="800"/>
</div>

**核心思想：**


`optimize_camera_extrinsics.py` 实现了基于深度图对齐的相机外参优化算法：

**算法流程：**
1. **数据准备：**
   - 加载第三人称相机和腕部相机的内参矩阵
   - 读取腕部相机的外参作为优化初始值

2. **逐帧优化：**
   - 从第三人称相机的深度图创建3D点云
   - 使用当前腕部相机外参将点云投影到腕部相机视图
   - 计算投影深度与实际深度的L1损失
   - 通过Adam优化器更新外参参数

3. **损失函数：**
   ```
   Loss = mean(|depth_projected - depth_ground_truth|)
   ```
   其中只计算两个深度图都有效的像素点


**使用方法：**
```bash
python optimize_camera_extrinsics.py \
  --cam1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam2 datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined \
  --max-iterations 50 \
  --lr 1e-4 \
  --visualize  # 可选：启用可视化
```

优化后的外参文件将保存为 `17368348/extrinsics_refined/17368348_left.npy`，可替换原始外参用于后续处理。