# Droid 数据处理



<div align="center">
  <img src="assets/overview.jpg" alt="overview" width="800"/>
</div>


## 环境准备

```bash
git clone https://github.com/Livioni/s2m2.git
conda env create -n droid -f environment.yml
conda activate droid
pip install -e .

# 安装pyzed follow the instructions in https://github.com/stereolabs/zed-python-api
# to verify pyzed
python -c "import pyzed.sl as sl"

# 安装Open3D用于相机对齐（ICP算法）
pip install open3d
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


<div align="center">
  <img src="assets/droid_gif.png" alt="gif" width="800"/>
</div>

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
  --cameras 17368348 \
  --model_type XL \
  --depth_storage npy \
  --depth_dtype float32 \
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

## 对齐左右相机

对齐左右两个相机。

`align_left_right_camera.py` 使用ICP（Iterative Closest Point）算法对齐两个静态第三人称相机。

**核心思想：**

由于第三人称相机是静态的，理论上只需要对齐一帧即可。但为了提高鲁棒性，该脚本使用前N帧（默认10帧）分别进行ICP对齐，然后对得到的变换矩阵取平均，最后将平均变换应用到所有帧。

**算法流程：**

1. **多帧对齐：**
   - 对前N帧（默认10帧）分别进行ICP对齐
   - 从两个相机的深度图创建3D点云
   - 将点云转换到世界坐标系
   - 使用点到面ICP算法对齐camera 2到camera 1
   - 记录每帧的变换矩阵、fitness和RMSE

2. **变换平均：**
   - 过滤掉fitness较低的对齐结果（< 0.5）
   - 对旋转矩阵和平移向量分别取平均
   - 使用SVD正交化平均后的旋转矩阵

3. **应用到所有帧：**
   - 将平均变换矩阵应用到camera 2的所有帧外参
   - 保持原有的帧数不变，输出形状为 [N, 3, 4]

**使用方法：**

```bash
# 基本用法
python align_left_right_camera.py \
  --cam1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp \
  --num-frames 10

# 自定义参数
python align_left_right_camera.py \
  --cam1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp \
  --num-frames 10 \
  --max-iterations 50 \
  --distance-threshold 0.05 \
  --voxel-size 0.01 \
  --visualize  # 可选：保存点云PLY文件用于可视化
```

**参数说明：**
- `--cam1`: 参考相机目录（camera 1，作为对齐基准）
- `--cam2`: 待对齐相机目录（camera 2）
- `--output-dir`: 输出目录，对齐后的外参保存路径
- `--num-frames`: 用于对齐的帧数（默认10）
- `--max-iterations`: 每帧ICP最大迭代次数（默认50）
- `--distance-threshold`: ICP对应点距离阈值，单位米（默认0.05）
- `--voxel-size`: 点云降采样体素大小，单位米（默认0.01）
- `--visualize`: 是否保存点云PLY文件用于可视化

**输出文件：**
- `{camera_id}_left.npy`: 对齐后的外参文件，形状为 [N, 3, 4]，N为原始帧数
- `alignment_stats.png`: 对齐统计图表（fitness和RMSE）
- `visualizations_alignment/`: 点云PLY文件（如果启用 --visualize）

对齐后的外参文件可以替换原始外参用于后续处理。


<div align="center">
  <img src="assets/projection_visualization.png" alt="projection" width="800"/>
</div>

## 优化wrist camera pose

使用两个已对齐的第三人称相机的组合点云来优化腕部相机的外参。`align_wrist_camera.py` 使用ICP算法将腕部相机与第三人称相机对齐。

**核心思想：**

利用两个第三人称相机提供更完整的场景覆盖，通过组合它们的点云来获得更鲁棒的对齐结果。

**算法流程：**

1. **创建组合点云：**
   - 从两个第三人称相机的深度图创建3D点云
   - 将两个点云转换到世界坐标系
   - 合并两个点云形成完整的场景点云

2. **投影到腕部相机：**
   - 使用当前腕部相机外参将组合点云投影到腕部相机坐标系
   - 过滤投影点，只保留腕部相机可见的最近点（处理遮挡）
   - 对于每个像素，只保留最接近腕部相机深度图的点

3. **ICP对齐：**
   - 源点云：过滤后的投影点云（来自第三人称相机）
   - 目标点云：腕部相机自身的深度点云
   - 使用点到面ICP算法进行对齐
   - 更新腕部相机外参

4. **逐帧优化：**
   - 对每一帧独立进行ICP对齐
   - 输出形状保持为 [N, 3, 4]，N为总帧数

**使用方法：**

```bash
# 基本用法（使用原始外参）
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam-ext2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined_icp

# 使用已对齐的第二个第三人称相机外参（推荐）
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam-ext2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
  --use-aligned-ext2 \
  --aligned-ext2-path datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp/27904255.npy \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined_icp

# 自定义参数
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam-ext2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
  --use-aligned-ext2 \
  --aligned-ext2-path datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp/27904255.npy \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined_icp \
  --max-iterations 50 \
  --distance-threshold 0.05 \
  --voxel-size 0.001 \
  --start-frame 0 \
  --end-frame 100 \
  --visualize  # 可选：保存点云PLY文件用于可视化
```

**参数说明：**
- `--cam-ext1`: 第一个第三人称相机目录（参考相机）
- `--cam-ext2`: 第二个第三人称相机目录
- `--cam-wrist`: 腕部相机目录
- `--use-aligned-ext2`: 是否使用已对齐的cam-ext2外参
- `--aligned-ext2-path`: 已对齐的cam-ext2外参文件路径
- `--output-dir`: 输出目录，对齐后的腕部相机外参保存路径
- `--max-iterations`: 每帧ICP最大迭代次数（默认50）
- `--distance-threshold`: ICP对应点距离阈值，单位米（默认0.05）
- `--voxel-size`: 点云降采样体素大小，单位米（默认0.001）
- `--start-frame`: 起始帧索引
- `--end-frame`: 结束帧索引
- `--visualize`: 是否保存点云PLY文件用于可视化

**输出文件：**
- `{wrist_camera_id}.npy`: 对齐后的腕部相机外参文件，形状为 [N, 3, 4]
- `wrist_alignment_stats.png`: 对齐统计图表（fitness和RMSE）
- `visualizations_wrist_alignment/`: 点云PLY文件（如果启用 --visualize）

**推荐工作流程：**

1. 首先使用 `align_left_right_camera.py` 对齐两个第三人称相机
2. 然后使用 `align_wrist_camera.py` 并指定 `--use-aligned-ext2` 来对齐腕部相机
3. 这样可以确保所有相机在同一个坐标系统中

对齐后的外参文件可以替换原始外参用于后续处理。

---

## 完整的相机对齐工作流程

### 步骤1：对齐两个第三人称相机

```bash
python align_left_right_camera.py \
  --cam1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp \
  --num-frames 10
```

**输出：** `datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp/27904255.npy`

### 步骤2：使用对齐后的第三人称相机对齐腕部相机

```bash
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam-ext2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
  --use-aligned-ext2 \
  --aligned-ext2-path datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp/27904255.npy \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined_icp
```

**输出：** `datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined_icp/17368348.npy`

### 步骤3：验证对齐结果

使用可视化工具检查对齐效果：

```bash
# 可视化单帧
python visualize_pointclouds.py --frame 100

# 可视化时序帧
python visualize_temporal_pointclouds.py
```

### 文件结构

对齐完成后的目录结构：

```
datasets/samples/Sun_Jun_11_15:52:37_2023/
├── 23897859/                          # 参考第三人称相机（不变）
│   ├── extrinsics/
│   │   └── 23897859_left.npy         # 原始外参
│   └── ...
├── 27904255/                          # 第二个第三人称相机
│   ├── extrinsics/
│   │   └── 27904255_left.npy         # 原始外参
│   ├── extrinsics_refined_icp/
│   │   ├── 27904255.npy              # 对齐后的外参 ✓
│   │   └── alignment_stats.png
│   └── ...
└── 17368348/                          # 腕部相机
    ├── extrinsics/
    │   └── 17368348_left.npy         # 原始外参
    ├── extrinsics_refined_icp/
    │   ├── 17368348.npy              # 对齐后的外参 ✓
    │   └── wrist_alignment_stats.png
    └── ...
```

### 关键点

1. **第三人称相机对齐**：使用前10帧的平均变换，因为这些相机是静态的
2. **腕部相机对齐**：逐帧对齐，因为腕部相机是移动的
3. **组合点云**：使用两个第三人称相机的组合点云提供更完整的场景覆盖
4. **遮挡处理**：自动过滤被遮挡的点，只保留腕部相机可见的最近点

