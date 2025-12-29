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

双目深度模型

- [S2M2](https://github.com/junhong-3dv/s2m2) SOTA, Wrist camera OOD
- [FoundationStereos](https://github.com/NVlabs/FoundationStereo) 比S2M2差
- [IGEV++](https://github.com/gangweiX/IGEV-plusplus) 垃圾
- [GGEV](https://github.com/JiaxinLiu-A/GGEV) 没开源


深度补全模型

- [PriorDepthAnything](https://github.com/SpatialVision/Prior-Depth-Anything) 效果差，OOD。
- [Camera Depth Model](https://github.com/ByteDance-Seed/manip-as-in-sim-suite) 效果差
- [Prompt Depth Anything](https://github.com/DepthAnything/PromptDA) 效果差


### 批量处理所有相机

使用 `process_camera_data.py` 提取数据后，可以使用修改后的批量脚本自动处理所有相机：

```bash
# 处理所有相机
python demo/batch_stereo_depth_pytorch.py \
  --dataset_root datasets/samples/Fri_Jul__7_09:42:23_2023 \
  --model_type XL \
  --confidence_threshold 0.9 \
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

## Camera Depth Model Refine

依赖：确保已安装 `torchvision`（CDM 推理会用到）。

```bash
python cdm_infer.py \
    --rgb-image cdm/example_data/color_12.png \
    --depth-image cdm/example_data/depth_12.png \  # 也支持 .npz/.npy（npz 默认 key=depth）
    --output visualization.png
```

批量处理一个场景目录（会自动遍历其下所有相机目录；输入 `images/left/*.png` + `depth_npy/*.npz`，输出到各相机的 `depth_refined/*.npy`）：

```bash
python cdm_infer.py \
  --root-dir high_conf/Sun_Jun_11_15:52:37_2023
```

## 初步可视化

### 同一时刻可视化
使用visualize_pointclouds.py初步可视化该场景是否有问题。

```bash
#修改默认相机列表（根据你的数据目录调整）
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

0. **使用MapAnything 初始化左右相机姿态：**
  有些场景左右相机标定太不准了，用`demo/mapanything_multimodal_extrinsics.py` 初始化相机相对姿态。

  ```bash
  python demo/mapanything_multimodal_extrinsics.py \
    --ref_cam datasets/samples/Fri_Jul__7_09:42:23_2023/22008760 \
    --tgt_cam datasets/samples/Fri_Jul__7_09:42:23_2023/24400334 \
    --frame 0 \
    --output_dir datasets/samples/Fri_Jul__7_09:42:23_2023/24400334/extrinsics_refined \
    --output_name 24400334_ma.npy
  ```

  该脚本使用MapAnything多模态推理来估计两个视图之间的相对相机姿态，特别适用于左右相机标定不准的场景。

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
# 基本用法（推荐使用robust方法）
python align_left_right_camera.py \
  --cam1 datasets/samples/Fri_Jul__7_09:42:23_2023/22008760 \
  --cam2 datasets/samples/Fri_Jul__7_09:42:23_2023/24400334 \
  --output-dir datasets/samples/Fri_Jul__7_09:42:23_2023/24400334/extrinsics_refined \
  --num-frames 10

# 高级用法：使用多尺度ICP
python align_left_right_camera.py \
  --cam1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/27904255/extrinsics_refined_icp \
  --num-frames 10 \
  --method multiscale \
  --max-iterations 100 \
  --distance-threshold 0.05 \
  --voxel-size 0.001 \
  --max-depth 1.0 \
  --visualize

# RANSAC方法
python align_left_right_camera.py \
  --cam1 datasets/samples/.../camera1 \
  --cam2 datasets/samples/.../camera2 \
  --method ransac \
  --output-dir output/ \
  --num-frames 15
```

**参数说明：**
- `--cam1`: 参考相机目录（camera 1，作为对齐基准）
- `--cam2`: 待对齐相机目录（camera 2）
- `--output-dir`: 输出目录，对齐后的外参保存路径
- `--num-frames`: 用于对齐的帧数（默认10）
- `--method`: 对齐方法选择
  - `original`: 基本ICP对齐
  - `robust`: 质量过滤和加权平均（默认，推荐）
  - `multiscale`: 多尺度ICP（最鲁棒，推荐用于复杂场景）
  - `ransac`: 基于特征的RANSAC-ICP（适用于有明显几何特征的场景）
- `--max-iterations`: 每帧ICP最大迭代次数（默认100）
- `--distance-threshold`: ICP对应点距离阈值，单位米（默认0.05）
- `--voxel-size`: 点云降采样体素大小，单位米（默认0.001）
- `--max-depth`: 点云深度过滤阈值，单位米（默认1.0）
- `--visualize`: 是否保存点云PLY文件用于可视化
- `--min-fitness`: 最小fitness阈值（robust/multiscale方法，默认0.4）
- `--max-rmse`: 最大RMSE阈值（robust/multiscale方法，默认0.1）

**输出文件：**
- `{camera_id}.npy`: 对齐后的外参文件，形状为 [N, 3, 4]，N为原始帧数
- `alignment_stats.png`: 对齐统计图表（fitness和RMSE）
- `visualizations_alignment/`: 点云PLY文件（如果启用 --visualize）

对齐后的外参文件可以替换原始外参用于后续处理。


<div align="center">
  <img src="assets/projection_visualization.png" alt="projection" width="800"/>
</div>

## 优化wrist camera pose

使用一个或两个第三人称相机的点云来优化腕部相机的外参。`align_wrist_camera.py` 使用先进的ICP算法将腕部相机与第三人称相机对齐，支持多尺度ICP和智能点过滤。

**核心思想：**

通过将第三人称相机提供的场景点云投影到腕部相机坐标系，并与腕部相机自身的深度点云进行ICP对齐，实现精确的姿态优化。支持单相机和双相机模式，双相机模式提供更完整的场景覆盖。

**算法流程：**

1. **创建场景点云：**
   - 从一个或两个第三人称相机的深度图创建3D点云
   - 将点云转换到世界坐标系并合并
   - 支持自动加载已对齐的相机外参

2. **智能投影和过滤：**
   - 使用当前腕部相机外参将场景点云投影到腕部相机坐标系
   - 深度过滤：只保留腕部相机可见的最近点（处理遮挡）
   - 像素级过滤：对于每个像素只保留最接近GT深度的点

3. **多尺度ICP对齐：**
   - 源点云：过滤后的投影点云（来自第三人称相机）
   - 目标点云：腕部相机自身的深度点云
   - 粗到精的多尺度策略（默认启用）
   - 点到面ICP算法确保收敛精度

4. **逐帧优化：**
   - 对每一帧独立进行ICP对齐
   - 支持指定帧范围处理
   - 输出形状保持为 [N, 3, 4]，N为总帧数

**使用方法：**

```bash
# 双相机模式（推荐，场景覆盖更完整）
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/Fri_Jul__7_09:42:23_2023/22008760 \
  --cam-ext2 datasets/samples/Fri_Jul__7_09:42:23_2023/24400334 \
  --cam-wrist datasets/samples/Fri_Jul__7_09:42:23_2023/18026681 \
  --output-dir datasets/samples/Fri_Jul__7_09:42:23_2023/18026681/extrinsics_refined

# 单相机模式（适用于只有一个第三人称相机的情况）
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/Fri_Jul__7_09:42:23_2023/22008760 \
  --cam-wrist datasets/samples/Fri_Jul__7_09:42:23_2023/18026681 \
  --output-dir datasets/samples/Fri_Jul__7_09:42:23_2023/18026681/extrinsics_refined

# 高级用法：自定义参数和帧范围
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/Sun_Jun_11_15:52:37_2023/23897859 \
  --cam-ext2 datasets/samples/Sun_Jun_11_15:52:37_2023/27904255 \
  --cam-wrist datasets/samples/Sun_Jun_11_15:52:37_2023/17368348 \
  --output-dir datasets/samples/Sun_Jun_11_15:52:37_2023/17368348/extrinsics_refined \
  --max-iterations 80 \
  --distance-threshold 0.05 \
  --voxel-size 0.001 \
  --icp-levels 3 \
  --voxel-factor 2.0 \
  --max-depth 1.0 \
  --start-frame 0 \
  --end-frame 100 \
  --visualize

# 多尺度ICP
python align_wrist_camera.py \
  --cam-ext1 datasets/samples/.../camera1 \
  --cam-wrist datasets/samples/.../wrist \
  --multiscale \
  --output-dir output/
```

**参数说明：**
- `--cam-ext1`: 第一个第三人称相机目录（必需，作为参考）
- `--cam-ext2`: 第二个第三人称相机目录（可选，提供更完整场景覆盖）
- `--cam-wrist`: 腕部相机目录（必需）
- `--output-dir`: 输出目录，对齐后的腕部相机外参保存路径
- `--max-iterations`: 每帧ICP最大迭代次数（默认80）
- `--distance-threshold`: ICP对应点距离阈值，单位米（默认0.05）
- `--voxel-size`: 点云降采样体素大小，单位米（默认0.001）
- `--icp-levels`: 多尺度ICP层数（默认3，>=1）
- `--voxel-factor`: 相邻尺度体素大小倍数（默认2.0）
- `--max-depth`: 深度过滤阈值，单位米（默认1.0）
- `--start-frame`: 起始帧索引（默认0）
- `--end-frame`: 结束帧索引（默认全部帧）
- `--multiscale`: 禁用多尺度ICP，使用单尺度模式
- `--visualize`: 保存点云PLY文件用于可视化

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