#!/usr/bin/env python3
"""
批量处理所有场景的腕部相机立体深度（基于 demo/wrist_camera_stereo_depth_pytorch.py）。
默认使用 8 张 GPU 并行处理，每张卡同一时间只跑一个场景。
"""

import argparse
import os
import sys
import time
import subprocess
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    class tqdm:
        def __init__(self, total, desc="", unit="it"):
            self.total = total
            self.count = 0
            self.desc = desc
            self.unit = unit
            print(f"{desc}: 0/{total}")

        def update(self, n=1):
            self.count += n
            print(f"{self.desc}: {self.count}/{self.total}")

        def set_postfix(self, **kwargs):
            pass

        def write(self, msg):
            print(msg)

        def close(self):
            pass


def parse_gpu_list(gpus_arg):
    gpu_ids = []
    for item in gpus_arg.split(","):
        item = item.strip()
        if item == "":
            continue
        if not item.isdigit():
            raise ValueError(f"非法 GPU id: {item}")
        gpu_ids.append(int(item))
    if not gpu_ids:
        raise ValueError("GPU 列表为空")
    return gpu_ids


def find_scene_dirs(datasets_root):
    """
    通过查找 wrist_camera.txt 来定位场景目录。
    wrist_camera.txt 位于相机目录下，场景目录是相机目录的父目录。
    """
    datasets_root = Path(datasets_root)
    scene_dirs = set()
    for marker in datasets_root.rglob("wrist_camera.txt"):
        camera_dir = marker.parent
        scene_dir = camera_dir.parent
        if scene_dir.is_dir():
            scene_dirs.add(scene_dir)
    return sorted(scene_dirs)


def has_processed_wrist_depth(scene_dir):
    """
    判断是否已处理过腕部相机：存在 wrist_camera.txt 且 depth_npy 下有 PNG。
    """
    scene_dir = Path(scene_dir)
    for subdir in scene_dir.iterdir():
        if not subdir.is_dir():
            continue
        if (subdir / "wrist_camera.txt").exists():
            depth_dir = subdir / "depth_npy"
            if depth_dir.exists():
                if any(depth_dir.glob("*.png")):
                    return True
    return False


def build_command(scene_dir, gpu_id, args):
    script_path = Path(__file__).resolve().parent / "demo" / "wrist_camera_stereo_depth_pytorch.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset_root", str(scene_dir),
        "--device", f"cuda:{gpu_id}",
        "--model_type", args.model_type,
        "--num_refine", str(args.num_refine),
        "--confidence_threshold", str(args.confidence_threshold),
        "--image_extension", args.image_extension,
    ]
    if args.allow_negative:
        cmd.append("--allow_negative")
    if args.torch_compile:
        cmd.append("--torch_compile")
    if args.save_visualization:
        cmd.append("--save_visualization")
    if args.save_disparity:
        cmd.append("--save_disparity")
    if args.save_calibration:
        cmd.append("--save_calibration")
    return cmd


def safe_scene_name(scene_dir, datasets_root):
    scene_dir = Path(scene_dir)
    datasets_root = Path(datasets_root)
    try:
        rel = scene_dir.relative_to(datasets_root)
        return "__".join(rel.parts)
    except ValueError:
        return scene_dir.name


def now_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path, message):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(message)


def worker_loop(worker_id, gpu_id, task_queue, result_queue, args):
    while True:
        scene_dir = task_queue.get()
        if scene_dir is None:
            break

        append_log(
            Path(args.master_log_path),
            f"{now_timestamp()} | START | gpu={gpu_id} | worker={worker_id} | scene={scene_dir}\n",
        )
        start_time = time.time()
        cmd = build_command(scene_dir, gpu_id, args)

        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        scene_key = safe_scene_name(scene_dir, args.datasets_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{scene_key}_gpu{gpu_id}_{timestamp}.log"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            duration = time.time() - start_time
            ok = result.returncode == 0

            with open(log_path, "w") as f:
                f.write(f"Worker: {worker_id}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Scene: {scene_dir}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Duration: {duration:.1f}s\n")
                f.write("\nSTDOUT:\n")
                f.write(result.stdout or "")
                f.write("\nSTDERR:\n")
                f.write(result.stderr or "")

            result_queue.put({
                "scene_dir": str(scene_dir),
                "gpu_id": gpu_id,
                "success": ok,
                "duration": duration,
                "log_path": str(log_path),
                "returncode": result.returncode,
            })
        except Exception as e:
            duration = time.time() - start_time
            with open(log_path, "w") as f:
                f.write(f"Worker: {worker_id}\n")
                f.write(f"GPU: {gpu_id}\n")
                f.write(f"Scene: {scene_dir}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Exception: {e}\n")
                f.write(f"Duration: {duration:.1f}s\n")
            result_queue.put({
                "scene_dir": str(scene_dir),
                "gpu_id": gpu_id,
                "success": False,
                "duration": duration,
                "log_path": str(log_path),
                "returncode": -1,
                "error": str(e),
            })


def main():
    parser = argparse.ArgumentParser(
        description="使用 X 卡并行处理所有场景的腕部相机立体深度"
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="/fsx/home/lihao/datasets/droid_datasets/processed",
        help="包含所有场景的根目录",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="GPU 列表，例如: 0,1,2,3,4,5,6,7",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/wrist_depth_batch",
        help="日志输出目录",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过已生成深度图的场景（depth_npy 存在且含 PNG）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只列出将处理的场景，不实际运行",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="从指定索引开始处理（用于续跑）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理场景数量",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="XL",
        help="模型类型: S, M, L, XL",
    )
    parser.add_argument(
        "--num_refine",
        type=int,
        default=9,
        help="局部迭代 refinement 次数",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="使用 torch.compile 加速",
    )
    parser.add_argument(
        "--allow_negative",
        action="store_true",
        help="允许负视差",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.99,
        help="置信度阈值，小于此值的深度置0",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="保存深度图可视化",
    )
    parser.add_argument(
        "--save_disparity",
        action="store_true",
        help="保存视差图",
    )
    parser.add_argument(
        "--save_calibration",
        action="store_true",
        help="保存每帧标定参数",
    )
    parser.add_argument(
        "--image_extension",
        type=str,
        default="png",
        help="图像文件扩展名 (png, jpg, etc.)",
    )

    args = parser.parse_args()

    gpu_ids = parse_gpu_list(args.gpus)
    num_workers = len(gpu_ids)

    datasets_root = Path(args.datasets_root)
    if not datasets_root.exists():
        print(f"错误: datasets_root 不存在: {datasets_root}")
        return 1

    scene_dirs = find_scene_dirs(datasets_root)

    if args.skip_existing:
        filtered = []
        skipped = 0
        for scene_dir in scene_dirs:
            if has_processed_wrist_depth(scene_dir):
                skipped += 1
            else:
                filtered.append(scene_dir)
        scene_dirs = filtered
        pass

    if args.start_from > 0:
        scene_dirs = scene_dirs[args.start_from:]

    if args.limit is not None:
        scene_dirs = scene_dirs[:args.limit]

    if args.dry_run:
        print("\nDry run: 将处理以下场景:")
        for i, scene_dir in enumerate(scene_dirs, 1):
            print(f"{i:4d}. {scene_dir}")
        return 0

    if not scene_dirs:
        print("没有需要处理的场景。")
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = log_dir / f"batch_{batch_ts}.log"
    args.master_log_path = str(master_log)
    append_log(
        master_log,
        (
            f"{'='*80}\n"
            f"Batch start: {now_timestamp()}\n"
            f"Datasets root: {args.datasets_root}\n"
            f"GPUs: {gpu_ids}\n"
            f"Workers: {num_workers}\n"
            f"Total scenes: {len(scene_dirs)}\n"
            f"Skip existing: {args.skip_existing}\n"
            f"Start from: {args.start_from}\n"
            f"Limit: {args.limit}\n"
            f"Model type: {args.model_type}\n"
            f"Num refine: {args.num_refine}\n"
            f"Allow negative: {args.allow_negative}\n"
            f"Torch compile: {args.torch_compile}\n"
            f"Confidence threshold: {args.confidence_threshold}\n"
            f"Image extension: {args.image_extension}\n"
            f"Save visualization: {args.save_visualization}\n"
            f"Save disparity: {args.save_disparity}\n"
            f"Save calibration: {args.save_calibration}\n"
            f"{'='*80}\n"
        ),
    )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    workers = []
    for idx, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=worker_loop,
            args=(idx, gpu_id, task_queue, result_queue, args),
        )
        p.start()
        workers.append(p)

    for scene_dir in scene_dirs:
        task_queue.put(scene_dir)

    for _ in workers:
        task_queue.put(None)

    total = len(scene_dirs)
    completed = 0
    success = 0
    failed = 0

    start_time = time.time()
    progress = tqdm(total=total, desc="处理进度", unit="scene")
    while completed < total:
        result = result_queue.get()
        completed += 1
        if result.get("success"):
            success += 1
        else:
            failed += 1
        progress.update(1)
        progress.set_postfix(success=success, failed=failed)
        append_log(
            master_log,
            (
                f"{now_timestamp()} | "
                f"{'OK' if result.get('success') else 'FAIL'} | "
                f"gpu={result.get('gpu_id')} | "
                f"duration={result.get('duration'):.1f}s | "
                f"returncode={result.get('returncode')} | "
                f"scene={result.get('scene_dir')} | "
                f"log={result.get('log_path')}\n"
            ),
        )

    for p in workers:
        p.join()
    progress.close()

    total_time = time.time() - start_time
    append_log(
        master_log,
        (
            f"{'='*80}\n"
            f"Batch end: {now_timestamp()}\n"
            f"Total: {total}\n"
            f"Success: {success}\n"
            f"Failed: {failed}\n"
            f"Duration: {total_time/60:.1f} minutes\n"
            f"{'='*80}\n"
        ),
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
