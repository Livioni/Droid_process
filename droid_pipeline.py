#!/usr/bin/env python3
"""
Droid Data Processing Pipeline

Complete pipeline for processing Droid dataset scenes:
1. Batch stereo depth processing for all cameras
2. MapAnything initialization for left-right camera poses
3. Left-right camera alignment using ICP
4. Wrist camera pose optimization

Usage:
    python droid_pipeline.py --data_root processed_droid --scene_id SCENE_NAME
    python droid_pipeline.py --data_root processed_droid --batch_process
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import shutil
import numpy as np


def setup_logging(log_file: str = "droid_pipeline.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def fix_depth_data(depth_file: Path, min_depth: float = 0.1, max_depth: float = 10.0) -> bool:
    """
    Fix depth data to prevent CUBLAS errors by replacing problematic values

    Args:
        depth_file: Path to depth .npz file
        min_depth: Minimum valid depth value
        max_depth: Maximum valid depth value

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        # Load depth data
        with np.load(str(depth_file)) as data:
            depth = data["depth"]
            original_depth = depth.copy()

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        modified = False

        # Replace zeros with small positive values
        zero_mask = depth == 0
        if np.any(zero_mask):
            # Use median of non-zero depths as replacement
            non_zero_depths = depth[~zero_mask]
            if len(non_zero_depths) > 0:
                median_depth = np.median(non_zero_depths)
                depth[zero_mask] = median_depth
                modified = True

        # Clamp depth values to valid range
        if np.any(depth < min_depth):
            depth[depth < min_depth] = min_depth
            modified = True

        if np.any(depth > max_depth):
            depth[depth > max_depth] = max_depth
            modified = True

        # Handle NaN and Inf values
        if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
            # Replace with median of valid depths
            valid_mask = ~(np.isnan(depth) | np.isinf(depth))
            if np.any(valid_mask):
                median_depth = np.median(depth[valid_mask])
                depth[np.isnan(depth) | np.isinf(depth)] = median_depth
                modified = True

        if modified:
            # Save fixed depth data
            if original_depth.ndim == 3 and original_depth.shape[-1] == 1:
                # Keep original format
                fixed_depth = depth[..., np.newaxis] if depth.ndim == 2 else depth
            else:
                fixed_depth = depth

            np.savez_compressed(str(depth_file), depth=fixed_depth.astype(np.float32))
            return True

    except Exception as e:
        logging.warning(f"Failed to fix depth file {depth_file}: {e}")

    return False


def find_cameras_in_scene(scene_dir: Path) -> Dict[str, List[str]]:
    """
    Identify camera types in a scene by looking for camera identification files
    and camera directories.

    Returns:
        Dict with keys: 'wrist', 'ext1', 'ext2', each containing list of camera IDs
    """
    cameras = {'wrist': [], 'ext1': [], 'ext2': []}

    # First, check for camera identification files
    wrist_file = scene_dir / 'wrist_camera.txt'
    ext1_file = scene_dir / 'ext1.txt'
    ext2_file = scene_dir / 'ext2.txt'

    if wrist_file.exists():
        with open(wrist_file, 'r') as f:
            wrist_id = f.read().strip()
            if wrist_id:
                cameras['wrist'].append(wrist_id)

    if ext1_file.exists():
        with open(ext1_file, 'r') as f:
            ext1_id = f.read().strip()
            if ext1_id:
                cameras['ext1'].append(ext1_id)

    if ext2_file.exists():
        with open(ext2_file, 'r') as f:
            ext2_id = f.read().strip()
            if ext2_id:
                cameras['ext2'].append(ext2_id)

    # If no identification files found, try to infer from directory structure
    if not any(cameras.values()):
        logging.warning("No camera identification files found, inferring from directory structure")
        camera_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if camera_dirs:
            # Sort by ID and assign first as wrist, next two as ext1/ext2
            sorted_cams = sorted([d.name for d in camera_dirs])
            if len(sorted_cams) >= 1:
                cameras['wrist'] = [sorted_cams[0]]
            if len(sorted_cams) >= 2:
                cameras['ext1'] = [sorted_cams[1]]
            if len(sorted_cams) >= 3:
                cameras['ext2'] = [sorted_cams[2]]

    logging.info(f"Found cameras in scene {scene_dir.name}: {cameras}")
    return cameras


def run_command(
    cmd: List[str],
    description: str,
    cwd: Optional[Path] = None,
    conda_env: Optional[str] = None,
    log_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Run a command and return success status and output

    Args:
        cmd: Command to run as list of strings
        description: Description of the command for logging
        cwd: Working directory (optional)
        conda_env: Conda environment name to activate (optional)
    """
    try:
        # Activate conda environment if specified
        use_shell = False
        if conda_env:
            if conda_env == "mapanything":
                # For mapanything environment, use shell to unset LD_LIBRARY_PATH first
                cmd_str = f"unset LD_LIBRARY_PATH && conda run -n {conda_env} {' '.join(cmd)}"
                use_shell = True
                logging.info(f"Running (shell): {cmd_str}")
            else:
                cmd = ["conda", "run", "-n", conda_env] + cmd
                logging.info(f"Running: {' '.join(cmd)}")
        else:
            logging.info(f"Running: {' '.join(cmd)}")

        logging.info(f"Description: {description}")

        if use_shell:
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                cwd=cwd
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=cwd
            )

        if log_path:
            try:
                with open(log_path, "a") as log_file:
                    log_file.write(f"\n{'='*80}\n")
                    log_file.write(f"Description: {description}\n")
                    log_file.write(f"Command: {cmd_str if use_shell else ' '.join(cmd)}\n")
                    if cwd:
                        log_file.write(f"CWD: {cwd}\n")
                    log_file.write(f"Exit code: {result.returncode}\n")
                    if result.stdout:
                        log_file.write(f"STDOUT:\n{result.stdout}\n")
                    if result.stderr:
                        log_file.write(f"STDERR:\n{result.stderr}\n")
            except Exception as e:
                logging.warning(f"Failed to write command log to {log_path}: {e}")

        if result.returncode == 0:
            logging.info(f"✓ {description} completed successfully")
            return True, result.stdout
        else:
            error_msg = f"✗ {description} failed with exit code {result.returncode}"
            logging.error(error_msg)
            if result.stderr:
                logging.error(f"STDERR: {result.stderr}")
            return False, result.stderr

    except Exception as e:
        error_msg = f"✗ Exception running {description}: {e}"
        logging.error(error_msg)
        return False, str(e)


def step_1_batch_depth_processing(
    scene_dir: Path,
    cameras: Dict[str, List[str]],
    depth_method: str = "mixed",
    log_path: Optional[Path] = None,
) -> Tuple[bool, int]:
    """
    Step 1: Batch stereo depth processing for all cameras
    """
    logging.info("="*80)
    logging.info("STEP 1: Batch Stereo Depth Processing")
    logging.info("="*80)

    # Get camera IDs by type
    wrist_camera_ids = list(cameras.get("wrist", []))
    ext_camera_ids = list(cameras.get("ext1", [])) + list(cameras.get("ext2", []))
    all_camera_ids = wrist_camera_ids + ext_camera_ids

    if not all_camera_ids:
        logging.error("No cameras found for depth processing")
        return False, 0

    processed_count = 0

    def run_s2m2(target_ids: List[str], label: str) -> bool:
        if not target_ids:
            logging.info(f"No cameras found for {label}, skipping S2M2 depth processing")
            return True
        logging.info(f"Using S2M2 (PyTorch) depth pipeline for {label}")
        cmd = [
            "python", "demo/batch_stereo_depth_pytorch.py",
            "--dataset_root", str(scene_dir),
            "--cameras",
        ] + target_ids + [
            "--model_type", "XL",
            "--num_refine", "16",
            "--confidence_threshold", "0.9",  # Lower threshold to reduce zero depths
            "--depth_storage", "compressed",
            "--depth_dtype", "float16",
        ]
        success, _ = run_command(
            cmd,
            f"S2M2 depth processing for {label}: {', '.join(target_ids)}",
            conda_env="droid",
            log_path=log_path,
        )
        return success

    def run_foundation(target_ids: List[str], label: str) -> bool:
        if not target_ids:
            logging.info(f"No cameras found for {label}, skipping FoundationStereo depth processing")
            return True
        logging.info(f"Using FoundationStereo depth pipeline for {label}")
        cmd = [
            "python", "demo/batch_stereo_depth_foundation.py",
            "--dataset_root", str(scene_dir),
            "--cameras",
        ] + target_ids
        success, _ = run_command(
            cmd,
            f"FoundationStereo depth processing for {label}: {', '.join(target_ids)}",
            conda_env="foundation_stereo",
            log_path=log_path,
        )
        return success

    if depth_method == "s2m2":
        if not run_s2m2(all_camera_ids, "all cameras"):
            logging.error("Depth processing failed")
            return False, processed_count
    elif depth_method == "foundation_stereo":
        if not run_foundation(all_camera_ids, "all cameras"):
            logging.error("Depth processing failed")
            return False, processed_count
    else:
        # mixed: wrist -> s2m2, external -> foundation
        if not run_s2m2(wrist_camera_ids, "wrist cameras"):
            logging.error("Wrist depth processing failed")
            return False, processed_count
        if not run_foundation(ext_camera_ids, "external cameras"):
            logging.error("External depth processing failed")
            return False, processed_count

    # Verify depth files were created and check for excessive zeros
    for cam_id in all_camera_ids:
        depth_dir = scene_dir / cam_id / "depth_npy"
        if not depth_dir.exists():
            logging.error(f"Depth directory not created for camera {cam_id}")
            return False, processed_count
        depth_files = list(depth_dir.glob("*.npz"))
        if not depth_files:
            logging.warning(f"No depth files found for camera {cam_id}")
            continue

        logging.info(f"Created {len(depth_files)} depth files for camera {cam_id}")
        processed_count += 1

    return True, processed_count


def step_2_mapanything_initialization(
    scene_dir: Path,
    cameras: Dict[str, List[str]],
    log_path: Optional[Path] = None,
) -> Tuple[bool, int]:
    """
    Step 2: MapAnything initialization for left-right camera poses
    Only run if we have at least 2 external cameras
    This step is optional and will not fail the pipeline if it fails
    """
    logging.info("="*80)
    logging.info("STEP 2: MapAnything Initialization")
    logging.info("="*80)

    ext_cameras = cameras['ext1'] + cameras['ext2']
    if len(ext_cameras) < 2:
        logging.info("Less than 2 external cameras found, skipping MapAnything initialization")
        return True, 0

    # Use first two external cameras for initialization
    ref_cam_id = ext_cameras[0]
    tgt_cam_id = ext_cameras[1]

    ref_cam_dir = scene_dir / ref_cam_id
    tgt_cam_dir = scene_dir / tgt_cam_id

    if not ref_cam_dir.exists() or not tgt_cam_dir.exists():
        logging.error(f"Camera directories not found: {ref_cam_dir}, {tgt_cam_dir}")
        return False, 0

    # Check if depth files exist and have reasonable quality
    ref_depth_dir = ref_cam_dir / "depth_npy"
    tgt_depth_dir = tgt_cam_dir / "depth_npy"

    if not ref_depth_dir.exists() or not tgt_depth_dir.exists():
        logging.warning("Depth directories not found, skipping MapAnything initialization")
        return True, 0

    # Check depth quality for potential CUBLAS issues
    try:
        import numpy as np
        ref_depth_files = list(ref_depth_dir.glob("*.npz"))
        tgt_depth_files = list(tgt_depth_dir.glob("*.npz"))

        if ref_depth_files and tgt_depth_files:
            # Check zero ratio in first frame
            with np.load(str(ref_depth_files[0])) as data:
                ref_depth = data["depth"]
                if ref_depth.ndim == 3 and ref_depth.shape[-1] == 1:
                    ref_depth = ref_depth[..., 0]
                ref_zero_ratio = np.sum(ref_depth == 0) / ref_depth.size

            with np.load(str(tgt_depth_files[0])) as data:
                tgt_depth = data["depth"]
                if tgt_depth.ndim == 3 and tgt_depth.shape[-1] == 1:
                    tgt_depth = tgt_depth[..., 0]
                tgt_zero_ratio = np.sum(tgt_depth == 0) / tgt_depth.size

            if ref_zero_ratio > 0.7 or tgt_zero_ratio > 0.7:
                logging.warning("Skipping MapAnything initialization due to poor depth quality")
                return True, 0

    except Exception as e:
        logging.warning(f"Could not check depth quality: {e}")

    # Create output directory
    output_dir = tgt_cam_dir / "extrinsics_refined"
    output_dir.mkdir(exist_ok=True)

    # Try MapAnything initialization
    cmd = [
        "python", "demo/mapanything_multimodal_extrinsics.py",
        "--ref_cam", str(ref_cam_dir),
        "--tgt_cam", str(tgt_cam_dir),
        "--frame", "0",
        "--output_dir", str(output_dir),
        "--output_name", f"mapanything.npy"
    ]
    
    print("cmd: ", cmd)

    success, output = run_command(
        cmd,
        f"MapAnything initialization for {ref_cam_id} -> {tgt_cam_id}",
        conda_env="mapanything",
        log_path=log_path,
    )
    if not success:
        logging.warning("MapAnything initialization failed (possibly due to CUBLAS error from zero depths)")
        logging.warning("This is often caused by excessive zero depth values")
        logging.warning("Continuing with ICP alignment without MapAnything initialization")
        return True, 0  # Don't fail the pipeline for this optional step

    # Check if output file was created
    output_file = output_dir / f"mapanything.npy"
    if output_file.exists():
        logging.info(f"MapAnything initialization output: {output_file}")
        return True, 1
    else:
        logging.warning("MapAnything output file not found")
        return True, 0


def step_3_align_left_right_cameras(
    scene_dir: Path,
    cameras: Dict[str, List[str]],
    log_path: Optional[Path] = None,
) -> Tuple[bool, int]:
    """
    Step 3: Align left-right cameras using ICP
    """
    logging.info("="*80)
    logging.info("STEP 3: Left-Right Camera Alignment")
    logging.info("="*80)

    ext_cameras = cameras['ext1'] + cameras['ext2']
    if len(ext_cameras) < 2:
        logging.info("Less than 2 external cameras found, skipping left-right alignment")
        return True, 0

    # Align each pair of external cameras
    ref_cam_id = ext_cameras[0]
    ref_cam_dir = scene_dir / ref_cam_id
    processed_count = 0

    for tgt_cam_id in ext_cameras[1:]:
        tgt_cam_dir = scene_dir / tgt_cam_id

        # Create output directory
        output_dir = tgt_cam_dir / "extrinsics_refined"
        output_dir.mkdir(exist_ok=True)

        cmd = [
            "python", "align_left_right_camera.py",
            "--cam1", str(ref_cam_dir),
            "--cam2", str(tgt_cam_dir),
            "--output-dir", str(output_dir),
            "--num-frames", "10",
            "--max-depth", "0.8",
            "--method", "robust"
        ]

        success, output = run_command(
            cmd,
            f"Aligning cameras {ref_cam_id} -> {tgt_cam_id}",
            conda_env="droid",
            log_path=log_path,
        )
        if not success:
            logging.error(f"Failed to align cameras {ref_cam_id} -> {tgt_cam_id}")
            return False, processed_count

        # Check output files
        output_file = output_dir / f"{tgt_cam_id}.npy"
        stats_file = output_dir / "alignment_stats.png"

        if output_file.exists():
            logging.info(f"Alignment output: {output_file}")
            processed_count += 1
        else:
            logging.error(f"Alignment output file not found: {output_file}")
            return False, processed_count

    return True, processed_count


def step_4_optimize_wrist_camera(
    scene_dir: Path,
    cameras: Dict[str, List[str]],
    log_path: Optional[Path] = None,
) -> Tuple[bool, int]:
    """
    Step 4: Optimize wrist camera pose using external cameras
    """
    logging.info("="*80)
    logging.info("STEP 4: Wrist Camera Pose Optimization")
    logging.info("="*80)

    if not cameras['wrist']:
        logging.info("No wrist camera found, skipping wrist pose optimization")
        return True, 0

    wrist_cam_id = cameras['wrist'][0]
    wrist_cam_dir = scene_dir / wrist_cam_id

    if not wrist_cam_dir.exists():
        logging.error(f"Wrist camera directory not found: {wrist_cam_dir}")
        return False, 0

    ext_cameras = cameras['ext1'] + cameras['ext2']
    if not ext_cameras:
        logging.warning("No external cameras found for wrist pose optimization")
        return True

    # Use first external camera, and second if available
    ext1_cam_id = ext_cameras[0]
    ext1_cam_dir = scene_dir / ext1_cam_id

    # Create output directory
    output_dir = wrist_cam_dir / "extrinsics_refined"
    output_dir.mkdir(exist_ok=True)

    if len(ext_cameras) >= 2:
        # Use both external cameras
        ext2_cam_id = ext_cameras[1]
        ext2_cam_dir = scene_dir / ext2_cam_id

        cmd = [
            "python", "align_wrist_camera.py",
            "--cam-ext1", str(ext1_cam_dir),
            "--cam-ext2", str(ext2_cam_dir),
            "--cam-wrist", str(wrist_cam_dir),
            "--output-dir", str(output_dir)
        ]

        description = f"Wrist camera optimization using both {ext1_cam_id} and {ext2_cam_id}"
    else:
        # Use single external camera
        cmd = [
            "python", "align_wrist_camera.py",
            "--cam-ext1", str(ext1_cam_dir),
            "--cam-wrist", str(wrist_cam_dir),
            "--output-dir", str(output_dir)
        ]

        description = f"Wrist camera optimization using {ext1_cam_id}"

    success, output = run_command(
        cmd, description, conda_env="droid", log_path=log_path
    )
    if not success:
        logging.error("Wrist camera pose optimization failed")
        return False, 0

    # Check output files
    output_file = output_dir / f"{wrist_cam_id}.npy"
    stats_file = output_dir / "wrist_alignment_stats.png"

    if output_file.exists():
        logging.info(f"Wrist optimization output: {output_file}")
        return True, 1
    else:
        logging.error(f"Wrist optimization output file not found: {output_file}")
        return False, 0


def process_single_scene(scene_dir: Path, depth_method: str) -> bool:
    """
    Process a single scene through the complete pipeline
    """
    logging.info(f"\n{'='*100}")
    logging.info(f"Processing scene: {scene_dir.name}")
    logging.info(f"Scene path: {scene_dir}")
    logging.info(f"{'='*100}")

    if not scene_dir.exists():
        logging.error(f"Scene directory does not exist: {scene_dir}")
        return False

    # Find cameras in the scene
    cameras = find_cameras_in_scene(scene_dir)

    if not any(cameras.values()):
        logging.error("No cameras found in scene")
        return False

    # Initialize statistics
    stats = {
        'step1_depth_processed': 0,
        'step2_mapanything_processed': 0,
        'step3_alignment_processed': 0,
        'step4_wrist_processed': 0,
        'total_cameras': sum(len(cam_list) for cam_list in cameras.values())
    }

    logging.info(f"Found {stats['total_cameras']} cameras: wrist={len(cameras['wrist'])}, ext1={len(cameras['ext1'])}, ext2={len(cameras['ext2'])}")

    # Command log file per scene
    command_log_path = scene_dir / "pipeline_commands.log"

    # Execute pipeline steps
    success = True

    # Step 1: Batch depth processing
    step1_success, step1_processed = step_1_batch_depth_processing(
        scene_dir, cameras, depth_method=depth_method, log_path=command_log_path
    )
    stats['step1_depth_processed'] = step1_processed
    if not step1_success:
        success = False

    # Step 2: MapAnything initialization
    if success:
        step2_success, step2_processed = step_2_mapanything_initialization(
            scene_dir, cameras, log_path=command_log_path
        )
        stats['step2_mapanything_processed'] = step2_processed
        if not step2_success:
            success = False

    # Step 3: Left-right camera alignment
    if success:
        step3_success, step3_processed = step_3_align_left_right_cameras(
            scene_dir, cameras, log_path=command_log_path
        )
        stats['step3_alignment_processed'] = step3_processed
        if not step3_success:
            success = False

    # Step 4: Wrist camera optimization
    if success:
        step4_success, step4_processed = step_4_optimize_wrist_camera(
            scene_dir, cameras, log_path=command_log_path
        )
        stats['step4_wrist_processed'] = step4_processed
        if not step4_success:
            success = False

    # Print statistics
    print(f"\n{'='*60}")
    print(f"PIPELINE STATISTICS FOR SCENE: {scene_dir.name}")
    print(f"{'='*60}")
    print(f"Depth method (step 1): {depth_method}")
    print(f"Total cameras found: {stats['total_cameras']}")
    print(f"  - Wrist cameras: {len(cameras['wrist'])}")
    print(f"  - External cameras (ext1): {len(cameras['ext1'])}")
    print(f"  - External cameras (ext2): {len(cameras['ext2'])}")
    print()
    print(f"Step 1 - Depth Processing: {stats['step1_depth_processed']} cameras processed")
    print(f"Step 2 - MapAnything Init: {stats['step2_mapanything_processed']} camera pairs processed")
    print(f"Step 3 - Camera Alignment: {stats['step3_alignment_processed']} camera pairs processed")
    print(f"Step 4 - Wrist Optimization: {stats['step4_wrist_processed']} cameras processed")
    print(f"{'='*60}")

    if success:
        logging.info(f"✓ Pipeline completed successfully for scene: {scene_dir.name}")
    else:
        logging.error(f"✗ Pipeline failed for scene: {scene_dir.name}")

    return success


def find_all_scenes(data_root: Path) -> List[Path]:
    """
    Find all scene directories in the data root
    """
    scenes = []
    if not data_root.exists():
        logging.error(f"Data root directory does not exist: {data_root}")
        return scenes

    # Look for directories that contain camera subdirectories or camera ID files
    for item in data_root.iterdir():
        if item.is_dir():
            # Check if it looks like a scene directory
            has_cameras = any(subitem.is_dir() and subitem.name.isdigit()
                            for subitem in item.iterdir())
            has_camera_files = any(subitem.name in ['wrist_camera.txt', 'ext1.txt', 'ext2.txt']
                                 for subitem in item.iterdir())

            if has_cameras or has_camera_files:
                scenes.append(item)

    return sorted(scenes)


def main():
    parser = argparse.ArgumentParser(description='Droid Data Processing Pipeline')
    parser.add_argument('--data_root', type=str, default='/opt/dlami/nvme/datasets/processed_droid',
                       help='Root directory containing processed Droid scenes')
    parser.add_argument('--scene_id', type=str,
                       help='Specific scene ID to process (optional)')
    parser.add_argument('--batch_process', action='store_true',
                       help='Process all scenes in batch mode')
    parser.add_argument('--log_file', type=str, default='droid_pipeline.log',
                       help='Log file path')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip scenes that have already been processed')
    parser.add_argument(
        '--depth_method',
        type=str,
        default='mixed',
        choices=['s2m2', 'foundation_stereo', 'mixed'],
        help='Depth method for step 1: mixed (wrist=s2m2, ext=foundation), or single method'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    logging.info("Droid Processing Pipeline Started")
    logging.info(f"Arguments: {args}")

    data_root = Path(args.data_root)

    if args.scene_id:
        # Process single scene
        scene_dir = data_root / args.scene_id
        success = process_single_scene(scene_dir, depth_method=args.depth_method)
        return 0 if success else 1

    elif args.batch_process:
        # Process all scenes
        scenes = find_all_scenes(data_root)

        if not scenes:
            logging.error(f"No scenes found in {data_root}")
            return 1

        logging.info(f"Found {len(scenes)} scenes to process")

        successful = 0
        failed = 0
        skipped = 0

        for scene_dir in scenes:
            if args.skip_existing:
                # Check if scene has been processed (look for refined extrinsics)
                processed = False
                for cam_dir in scene_dir.iterdir():
                    if cam_dir.is_dir() and cam_dir.name.isdigit():
                        refined_dir = cam_dir / "extrinsics_refined"
                        if refined_dir.exists():
                            processed = True
                            break
                if processed:
                    logging.info(f"Skipping already processed scene: {scene_dir.name}")
                    skipped += 1
                    continue

            if process_single_scene(scene_dir, depth_method=args.depth_method):
                successful += 1
            else:
                failed += 1

        logging.info(f"\n{'='*80}")
        logging.info("Batch processing completed")
        logging.info(f"Total scenes: {len(scenes)}")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Skipped: {skipped}")
        logging.info(f"{'='*80}")

        return 0 if failed == 0 else 1

    else:
        logging.error("Must specify either --scene_id or --batch_process")
        return 1


if __name__ == "__main__":
    exit(main())