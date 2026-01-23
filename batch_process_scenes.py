#!/usr/bin/env python3
"""
Batch processing script for all scene directories in the datasets folder.

This script recursively finds all scene directories (e.g., Fri_Jul__7_09:45:39_2023)
and processes them using process_camera_data.py with support for parallel processing.

Usage:
    python batch_process_scenes.py [--datasets_root datasets] [--output_dir /opt/dlami/nvme/datasets/processed_droid] [--num_workers 4]
"""

import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import shutil
import json


def is_safe_to_delete(scene_dir, datasets_root):
    """
    Safety check: Ensure the directory is within datasets_root and looks like a scene directory.
    
    Args:
        scene_dir: Path to the scene directory to delete
        datasets_root: Path to the datasets root directory
    
    Returns:
        bool: True if safe to delete, False otherwise
    """
    scene_dir = Path(scene_dir).resolve()
    datasets_root = Path(datasets_root).resolve()
    
    # Check 1: Must be within datasets_root
    try:
        scene_dir.relative_to(datasets_root)
    except ValueError:
        return False
    
    # Check 2: Must not be datasets_root itself
    if scene_dir == datasets_root:
        return False
    
    # Check 3: Must be at least 3 levels deep (e.g., datasets/LAB/success/DATE/SCENE)
    rel_path = scene_dir.relative_to(datasets_root)
    if len(rel_path.parts) < 3:
        return False
    
    # Check 4: Must contain typical scene markers
    has_recordings = (scene_dir / "recordings").exists()
    has_trajectory = (scene_dir / "trajectory.h5").exists()
    has_metadata = any(f.name.startswith('metadata') and f.name.endswith('.json') 
                      for f in scene_dir.glob('*.json'))
    
    if not (has_recordings or has_trajectory or has_metadata):
        return False
    
    return True


def validate_scene_completeness(scene_dir):
    """
    Validate if a scene directory has all required files based on metadata.
    
    Args:
        scene_dir: Path to the scene directory
    
    Returns:
        Tuple of (is_valid: bool, missing_files: list, metadata_path: Path or None)
    """
    scene_dir = Path(scene_dir)
    
    # Find metadata file
    metadata_files = list(scene_dir.glob("metadata*.json"))
    if not metadata_files:
        return (False, ["No metadata file found"], None)
    
    metadata_path = metadata_files[0]
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        return (False, [f"Failed to read metadata: {e}"], metadata_path)
    
    missing_files = []
    
    # Check trajectory.h5
    if 'hdf5_path' in metadata:
        traj_file = scene_dir / "trajectory.h5"
        if not traj_file.exists():
            missing_files.append("trajectory.h5")
    
    # Check recordings directory exists
    recordings_dir = scene_dir / "recordings"
    if not recordings_dir.exists():
        missing_files.append("recordings/")
    else:
        # Check SVO files
        svo_paths = ['wrist_svo_path', 'ext1_svo_path', 'ext2_svo_path']
        for svo_key in svo_paths:
            if svo_key in metadata:
                svo_path = metadata[svo_key]
                # Extract just the filename from the full path
                svo_filename = Path(svo_path).name
                svo_file = scene_dir / "recordings" / "SVO" / svo_filename
                if not svo_file.exists():
                    missing_files.append(str(svo_file.relative_to(scene_dir)))
        
        # Check MP4 files (optional - some scenes might not have MP4s)
        mp4_paths = ['wrist_mp4_path', 'ext1_mp4_path', 'ext2_mp4_path']
        for mp4_key in mp4_paths:
            if mp4_key in metadata:
                mp4_path = metadata[mp4_key]
                # Extract just the filename from the full path
                mp4_filename = Path(mp4_path).name
                mp4_file = scene_dir / "recordings" / "MP4" / mp4_filename
                # Note: MP4 files are less critical, but we still report if missing
                if not mp4_file.exists():
                    missing_files.append(str(mp4_file.relative_to(scene_dir)))
                # Also require the stereo MP4 variant: xxxx-stereo.mp4
                mp4_stem = Path(mp4_filename).stem
                stereo_mp4_filename = f"{mp4_stem}-stereo.mp4"
                stereo_mp4_file = scene_dir / "recordings" / "MP4" / stereo_mp4_filename
                if not stereo_mp4_file.exists():
                    missing_files.append(str(stereo_mp4_file.relative_to(scene_dir)))
    
    is_valid = len(missing_files) == 0
    return (is_valid, missing_files, metadata_path)


def find_scene_directories(datasets_root):
    """
    Find all scene directories (final level with actual data files).
    Scene directories should contain recordings/ folder and trajectory.h5
    
    Returns a list of Path objects for all scene directories.
    """
    datasets_root = Path(datasets_root)
    scene_dirs = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(datasets_root):
        root_path = Path(root)
        
        # Check if this directory contains the expected structure
        # Scene directories should have recordings/ and trajectory.h5
        has_recordings = (root_path / "recordings").exists()
        has_trajectory = (root_path / "trajectory.h5").exists()
        has_metadata = any(f.endswith('.json') and f.startswith('metadata') for f in files)
        
        if has_recordings or has_trajectory or has_metadata:
            scene_dirs.append(root_path)
    
    return sorted(scene_dirs)


def process_scene(scene_dir, output_dir, log_file_path=None, delete_after_success=False, 
                 validate_before_process=False, datasets_root=None):
    """
    Process a single scene directory using process_camera_data.py
    
    Args:
        scene_dir: Path to the scene directory
        output_dir: Base output directory
        log_file_path: Optional path to log file for logging
        delete_after_success: If True, delete the original directory after successful processing
        validate_before_process: If True, validate scene completeness before processing
        datasets_root: Root datasets directory for safety checks
    
    Returns:
        Tuple of (success: bool, scene_dir: Path, message: str, deleted: bool)
    """
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    
    # Validate scene completeness if requested
    if validate_before_process:
        is_valid, missing_files, metadata_path = validate_scene_completeness(scene_dir)
        
        if not is_valid:
            error_msg = f"✗ Invalid scene (missing files): {scene_dir}\n"
            error_msg += f"  Missing files: {', '.join(missing_files)}\n"
            
            log_msg = f"\n{'='*80}\n"
            log_msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            log_msg += f"{error_msg}\n"
            
            # Safety check before deletion
            if datasets_root and is_safe_to_delete(scene_dir, datasets_root):
                missing_detail = ", ".join(missing_files) if missing_files else "unknown"
                delete_reason = f"incomplete scene (missing: {missing_detail})"
                print(f"Deleting scene: {scene_dir} | Reason: {delete_reason}")
                error_msg += f"  Deleting incomplete scene directory..."
                
                # Delete the incomplete scene directory
                try:
                    shutil.rmtree(scene_dir)
                    delete_msg = f"  Successfully deleted incomplete scene: {scene_dir}"
                    error_msg += f"\n{delete_msg}"
                    log_msg += f"{delete_msg}\n"
                except Exception as e:
                    delete_error = f"  Warning: Failed to delete {scene_dir}: {e}"
                    error_msg += f"\n{delete_error}"
                    log_msg += f"{delete_error}\n"
            else:
                safety_msg = f"  Safety check failed - NOT deleting: {scene_dir}"
                error_msg += f"\n{safety_msg}"
                log_msg += f"{safety_msg}\n"
            
            log_msg += f"{'='*80}\n"
            
            # Write to log file if provided
            if log_file_path:
                with open(log_file_path, 'a') as f:
                    f.write(log_msg)
            
            return (False, scene_dir, error_msg, True)  # deleted=True
    
    # Build the command
    cmd = [
        "python",
        "process_camera_data.py",
        "--input_dir", str(scene_dir),
        "--output_dir", str(output_dir)
    ]
    
    start_time = datetime.now()
    log_msg = f"\n{'='*80}\n"
    log_msg += f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_msg += f"Processing: {scene_dir}\n"
    log_msg += f"Command: {' '.join(cmd)}\n"
    log_msg += f"{'='*80}\n"
    
    try:
        # Run the processing command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Collect output
        log_msg += result.stdout if result.stdout else ""
        log_msg += f"STDERR: {result.stderr}\n" if result.stderr else ""
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            success_msg = f"✓ Successfully processed: {scene_dir} (took {duration:.1f}s)"
            log_msg += f"{success_msg}\n"
            
            # Delete original directory if requested
            if delete_after_success:
                try:
                    delete_reason = "processing completed successfully (delete_after_success)"
                    print(f"Deleting scene: {scene_dir} | Reason: {delete_reason}")
                    shutil.rmtree(scene_dir)
                    delete_msg = f"  Deleted original directory: {scene_dir}"
                    success_msg += f"\n{delete_msg}"
                    log_msg += f"{delete_msg}\n"
                except Exception as e:
                    delete_error = f"  Warning: Failed to delete {scene_dir}: {e}"
                    success_msg += f"\n{delete_error}"
                    log_msg += f"{delete_error}\n"
            
            # Write to log file if provided
            if log_file_path:
                with open(log_file_path, 'a') as f:
                    f.write(log_msg)
            
            return (True, scene_dir, success_msg, False)  # deleted=False
        else:
            error_msg = f"✗ Failed to process: {scene_dir} (exit code: {result.returncode}, took {duration:.1f}s)"
            log_msg += f"{error_msg}\n"
            
            # Write to log file if provided
            if log_file_path:
                with open(log_file_path, 'a') as f:
                    f.write(log_msg)
            
            return (False, scene_dir, error_msg, False)  # deleted=False
            
    except Exception as e:
        error_msg = f"✗ Error processing {scene_dir}: {e}"
        log_msg += f"{error_msg}\n"
        
        # Write to log file if provided
        if log_file_path:
            with open(log_file_path, 'a') as f:
                f.write(log_msg)
        
        return (False, scene_dir, error_msg, False)  # deleted=False


def process_scene_wrapper(args):
    """Wrapper function for parallel processing"""
    scene_dir, output_dir, log_file_path, delete_after_success, validate_before_process, datasets_root = args
    return process_scene(scene_dir, output_dir, log_file_path, delete_after_success, 
                        validate_before_process, datasets_root)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process all scene directories in the datasets folder'
    )
    parser.add_argument(
        '--datasets_root',
        type=str,
        default='/opt/dlami/nvme/datasets/droid_datasets',
        help='Root directory containing all datasets (default: datasets)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/opt/dlami/nvme/datasets/processed_droid',
        help='Output directory for processed data (default: /opt/dlami/nvme/datasets/processed_droid)'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='batch_processing.log',
        help='Log file path (default: batch_processing.log)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only list the directories that would be processed without actually processing them'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip scenes that have already been processed (check if output directory exists)'
    )
    parser.add_argument(
        '--start_from',
        type=int,
        default=0,
        help='Start processing from this index (useful for resuming)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of scenes to process'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, sequential processing)'
    )
    parser.add_argument(
        '--delete_after_success',
        action='store_true',
        help='Delete original scene directory after successful processing to save space'
    )
    parser.add_argument(
        '--validate_scenes',
        action='store_true',
        help='Validate scene completeness before processing. Delete incomplete scenes.'
    )
    
    args = parser.parse_args()
    
    # Find all scene directories
    print("Scanning for scene directories...")
    scene_dirs = find_scene_directories(args.datasets_root)
    
    print(f"\nFound {len(scene_dirs)} scene directories")
    
    if args.dry_run:
        print("\nDry run mode - listing directories that would be processed:\n")
        for i, scene_dir in enumerate(scene_dirs):
            print(f"{i+1:4d}. {scene_dir}")
        return 0
    
    # Apply start_from and limit
    if args.start_from > 0:
        scene_dirs = scene_dirs[args.start_from:]
        print(f"Starting from index {args.start_from}")
    
    if args.limit is not None:
        scene_dirs = scene_dirs[:args.limit]
        print(f"Limited to {args.limit} scenes")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Validate num_workers
    max_workers = multiprocessing.cpu_count()
    if args.num_workers > max_workers:
        print(f"Warning: num_workers ({args.num_workers}) exceeds CPU count ({max_workers}). Using {max_workers} instead.")
        args.num_workers = max_workers
    
    print(f"Using {args.num_workers} parallel worker(s)")
    
    if args.validate_scenes:
        print("✓ Scene validation enabled: Incomplete scenes will be identified and deleted.")
    
    if args.delete_after_success or args.validate_scenes:
        print("\n" + "="*80)
        print("⚠️  WARNING: DELETION MODE ENABLED!")
        print("="*80)
        if args.validate_scenes:
            print("• Incomplete scenes will be DELETED")
        if args.delete_after_success:
            print("• Successfully processed scenes will be DELETED")
        print(f"\nDatasets root: {Path(args.datasets_root).resolve()}")
        print(f"Output directory: {output_dir.resolve()}")
        print("="*80 + "\n")
    
    # Initialize log file
    with open(args.log_file, 'a') as log_file:
        log_file.write(f"\n\n{'='*80}\n")
        log_file.write(f"Batch processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total scenes to process: {len(scene_dirs)}\n")
        log_file.write(f"Output directory: {output_dir}\n")
        log_file.write(f"Number of workers: {args.num_workers}\n")
        log_file.write(f"Validate scenes: {args.validate_scenes}\n")
        log_file.write(f"Delete after success: {args.delete_after_success}\n")
        log_file.write(f"{'='*80}\n\n")
    
    # Filter scenes to process (skip existing if needed)
    scenes_to_process = []
    skipped = 0
    
    for scene_dir in scene_dirs:
        if args.skip_existing:
            scene_name = scene_dir.name
            output_scene_dir = output_dir / scene_name
            if output_scene_dir.exists():
                skip_msg = f"Skipping (already exists): {scene_dir}"
                print(skip_msg)
                with open(args.log_file, 'a') as log_file:
                    log_file.write(f"{skip_msg}\n")
                skipped += 1
                continue
        scenes_to_process.append(scene_dir)
    
    print(f"\nProcessing {len(scenes_to_process)} scenes (skipped {skipped})")
    
    # Process all scenes
    successful = 0
    failed = 0
    completed = 0
    deleted_invalid = 0
    
    # Use lock for thread-safe printing
    print_lock = threading.Lock()
    
    start_time = datetime.now()
    
    if args.num_workers == 1:
        # Sequential processing
        for i, scene_dir in enumerate(scenes_to_process, 1):
            with print_lock:
                print(f"\n[{i}/{len(scenes_to_process)}] Processing scene...")
            
            success, scene_path, message, was_deleted = process_scene(
                scene_dir, output_dir, args.log_file, args.delete_after_success, 
                args.validate_scenes, args.datasets_root
            )
            
            with print_lock:
                print(message)
                completed += 1
                if success:
                    successful += 1
                else:
                    failed += 1
                    if was_deleted and args.validate_scenes:
                        deleted_invalid += 1
                
                # Print progress summary
                print(f"\nProgress: {completed}/{len(scenes_to_process)} scenes completed")
                print(f"  ✓ Successful: {successful}")
                print(f"  ✗ Failed: {failed}")
                if args.validate_scenes and deleted_invalid > 0:
                    print(f"  🗑 Deleted (invalid): {deleted_invalid}")
                if skipped > 0:
                    print(f"  ⊘ Skipped: {skipped}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all jobs
            future_to_scene = {
                executor.submit(
                    process_scene, scene_dir, output_dir, args.log_file, 
                    args.delete_after_success, args.validate_scenes, args.datasets_root
                ): scene_dir 
                for scene_dir in scenes_to_process
            }
            
            # Process results as they complete
            for future in as_completed(future_to_scene):
                scene_dir = future_to_scene[future]
                try:
                    success, scene_path, message, was_deleted = future.result()
                    
                    with print_lock:
                        print(message)
                        completed += 1
                        if success:
                            successful += 1
                        else:
                            failed += 1
                            if was_deleted and args.validate_scenes:
                                deleted_invalid += 1
                        
                        # Print progress summary
                        print(f"\nProgress: {completed}/{len(scenes_to_process)} scenes completed")
                        print(f"  ✓ Successful: {successful}")
                        print(f"  ✗ Failed: {failed}")
                        if args.validate_scenes and deleted_invalid > 0:
                            print(f"  🗑 Deleted (invalid): {deleted_invalid}")
                        if skipped > 0:
                            print(f"  ⊘ Skipped: {skipped}")
                        
                except Exception as e:
                    with print_lock:
                        error_msg = f"✗ Exception processing {scene_dir}: {e}"
                        print(error_msg)
                        with open(args.log_file, 'a') as log_file:
                            log_file.write(f"{error_msg}\n")
                        failed += 1
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Final summary
    summary = f"""
{'='*80}
Batch processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)
{'='*80}
Total scenes: {len(scene_dirs)}
Successful: {successful}
Failed: {failed}"""
    
    if args.validate_scenes:
        summary += f"\nDeleted (invalid): {deleted_invalid}"
    
    summary += f"""
Skipped: {skipped}
{'='*80}
"""
    
    print(summary)
    with open(args.log_file, 'a') as log_file:
        log_file.write(summary)
    
    print(f"\nLog file saved to: {args.log_file}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
