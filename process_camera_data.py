#!/usr/bin/env python3
"""
Unified script to process camera data from a dataset folder.
Extracts intrinsics, extrinsics, and stereo frames for all cameras.

Usage:
    python process_camera_data.py /path/to/dataset/folder
"""

import pyzed.sl as sl
import cv2
import numpy as np
import h5py
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
import argparse


def extract_intrinsics_from_svo(svo_path):
    """Extract left and right camera intrinsic matrices from a .svo file"""
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.2

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise Exception(f"Error reading camera data from {svo_path}: {err}")

    params = zed.get_camera_information().camera_configuration.calibration_parameters

    left_intrinsic_mat = np.array([
        [params.left_cam.fx, 0, params.left_cam.cx],
        [0, params.left_cam.fy, params.left_cam.cy],
        [0, 0, 1],
    ])

    right_intrinsic_mat = np.array([
        [params.right_cam.fx, 0, params.right_cam.cx],
        [0, params.right_cam.fy, params.right_cam.cy],
        [0, 0, 1],
    ])

    zed.close()
    return left_intrinsic_mat, right_intrinsic_mat


def extract_raw_depth_from_svo(svo_path, raw_depth_output_dir):
    """
    Extract per-frame raw depth (in meters) from a ZED .svo file and save to disk.

    Saved format:
      - One .npy file per frame: {frame_idx:06d}.npy
      - dtype: float32
      - shape: (H, W)
    """
    raw_depth_output_dir = Path(raw_depth_output_dir)
    raw_depth_output_dir.mkdir(parents=True, exist_ok=True)

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.2

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise Exception(f"Error reading depth from {svo_path}: {err}")

    runtime_params = sl.RuntimeParameters()
    depth_mat = sl.Mat()

    try:
        total_frames = zed.get_svo_number_of_frames()
    except Exception:
        total_frames = None

    frame_idx = 0
    while True:
        grab_err = zed.grab(runtime_params)
        if grab_err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if grab_err != sl.ERROR_CODE.SUCCESS:
            print(f"Warning: grab() failed at frame {frame_idx}: {grab_err}")
            continue

        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth_np = depth_mat.get_data()

        # ZED SDK sometimes returns (H, W, 1). Normalize to (H, W).
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[:, :, 0]

        depth_np = depth_np.astype(np.float32, copy=False)
        np.save(raw_depth_output_dir / f"{frame_idx:06d}.npy", depth_np)

        frame_idx += 1
        if frame_idx % 100 == 0:
            if total_frames is not None:
                print(f"  Saved raw depth {frame_idx}/{total_frames} frames...")
            else:
                print(f"  Saved raw depth {frame_idx} frames...")

    zed.close()
    print(f"Done! Saved {frame_idx} raw depth frames to {raw_depth_output_dir}")


def extract_stereo_frames(video_path, left_output_dir, right_output_dir):
    """
    Extract left and right frames from a stereo video (side-by-side format).
    Splits each frame horizontally into left and right images.
    """
    print(f"Processing {video_path.name}...")
    print(f"Left frames output: {left_output_dir}")
    print(f"Right frames output: {right_output_dir}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")

    # Check if width is even (required for splitting)
    if width % 2 != 0:
        print(f"Warning: Video width ({width}) is not even. Splitting may not work correctly.")

    half_width = width // 2

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split frame into left and right halves
        left_frame = frame[:, :half_width]  # Left half
        right_frame = frame[:, half_width:]  # Right half

        # Save left and right frames
        left_frame_path = left_output_dir / f"{frame_count:06d}.png"
        right_frame_path = right_output_dir / f"{frame_count:06d}.png"

        cv2.imwrite(str(left_frame_path), left_frame)
        cv2.imwrite(str(right_frame_path), right_frame)

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    cap.release()

    print(f"Done! Extracted {frame_count} left and right frame pairs.")


def extrinsics_to_matrix(extrinsics, format='3x4'):
    """
    Convert extrinsics to transformation matrix

    Args:
        extrinsics: [tx, ty, tz, rx, ry, rz] format extrinsics
        format: '3x4' returns [R|t] 3x4 matrix, '4x4' returns complete 4x4 transformation matrix

    Returns:
        3x4 or 4x4 transformation matrix
    """
    translation = extrinsics[:3]
    rotation_euler = extrinsics[3:]

    # Create rotation matrix from euler angles
    rotation_matrix = Rotation.from_euler('xyz', rotation_euler).as_matrix()

    if format == '3x4':
        # Build 3x4 [R|t] matrix
        transform = np.zeros((3, 4))
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        return transform
    else:
        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        return transform


def process_camera_data(input_dir, output_dir):
    """
    Process all camera data from the input directory.
    Extracts intrinsics, extrinsics, and stereo frames.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    print(f"Processing camera data from: {input_dir.absolute()}")

    # Create output base directory
    input_dir_name = input_dir.name
    output_dir = Path(output_dir)
    output_base = output_dir / input_dir_name
    output_base.mkdir(parents=True, exist_ok=True)

    # Read metadata to get camera serial numbers
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No metadata JSON file found in {input_dir}")

    with open(json_files[0], "r") as f:
        metadata = json.load(f)

    print(f"Found metadata file: {json_files[0].name}")

    # Camera name to serial mapping
    camera_names = ["ext1", "ext2", "wrist"]
    camera_serials = {}

    for camera_name in camera_names:
        serial_key = f"{camera_name}_cam_serial"
        if serial_key in metadata:
            camera_serials[camera_name] = metadata[serial_key]
            print(f"Camera {camera_name}: serial {metadata[serial_key]}")
        else:
            print(f"Warning: {serial_key} not found in metadata")

    # Process each camera
    for camera_name, camera_id in camera_serials.items():
        print(f"\n{'='*60}")
        print(f"Processing camera: {camera_name} (ID: {camera_id})")
        print(f"{'='*60}")

        # Create output directories for this camera
        camera_output_dir = output_base / str(camera_id)
        images_dir = camera_output_dir / "images"
        left_images_dir = images_dir / "left"
        right_images_dir = images_dir / "right"
        raw_depth_dir = camera_output_dir / "raw_depth"
        intrinsics_dir = camera_output_dir / "intrinsics"
        extrinsics_dir = camera_output_dir / "extrinsics"

        left_images_dir.mkdir(parents=True, exist_ok=True)
        right_images_dir.mkdir(parents=True, exist_ok=True)
        raw_depth_dir.mkdir(parents=True, exist_ok=True)
        intrinsics_dir.mkdir(parents=True, exist_ok=True)
        extrinsics_dir.mkdir(parents=True, exist_ok=True)

        # Create camera type identifier file
        if camera_name == "wrist":
            identifier_file = camera_output_dir / "wrist_camera.txt"
        elif camera_name == "ext1":
            identifier_file = camera_output_dir / "ext1.txt"
        elif camera_name == "ext2":
            identifier_file = camera_output_dir / "ext2.txt"
        else:
            # Fallback for unknown camera types
            identifier_file = camera_output_dir / f"{camera_name}_camera.txt"

        # Create an empty identifier file
        identifier_file.touch()
        print(f"Created camera identifier: {identifier_file}")

        # 1. Extract intrinsics from SVO file
        svo_dir = input_dir / "recordings" / "SVO"
        svo_file = svo_dir / f"{camera_id}.svo"

        if svo_file.exists():
            print(f"Extracting intrinsics from {svo_file.name}...")
            try:
                left_intrinsic, right_intrinsic = extract_intrinsics_from_svo(svo_file)

                # Save intrinsics
                left_intrinsic_path = intrinsics_dir / f"{camera_id}_left.npy"
                right_intrinsic_path = intrinsics_dir / f"{camera_id}_right.npy"

                np.save(left_intrinsic_path, left_intrinsic)
                np.save(right_intrinsic_path, right_intrinsic)

                print(f"Saved left intrinsic to {left_intrinsic_path}")
                print(f"Saved right intrinsic to {right_intrinsic_path}")

            except Exception as e:
                print(f"Error extracting intrinsics from {svo_file.name}: {e}")

            print(f"Extracting raw depth from {svo_file.name}...")
            try:
                extract_raw_depth_from_svo(svo_file, raw_depth_dir)
            except Exception as e:
                print(f"Error extracting raw depth from {svo_file.name}: {e}")
        else:
            print(f"SVO file not found: {svo_file}")

        # 2. Extract stereo frames from MP4 video
        mp4_dir = input_dir / "recordings" / "MP4"
        stereo_video = mp4_dir / f"{camera_id}-stereo.mp4"

        if stereo_video.exists():
            print(f"Extracting stereo frames from {stereo_video.name}...")
            try:
                extract_stereo_frames(stereo_video, left_images_dir, right_images_dir)
            except Exception as e:
                print(f"Error extracting stereo frames from {stereo_video.name}: {e}")
        else:
            print(f"Stereo video not found: {stereo_video}")

    # 3. Extract extrinsics from trajectory.h5 (this is done once for all cameras)
    h5_path = input_dir / "trajectory.h5"
    if h5_path.exists():
        print(f"\n{'='*60}")
        print("Extracting extrinsics from trajectory.h5")
        print(f"{'='*60}")

        try:
            with h5py.File(h5_path, "r") as traj:
                camera_extrinsics = traj["observation"]["camera_extrinsics"]

                for camera_name, camera_id in camera_serials.items():
                    left_key = f"{camera_id}_left"
                    right_key = f"{camera_id}_right"

                    extrinsics_dir = output_base / str(camera_id) / "extrinsics"

                    # Process left camera extrinsics
                    if left_key in camera_extrinsics:
                        left_extrinsics = np.array(camera_extrinsics[left_key])
                        num_frames = len(left_extrinsics)
                        extrinsics_3x4 = np.zeros((num_frames, 3, 4))

                        for frame_idx in range(num_frames):
                            extrinsics_3x4[frame_idx] = extrinsics_to_matrix(
                                left_extrinsics[frame_idx], format='3x4'
                            )

                        left_output_path = extrinsics_dir / f"{camera_id}_left.npy"
                        np.save(left_output_path, extrinsics_3x4)
                        print(f"Saved left extrinsics to {left_output_path} (shape: {extrinsics_3x4.shape})")

                    # Process right camera extrinsics
                    if right_key in camera_extrinsics:
                        right_extrinsics = np.array(camera_extrinsics[right_key])
                        num_frames = len(right_extrinsics)
                        extrinsics_3x4 = np.zeros((num_frames, 3, 4))

                        for frame_idx in range(num_frames):
                            extrinsics_3x4[frame_idx] = extrinsics_to_matrix(
                                right_extrinsics[frame_idx], format='3x4'
                            )

                        right_output_path = extrinsics_dir / f"{camera_id}_right.npy"
                        np.save(right_output_path, extrinsics_3x4)
                        print(f"Saved right extrinsics to {right_output_path} (shape: {extrinsics_3x4.shape})")

        except Exception as e:
            print(f"Error extracting extrinsics: {e}")
    else:
        print(f"Trajectory file not found: {h5_path}")

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Output saved to: {output_base.absolute()}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Process camera data: extract intrinsics, extrinsics, and stereo frames'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Input directory containing camera data (e.g., datasets/Sun_Jun_11_15:52:37_2023)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory containing camera data (e.g., datasets/samples/Sun_Jun_11_15:52:37_2023)'
    )

    args = parser.parse_args()

    try:
        process_camera_data(args.input_dir, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())