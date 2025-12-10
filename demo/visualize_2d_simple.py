import argparse
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
import torch._dynamo
from s2m2.core.utils.model_utils import load_model, run_stereo_matching
from s2m2.core.utils.image_utils import read_images
from s2m2.core.utils.vis_utils import visualize_stereo_results_2d
import matplotlib.pyplot as plt

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch._dynamo.config.verbose = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='XL', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=3, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--torch_compile', action='store_true', help='apply torch_compile')
    parser.add_argument('--allow_negative', action='store_true', help='allow negative disparity for imperfect rectification')
    parser.add_argument('--focal_length', default=733.300171, type=float,
                        help='focal length in pixels for depth calculation')
    parser.add_argument('--baseline', default=130.0, type=float,
                        help='baseline distance in mm for depth calculation')
    parser.add_argument('--doffs', default=11.784, type=float,
                        help='disparity offset for depth calculation')
    return parser


def main(args):
    # load stereo model
    model = load_model(os.path.join(project_root, "weights/pretrain_weights"), args.model_type, not args.allow_negative, args.num_refine, device)
    if args.torch_compile:
        model = torch.compile(model)

    # load web stereo images
    if args.allow_negative:
        left_path = 'Web/64648_pbz98_3D_MPO_70pc_L.jpg'
        right_path = 'Web/64648_pbz98_3D_MPO_70pc_R.jpg'
    else:
        left_path = '/mnt/disk3.8-5/phs_github/s2m2/datasets/l-r_images/17368348-stereo_left/0.png'
        right_path = '/mnt/disk3.8-5/phs_github/s2m2/datasets/l-r_images/17368348-stereo_right/0.png'

    left_path = os.path.join(project_root, "data", "samples", left_path)
    right_path = os.path.join(project_root, "data", "samples", right_path)

    # load stereo images
    left, right = read_images(left_path, right_path)

    img_height, img_width = left.shape[:2]
    print(f"original image size: img_height({img_height}), img_width({img_width})")

    # img_height = (img_height // 32) * 32
    # img_width = (img_width // 32) * 32
    # print(f"cropped image size: img_height({img_height}), img_width({img_width})")

    # # image crop
    # left = left[:img_height, :img_width]
    # right = right[:img_height, :img_width]

    # to torch tensor
    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).to(device)

    # run stereo matching
    _ = run_stereo_matching(model, left_torch, right_torch, device) #pre-run
    pred_disp, pred_occ, pred_conf, avg_conf_score, avg_run_time = run_stereo_matching(model, left_torch, right_torch, device, N_repeat=5)
    print(F"torch avg inference time:{(avg_run_time)/1000}, FPS:{1000/(avg_run_time)}")

    # opencv 2D visualization
    pred_disp, pred_occ, pred_conf = pred_disp.cpu().numpy(), pred_occ.cpu().numpy(), pred_conf.cpu().numpy()
    visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf)

    # calculate and save depth map
    depth_map = calculate_depth_from_disparity(pred_disp, args.focal_length, args.baseline, args.doffs)

    # filter depth values where confidence is less than 10%
    confidence_threshold = 0.4
    depth_map[pred_conf < confidence_threshold] = 0

    save_depth_visualization(depth_map)



def calculate_depth_from_disparity(disparity, focal_length, baseline, doffs=0):
    """
    Calculate depth map from disparity map using the formula:
    depth = (focal_length * baseline) / (disparity + doffs)

    Args:
        disparity (np.ndarray): disparity map [H, W]
        focal_length (float): focal length in pixels
        baseline (float): baseline distance in mm
        doffs (float): disparity offset

    Returns:
        depth (np.ndarray): depth map in mm [H, W]
    """
    # Avoid division by zero
    disparity = np.where(disparity <= 0, 1e-6, disparity)

    # Calculate depth: depth = (focal_length * baseline) / (disparity + doffs)
    depth = (focal_length * baseline) / (disparity + doffs)

    # Set invalid depths to a large value
    depth = np.where(disparity <= 1e-6, 1e9, depth)

    return depth


def save_depth_visualization(depth_map):
    """
    Save depth map visualization

    Args:
        depth_map (np.ndarray): depth map in mm [H, W]
    """
    import os

    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create depth visualization
    depth_vis = depth_map.copy()

    # Create a mask for valid depth values (non-zero)
    valid_mask = depth_vis > 0

    if np.any(valid_mask):
        # Clip depth values for better visualization (e.g., 0.1m to 10m)
        depth_vis_clipped = np.clip(depth_vis, 100, 10000)  # 100mm to 10000mm

        # Apply logarithmic scaling for better visualization
        depth_vis_log = np.log(depth_vis_clipped)

        # Normalize to 0-255 range using only valid values
        min_val = np.min(depth_vis_log[valid_mask])
        max_val = np.max(depth_vis_log[valid_mask])
        depth_vis_norm = np.zeros_like(depth_vis_log, dtype=np.uint8)
        depth_vis_norm[valid_mask] = ((depth_vis_log[valid_mask] - min_val) / (max_val - min_val)) * 255
        depth_vis = depth_vis_norm.astype(np.uint8)
    else:
        # If no valid depths, create black image
        depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # Save the depth visualization
    output_path = os.path.join(output_dir, 'depth_map.png')
    cv2.imwrite(output_path, depth_colored)

    # Also save raw depth data as numpy array for further processing
    np.save(os.path.join(output_dir, 'depth_map.npy'), depth_map)

    print(f"Depth map saved to {output_path}")

    # Calculate statistics for valid depths (non-zero after confidence filtering)
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) > 0:
        print(f"Depth range: {valid_depths.min():.3f}mm - {valid_depths.max():.3f}mm")
        print(f"Mean depth: {valid_depths.mean():.3f}mm")
    else:
        print("No valid depths after confidence filtering")

    print(f"Valid depth pixels after confidence filtering: {np.sum(depth_map > 0)} / {depth_map.size} "
          f"({100 * np.sum(depth_map > 0) / depth_map.size:.1f}%)")


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)
