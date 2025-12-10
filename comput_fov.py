import numpy as np  
import cv2

def compute_fov(image_path, fx, fy):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))
    return fov_x, fov_y

if __name__ == "__main__":
    image_path = "datasets/samples/17368348/images/0.png"
    intrinsic = np.load("datasets/Sun_Jun_11_15:52:37_2023/intrinsics/17368348_left_intrinsic.npy")
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    fov_x, fov_y = compute_fov(image_path, fx, fy)
    fov_x_deg = np.degrees(fov_x)
    fov_y_deg = np.degrees(fov_y)
    print(f"FOV_X: {fov_x_deg}, FOV_Y: {fov_y_deg}")