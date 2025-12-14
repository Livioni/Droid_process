import h5py
import numpy as np

def load_dataset_or_group(item, name):
    """加载数据集或递归处理组"""
    if isinstance(item, h5py.Dataset):
        try:
            return item[:]
        except TypeError:
            # 如果是标量数据集
            return item[()]
    elif isinstance(item, h5py.Group):
        # 如果是组，递归加载其内容
        group_data = {}
        for key in item.keys():
            group_data[key] = load_dataset_or_group(item[key], f"{name}/{key}")
        return group_data
    else:
        return None

def read_trajectory_data(file_path):
    """读取trajectory.h5文件中的所有数据"""
    data = {}

    with h5py.File(file_path, "r") as traj:
        print("=== Loading Trajectory Data ===")

        # 递归加载所有数据
        for key in traj.keys():
            print(f"Loading {key}...")
            data[key] = load_dataset_or_group(traj[key], key)

        # 打印一些关键信息
        if 'action' in data:
            if isinstance(data['action'], dict):
                print(f"Action is a group with keys: {list(data['action'].keys())}")
            elif hasattr(data['action'], 'shape'):
                print(f"Action shape: {data['action'].shape}, dtype: {data['action'].dtype}")

        if 'observation' in data and isinstance(data['observation'], dict):
            obs = data['observation']
            print(f"\nObservation keys: {list(obs.keys())}")

            if 'robot_state' in obs and isinstance(obs['robot_state'], dict):
                rs = obs['robot_state']
                print(f"Robot state keys: {list(rs.keys())}")

                # 显示一些关键数据集的信息
                for key in ['cartesian_position', 'gripper_position', 'joint_positions']:
                    if key in rs and hasattr(rs[key], 'shape'):
                        print(f"  {key}: shape={rs[key].shape}, dtype={rs[key].dtype}")

    return data

def inspect_trajectory_structure(file_path):
    """快速检查trajectory.h5文件的结构，不加载所有数据"""
    try:
        with h5py.File(file_path, "r") as traj:
            print("=== Trajectory File Structure ===")

            def print_structure(group, prefix=""):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        print(f"{prefix}{key}/ (Group)")
                        print_structure(item, prefix + "  ")
                    else:
                        print(f"{prefix}{key}: shape={item.shape}, dtype={item.dtype}")

            print_structure(traj)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Please update the file_path variable to point to your trajectory.h5 file.")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # 修改这个路径为你的trajectory.h5文件路径
    file_path = "datasets/Sun_Jun_11_15:52:37_2023/trajectory.h5"

    # 首先检查文件结构
    print("Checking file structure...")
    inspect_trajectory_structure(file_path)

    # 如果文件存在，则加载数据
    try:
        data = read_trajectory_data(file_path)
        print("\n=== Data Loading Complete ===")
        print(f"Total trajectory length: {len(data['cartesian_position']) if 'cartesian_position' in data and data['cartesian_position'] is not None else 'Unknown'}")
        print("All data has been loaded into memory. You can now process it further.")

        # 示例：访问action数据
        if 'action' in data and data['action'] is not None:
            if isinstance(data['action'], dict):
                print(f"\nAction is a group with {len(data['action'])} keys:")
                for key in data['action'].keys():
                    if hasattr(data['action'][key], 'shape'):
                        print(f"  {key}: shape={data['action'][key].shape}")
                    else:
                        print(f"  {key}: {type(data['action'][key])}")
            elif hasattr(data['action'], 'shape'):
                print(f"\nAction data shape: {data['action'].shape}")
                print(f"First action: {data['action'][0] if len(data['action']) > 0 else 'Empty'}")

        # 示例：访问机器人状态数据
        if 'observation' in data and isinstance(data['observation'], dict):
            if 'robot_state' in data['observation']:
                rs = data['observation']['robot_state']
                if 'cartesian_position' in rs:
                    cart_pos = rs['cartesian_position']
                    print(f"\nCartesian position shape: {cart_pos.shape}")
                    print(f"First few cartesian positions:\n{cart_pos[:3]}")
                    print(f"Total trajectory length: {len(cart_pos)} steps")

    except FileNotFoundError:
        print(f"\nFile {file_path} not found. Please:")
        print("1. Download or locate your trajectory.h5 file")
        print("2. Update the file_path variable above")
        print("3. Or place the file in the expected location")