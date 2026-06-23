import argparse
import os
import tqdm
import traceback
import json
import numpy as np

DIFF_THRESHOLD = 1e-3

def warn(msg: str):
    print(f"[WARN] {msg}")

def replace_prefix(old_path: str, new_prefix: str) -> str:
    """
    old: data/real_0_0_255/episodes/episode0/xxx
    new: data/newprefix/episodes/episode0/xxx
    """
    old_path_norm = old_path.replace("\\", "/")
    new_root_norm = new_prefix.replace("\\", "/")

    old_parts = old_path_norm.split("/")
    new_root_parts = new_root_norm.split("/")

    if len(new_root_parts) < 2:
        raise ValueError("new_root_2levels must have at least two parts")

    old_parts[:2] = new_root_parts[:2]
    return "/".join(old_parts)

def average_value(k: str, values) -> float:
    values = list(values)
    if max(values) - min(values) > DIFF_THRESHOLD:
        warn(f"{k} difference too large: {values}")
    return sum(values) / len(values)

def check_equal(key: str, values):
    values = list(values)
    first_value = values[0]
    if any(v != first_value for v in values[1:]):
        warn(f"{key} mismatch: {values}")
    return first_value

def normalize_weights(num_inputs: int, weights=None):
    if weights is None:
        return [1.0 / num_inputs] * num_inputs
    if len(weights) != num_inputs:
        raise ValueError(f"Expected {num_inputs} weights, got {len(weights)}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return [float(w) / total for w in weights]

def safe_weighted_average_raw16(images, weights=None, white_level: int = 65535, black_level: int = 0) -> np.ndarray:
    images = list(images)
    if not images:
        raise ValueError("At least one image is required")

    base_shape = images[0].shape
    for idx, img in enumerate(images):
        if img.shape != base_shape:
            raise ValueError(f"Image shape mismatch at index {idx}: {img.shape} vs {base_shape}")
        if img.dtype != np.uint16:
            raise ValueError(f"Image at index {idx} must be uint16, got {img.dtype}")

    weights = normalize_weights(len(images), weights)
    img_out = np.zeros(base_shape, dtype=np.float32)
    for img, weight in zip(images, weights):
        img_out += img.astype(np.float32) * weight

    img_out = np.clip(img_out, black_level, white_level)
    return img_out.round().astype(np.uint16)

def safe_average_traj(traj_paths, traj_path_save: str):
    """
    Safely average multiple trajectory JSON files and save the result.

    Args:
        traj_paths (list[str]): Paths to trajectory JSON files.
        traj_path_save (str): Path to save the averaged trajectory JSON file.
    """
    trajectories = []
    for traj_path in traj_paths:
        with open(traj_path, 'r') as f:
            trajectories.append(json.load(f))

    traj_lengths = [len(traj) for traj in trajectories]
    if len(set(traj_lengths)) != 1:
        raise ValueError(f"Trajectory length mismatch: {traj_lengths}")
    

    averaged_traj = []
    for frames in zip(*trajectories):
        out = {}

        # ******************** State information ********************
        # ======== 1) state_index ========
        out["state_index"] = check_equal("state_index", [frame["state_index"] for frame in frames])
        # ======== 2) timestamp ========
        out["state_timestamp"] = frames[0]["state_timestamp"] # just take the first one
        # ======== 3) joint, pos values ========
        keys_to_avg = ["j1","j2","j3","j4","j5","j6","j7",
                       "x","y","z","roll","pitch","yaw"]
        for k in keys_to_avg:
            out[k] = average_value(k, [frame[k] for frame in frames])
        # ======== 4) gripper state ========
        gripper_states = [frame["gripper_state"] for frame in frames]
        gripper_state = {}
        gripper_state["type"] = check_equal("gripper_state.type", [gs["type"] for gs in gripper_states])
        gripper_state["pos"] = average_value("gripper_state.pos", [gs["pos"] for gs in gripper_states])
        out["gripper_state"] = gripper_state
        # ======== 5) image paths ======== replace first two prefix
        path_keys = ["rgb_0_path", "depth_0_path", "rgb_1_path", "depth_1_path"]

        for p in path_keys:
            old_path = frames[0][p]
            new_path = replace_prefix(old_path, traj_path_save)
            out[p] = new_path
        # ************************************************************
        # ******************** Action information ********************
        actions = [frame["action"] for frame in frames]
        action = {}

        action_avg_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        for k in action_avg_keys:
            action[k] = average_value(f"action.{k}", [action_item[k] for action_item in actions])

        # action.gripper_state
        action_gripper_states = [action_item["gripper_state"] for action_item in actions]
        ag = {}
        ag["type"] = check_equal("action.gripper_state.type", [ag_state["type"] for ag_state in action_gripper_states])
        ag["pos"] = average_value("action.gripper_state.pos", [ag_state["pos"] for ag_state in action_gripper_states])
        action["gripper_state"] = ag

        out["action"] = action

        averaged_traj.append(out)

    # ====== save json ======
    with open(traj_path_save, "w") as f:
        json.dump(averaged_traj, f, indent=4)

    print(f"[INFO] Averaged trajectory saved to {traj_path_save}")


def batch_curation(data_dirs, dir_save: str = None, stard_id: int = None, end_id: int = None, weights=None):
    """
    Args:
        data_dirs (list[str] | str): Dataset directories to fuse, or the first dataset directory
            when using the old two-input call style.
        dir_save (str): Path to save the curated dataset.
        stard_id (int): First episode id, inclusive.
        end_id (int): Last episode id, exclusive.
        weights (list[float] | None): Optional image fusion weights. Defaults to equal weights.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs, dir_save]
        dir_save = stard_id
        stard_id = end_id
        end_id = weights
        weights = None

    if not data_dirs or len(data_dirs) < 2:
        raise ValueError("At least two dataset directories are required")
    if dir_save is None or stard_id is None or end_id is None:
        raise ValueError("dir_save, stard_id, and end_id are required")

    weights = normalize_weights(len(data_dirs), weights)
    episodes_save_root = os.path.join(dir_save, 'episodes')
    os.makedirs(episodes_save_root, exist_ok=True)

    episodes_dirs = [os.path.join(data_dir, 'episodes') for data_dir in data_dirs]

    missing_episode_dirs = [episodes_dir for episodes_dir in episodes_dirs if not os.path.exists(episodes_dir)]
    if missing_episode_dirs:
        raise FileNotFoundError(f"Directory not found: {missing_episode_dirs}")
    
    # for episode_name in tqdm.tqdm(sorted(os.listdir(episodes_dirs[0])), desc="Processing episodes"):
    for id in tqdm.tqdm(range(stard_id, end_id), desc="Processing episodes"):
        episode_name = f"episode{id}"
        episode_dirs = [os.path.join(episodes_dir, episode_name) for episodes_dir in episodes_dirs]
        episode_dir_save = os.path.join(episodes_save_root, episode_name)
        os.makedirs(episode_dir_save, exist_ok=True)

        rgb0_dirs = [os.path.join(episode_dir, 'rgb_0') for episode_dir in episode_dirs]
        rgb0_dir_save = os.path.join(episode_dir_save, 'rgb_0')
        os.makedirs(rgb0_dir_save, exist_ok=True)

        traj_paths = [os.path.join(episode_dir, 'state_action.json') for episode_dir in episode_dirs]
        traj_path_save = os.path.join(episode_dir_save, 'state_action.json')

        # curate the trajectory and save to new directory
        safe_average_traj(traj_paths, traj_path_save)

        # curate rgb_0 images
        img_files1 = sorted([
            f for f in os.listdir(rgb0_dirs[0])
            if f.lower().endswith(".npy")
        ])

        for img_name in tqdm.tqdm(img_files1, desc=f"{episode_name}/rgb_0", leave=False):
            src_paths = [os.path.join(rgb0_dir, img_name) for rgb0_dir in rgb0_dirs]
            missing_paths = [src_path for src_path in src_paths if not os.path.exists(src_path)]
            if missing_paths:
                print(f"[Warning] Missing {img_name}: {missing_paths}")
                continue

            dst_path = os.path.join(rgb0_dir_save, img_name)
            if os.path.exists(dst_path):
                print(f"{dst_path} already exists, skip image {img_name}")
                continue

            try:
                imgs = [np.load(src_path) for src_path in src_paths]
                img_fused = safe_weighted_average_raw16(imgs, weights=weights)
                np.save(dst_path, img_fused)
            except Exception as e:
                print(f"[Error] Failed to process {img_name}: {e}")
                traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate aligned RoboLight datasets into a synthesized dataset."
    )
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        required=True,
        help="Input dataset roots to interpolate. Each root must contain an episodes/ directory.",
    )
    parser.add_argument(
        "--dir_save",
        required=True,
        help="Output dataset root for the synthesized dataset.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        required=True,
        help="Start episode ID, inclusive.",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        required=True,
        help="End episode ID, exclusive.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional interpolation weights. Defaults to equal weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_curation(
        data_dirs=args.data_dirs,
        dir_save=args.dir_save,
        stard_id=args.start_id,
        end_id=args.end_id,
        weights=args.weights,
    )