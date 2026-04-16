import os
import numpy as np
import shutil
import tqdm


# src_folder_path = "/data/user/rencw/ICL-I2PReg/dst_folder"
# dst_folder_path = "/data/user/rencw/ICL-I2PReg/dst_folder_full"
src_folder_path = "/share/volumes/csi/renchengwei/blender_renderings/dst_folder_tube_vid"
dst_folder_path = "/share/volumes/csi/renchengwei/blender_renderings/dst_folder_tube_full_vid"

os.makedirs(dst_folder_path, exist_ok=True)

os.makedirs(os.path.join(dst_folder_path, "depth"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "mask"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "mask_visib"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "npz"), exist_ok=True)
os.makedirs(os.path.join(dst_folder_path, "rgb"), exist_ok=True)


# depth
global_cnt = 0
global_scale = 20.
# move
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "depth")
    shutil.copytree(each_src_folder_path, os.path.join(dst_folder_path, "depth", f"{global_cnt:06d}"))
    global_cnt += 1
# post-process
for each_dst_folder_path in tqdm.tqdm(os.listdir(os.path.join(dst_folder_path, "depth"))):
    each_dst_folder_path = os.path.join(dst_folder_path, "depth", each_dst_folder_path)
    for each_depth_file in os.listdir(each_dst_folder_path):
        each_depth_file_path = os.path.join(each_dst_folder_path, each_depth_file)
        depth = np.load(each_depth_file_path)
        depth = depth.astype(np.float32)
        depth = depth / global_scale
        np.save(each_depth_file_path, depth)


# mask
global_cnt = 0
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "mask")
    shutil.copytree(each_src_folder_path, os.path.join(dst_folder_path, "mask", f"{global_cnt:06d}"))
    global_cnt += 1



# mask_visib
global_cnt = 0
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "mask_visib")
    shutil.copytree(each_src_folder_path, os.path.join(dst_folder_path, "mask_visib", f"{global_cnt:06d}"))
    global_cnt += 1



# npz
global_cnt = 0
idx_max = 12
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "npz")
    os.makedirs(os.path.join(dst_folder_path, "npz", f"{global_cnt:06d}"), exist_ok=True)
    for i in range(idx_max):
        each_npz_path = os.path.join(each_src_folder_path, f"{i:06d}.npz")
        if os.path.exists(each_npz_path):
            shutil.copy(each_npz_path, os.path.join(dst_folder_path, "npz", f"{global_cnt:06d}", f"{i:06d}.npz"))
            # post-process
            npz_data = np.load(each_npz_path)
            pose = npz_data["obj2cam_poses"].copy()
            pose_scale_x = np.linalg.norm(pose[0, :3])
            pose_scale_y = np.linalg.norm(pose[1, :3])
            pose_scale_z = np.linalg.norm(pose[2, :3])
            pose[:3, :3] = pose[:3, :3] / np.array([pose_scale_x, pose_scale_y, pose_scale_z])
            pose[:3, 3] = pose[:3, 3] / global_scale
            new_dict = {
                "table": npz_data["table"],
                "objects": npz_data["objects"],
                "K": npz_data["K"],
                "RT": npz_data["RT"],
                "obj2cam_poses": pose,
                "azimuth": npz_data["azimuth"],
                "elevation": npz_data["elevation"],
                "distance": npz_data["distance"],
            }
            np.savez(os.path.join(dst_folder_path, "npz", f"{global_cnt:06d}", f"{i:06d}.npz"), **new_dict)
        else:
            print(f"NPZ path not found: {each_npz_path}")
    global_cnt += 1


# rgb
global_cnt = 0
for each_src_folder_path in tqdm.tqdm(os.listdir(src_folder_path)):
    each_src_folder_path = os.path.join(src_folder_path, each_src_folder_path, "rgb")
    shutil.copytree(each_src_folder_path, os.path.join(dst_folder_path, "rgb", f"{global_cnt:06d}"))
    global_cnt += 1