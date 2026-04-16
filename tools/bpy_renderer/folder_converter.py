import os
import numpy as np
import shutil
import tqdm

# folder_path = "/data/user/rencw/ICL-I2PReg/views/plane_furn__test_tube_rack_mass_chalice"
# dst_folder_path = "/data/user/rencw/ICL-I2PReg/dst_folder/plane_furn__test_tube_rack_mass_chalice"
src_folder_path = "/share/volumes/csi/renchengwei/blender_renderings/views_tube_vid"
dst_folder = "/share/volumes/csi/renchengwei/blender_renderings/dst_folder_tube_vid"
os.makedirs(dst_folder, exist_ok=True)

folder_path_list = os.listdir(src_folder_path)
dst_folder_path_list = []

for folder_path in folder_path_list:
    each_dst_folder_path = os.path.join(dst_folder, folder_path)
    dst_folder_path_list.append(each_dst_folder_path)

    
src_folder_path_list = []

for folder_path in folder_path_list:
    each_src_folder_path = os.path.join(src_folder_path, folder_path)
    src_folder_path_list.append(each_src_folder_path)

for src_folder_path, dst_folder_path in tqdm.tqdm(zip(src_folder_path_list, dst_folder_path_list)):
    idx_list = range(0, 12)
    
    # mkdir dst_folder_path
    os.makedirs(dst_folder_path, exist_ok=True)
    os.makedirs(os.path.join(dst_folder_path, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder_path, "mask_visib"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(dst_folder_path, "npz"), exist_ok=True)
    
    for idx in idx_list:
        rgb_path = os.path.join(src_folder_path, f"{idx:03d}.png")
        mask_path = os.path.join(src_folder_path, f"{idx:03d}_mask.png")
        # mask_visib_path = os.path.join(src_folder_path, f"{idx:03d}_mask.png")
        mask_visib_path = os.path.join(src_folder_path, f"{idx:03d}_mask_visib.png")
        depth_path = os.path.join(src_folder_path, f"{idx:03d}_depth.npy")
        npz_path = os.path.join(src_folder_path, f"{idx:03d}.npz")
        
        if os.path.exists(rgb_path):
            shutil.copy(rgb_path, os.path.join(dst_folder_path, "rgb", f"{idx:06d}.png"))
        else:
            print(f"RGB path not found: {rgb_path}")
        if os.path.exists(mask_path):
            shutil.copy(mask_path, os.path.join(dst_folder_path, "mask", f"{idx:06d}_000000.png"))
        else:
            print(f"Mask path not found: {mask_path}")
        if os.path.exists(mask_visib_path):
            shutil.copy(mask_visib_path, os.path.join(dst_folder_path, "mask_visib", f"{idx:06d}_000000.png"))
        else:
            print(f"Mask visib path not found: {mask_visib_path}")
        if os.path.exists(depth_path):
            shutil.copy(depth_path, os.path.join(dst_folder_path, "depth", f"{idx:06d}.npy"))
        else:
            print(f"Depth path not found: {depth_path}")
        if os.path.exists(npz_path):
            shutil.copy(npz_path, os.path.join(dst_folder_path, "npz", f"{idx:06d}.npz"))
        else:
            print(f"NPZ path not found: {npz_path}")
    