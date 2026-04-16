import os
import shutil

src_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim_v1"
dst_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim"

sub_num = len(os.listdir(os.path.join(src_path, "train"))) - 1 # exclude the xyz_crop folder

scene_frame_num = 12

# split = "train"
split = "test"


# train split
for i in range(1, sub_num + 1):
    src_scene_path = os.path.join(src_path, split, f"{i:06d}")
    dst_scene_path = os.path.join(dst_path, split, f"{i:06d}")
    os.makedirs(dst_scene_path, exist_ok=True)
    # depth
    src_depth_path = os.path.join(src_scene_path, "depth")
    dst_depth_path = os.path.join(dst_scene_path, "depth")
    os.makedirs(dst_depth_path, exist_ok=True)
    for depth_file in os.listdir(src_depth_path):
        depth_idx = int(depth_file.split(".")[0])
        depth_folder_idx = depth_idx // scene_frame_num
        depth_file_idx = depth_idx % scene_frame_num
        os.makedirs(os.path.join(dst_depth_path, f"{depth_folder_idx:06d}"), exist_ok=True)
        print(f"Copying {os.path.join(src_depth_path, depth_file)} to {os.path.join(dst_depth_path, f'{depth_folder_idx:06d}', f'{depth_file_idx:06d}.npy')}")
        shutil.copy(os.path.join(src_depth_path, depth_file), os.path.join(dst_depth_path, f"{depth_folder_idx:06d}", f"{depth_file_idx:06d}.npy"))
    
    # mask
    src_mask_path = os.path.join(src_scene_path, "mask")
    dst_mask_path = os.path.join(dst_scene_path, "mask")
    os.makedirs(dst_mask_path, exist_ok=True)
    for mask_file in os.listdir(src_mask_path):
        mask_idx = int(mask_file.split("_")[0])
        mask_folder_idx = mask_idx // scene_frame_num
        mask_file_idx = mask_idx % scene_frame_num
        os.makedirs(os.path.join(dst_mask_path, f"{mask_folder_idx:06d}"), exist_ok=True)
        print(f"Copying {os.path.join(src_mask_path, mask_file)} to {os.path.join(dst_mask_path, f'{mask_folder_idx:06d}', f'{mask_file_idx:06d}_000000.png')}")
        shutil.copy(os.path.join(src_mask_path, mask_file), os.path.join(dst_mask_path, f"{mask_folder_idx:06d}", f"{mask_file_idx:06d}_000000.png"))
    
    
    # mask_visib
    src_mask_visib_path = os.path.join(src_scene_path, "mask_visib")
    dst_mask_visib_path = os.path.join(dst_scene_path, "mask_visib")
    os.makedirs(dst_mask_visib_path, exist_ok=True)
    for mask_visib_file in os.listdir(src_mask_visib_path):
        mask_visib_idx = int(mask_visib_file.split("_")[0])
        mask_visib_folder_idx = mask_visib_idx // scene_frame_num
        mask_visib_file_idx = mask_visib_idx % scene_frame_num
        os.makedirs(os.path.join(dst_mask_visib_path, f"{mask_visib_folder_idx:06d}"), exist_ok=True)
        print(f"Copying {os.path.join(src_mask_visib_path, mask_visib_file)} to {os.path.join(dst_mask_visib_path, f'{mask_visib_folder_idx:06d}', f'{mask_visib_file_idx:06d}_000000.png')}")
        shutil.copy(os.path.join(src_mask_visib_path, mask_visib_file), os.path.join(dst_mask_visib_path, f"{mask_visib_folder_idx:06d}", f"{mask_visib_file_idx:06d}_000000.png"))
    
    # npz
    src_npz_path = os.path.join(src_scene_path, "npz")
    dst_npz_path = os.path.join(dst_scene_path, "npz")
    os.makedirs(dst_npz_path, exist_ok=True)
    for npz_file in os.listdir(src_npz_path):
        npz_idx = int(npz_file.split(".")[0])
        npz_folder_idx = npz_idx // scene_frame_num
        npz_file_idx = npz_idx % scene_frame_num
        os.makedirs(os.path.join(dst_npz_path, f"{npz_folder_idx:06d}"), exist_ok=True)
        print(f"Copying {os.path.join(src_npz_path, npz_file)} to {os.path.join(dst_npz_path, f'{npz_folder_idx:06d}', f'{npz_file_idx:06d}.npz')}")
        shutil.copy(os.path.join(src_npz_path, npz_file), os.path.join(dst_npz_path, f"{npz_folder_idx:06d}", f"{npz_file_idx:06d}.npz"))
        
    # rgb
    src_rgb_path = os.path.join(src_scene_path, "rgb")
    dst_rgb_path = os.path.join(dst_scene_path, "rgb")
    os.makedirs(dst_rgb_path, exist_ok=True)
    for rgb_file in os.listdir(src_rgb_path):
        rgb_idx = int(rgb_file.split(".")[0])
        rgb_folder_idx = rgb_idx // scene_frame_num
        rgb_file_idx = rgb_idx % scene_frame_num
        os.makedirs(os.path.join(dst_rgb_path, f"{rgb_folder_idx:06d}"), exist_ok=True)
        print(f"Copying {os.path.join(src_rgb_path, rgb_file)} to {os.path.join(dst_rgb_path, f'{rgb_folder_idx:06d}', f'{rgb_file_idx:06d}.png')}")
        shutil.copy(os.path.join(src_rgb_path, rgb_file), os.path.join(dst_rgb_path, f"{rgb_folder_idx:06d}", f"{rgb_file_idx:06d}.png"))
    
    # xyz_crop
    src_xyz_crop_path = os.path.join(src_path, split, "xyz_crop", f"{i:06d}")
    dst_xyz_crop_path = os.path.join(dst_path, split, "xyz_crop", f"{i:06d}")
    os.makedirs(dst_xyz_crop_path, exist_ok=True)
    for xyz_crop_file in os.listdir(src_xyz_crop_path):
        xyz_crop_idx = int(xyz_crop_file.split("_")[0])
        xyz_crop_folder_idx = xyz_crop_idx // scene_frame_num
        xyz_crop_file_idx = xyz_crop_idx % scene_frame_num
        os.makedirs(os.path.join(dst_xyz_crop_path, f"{xyz_crop_folder_idx:06d}"), exist_ok=True)
        print(f"Copying {os.path.join(src_xyz_crop_path, xyz_crop_file)} to {os.path.join(dst_xyz_crop_path, f'{xyz_crop_folder_idx:06d}', f'{xyz_crop_file_idx:06d}_000000.pkl')}")
        shutil.copy(os.path.join(src_xyz_crop_path, xyz_crop_file), os.path.join(dst_xyz_crop_path, f"{xyz_crop_folder_idx:06d}", f"{xyz_crop_file_idx:06d}_000000.pkl"))

