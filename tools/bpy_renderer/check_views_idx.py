import os
import numpy as np
import tqdm

# views_dir = "./views_tube"
# views_dir = "/share/volumes/csi/renchengwei/blender_renderings/views_tube"
# views_dir = "./views_tube_1"
# views_dir = "./views_tube_fixed_cam_1"
# views_dir = "./views_tube_fixed_cam"
views_dir = "./views_tube_fixed_cam_2"

for views_sub_dir in tqdm.tqdm(os.listdir(views_dir)):
    views_sub_dir_path = os.path.join(views_dir, views_sub_dir)
    for i in range(12):
        # check depth
        if not os.path.exists(os.path.join(views_sub_dir_path, f"{i:03d}_depth.exr")):
            print(f"Depth file {i:03d}_depth.exr not found")
            
        # check mask
        if not os.path.exists(os.path.join(views_sub_dir_path, f"{i:03d}_mask.png")):
            print(f"Mask file {i:03d}_mask.png not found")
        
        # check mask_visib
        if not os.path.exists(os.path.join(views_sub_dir_path, f"{i:03d}_mask_visib.png")):
            print(f"Mask_visib file {i:03d}_mask_visib.png not found")
            
        # check .png
        if not os.path.exists(os.path.join(views_sub_dir_path, f"{i:03d}.png")):
            print(f"PNG file {i:03d}.png not found")