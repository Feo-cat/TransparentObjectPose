import os
import numpy as np

# depth_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/depth"
# save_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/depth"
# depth_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/test/000001/depth"
# save_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/test/000001/depth"
depth_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/depth"
save_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/depth"

global_scale = 20.

for depth_file in os.listdir(depth_dir):
    depth = np.load(os.path.join(depth_dir, depth_file))
    depth = depth / global_scale
    np.save(os.path.join(save_dir, depth_file), depth)