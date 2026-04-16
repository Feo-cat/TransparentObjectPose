import os
import numpy as np
import json

split = "train"
# split = "test"
obj_id = 2
# obj_id = 1

# npzs_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/npz"
# save_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/scene_gt.json"
# npzs_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/npz"
# save_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/scene_gt.json"
npzs_dir = f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/{split}/{obj_id:06d}/npz"
save_path = f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/{split}/{obj_id:06d}/scene_gt.json"

save_dict = {}

# global_scale = 20.

scene_frame_num = 12


for npz_sub_dir in os.listdir(npzs_dir):
    for npz_file in os.listdir(os.path.join(npzs_dir, npz_sub_dir)):
        npz = np.load(os.path.join(npzs_dir, npz_sub_dir, npz_file))
        m2c_pose = npz["obj2cam_poses"]
        m2c_R = m2c_pose[:3, :3]
        # scale_x = np.linalg.norm(m2c_R[0, :])
        # scale_y = np.linalg.norm(m2c_R[1, :])
        # scale_z = np.linalg.norm(m2c_R[2, :])
        # scale = np.array([scale_x, scale_y, scale_z])
        # m2c_R = m2c_R / scale
        # m2c_t = m2c_pose[:3, 3] / global_scale
        m2c_t = m2c_pose[:3, 3]
        
        m2c_R_list = [m2c_R[0, 0], m2c_R[0, 1], m2c_R[0, 2], m2c_R[1, 0], m2c_R[1, 1], m2c_R[1, 2], m2c_R[2, 0], m2c_R[2, 1], m2c_R[2, 2]]
        m2c_t_list = [m2c_t[0], m2c_t[1], m2c_t[2]]
        folder_idx = int(npz_sub_dir)
        file_idx = int(npz_file.split(".")[0])
        # save_dict[id] = [{"cam_R_m2c": m2c_R_list, "cam_t_m2c": m2c_t_list, "obj_id": obj_id}]
        save_dict[f"{folder_idx:06d}_{file_idx:06d}"] = [{"cam_R_m2c": m2c_R_list, "cam_t_m2c": m2c_t_list, "obj_id": obj_id}]

# with open("m2c.json", "w") as f:
#     json.dump(save_dict, f)

# write each line
with open(save_path, "w") as f:
    f.write("{\n")
    for id in save_dict:
        if id == list(save_dict.keys())[-1]:
            f.write(f"\t\"{id}\": {json.dumps(save_dict[id])}\n")
        else:
            f.write(f"\t\"{id}\": {json.dumps(save_dict[id])},\n")
    f.write("}")