import os
import json

# idx = range(1143)
# idx = range(388)
# idx = range(13)
# idx = range(2400)

split = "train"
# split = "test"
obj_id = 2
# obj_id = 1

scene_frame_num = 12

idx_num = 0
tmp_path = f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/{split}/{obj_id:06d}/rgb"
for rgb_sub_dir in os.listdir(tmp_path):
    for rgb_file in os.listdir(os.path.join(tmp_path, rgb_sub_dir)):
        idx_num += 1
idx = range(idx_num)
print(f"Index number: {idx_num}")

camera_gt_dict = {}

for i in idx:
    folder_idx = i // scene_frame_num
    file_idx = i % scene_frame_num
    camera_gt_dict[f"{folder_idx:06d}_{file_idx:06d}"] = {
        "cam_K": [700, 0.0, 320, 0.0, 700, 240, 0.0, 0.0, 1.0],
        "depth_scale": 1.0,
        "view_level": 1
    }

# dst_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/scene_camera.json"
# dst_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/scene_camera.json"
dst_path = f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/{split}/{obj_id:06d}/scene_camera.json"
with open(dst_path, "w") as f:
    f.write("{\n")
    for i in camera_gt_dict:
        if i == list(camera_gt_dict.keys())[-1]:
            f.write(f"\t\"{i}\": {json.dumps(camera_gt_dict[i])}\n")
        else:
            f.write(f"\t\"{i}\": {json.dumps(camera_gt_dict[i])},\n")
    f.write("}")