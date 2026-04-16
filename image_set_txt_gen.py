import os


# idx = range(1143)
# idx = range(388)
# idx = range(2400)
# idx = range(12)
# split = "test"
# split = "train"
# obj_id = 2
# obj_id = 1
splits = ["train", "test"]
# obj_ids = [1, 2]
obj_ids = [2]

id2obj = {
    1: "test_tube_rack",
    2: "tube",
}

scene_frame_num = 12



for split in splits:
    for obj_id in obj_ids:
        idx_num = 0
        tmp_path = f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/{split}/{obj_id:06d}/rgb"
        for rgb_sub_dir in os.listdir(tmp_path):
            for rgb_file in os.listdir(os.path.join(tmp_path, rgb_sub_dir)):
                idx_num += 1
        print(f"Index number: {idx_num}")
        idx = range(idx_num)

        with open(f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/image_set/{id2obj[obj_id]}_{split}.txt", "w") as f:
        # with open(f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/image_set/tube_{split}.txt", "w") as f:
        # with open(f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/image_set/tube_{split}.txt", "w") as f:
            for i in idx:
                folder_idx = i // scene_frame_num
                file_idx = i % scene_frame_num
                # f.write(f"{folder_idx:06d}_{file_idx:06d}\n")
                if file_idx == 0:
                    f.write(f"{folder_idx:06d}_000000\n")