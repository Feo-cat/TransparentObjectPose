import numpy as np
import os
import cv2
import json

def mask_to_bbox(mask):
    """
    mask: H x W 的二值数组，前景为 True 或 1，背景为 False 或 0
    return: bbox = [xmin, ymin, xmax, ymax]
    """
    ys, xs = np.where(mask)  # 找到所有前景像素的坐标
    if len(xs) == 0 or len(ys) == 0:
        return None  # 没有前景
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    # return [int(xmin), int(ymin), int(xmax), int(ymax)]
    return [int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin)]



# split = "train"
# split = "test"
# obj_id = 2
# obj_id = 1
splits = ["train", "test"]
# obj_ids = [1, 2]
obj_ids = [2]



for split in splits:
    for obj_id in obj_ids:
        # masks_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/mask_visib"
        # dst_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/scene_gt_info.json"
        # masks_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/mask_visib"
        # dst_dir = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/scene_gt_info.json"
        masks_dir = f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/{split}/{obj_id:06d}/mask_visib"
        dst_dir = f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/{split}/{obj_id:06d}/scene_gt_info.json"

        # for debug
        # for mask_sub_dir in os.listdir(masks_dir):
        #     for mask_file in os.listdir(os.path.join(masks_dir, mask_sub_dir)):
        #         mask = cv2.imread(os.path.join(masks_dir, mask_sub_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        #         bbox = mask_to_bbox(mask)
        #         print(bbox)
            
        # save like this:
        # {
        #   "0": [{"bbox_obj": [274, 188, 99, 106], "bbox_visib": [274, 188, 99, 106], "px_count_all": 7067, "px_count_valid": 7067, "px_count_visib": 7067, "visib_fract": 1.0}],
        #   "1": [{"bbox_obj": [267, 193, 107, 88], "bbox_visib": [267, 193, 107, 88], "px_count_all": 7184, "px_count_valid": 7184, "px_count_visib": 7184, "visib_fract": 1.0}],
        #   "2": [{"bbox_obj": [276, 191, 100, 106], "bbox_visib": [276, 191, 100, 106], "px_count_all": 7084, "px_count_valid": 7084, "px_count_visib": 7084, "visib_fract": 1.0}],
        #   "3": [{"bbox_obj": [278, 187, 94, 103], "bbox_visib": [278, 187, 94, 103], "px_count_all": 7025, "px_count_valid": 7025, "px_count_visib": 7025, "visib_fract": 1.0}],
        #   "4": [{"bbox_obj": [276, 208, 106, 87], "bbox_visib": [276, 208, 106, 87], "px_count_all": 7114, "px_count_valid": 7114, "px_count_visib": 7114, "visib_fract": 1.0}],
        #   "5": [{"bbox_obj": [274, 189, 99, 107], "bbox_visib": [274, 189, 99, 107], "px_count_all": 7154, "px_count_valid": 7154, "px_count_visib": 7154, "visib_fract": 1.0}],
        #   "6": [{"bbox_obj": [278, 198, 93, 102], "bbox_visib": [278, 198, 93, 102], "px_count_all": 7184, "px_count_valid": 7184, "px_count_visib": 7184, "visib_fract": 1.0}],
        #   "7": [{"bbox_obj": [272, 194, 107, 104], "bbox_visib": [272, 194, 107, 104], "px_count_all": 7331, "px_count_valid": 7331, "px_count_visib": 7331, "visib_fract": 1.0}],
        #   "8": [{"bbox_obj": [267, 195, 107, 90], "bbox_visib": [267, 195, 107, 90], "px_count_all": 7338, "px_count_valid": 7338, "px_count_visib": 7338, "visib_fract": 1.0}],
        #   "9": [{"bbox_obj": [269, 199, 102, 95], "bbox_visib": [269, 199, 102, 95], "px_count_all": 7209, "px_count_valid": 7209, "px_count_visib": 7209, "visib_fract": 1.0}],
        #   "10": [{"bbox_obj": [276, 193, 100, 105], "bbox_visib": [276, 193, 100, 105], "px_count_all": 7150, "px_count_valid": 7150, "px_count_visib": 7150, "visib_fract": 1.0}],
        #   "11": [{"bbox_obj": [288, 188, 88, 105], "bbox_visib": [288, 188, 88, 105], "px_count_all": 7031, "px_count_valid": 7031, "px_count_visib": 7031, "visib_fract": 1.0}]
        # }
        
        save_dict = {}
        for mask_sub_dir in os.listdir(masks_dir):
            for mask_file in os.listdir(os.path.join(masks_dir, mask_sub_dir)):
                mask = cv2.imread(os.path.join(masks_dir, mask_sub_dir, mask_file), cv2.IMREAD_GRAYSCALE)
                bbox = mask_to_bbox(mask)
                folder_idx = int(mask_sub_dir)
                file_idx = int(mask_file.split("_")[0])
                save_list = []
                save_list.append({"bbox_obj": bbox, "bbox_visib": bbox, "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0})
                print(save_list)
                save_dict[f"{folder_idx:06d}_{file_idx:06d}"] = save_list
        
        # with open("bbox.json", "w") as f:
        #     # change line
        #     json.dump(save_dict, f)
        
        # write each line
        with open(dst_dir, "w") as f:
            f.write("{\n")
            for id in save_dict:
                if id == list(save_dict.keys())[-1]:
                    f.write(f"\t\"{id}\": {json.dumps(save_dict[id])}\n")
                else:
                    f.write(f"\t\"{id}\": {json.dumps(save_dict[id])},\n")
            f.write("}")