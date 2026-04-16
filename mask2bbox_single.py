import numpy as np
import os
import cv2


mask_file = "/home/renchengwei/GDR-Net/test_repo/003_mask.png"
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
mask = mask / 255.0
mask = mask.astype(np.uint8)


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

bbox = mask_to_bbox(mask)
print("XYWH: {0} {1} {2} {3}".format(bbox[0], bbox[1], bbox[2], bbox[3]))
print("XYXY: {0} {1} {2} {3}".format(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))