from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
import mmcv
import numpy as np

mask = mmcv.imread("/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/mask_visib/000004_000000.png", "unchanged")
# mask = mmcv.imread("/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/lm/train/000011/mask_visib/000002_000000.png", "unchanged")
mask = mask / 255.0
mask = mask.astype(np.uint8)
# mask = np.asfortranarray(mask)
print(mask.shape)
rle = binary_mask_to_rle(mask, compressed=True)
print(rle)

mask_2 = cocosegm2mask(rle, h=mask.shape[0], w=mask.shape[1])
print(mask_2.shape)
from PIL import Image
Image.fromarray(mask_2*255).save(f"/home/renchengwei/GDR-Net/debug/debug_rle_mask.png")