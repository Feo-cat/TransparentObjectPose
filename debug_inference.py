#!/usr/bin/env python3
"""
Debug script to compare training and inference data processing.
Run this to check if the preprocessing is consistent.
"""

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np
import torch

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)

from mmcv import Config
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T

from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.utils.data_utils import (
    read_image_cv2,
    crop_resize_by_warp_affine,
    get_2d_coord_np,
)
import ref
from lib.pysixd import inout


def debug_preprocess(cfg, image, bbox_xyxy, obj_cls, dataset_name):
    """Debug preprocessing to print intermediate values."""
    
    print("\n" + "="*60)
    print("DEBUG: Preprocessing Steps")
    print("="*60)
    
    im_H_ori, im_W_ori = image.shape[:2]
    print(f"1. Original image shape: {image.shape} (H={im_H_ori}, W={im_W_ori})")
    print(f"   Image dtype: {image.dtype}, min={image.min()}, max={image.max()}")
    
    # Apply augmentation (mainly resize)
    augmentation = []
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    print(f"\n2. Resize config: MIN_SIZE_TEST={min_size}, MAX_SIZE_TEST={max_size}")
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, "choice"))
    
    image_transformed, transforms = T.apply_augmentations(augmentation, image.copy())
    im_H, im_W = image_transformed.shape[:2]
    print(f"   After resize: H={im_H}, W={im_W}")
    print(f"   Is divisible by 16? H%16={im_H % 16}, W%16={im_W % 16}")
    
    # Transform bbox
    bbox_xyxy = np.array(bbox_xyxy, dtype=np.float32)
    print(f"\n3. Original bbox: {bbox_xyxy}")
    bbox_xyxy_transformed = transforms.apply_box([bbox_xyxy])[0]
    print(f"   Transformed bbox: {bbox_xyxy_transformed}")
    
    # Calculate bbox center and scale
    patch_size = 16
    x1, y1, x2, y2 = bbox_xyxy_transformed
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    print(f"\n4. Bbox center: ({cx:.2f}, {cy:.2f}), bw={bw:.2f}, bh={bh:.2f}")
    
    # Calculate scale
    scale = max(bh, bw) * cfg.INPUT.DZI_PAD_SCALE
    scale = min(scale, max(im_H, im_W)) * 1.0
    print(f"   DZI_PAD_SCALE={cfg.INPUT.DZI_PAD_SCALE}")
    print(f"   Initial scale: {scale:.2f}")
    
    # Patch alignment
    bbox_center = np.array([cx, cy])
    xmin = bbox_center[0] - scale / 2
    ymin = bbox_center[1] - scale / 2
    xmax = bbox_center[0] + scale / 2
    ymax = bbox_center[1] + scale / 2
    
    px_min = np.floor(xmin / patch_size)
    py_min = np.floor(ymin / patch_size)
    px_max = np.ceil(xmax / patch_size)
    py_max = np.ceil(ymax / patch_size)
    
    pw = px_max - px_min
    ph = py_max - py_min
    p = max(pw, ph)
    if p % 2 == 1:
        p += 1
    
    cx_p = (px_min + px_max) / 2
    cy_p = (py_min + py_max) / 2
    px_min = cx_p - p / 2
    px_max = cx_p + p / 2
    py_min = cy_p - p / 2
    py_max = cy_p + p / 2
    
    xmin = px_min * patch_size
    xmax = px_max * patch_size
    ymin = py_min * patch_size
    ymax = py_max * patch_size
    
    bbox_center = np.array([
        (xmin + xmax) / 2,
        (ymin + ymax) / 2
    ], dtype=np.float32)
    
    scale = float(xmax - xmin)
    print(f"\n5. After patch alignment:")
    print(f"   Aligned bbox_center: ({bbox_center[0]:.2f}, {bbox_center[1]:.2f})")
    print(f"   Aligned scale: {scale:.2f}")
    print(f"   Patch range: px=[{px_min:.0f}, {px_max:.0f}), py=[{py_min:.0f}, {py_max:.0f})")
    
    # ROI image
    input_res = cfg.MODEL.CDPN.BACKBONE.INPUT_RES
    out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES
    print(f"\n6. Model resolution: INPUT_RES={input_res}, OUTPUT_RES={out_res}")
    
    roi_img = crop_resize_by_warp_affine(
        image_transformed, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
    ).transpose(2, 0, 1)
    print(f"   ROI image shape: {roi_img.shape}")
    print(f"   ROI image range: [{roi_img.min():.2f}, {roi_img.max():.2f}]")
    
    # Normalize ROI
    pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN).reshape(-1, 1, 1)
    pixel_std = np.array(cfg.MODEL.PIXEL_STD).reshape(-1, 1, 1)
    print(f"\n7. ROI normalization:")
    print(f"   PIXEL_MEAN: {cfg.MODEL.PIXEL_MEAN}")
    print(f"   PIXEL_STD: {cfg.MODEL.PIXEL_STD}")
    roi_img_norm = (roi_img - pixel_mean) / pixel_std
    print(f"   Normalized ROI range: [{roi_img_norm.min():.4f}, {roi_img_norm.max():.4f}]")
    
    # Input image for DINO
    print(f"\n8. Input image for DINO encoder:")
    # BGR to RGB conversion
    input_image = cv2.cvtColor(image_transformed, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(f"   After BGR->RGB conversion, shape: {input_image.shape}")
    input_image = input_image.transpose(2, 0, 1)
    input_image = input_image / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    input_image = (input_image - imagenet_mean) / imagenet_std
    print(f"   Normalized range per channel:")
    print(f"   - R: [{input_image[0].min():.4f}, {input_image[0].max():.4f}]")
    print(f"   - G: [{input_image[1].min():.4f}, {input_image[1].max():.4f}]")
    print(f"   - B: [{input_image[2].min():.4f}, {input_image[2].max():.4f}]")
    
    # Patch mask
    h_p, w_p = im_H // patch_size, im_W // patch_size
    roi_patch_mask = np.zeros((h_p, w_p), dtype=bool)
    patch_y1 = max(0, int(py_min))
    patch_y2 = min(h_p, int(py_max))
    patch_x1 = max(0, int(px_min))
    patch_x2 = min(w_p, int(px_max))
    roi_patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = True
    print(f"\n9. Patch mask:")
    print(f"   Patch grid size: ({h_p}, {w_p})")
    print(f"   ROI patch range: y=[{patch_y1}, {patch_y2}), x=[{patch_x1}, {patch_x2})")
    print(f"   Number of True patches: {roi_patch_mask.sum()}")
    print(f"   Patch mask shape: {roi_patch_mask.shape}")
    
    # Resize ratio
    resize_ratio = out_res / scale
    print(f"\n10. Resize ratio: {resize_ratio:.6f}")
    
    # Camera
    print(f"\n11. Camera handling:")
    scale_x = im_W / im_W_ori
    scale_y = im_H / im_H_ori
    print(f"   Scale factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Debug GDRN Preprocessing")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, help="Bbox: x1 y1 x2 y2")
    parser.add_argument("--obj_cls", type=int, required=True, help="Object class id (0-based)")
    parser.add_argument("--dataset_name", type=str, default="labsim_test", help="Dataset name")
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    register_datasets_in_cfg(cfg)
    
    # Read image
    print(f"Reading image from {args.image}...")
    image = read_image_cv2(args.image, format=cfg.INPUT.FORMAT)
    print(f"Image format: {cfg.INPUT.FORMAT}")
    
    # Debug preprocessing
    debug_preprocess(cfg, image, args.bbox, args.obj_cls, args.dataset_name)


if __name__ == "__main__":
    main()
