#!/usr/bin/env python3
"""
Inference script for GDRN model.
Given an image and a bbox, predict the 6D pose.

Usage:
    python inference_single_image.py \
        --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
        --weights path/to/model.pth \
        --image path/to/image.jpg \
        --bbox x1 y1 x2 y2 \
        --obj_cls 0 \
        --dataset_name labsim_test
    
    Example:
        python inference_single_image.py \
            --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
            --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
            --image test_image.jpg \
            --bbox 100 50 300 250 \
            --obj_cls 0 \
            --dataset_name labsim_test \
            --cam 572.4114 0 325.2611 0 573.57043 242.04899 0 0 1
    
    Note:
        - bbox format: [x1, y1, x2, y2] in image coordinates
        - obj_cls: 0-based object class id (check dataset metadata for class names)
        - cam: optional camera intrinsic matrix (9 values: fx 0 cx 0 fy cy 0 0 1)
               If not provided, a default camera will be used based on image size
"""

import argparse
import os
import os.path as osp
import sys
import cv2
import numpy as np
import torch
import json

# Add project root to path
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)

from mmcv import Config
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.models.GDRN import GDRN, get_xyz_mask_region_out_dim
import logging
from core.gdrn_modeling.models.resnet_backbone import ResNetBackboneNet, resnet_spec
from core.gdrn_modeling.models.cdpn_rot_head_region import RotWithRegionHead
from core.gdrn_modeling.models.cdpn_trans_head import TransHeadNet
from core.gdrn_modeling.models.conv_pnp_net import ConvPnPNet
from core.gdrn_modeling.models.point_pnp_net import PointPnPNet, SimplePointPnPNet
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.data_utils import (
    read_image_cv2,
    crop_resize_by_warp_affine,
    get_2d_coord_np,
)
from core.base_data_loader import Base_DatasetFromList
import ref
from lib.pysixd import inout

import time

import sys
sys.path.append("/home/renchengwei/GDR-Net/tools")
from visualize_3d_bbox import visualize_example


def get_extent_from_dataset(dataset_name, obj_cls):
    """Get object extent from dataset metadata."""
    dset_meta = MetadataCatalog.get(dataset_name)
    ref_key = dset_meta.ref_key
    data_ref = ref.__dict__[ref_key]
    objs = dset_meta.objs
    
    obj_name = objs[obj_cls]
    obj_id = data_ref.obj2id[obj_name]
    model_path = osp.join(data_ref.model_dir, f"obj_{obj_id:06d}.ply")
    model = inout.load_ply(model_path, vertex_scale=data_ref.vertex_scale)
    pts = model["pts"]
    
    xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
    ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
    zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
    
    extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float32)
    return extent


def preprocess_image_and_bbox(cfg, image, bbox_xyxy, obj_cls, dataset_name):
    """
    Preprocess image and bbox to get ROI inputs for model.
    
    Args:
        cfg: config object
        image: HWC image (BGR format from cv2)
        bbox_xyxy: [x1, y1, x2, y2] in image coordinates
        obj_cls: object class id (0-based)
        dataset_name: dataset name for getting extent
    
    Returns:
        dict with preprocessed inputs for model
    """
    im_H_ori, im_W_ori = image.shape[:2]
    
    # Apply augmentation (mainly resize)
    from detectron2.data import transforms as T
    augmentation = []
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, "choice"))
    
    image_transformed, transforms = T.apply_augmentations(augmentation, image.copy())
    im_H, im_W = image_transformed.shape[:2]
    
    # Transform bbox
    bbox_xyxy = np.array(bbox_xyxy, dtype=np.float32)
    bbox_xyxy_transformed = transforms.apply_box([bbox_xyxy])[0]
    
    # Get extent
    extent = get_extent_from_dataset(dataset_name, obj_cls)
    
    # Calculate bbox center and scale, aligned to patch grid (same as data_loader.aug_bbox)
    patch_size = 16
    x1, y1, x2, y2 = bbox_xyxy_transformed
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    
    # No random augmentation for inference, just use center and max dimension
    bbox_center = np.array([cx, cy])
    scale = max(bh, bw) * cfg.INPUT.DZI_PAD_SCALE
    scale = min(scale, max(im_H, im_W)) * 1.0
    
    # Conditionally align bbox to patch grid (same as data_loader.py)
    dzi_patch_grid_adsorption = getattr(cfg.INPUT, 'DZI_PATCH_GRID_ADSORPTION', False)
    if dzi_patch_grid_adsorption:
        # Re-shift the bbox's four corners to align with patch grid
        xmin = bbox_center[0] - scale / 2
        ymin = bbox_center[1] - scale / 2
        xmax = bbox_center[0] + scale / 2
        ymax = bbox_center[1] + scale / 2
        
        # Convert to patch coordinates
        px_min = np.floor(xmin / patch_size)
        py_min = np.floor(ymin / patch_size)
        px_max = np.ceil(xmax / patch_size)
        py_max = np.ceil(ymax / patch_size)
        
        # Enforce square & even number of patches
        pw = px_max - px_min
        ph = py_max - py_min
        p = max(pw, ph)
        
        # Make patch count even → center exactly on patch grid
        if p % 2 == 1:
            p += 1
        
        cx_p = (px_min + px_max) / 2
        cy_p = (py_min + py_max) / 2
        px_min = cx_p - p / 2
        px_max = cx_p + p / 2
        py_min = cy_p - p / 2
        py_max = cy_p + p / 2
        
        # Back to pixel coordinates
        xmin = px_min * patch_size
        xmax = px_max * patch_size
        ymin = py_min * patch_size
        ymax = py_max * patch_size
        
        # Update center, scale, bw, bh after alignment (same as data_loader.py)
        bbox_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)
        scale = float(xmax - xmin)  # guaranteed multiple of patch_size
        bw = max(xmax - xmin, 1)
        bh = max(ymax - ymin, 1)
    
    # Get input and output resolution
    input_res = cfg.MODEL.CDPN.BACKBONE.INPUT_RES
    out_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES
    
    # Crop and resize ROI image
    roi_img = crop_resize_by_warp_affine(
        image_transformed, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
    ).transpose(2, 0, 1)  # HWC -> CHW
    
    # Normalize image
    pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN).reshape(-1, 1, 1)
    pixel_std = np.array(cfg.MODEL.PIXEL_STD).reshape(-1, 1, 1)
    roi_img = (roi_img - pixel_mean) / pixel_std
    
    # Get 2D coordinates
    coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)  # HWC
    roi_coord_2d = crop_resize_by_warp_affine(
        coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
    ).transpose(2, 0, 1)  # HWC -> CHW
    
    # Calculate resize ratio
    resize_ratio = out_res / scale
    
    # Prepare camera intrinsic (if not provided, use a default one)
    # You may need to adjust this based on your actual camera
    if not hasattr(cfg, 'TEST_CAM') or cfg.TEST_CAM is None:
        # Default camera intrinsic (adjust based on your setup)
        # Using a reasonable default: assume focal length is similar to image width
        fx = fy = im_W_ori  # Use original image size for focal length
        cx = im_W_ori / 2.0
        cy = im_H_ori / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    else:
        K = np.array(cfg.TEST_CAM, dtype=np.float32)
        if K.shape == (3, 3):
            pass  # Already 3x3
        elif K.shape == (9,):
            K = K.reshape(3, 3)
        else:
            raise ValueError(f"Invalid camera shape: {K.shape}")
    
    # Scale camera if image was resized
    scale_x = im_W / im_W_ori
    scale_y = im_H / im_H_ori
    K[0, 0] *= scale_x  # fx
    K[0, 2] *= scale_x  # cx
    K[1, 1] *= scale_y  # fy
    K[1, 2] *= scale_y  # cy
    
    # ========== Prepare input_images for the model ==========
    # input_images is always required (model uses it to extract B, N dimensions)
    # Prepare full input image normalized with ImageNet stats (same as training in data_loader.py)
    # IMPORTANT: Convert BGR to RGB first (same as training)
    input_image = cv2.cvtColor(image_transformed, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_image = input_image.transpose(2, 0, 1)  # HWC -> CHW
    input_image = input_image / 255.0  # Convert to [0, 1] range first
    # Normalize with ImageNet mean/std (standard normalization)
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    input_image = (input_image - imagenet_mean) / imagenet_std
    
    # Prepare batch data
    # Note: For multiview support, add N (view) dimension where N=1 for single view inference
    # Format: [B, N, ...] for parameters that need view dimension
    batch = {
        "roi_img": torch.as_tensor(roi_img[None, None], dtype=torch.float32),  # [B, N, C, H, W]
        "roi_coord_2d": torch.as_tensor(roi_coord_2d[None, None], dtype=torch.float32),  # [B, N, C, H, W]
        "roi_cls": torch.as_tensor([obj_cls], dtype=torch.long),  # [B]
        "roi_extent": torch.as_tensor(extent[None, None], dtype=torch.float32),  # [B, N, 3]
        "roi_cam": torch.as_tensor(K[None, None], dtype=torch.float32),  # [B, N, 3, 3]
        "roi_center": torch.as_tensor(bbox_center[None, None], dtype=torch.float32),  # [B, N, 2]
        "scale": torch.as_tensor([[[scale]]], dtype=torch.float32),  # [B, N, 1]
        "roi_wh": torch.as_tensor(np.array([[[bw, bh]]], dtype=np.float32)),  # [B, N, 2]
        "resize_ratio": torch.as_tensor([[resize_ratio]], dtype=torch.float32),  # [B, N]
        "input_images": torch.as_tensor(input_image[None, None], dtype=torch.float32),  # [B, N, C, H, W]
    }
    
    # Conditionally add patch_mask (only used when VGGT_BACKBONE is enabled)
    vggt_backbone = getattr(cfg.MODEL.CDPN, 'VGGT_BACKBONE', False)
    if vggt_backbone and dzi_patch_grid_adsorption:
        # Ensure image is divisible by patch_size for VGGT/DINO encoder
        assert im_H % patch_size == 0 and im_W % patch_size == 0, \
            f"Image size ({im_H}x{im_W}) must be divisible by patch size ({patch_size})"
        
        # Create roi_patch_mask based on aligned patch coordinates
        h_p, w_p = im_H // patch_size, im_W // patch_size
        roi_patch_mask = np.zeros((h_p, w_p), dtype=bool)
        patch_y1 = max(0, int(py_min))
        patch_y2 = min(h_p, int(py_max))
        patch_x1 = max(0, int(px_min))
        patch_x2 = min(w_p, int(px_max))
        roi_patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = True
        batch["patch_mask"] = torch.as_tensor(roi_patch_mask[None, None], dtype=bool)  # [B, N, h_p, w_p]
    
    return batch


def build_model_only(cfg, device=None):
    """Build model only without optimizer (for inference).
    
    Respects the following config flags (consistent with training in GDRN.py / engine.py):
      - cfg.MODEL.CDPN.RESNET_BACKBONE   : whether to create the ResNet backbone
      - cfg.MODEL.CDPN.VGGT_BACKBONE     : whether to use DINO/VGGT image encoder
      - cfg.MODEL.CDPN.POSE_ENCODER      : whether to use the pose encoder
      - cfg.MODEL.CDPN.FREEZE_IMAGE_ENCODER : whether to freeze the DINO image encoder
      - cfg.INPUT.DZI_PATCH_GRID_ADSORPTION : whether to align bbox to patch grid
    """
    logger = logging.getLogger(__name__)
    
    backbone_cfg = cfg.MODEL.CDPN.BACKBONE
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

    # Read key config flags and log them
    resnet_backbone = getattr(cfg.MODEL.CDPN, 'RESNET_BACKBONE', True)
    vggt_backbone = getattr(cfg.MODEL.CDPN, 'VGGT_BACKBONE', False)
    pose_encoder = getattr(cfg.MODEL.CDPN, 'POSE_ENCODER', False)
    freeze_image_encoder = getattr(cfg.MODEL.CDPN, 'FREEZE_IMAGE_ENCODER', False)
    dzi_patch_grid_adsorption = getattr(cfg.INPUT, 'DZI_PATCH_GRID_ADSORPTION', False)
    
    logger.info(f"[Inference] RESNET_BACKBONE={resnet_backbone}, VGGT_BACKBONE={vggt_backbone}, "
                f"POSE_ENCODER={pose_encoder}, FREEZE_IMAGE_ENCODER={freeze_image_encoder}, "
                f"DZI_PATCH_GRID_ADSORPTION={dzi_patch_grid_adsorption}")

    if "resnet" in backbone_cfg.ARCH:
        block_type, layers, channels, name = resnet_spec[backbone_cfg.NUM_LAYERS]
        if resnet_backbone:
            backbone_net = ResNetBackboneNet(
                block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, rot_concat=r_head_cfg.ROT_CONCAT
            )
        else:
            backbone_net = None

        # rotation head net
        r_out_dim, mask_out_dim, region_out_dim = get_xyz_mask_region_out_dim(cfg)
        rot_head_net = RotWithRegionHead(
            cfg,
            channels[-1],
            r_head_cfg.NUM_LAYERS,
            r_head_cfg.NUM_FILTERS,
            r_head_cfg.CONV_KERNEL_SIZE,
            r_head_cfg.OUT_CONV_KERNEL_SIZE,
            rot_output_dim=r_out_dim,
            mask_output_dim=mask_out_dim,
            freeze=r_head_cfg.FREEZE,
            num_classes=r_head_cfg.NUM_CLASSES,
            rot_class_aware=r_head_cfg.ROT_CLASS_AWARE,
            mask_class_aware=r_head_cfg.MASK_CLASS_AWARE,
            num_regions=r_head_cfg.NUM_REGIONS,
            region_class_aware=r_head_cfg.REGION_CLASS_AWARE,
            norm=r_head_cfg.NORM,
            num_gn_groups=r_head_cfg.NUM_GN_GROUPS,
        )
        if r_head_cfg.FREEZE:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False

        # translation head net
        if not t_head_cfg.ENABLED:
            trans_head_net = None
            assert not pnp_net_cfg.R_ONLY, "if pnp_net is R_ONLY, trans_head must be enabled!"
        else:
            trans_head_net = TransHeadNet(
                channels[-1],
                t_head_cfg.NUM_LAYERS,
                t_head_cfg.NUM_FILTERS,
                t_head_cfg.CONV_KERNEL_SIZE,
                t_head_cfg.OUT_CHANNEL,
                freeze=t_head_cfg.FREEZE,
                norm=t_head_cfg.NORM,
                num_gn_groups=t_head_cfg.NUM_GN_GROUPS,
            )
            if t_head_cfg.FREEZE:
                for param in trans_head_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False

        # PnP net
        pnp_net_in_channel = get_pnp_net_input_channels(cfg.MODEL.CDPN)

        if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
            rot_dim = 4
        elif pnp_net_cfg.ROT_TYPE in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
            rot_dim = 3
        elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
            rot_dim = 6
        else:
            raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

        pnp_head_cfg = pnp_net_cfg.PNP_HEAD_CFG.copy()  # Make a copy to avoid modifying original
        pnp_head_type = pnp_head_cfg.pop("type")
        if pnp_head_type == "ConvPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                num_regions=r_head_cfg.NUM_REGIONS,
                featdim=128,
                num_layers=3,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
            )
            pnp_net = ConvPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "PointPnPNet":
            pnp_head_cfg.update(nIn=pnp_net_in_channel, rot_dim=rot_dim, num_regions=r_head_cfg.NUM_REGIONS)
            pnp_net = PointPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "SimplePointPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
            )
            pnp_net = SimplePointPnPNet(**pnp_head_cfg)
        else:
            raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

        if pnp_net_cfg.FREEZE:
            for param in pnp_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False

        # Build model
        model = GDRN(cfg, backbone_net, rot_head_net, trans_head_net=trans_head_net, pnp_net=pnp_net)

    # Note: backbone initialization is no longer needed since we use DINO encoder
    # The DINO encoder weights are loaded in GDRN.__init__

    # Freeze image encoder if configured (consistent with build_model_optimizer in GDRN.py)
    if freeze_image_encoder and vggt_backbone and hasattr(model, 'image_encoder'):
        logger.info("[Inference] Freezing image encoder weights")
        for param in model.image_encoder.parameters():
            param.requires_grad = False

    model_device = device if device is not None else cfg.MODEL.DEVICE
    model.to(torch.device(model_device))
    return model


def inference(cfg, model, batch, device="cuda"):
    """Run inference on a batch."""
    model.eval()
    
    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    
    # For single view inference, target_idx is -1 (or 0, since N=1)
    target_idx = -1
    
    # NOTE: model_infos is not passed (defaults to None), which triggers the
    # pose encoder to use all-zero auxiliary codes — meaning no GT pose is
    # provided and the model must predict the pose entirely on its own.
    with torch.no_grad():
        out_dict = model(
            batch["roi_img"],
            roi_classes=batch["roi_cls"],
            roi_cams=batch["roi_cam"],
            roi_centers=batch["roi_center"],
            scales=batch["scale"],
            roi_whs=batch["roi_wh"],
            resize_ratios=batch["resize_ratio"],
            roi_coord_2d=batch.get("roi_coord_2d", None),
            roi_extents=batch["roi_extent"],
            input_images=batch["input_images"],
            target_idx=target_idx,
            do_loss=False,
        )
    
    return out_dict


def main():
    parser = argparse.ArgumentParser(description="GDRN Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, help="Bbox in format: x1 y1 x2 y2")
    parser.add_argument("--obj_cls", type=int, required=True, help="Object class id (0-based)")
    parser.add_argument("--dataset_name", type=str, default="labsim_test", help="Dataset name for getting extent")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--cam", type=float, nargs=9, default=None, 
                        help="Camera intrinsic as 3x3 matrix (row-major, 9 values: fx 0 cx 0 fy cy 0 0 1)")
    parser.add_argument("--output_dir", type=str, default="/home/renchengwei/GDR-Net/debug", help="Output directory for visualization")
    parser.add_argument("--output_name", type=str, default="test_pred.png", help="Output name for visualization")
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    # Register datasets
    register_datasets_in_cfg(cfg)
    
    # Set camera if provided
    if args.cam is not None:
        # Camera matrix as 9 values: fx 0 cx 0 fy cy 0 0 1
        cfg.TEST_CAM = np.array(args.cam, dtype=np.float32).reshape(3, 3)
    else:
        cfg.TEST_CAM = None
    
    # Build model (without optimizer for inference)
    print("Building model...")
    model = build_model_only(cfg, device=args.device)
    
    # Load weights
    print(f"Loading weights from {args.weights}...")
    checkpointer = MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(args.weights, resume=False)
    
    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)
    
    # Read image
    torch.cuda.synchronize()
    start_time = time.time()
    print(f"Reading image from {args.image}...")
    image = read_image_cv2(args.image, format=cfg.INPUT.FORMAT)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Time taken to read image: {end_time - start_time} seconds")
    
    # Preprocess
    torch.cuda.synchronize()
    start_time = time.time()
    print("Preprocessing image and bbox...")
    batch = preprocess_image_and_bbox(
        cfg, image, args.bbox, args.obj_cls, args.dataset_name
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Time taken to preprocess image and bbox: {end_time - start_time} seconds")
    
    # Inference
    torch.cuda.synchronize()
    start_time = time.time()
    print("Running inference...")
    out_dict = inference(cfg, model, batch, device=args.device)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Time taken to inference: {end_time - start_time} seconds")
    
    # Get results
    pred_rot = out_dict["rot"][0].cpu().numpy()  # 3x3 rotation matrix
    pred_trans = out_dict["trans"][0].cpu().numpy()  # 3D translation vector
    
    print("\n" + "="*50)
    print("Predicted Pose:")
    print("="*50)
    print(f"Rotation matrix (3x3):")
    print(pred_rot)
    print(f"\nTranslation vector (3x1):")
    print(pred_trans)
    print("="*50)
    
    # Save results if needed
    result = {
        "rotation": pred_rot.tolist(),
        "translation": pred_trans.tolist(),
    }
    
    """Get object extent from dataset metadata."""
    dset_meta = MetadataCatalog.get(args.dataset_name)
    ref_key = dset_meta.ref_key
    data_ref = ref.__dict__[ref_key]
    
    model_info = data_ref.get_models_info()[str(args.obj_cls + 1)] # since the obj_cls is 0-based index
    mins = [model_info["min_x"] / 1000.0, model_info["min_y"] / 1000.0, model_info["min_z"] / 1000.0]
    sizes = [model_info["size_x"] / 1000.0, model_info["size_y"] / 1000.0, model_info["size_z"] / 1000.0]
    size = mins + sizes
    
    # visualize the results
    visualize_example(
        K=cfg.TEST_CAM, 
        image=image, 
        RT=np.concatenate([pred_rot, pred_trans[..., None]], axis=1), 
        size=size, 
        output_path=osp.join(args.output_dir, args.output_name)
    )
    
    return result


if __name__ == "__main__":
    main()
