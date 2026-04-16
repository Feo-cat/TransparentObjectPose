#!/usr/bin/env python3
"""
Multiview video inference script for GDRN.

Given a folder of ordered frames (default 12) and bbox json, run one multiview
forward pass and predict target views "all_except_first" (indices 1..N-1).
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)

from mmcv import Config
from detectron2.data import MetadataCatalog

from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.data_utils import read_image_cv2
from core.utils.utils import depth_to_cam_points, cam_to_obj, write_ply_xyzrgb
import ref

# Reuse tested preprocessing/model-building utilities from single/batch inference.
from inference_batch import (
    load_bbox_json,
    preprocess_single_view,
    build_model_only,
    stabilize_rotation_given_axis_single,
)

sys.path.append("/home/renchengwei/GDR-Net/tools")
from visualize_3d_bbox import visualize_symmetric_object, visualize_example


def make_multiview_batch(view_data_list, obj_cls):
    """Pack N preprocessed views into one multiview batch with shape [B=1, N, ...]."""
    if len(view_data_list) == 0:
        raise ValueError("view_data_list is empty")

    roi_img = np.stack([v["roi_img"] for v in view_data_list], axis=0)
    roi_coord_2d = np.stack([v["roi_coord_2d"] for v in view_data_list], axis=0)
    roi_extent = np.stack([v["roi_extent"] for v in view_data_list], axis=0)
    roi_cam = np.stack([v["roi_cam"] for v in view_data_list], axis=0)
    roi_center = np.stack([v["roi_center"] for v in view_data_list], axis=0)
    scale = np.array([[v["scale"]] for v in view_data_list], dtype=np.float32)  # [N, 1]
    roi_wh = np.stack([v["roi_wh"] for v in view_data_list], axis=0)
    resize_ratio = np.array([v["resize_ratio"] for v in view_data_list], dtype=np.float32)  # [N]
    input_images = np.stack([v["input_image"] for v in view_data_list], axis=0)

    batch = {
        "roi_img": torch.as_tensor(roi_img[None], dtype=torch.float32),  # [1, N, C, H, W]
        "roi_coord_2d": torch.as_tensor(roi_coord_2d[None], dtype=torch.float32),  # [1, N, C, H, W]
        "roi_cls": torch.as_tensor([obj_cls], dtype=torch.long),  # [1]
        "roi_extent": torch.as_tensor(roi_extent[None], dtype=torch.float32),  # [1, N, 3]
        "roi_cam": torch.as_tensor(roi_cam[None], dtype=torch.float32),  # [1, N, 3, 3]
        "roi_center": torch.as_tensor(roi_center[None], dtype=torch.float32),  # [1, N, 2]
        "scale": torch.as_tensor(scale[None], dtype=torch.float32),  # [1, N, 1]
        "roi_wh": torch.as_tensor(roi_wh[None], dtype=torch.float32),  # [1, N, 2]
        "resize_ratio": torch.as_tensor(resize_ratio[None], dtype=torch.float32),  # [1, N]
        "input_images": torch.as_tensor(input_images[None], dtype=torch.float32),  # [1, N, C, H, W]
    }

    # Optional patch mask path for VGGT + patch-grid settings.
    if all("patch_mask" in v for v in view_data_list):
        patch_mask = np.stack([v["patch_mask"] for v in view_data_list], axis=0)
        batch["patch_mask"] = torch.as_tensor(patch_mask[None], dtype=bool)  # [1, N, h_p, w_p]

    return batch


def inference_multiview(model, batch, target_idx, device="cuda", roi_mode="external_bbox", mask_thr=0.5, min_mask_pixels=16):
    """Run one multiview forward pass with explicit target_idx."""
    model.eval()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    with torch.no_grad():
        if roi_mode == "auto_mask":
            out_dict = model.forward_infer(
                input_images=batch["input_images"],
                roi_classes=batch["roi_cls"],
                roi_cams=batch["roi_cam"],
                roi_extents=batch["roi_extent"],
                target_idx=target_idx,
                mask_thr=mask_thr,
                min_mask_pixels=min_mask_pixels,
            )
        else:
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


def _get_pred_depth_mask_by_out_idx(out_dict, out_i):
    """Fetch one target-view depth/mask pair from out_dict by output index."""
    if "target_view_depth" not in out_dict or "target_view_mask" not in out_dict:
        return None, None

    depth_out = out_dict["target_view_depth"]
    mask_out = out_dict["target_view_mask"]
    if depth_out.ndim not in (4, 5) or mask_out.ndim not in (4, 5):
        return None, None
    if depth_out.ndim == 5:
        # [B, T, 1, H, W]
        if out_i < 0 or out_i >= depth_out.shape[1] or out_i >= mask_out.shape[1]:
            return None, None
        pred_depth = depth_out[0, out_i, 0].detach().cpu().numpy()
        pred_mask_raw = mask_out[0, out_i, 0].detach().cpu().numpy()
    else:
        # [T, 1, H, W]
        if out_i < 0 or out_i >= depth_out.shape[0] or out_i >= mask_out.shape[0]:
            return None, None
        pred_depth = depth_out[out_i, 0].detach().cpu().numpy()
        pred_mask_raw = mask_out[out_i, 0].detach().cpu().numpy()

    if pred_depth.ndim != 2 or pred_mask_raw.ndim != 2:
        return None, None

    return pred_depth, pred_mask_raw


def save_depth_and_mask_multiview(out_dict, out_i, image_id, results_dir):
    """Save depth/mask visualization for one target view in multiview inference."""
    pred_depth, pred_mask_raw = _get_pred_depth_mask_by_out_idx(out_dict, out_i)
    if pred_depth is None:
        return {}

    if pred_mask_raw.min() >= 0.0 and pred_mask_raw.max() <= 1.0:
        pred_mask = pred_mask_raw
    else:
        pred_mask = 1.0 / (1.0 + np.exp(-pred_mask_raw))

    depth_dir = osp.join(results_dir, "depth")
    mask_dir = osp.join(results_dir, "mask")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    depth_valid = np.isfinite(pred_depth) & (pred_depth > 0)
    mask_bin = pred_mask > 0.5
    masked_valid = depth_valid & mask_bin

    mask_u8 = (np.clip(pred_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_path = osp.join(mask_dir, f"{image_id}_mask.png")
    cv2.imwrite(mask_path, mask_u8)

    if np.any(masked_valid):
        dmin = float(np.percentile(pred_depth[masked_valid], 2))
        dmax = float(np.percentile(pred_depth[masked_valid], 98))
    elif np.any(depth_valid):
        dmin = float(np.percentile(pred_depth[depth_valid], 2))
        dmax = float(np.percentile(pred_depth[depth_valid], 98))
    else:
        dmin, dmax = 0.0, 1.0
    if abs(dmax - dmin) < 1e-8:
        dmax = dmin + 1e-8

    depth_norm = np.clip((pred_depth - dmin) / (dmax - dmin), 0.0, 1.0)
    depth_gray = np.zeros_like(pred_depth, dtype=np.uint8)
    depth_gray[masked_valid] = (depth_norm[masked_valid] * 255.0).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)
    depth_vis[~mask_bin] = 0
    depth_path = osp.join(depth_dir, f"{image_id}_depth.png")
    cv2.imwrite(depth_path, depth_vis)

    return {"depth_path": depth_path, "mask_path": mask_path}


def save_fused_pointcloud_multiview(out_dict, batch, fused_meta, results_dir, max_points=200000):
    """Fuse all target-view depth maps into one object-frame point cloud."""
    if len(fused_meta) == 0:
        return ""

    pcd_dir = osp.join(results_dir, "pointcloud")
    os.makedirs(pcd_dir, exist_ok=True)

    view_palette = np.array(
        [
            [255, 64, 64],
            [64, 255, 64],
            [64, 128, 255],
            [255, 255, 64],
            [255, 64, 255],
            [64, 255, 255],
        ],
        dtype=np.uint8,
    )

    points_obj_all = []
    colors_all = []
    for local_i, (out_i, frame_i, cur_rot, cur_trans) in enumerate(fused_meta):
        pred_depth, pred_mask_raw = _get_pred_depth_mask_by_out_idx(out_dict, out_i)
        if pred_depth is None:
            continue

        if pred_mask_raw.min() >= 0.0 and pred_mask_raw.max() <= 1.0:
            pred_mask = pred_mask_raw
        else:
            pred_mask = 1.0 / (1.0 + np.exp(-pred_mask_raw))

        valid = np.isfinite(pred_depth) & (pred_depth > 0) & (pred_mask > 0.5)
        if not np.any(valid):
            continue

        K = batch["roi_cam"][0, frame_i].detach().cpu().numpy().astype(np.float32)
        pts_cam = depth_to_cam_points(pred_depth, K, valid)
        pts_obj = cam_to_obj(pts_cam.astype(np.float32), cur_rot.astype(np.float32), cur_trans.astype(np.float32))
        if pts_obj.shape[0] == 0:
            continue

        color_i = np.tile(view_palette[local_i % len(view_palette)][None, :], (pts_obj.shape[0], 1))
        points_obj_all.append(pts_obj)
        colors_all.append(color_i)

    if len(points_obj_all) == 0:
        return ""

    points_obj = np.concatenate(points_obj_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    if points_obj.shape[0] > max_points:
        keep_idx = np.random.choice(points_obj.shape[0], max_points, replace=False)
        points_obj = points_obj[keep_idx]
        colors = colors[keep_idx]

    fused_pcd_path = osp.join(pcd_dir, "pred_depth_fused_obj.ply")
    write_ply_xyzrgb(fused_pcd_path, points_obj.astype(np.float32), colors.astype(np.uint8))
    return fused_pcd_path


def _find_bbox_for_image(image_name, bbox_dict):
    image_id = osp.splitext(image_name)[0]
    if image_name in bbox_dict:
        return bbox_dict[image_name]
    if image_id in bbox_dict:
        return bbox_dict[image_id]
    return None


def main():
    parser = argparse.ArgumentParser(description="GDRN Multiview Video Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to frame folder")
    parser.add_argument("--bbox_json", type=str, default=None, help="Path to bbox JSON (required for external_bbox mode)")
    parser.add_argument("--obj_cls", type=int, required=True, help="Object class id (0-based default)")
    parser.add_argument("--dataset_name", type=str, default="labsim_test", help="Dataset name")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument(
        "--cam",
        type=float,
        nargs=9,
        default=None,
        help="Camera intrinsic 3x3 matrix (row-major, 9 values: fx 0 cx 0 fy cy 0 0 1)",
    )
    parser.add_argument("--image_ext", type=str, default=".png", help="Image extension")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization images")
    parser.add_argument("--expect_num_frames", type=int, default=12, help="Expected frame count (<=0 to disable)")
    parser.add_argument(
        "--roi_mode",
        type=str,
        default="external_bbox",
        choices=["external_bbox", "auto_mask"],
        help="ROI source: external bbox json or model-predicted mask",
    )
    parser.add_argument("--mask_thr", type=float, default=0.5, help="Mask threshold for auto_mask ROI extraction")
    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=16,
        help="Fallback to full image ROI if predicted mask has fewer foreground pixels",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="all_except_first",
        choices=["all_except_first"],
        help="Target-view selection mode",
    )
    parser.add_argument("--symm_mode", type=str, default="none", choices=["none", "continuous", "discrete"])
    parser.add_argument("--symm_axis", type=float, nargs=3, default=None, help="Symmetry axis for continuous mode")

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    register_datasets_in_cfg(cfg)

    if args.cam is not None:
        cfg.TEST_CAM = np.array(args.cam, dtype=np.float32).reshape(3, 3)
    else:
        cfg.TEST_CAM = None

    print("Building model...")
    model = build_model_only(cfg, device=args.device)
    print(f"Loading weights from {args.weights}...")
    checkpointer = MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(args.weights, resume=False)
    model = model.to(torch.device(args.device))

    bbox_dict = None
    if args.roi_mode == "external_bbox":
        if args.bbox_json is None:
            raise ValueError("--bbox_json is required when --roi_mode=external_bbox")
        print(f"Loading bbox JSON from {args.bbox_json}...")
        bbox_dict = load_bbox_json(args.bbox_json)
        print(f"Loaded {len(bbox_dict)} bbox entries")
    else:
        print("ROI mode: auto_mask (bbox_json is ignored)")

    image_exts = [args.image_ext, args.image_ext.lower(), args.image_ext.upper()]
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(osp.join(args.image_dir, f"*{ext}")))
        image_files.extend(glob.glob(osp.join(args.image_dir, f"*{ext.replace('.', '')}")))
    image_files = sorted(list(set(image_files)))

    if args.expect_num_frames > 0 and len(image_files) != args.expect_num_frames:
        raise ValueError(
            f"Frame count mismatch: found {len(image_files)} in {args.image_dir}, "
            f"expected {args.expect_num_frames}"
        )
    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 frames, got {len(image_files)}")

    results_dir = args.output_dir if args.output_dir is not None else osp.join(args.image_dir, "results_vid")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(osp.join(results_dir, "depth"), exist_ok=True)
    os.makedirs(osp.join(results_dir, "mask"), exist_ok=True)
    os.makedirs(osp.join(results_dir, "pointcloud"), exist_ok=True)
    print(f"Found {len(image_files)} frames")
    print(f"Results will be saved to {results_dir}")

    dset_meta = MetadataCatalog.get(args.dataset_name)
    data_ref = ref.__dict__[dset_meta.ref_key]
    model_info = data_ref.get_models_info()[str(args.obj_cls + 1)]
    mins = [model_info["min_x"] / 1000.0, model_info["min_y"] / 1000.0, model_info["min_z"] / 1000.0]
    sizes = [model_info["size_x"] / 1000.0, model_info["size_y"] / 1000.0, model_info["size_z"] / 1000.0]
    model_size = mins + sizes

    view_data_list = []
    image_names = []
    obj_cls_list = []
    print("Preprocessing frames...")
    for image_path in tqdm(image_files, desc="Loading frames"):
        image_name = osp.basename(image_path)
        image = read_image_cv2(image_path, format=cfg.INPUT.FORMAT)
        if args.roi_mode == "external_bbox":
            bbox_info = _find_bbox_for_image(image_name, bbox_dict)
            if bbox_info is None:
                raise ValueError(f"No bbox found for frame {image_name}")

            if isinstance(bbox_info, dict):
                bbox = bbox_info.get("bbox")
                obj_cls = bbox_info.get("obj_cls", args.obj_cls)
                if obj_cls is None:
                    obj_cls = args.obj_cls
            elif isinstance(bbox_info, list) and len(bbox_info) == 4:
                bbox = bbox_info
                obj_cls = args.obj_cls
            else:
                raise ValueError(f"Invalid bbox format for {image_name}: {type(bbox_info)}")

            if bbox is None or not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox for {image_name}: {bbox}")
        else:
            # auto_mask mode: initialize from full image, ROI will be re-estimated from model mask.
            h, w = image.shape[:2]
            bbox = [0.0, 0.0, float(max(w - 1, 1)), float(max(h - 1, 1))]
            obj_cls = args.obj_cls

        view_data, obj_cls = preprocess_single_view(cfg, image, bbox, obj_cls, args.dataset_name)
        view_data_list.append(view_data)
        image_names.append(image_name)
        obj_cls_list.append(int(obj_cls))

    obj_cls_mv = obj_cls_list[0]
    if any(c != obj_cls_mv for c in obj_cls_list):
        raise ValueError("Inconsistent obj_cls across frames in one clip")

    batch = make_multiview_batch(view_data_list, obj_cls_mv)

    # all_except_first
    target_idx = list(range(1, len(view_data_list)))
    print(f"Running multiview inference with target_idx={target_idx}, roi_mode={args.roi_mode}")
    out_dict = inference_multiview(
        model,
        batch,
        target_idx=target_idx,
        device=args.device,
        roi_mode=args.roi_mode,
        mask_thr=args.mask_thr,
        min_mask_pixels=args.min_mask_pixels,
    )

    pred_rot = out_dict["rot"].cpu().numpy()
    pred_trans = out_dict["trans"].cpu().numpy()
    if pred_rot.shape[0] != len(target_idx) or pred_trans.shape[0] != len(target_idx):
        raise RuntimeError(
            f"Unexpected output shape: rot={pred_rot.shape}, trans={pred_trans.shape}, "
            f"expected first dim={len(target_idx)}"
        )

    all_results = {}
    x_ref = None
    fused_meta = []
    for out_i, frame_i in enumerate(target_idx):
        cur_rot = pred_rot[out_i]
        cur_trans = pred_trans[out_i]
        if args.symm_mode == "continuous":
            if args.symm_axis is None:
                raise ValueError("symm_mode=continuous requires --symm_axis")
            cur_rot, x_ref = stabilize_rotation_given_axis_single(cur_rot, np.array(args.symm_axis), x_ref)

        image_name = image_names[frame_i]
        image_id = osp.splitext(image_name)[0]
        all_results[image_name] = {
            "rotation": cur_rot.tolist(),
            "translation": cur_trans.tolist(),
            "source_index": int(frame_i),
            "target_mode": "all_except_first",
        }
        all_results[image_name].update(
            save_depth_and_mask_multiview(
                out_dict=out_dict,
                out_i=out_i,
                image_id=image_id,
                results_dir=results_dir,
            )
        )
        fused_meta.append((out_i, frame_i, cur_rot, cur_trans))

        if args.save_vis:
            vis_output_path = osp.join(results_dir, f"{image_id}_pred.png")
            # if not symmetric
            if args.symm_mode == "none":
                visualize_example(
                    K=cfg.TEST_CAM,
                    image=view_data_list[frame_i]["original_image"],
                    RT=np.concatenate([cur_rot, cur_trans[..., None]], axis=1),
                    size=model_size,
                    output_path=vis_output_path,
                )
            else:
                visualize_symmetric_object(
                    K=cfg.TEST_CAM,
                    image=view_data_list[frame_i]["original_image"],
                    RT=np.concatenate([cur_rot, cur_trans[..., None]], axis=1),
                    size=model_size,
                    symmetry_axis="z",
                    output_path=vis_output_path,
                    color=(0, 255, 0),
                    thickness=1,
                )

    fused_pcd_path = save_fused_pointcloud_multiview(
        out_dict=out_dict,
        batch=batch,
        fused_meta=fused_meta,
        results_dir=results_dir,
    )
    if fused_pcd_path:
        for image_name in all_results:
            all_results[image_name]["pointcloud_fused_obj_path"] = fused_pcd_path

    results_json_path = osp.join(results_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_json_path}")
    print(f"Predicted frames: {len(all_results)} / {len(image_files)} (excluded first frame by design)")


if __name__ == "__main__":
    main()
