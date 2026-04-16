#!/usr/bin/env python3
"""
Autoregressive multiview video inference script for GDRN.

Inference schedule (aligned with training config TRAIN_NUM_CONTEXT_VIEWS / TRAIN_NUM_TARGET_VIEWS):
1) First chunk: chunk_size views, all are target views.
2) Following chunks: first ar_context_size views are context, last ar_target_size views are target.
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
from scipy.spatial.transform import Rotation as Rot, Slerp

# Add project root to path
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, cur_dir)

from mmcv import Config
from detectron2.data import MetadataCatalog

from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.models.GDRN import get_pnp_net_input_channels
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.data_utils import read_image_cv2
from core.utils.utils import depth_to_cam_points, cam_to_obj, write_ply_xyzrgb
import ref

# Reuse tested preprocessing/model-building utilities from single/batch inference.
from inference_batch import (
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

    if all("input_obj_mask" in v for v in view_data_list):
        input_obj_masks = np.stack([v["input_obj_mask"] for v in view_data_list], axis=0)
        batch["input_obj_masks"] = torch.as_tensor(input_obj_masks[None], dtype=torch.float32)  # [1, N, H, W]

    return batch


def inference_multiview(
    model,
    batch,
    target_idx,
    model_infos=None,
    device="cuda",
    mask_thr=0.5,
    min_mask_pixels=16,
    mask_postproc="none",
    mask_prev_dilate_kernel=11,
    mask_prev_gate=True,
    mask_post_open_kernel=0,
    mask_post_dilate_kernel=3,
    mask_post_close_kernel=3,
    mask_fallback_to_prev=False,
    external_mask_pad_scale=None,
):
    """Run one multiview forward pass with explicit target_idx."""
    model.eval()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    with torch.no_grad():
        out_dict = model.forward_infer(
            input_images=batch["input_images"],
            roi_classes=batch["roi_cls"],
            roi_cams=batch["roi_cam"],
            roi_extents=batch["roi_extent"],
            model_infos=model_infos,
            gt_ego_rot=batch.get("gt_ego_rot", None),
            gt_trans=batch.get("gt_trans", None),
            gt_pose_valid=batch.get("gt_pose_valid", None),
            seed_roi_centers=batch.get("roi_center", None),
            seed_scales=batch.get("scale", None),
            target_idx=target_idx,
            mask_thr=mask_thr,
            min_mask_pixels=min_mask_pixels,
            external_target_masks=batch.get("input_obj_masks", None),
            prev_target_mask=batch.get("prev_target_mask", None),
            mask_postproc=mask_postproc,
            mask_prev_dilate_kernel=mask_prev_dilate_kernel,
            mask_prev_gate=mask_prev_gate,
            mask_post_open_kernel=mask_post_open_kernel,
            mask_post_dilate_kernel=mask_post_dilate_kernel,
            mask_post_close_kernel=mask_post_close_kernel,
            mask_fallback_to_prev=mask_fallback_to_prev,
            external_mask_pad_scale=external_mask_pad_scale,
        )
    return out_dict


def build_window_pose_context(window_global_ids, all_results):
    """Build pseudo GT pose tensors for current window from already-predicted frames."""
    n = len(window_global_ids)
    rot = np.tile(np.eye(3, dtype=np.float32)[None], (n, 1, 1))
    trans = np.zeros((n, 3), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    for local_i, gid in enumerate(window_global_ids):
        item = all_results.get(int(gid), None)
        if item is None:
            continue
        if ("rotation" not in item) or ("translation" not in item):
            continue
        context_pose_valid = bool(item.get("context_pose_valid", True))
        if not context_pose_valid:
            continue
        rot[local_i] = np.asarray(item["rotation"], dtype=np.float32)
        trans[local_i] = np.asarray(item["translation"], dtype=np.float32)
        valid[local_i] = True
    return (
        torch.as_tensor(rot[None], dtype=torch.float32),   # [1, N, 3, 3]
        torch.as_tensor(trans[None], dtype=torch.float32), # [1, N, 3]
        valid,
    )


def smooth_pose_temporally(cur_rot, cur_trans, prev_rot, prev_trans, rot_alpha=1.0, trans_alpha=1.0):
    """Causal temporal smoothing for one pose using SLERP on rotation and EMA on translation.

    alpha=1 keeps the current prediction unchanged.
    Smaller alpha values increase smoothing strength.
    """
    cur_rot = np.asarray(cur_rot, dtype=np.float32)
    cur_trans = np.asarray(cur_trans, dtype=np.float32)
    if prev_rot is None or prev_trans is None:
        return cur_rot, cur_trans

    rot_alpha = float(np.clip(rot_alpha, 0.0, 1.0))
    trans_alpha = float(np.clip(trans_alpha, 0.0, 1.0))
    if rot_alpha >= 1.0 and trans_alpha >= 1.0:
        return cur_rot, cur_trans

    prev_rot = np.asarray(prev_rot, dtype=np.float32)
    prev_trans = np.asarray(prev_trans, dtype=np.float32)

    if rot_alpha >= 1.0:
        smooth_rot = cur_rot
    elif rot_alpha <= 0.0:
        smooth_rot = prev_rot
    else:
        prev_q = Rot.from_matrix(prev_rot).as_quat()
        cur_q = Rot.from_matrix(cur_rot).as_quat()
        if np.dot(prev_q, cur_q) < 0.0:
            cur_q = -cur_q
        key_rots = Rot.from_quat(np.stack([prev_q, cur_q], axis=0))
        smooth_rot = Slerp([0.0, 1.0], key_rots)(rot_alpha).as_matrix().astype(np.float32)

    if trans_alpha >= 1.0:
        smooth_trans = cur_trans
    elif trans_alpha <= 0.0:
        smooth_trans = prev_trans
    else:
        smooth_trans = ((1.0 - trans_alpha) * prev_trans + trans_alpha * cur_trans).astype(np.float32)

    return smooth_rot, smooth_trans


def _rotation_angle_deg_np(r1, r2):
    rel = np.asarray(r1, dtype=np.float64) @ np.asarray(r2, dtype=np.float64).T
    val = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(val)))


def _axis_angle_deg_np(r1, r2, axis=(0.0, 0.0, 1.0)):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    v1 = np.asarray(r1, dtype=np.float64) @ axis
    v2 = np.asarray(r2, dtype=np.float64) @ axis
    v1 = v1 / max(np.linalg.norm(v1), 1e-12)
    v2 = v2 / max(np.linalg.norm(v2), 1e-12)
    val = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(val)))


def _maybe_append_jsonl(path, payload):
    if not path:
        return
    path = str(path).strip()
    if not path:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _project_translation_to_image_xy(trans, cam, eps=1e-6):
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    cam = np.asarray(cam, dtype=np.float64).reshape(3, 3)
    z = float(trans[2])
    if abs(z) < eps:
        return None
    fx = float(cam[0, 0])
    fy = float(cam[1, 1])
    px = float(cam[0, 2])
    py = float(cam[1, 2])
    cx = fx * float(trans[0]) / z + px
    cy = fy * float(trans[1]) / z + py
    return [float(cx), float(cy)]


def smooth_roi_temporally(
    cur_roi_center,
    cur_scale,
    cur_roi_wh,
    prev_roi_center,
    prev_scale,
    prev_roi_wh,
    smooth_type="ema",
    alpha=0.4,
    min_alpha=0.15,
    max_alpha=0.7,
):
    """Causal temporal smoothing for ROI parameters used by the next AR chunk."""
    cur_roi_center = np.asarray(cur_roi_center, dtype=np.float32).reshape(2)
    cur_scale = np.asarray(cur_scale, dtype=np.float32).reshape(-1)
    cur_roi_wh = np.asarray(cur_roi_wh, dtype=np.float32).reshape(2)

    if prev_roi_center is None or prev_scale is None or prev_roi_wh is None:
        return cur_roi_center, cur_scale, cur_roi_wh, 1.0

    prev_roi_center = np.asarray(prev_roi_center, dtype=np.float32).reshape(2)
    prev_scale = np.asarray(prev_scale, dtype=np.float32).reshape(-1)
    prev_roi_wh = np.asarray(prev_roi_wh, dtype=np.float32).reshape(2)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
    max_alpha = float(np.clip(max_alpha, 0.0, 1.0))
    if min_alpha > max_alpha:
        min_alpha, max_alpha = max_alpha, min_alpha

    if smooth_type == "ema":
        alpha_used = alpha
    elif smooth_type == "adaptive_ema":
        prev_scale_scalar = max(float(prev_scale[0]), 1e-6)
        cur_scale_scalar = max(float(cur_scale[0]), 1e-6)
        center_motion = np.linalg.norm(cur_roi_center - prev_roi_center) / prev_scale_scalar
        scale_motion = abs(np.log(cur_scale_scalar / prev_scale_scalar))
        wh_motion = np.linalg.norm((cur_roi_wh - prev_roi_wh) / np.maximum(prev_roi_wh, 1e-6))
        motion_score = max(center_motion / 0.12, scale_motion / 0.18, wh_motion / 0.25)
        motion_score = float(np.clip(motion_score, 0.0, 1.0))
        alpha_used = min_alpha + motion_score * (max_alpha - min_alpha)
    else:
        raise ValueError(f"Unknown ROI smooth type: {smooth_type}")

    smooth_roi_center = ((1.0 - alpha_used) * prev_roi_center + alpha_used * cur_roi_center).astype(np.float32)
    smooth_scale = ((1.0 - alpha_used) * prev_scale + alpha_used * cur_scale).astype(np.float32)
    smooth_roi_wh = ((1.0 - alpha_used) * prev_roi_wh + alpha_used * cur_roi_wh).astype(np.float32)
    return smooth_roi_center, smooth_scale, smooth_roi_wh, float(alpha_used)


def _get_pred_depth_mask_by_out_idx(out_dict, out_i):
    """Fetch one target-view depth/mask pair from out_dict by output index."""
    if "target_view_depth" not in out_dict:
        return None, None

    depth_out = out_dict["target_view_depth"]
    mask_out = out_dict.get("target_view_mask_clean", out_dict.get("target_view_mask", None))
    if mask_out is None:
        return None, None
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


def _get_target_mask_by_out_idx(out_dict, out_i, mask_key):
    """Fetch one target-view mask by output index from a specific out_dict key."""
    if mask_key not in out_dict:
        return None
    mask_out = out_dict[mask_key]
    if mask_out.ndim not in (4, 5):
        return None
    if mask_out.ndim == 5:
        if out_i < 0 or out_i >= mask_out.shape[1]:
            return None
        pred_mask = mask_out[0, out_i, 0].detach().cpu().numpy()
    else:
        if out_i < 0 or out_i >= mask_out.shape[0]:
            return None
        pred_mask = mask_out[out_i, 0].detach().cpu().numpy()
    return pred_mask


def _get_target_tensor_by_out_idx(out_dict, out_i, tensor_key):
    """Fetch one target-view tensor (C,H,W) from out_dict by output index."""
    if tensor_key not in out_dict:
        return None
    tensor_out = out_dict[tensor_key]
    if not torch.is_tensor(tensor_out):
        return None
    if tensor_out.ndim == 5:
        # [B, T, C, H, W]
        if out_i < 0 or out_i >= tensor_out.shape[1]:
            return None
        tensor_item = tensor_out[0, out_i]
    elif tensor_out.ndim == 4:
        # [T, C, H, W]
        if out_i < 0 or out_i >= tensor_out.shape[0]:
            return None
        tensor_item = tensor_out[out_i]
    elif tensor_out.ndim == 3:
        # [C, H, W] for single target view
        if out_i != 0:
            return None
        tensor_item = tensor_out
    else:
        return None
    return tensor_item.detach().cpu().numpy().astype(np.float32, copy=False)


def _decode_xyz_expectation_from_coor_logits_np(coor_x, coor_y, coor_z):
    """Decode xyz expectation map from per-axis coordinate logits."""
    coor_x = np.asarray(coor_x, dtype=np.float32)
    coor_y = np.asarray(coor_y, dtype=np.float32)
    coor_z = np.asarray(coor_z, dtype=np.float32)
    if coor_x.ndim != 3 or coor_y.ndim != 3 or coor_z.ndim != 3:
        raise ValueError("coor_x/coor_y/coor_z must be 3D [C,H,W]")
    if coor_x.shape[0] != coor_y.shape[0] or coor_x.shape[0] != coor_z.shape[0]:
        raise ValueError("coor_x/coor_y/coor_z channel counts must match")

    if coor_x.shape[0] == 1:
        return np.concatenate([coor_x, coor_y, coor_z], axis=0)

    # Match model decode path in forward_infer: drop the last channel for expectation.
    logits_x = coor_x[:-1]
    logits_y = coor_y[:-1]
    logits_z = coor_z[:-1]
    num_bins = logits_x.shape[0]
    if num_bins <= 0:
        raise ValueError(f"Invalid logits channel count: {coor_x.shape[0]}")

    bins = np.linspace(0.0, 1.0, num_bins, dtype=np.float32).reshape(num_bins, 1, 1)

    def _softmax_channel(logits):
        logits = logits - np.max(logits, axis=0, keepdims=True)
        expv = np.exp(logits)
        denom = np.sum(expv, axis=0, keepdims=True) + 1e-12
        return expv / denom

    x = (_softmax_channel(logits_x) * bins).sum(axis=0, keepdims=True)
    y = (_softmax_channel(logits_y) * bins).sum(axis=0, keepdims=True)
    z = (_softmax_channel(logits_z) * bins).sum(axis=0, keepdims=True)
    return np.concatenate([x, y, z], axis=0).astype(np.float32, copy=False)


def _sigmoid_numpy(x):
    """Numerically stable sigmoid for raw mask logits."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


def save_depth_and_mask_multiview(out_dict, out_i, image_id, results_dir):
    """Save depth/mask visualization for one target view in multiview inference."""
    pred_depth, pred_mask_raw = _get_pred_depth_mask_by_out_idx(out_dict, out_i)
    if pred_depth is None:
        return {}

    if pred_mask_raw.min() >= 0.0 and pred_mask_raw.max() <= 1.0:
        pred_mask = pred_mask_raw
    else:
        pred_mask = _sigmoid_numpy(pred_mask_raw)

    depth_dir = osp.join(results_dir, "depth")
    mask_dir = osp.join(results_dir, "mask")
    mask_raw_dir = osp.join(results_dir, "mask_raw")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask_raw_dir, exist_ok=True)

    depth_valid = np.isfinite(pred_depth) & (pred_depth > 0)
    mask_bin = pred_mask > 0.5
    masked_valid = depth_valid & mask_bin

    mask_u8 = (np.clip(pred_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_path = osp.join(mask_dir, f"{image_id}_mask.png")
    cv2.imwrite(mask_path, mask_u8)

    raw_mask_path = ""
    raw_mask_item = _get_target_mask_by_out_idx(out_dict, out_i, "target_view_mask_raw")
    if raw_mask_item is not None:
        if raw_mask_item.min() >= 0.0 and raw_mask_item.max() <= 1.0:
            raw_mask_prob = raw_mask_item
        else:
            raw_mask_prob = _sigmoid_numpy(raw_mask_item)
        raw_mask_u8 = (np.clip(raw_mask_prob, 0.0, 1.0) * 255.0).astype(np.uint8)
        raw_mask_path = osp.join(mask_raw_dir, f"{image_id}_mask_raw.png")
        cv2.imwrite(raw_mask_path, raw_mask_u8)

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

    return {"depth_path": depth_path, "mask_path": mask_path, "mask_raw_path": raw_mask_path}


def save_coor_xyz_npy_multiview(out_dict, out_i, image_id, results_dir):
    """Save per-target coor_z logits and decoded xyz map as .npy files."""
    coor_z = _get_target_tensor_by_out_idx(out_dict, out_i, "coor_z")
    coor_x = _get_target_tensor_by_out_idx(out_dict, out_i, "coor_x")
    coor_y = _get_target_tensor_by_out_idx(out_dict, out_i, "coor_y")
    if coor_z is None or coor_x is None or coor_y is None:
        return {}

    coor_z_dir = osp.join(results_dir, "coor_z_npy")
    xyz_dir = osp.join(results_dir, "xyz_npy")
    os.makedirs(coor_z_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)

    coor_z_path = osp.join(coor_z_dir, f"{image_id}_coor_z.npy")
    np.save(coor_z_path, coor_z.astype(np.float32, copy=False))

    xyz = _decode_xyz_expectation_from_coor_logits_np(coor_x, coor_y, coor_z)
    xyz_path = osp.join(xyz_dir, f"{image_id}_xyz.npy")
    np.save(xyz_path, xyz.astype(np.float32, copy=False))

    return {"coor_z_npy_path": coor_z_path, "xyz_npy_path": xyz_path}


def save_pnp_decode_debug_npy(image_id, results_dir, pnp_t_raw=None, pnp_rot_raw=None):
    """Save raw pnp head outputs as .npy files for decode debugging."""
    if pnp_t_raw is None and pnp_rot_raw is None:
        return {}
    debug_dir = osp.join(results_dir, "pnp_decode_npy")
    os.makedirs(debug_dir, exist_ok=True)

    out = {}
    if pnp_t_raw is not None:
        t_path = osp.join(debug_dir, f"{image_id}_pnp_t_raw.npy")
        np.save(t_path, np.asarray(pnp_t_raw, dtype=np.float32))
        out["pnp_t_raw_npy_path"] = t_path
    if pnp_rot_raw is not None:
        r_path = osp.join(debug_dir, f"{image_id}_pnp_rot_raw.npy")
        np.save(r_path, np.asarray(pnp_rot_raw, dtype=np.float32))
        out["pnp_rot_raw_npy_path"] = r_path
    return out


def save_pnp_input_debug_npy_multiview(out_dict, out_i, image_id, results_dir):
    """Save per-target PnP input-related internal tensors as .npy files."""
    debug_dir = osp.join(results_dir, "pnp_input_npy")
    os.makedirs(debug_dir, exist_ok=True)

    tensor_key_to_name = {
        "dbg_pnp_coor_feat_base": "coor_feat_base",
        "dbg_pnp_coor_feat_final": "coor_feat_final",
        "dbg_pnp_xyz_single": "xyz_single",
        "dbg_pnp_roi_coord_2d": "roi_coord_2d",
        "dbg_pnp_region_softmax": "region_softmax",
        "dbg_pnp_region_atten": "region_atten",
        "dbg_pnp_mask_atten": "mask_atten",
        "dbg_pnp_prior_xyz": "prior_xyz",
        "dbg_pnp_prior_conf": "prior_conf",
        "dbg_pnp_prior_residual": "prior_residual",
    }

    out = {}
    for tensor_key, short_name in tensor_key_to_name.items():
        arr = _get_target_tensor_by_out_idx(out_dict, out_i, tensor_key)
        if arr is None:
            continue
        # Use float16 to reduce disk footprint for large per-frame feature maps.
        arr_to_save = arr.astype(np.float16, copy=False)
        out_path = osp.join(debug_dir, f"{image_id}_{short_name}.npy")
        np.save(out_path, arr_to_save)
        out[f"pnp_{short_name}_npy_path"] = out_path
    return out


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
            pred_mask = _sigmoid_numpy(pred_mask_raw)

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


def _find_mask_path_for_image(image_name, mask_dir):
    image_id = osp.splitext(image_name)[0]
    exact_path = osp.join(mask_dir, image_name)
    if osp.isfile(exact_path):
        return exact_path

    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = osp.join(mask_dir, f"{image_id}{ext}")
        if osp.isfile(candidate):
            return candidate

    glob_matches = sorted(glob.glob(osp.join(mask_dir, f"{image_id}.*")))
    if glob_matches:
        return glob_matches[0]
    return None


def _load_external_mask(mask_path, target_hw):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Failed to read external mask: {mask_path}")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = (mask > 0.5).astype(np.float32)
    return mask


def build_autoregressive_windows(num_frames, chunk_size=6, context_size=3, target_size=3):
    """Build AR windows aligned with training:
    - chunk0: all target
    - chunk>=1: first context_size as context, last target_size as target
    """
    if num_frames < 1:
        return []
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if context_size < 0 or target_size < 1:
        raise ValueError(
            f"context_size/target_size must satisfy context_size>=0 and target_size>=1, "
            f"got context_size={context_size}, target_size={target_size}"
        )
    if context_size + target_size != chunk_size:
        raise ValueError(
            f"context_size + target_size must equal chunk_size, got "
            f"{context_size}+{target_size}!={chunk_size}"
        )

    windows = []
    first_end = min(chunk_size, num_frames)
    first_ids = list(range(0, first_end))
    windows.append(
        {
            "window_global_ids": first_ids,
            "target_local_ids": list(range(len(first_ids))),
            "mode": "all_target",
        }
    )
    next_unseen = first_end

    while next_unseen < num_frames:
        # Try to build a full chunk [context_size previous + target_size new].
        start = max(0, next_unseen - context_size)
        end = start + chunk_size
        if end > num_frames:
            # Keep chunk as large as possible near the tail.
            end = num_frames
            start = max(0, end - chunk_size)
        cur_ids = list(range(start, end))

        # Target views are unseen frames in this chunk.
        target_local_ids = [local_i for local_i, gid in enumerate(cur_ids) if gid >= next_unseen]
        if len(target_local_ids) == 0:
            break

        windows.append(
            {
                "window_global_ids": cur_ids,
                "target_local_ids": target_local_ids,
                "mode": "context_then_target",
            }
        )
        next_unseen += len(target_local_ids)
    return windows


def main():
    parser = argparse.ArgumentParser(description="GDRN Multiview Video Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to frame folder")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Optional directory of external per-frame masks used as the mask source for ROI extraction, mask attention, and refiner observation",
    )
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
    parser.add_argument(
        "--save_coor_xyz_npy",
        action="store_true",
        default=bool(int(os.environ.get("GDRN_SAVE_COOR_XYZ_NPY", "0"))),
        help=(
            "Save per-target coor_z logits and decoded xyz maps as .npy under "
            "<results_dir>/coor_z_npy and <results_dir>/xyz_npy"
        ),
    )
    parser.add_argument(
        "--save_pnp_decode_debug",
        action="store_true",
        default=bool(int(os.environ.get("GDRN_SAVE_PNP_DECODE_DEBUG", "0"))),
        help=(
            "Save raw PnP head outputs (pred_t_raw/pred_rot_raw) and per-frame decode diagnostics "
            "for z-jitter attribution"
        ),
    )
    parser.add_argument(
        "--save_pnp_input_debug",
        action="store_true",
        default=bool(int(os.environ.get("GDRN_SAVE_PNP_INPUT_DEBUG", "0"))),
        help=(
            "Save per-target PnP input-related internal tensors (coor_feat/region/mask/prior/coord2d) "
            "as .npy for jitter attribution"
        ),
    )
    parser.add_argument(
        "--roi_mode",
        type=str,
        default="auto_mask",
        choices=["auto_mask", "external_bbox"],
        help=(
            "ROI source mode. 'auto_mask' uses the model-predicted mask for ROI extraction (default). "
            "'external_bbox' is not yet implemented for multiview AR inference; use --mask_dir instead."
        ),
    )
    parser.add_argument(
        "--bbox_json",
        type=str,
        default=None,
        help="Path to per-frame bounding box JSON (only used with --roi_mode external_bbox, not yet implemented).",
    )
    parser.add_argument("--mask_thr", type=float, default=0.5, help="Mask threshold for model-predicted mask ROI extraction")
    parser.add_argument(
        "--min_mask_pixels",
        type=int,
        default=16,
        help="Fallback to full image ROI if predicted mask has fewer foreground pixels",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="autoregressive",
        choices=["autoregressive"],
        help="Target-view selection mode (fixed AR chunk schedule)",
    )
    parser.add_argument(
        "--pose_source",
        type=str,
        default="auto",
        choices=[
            "auto",
            "abs_head",
        ],
        help=(
            "Pose output source for each chunk. "
            "'auto' and 'abs_head' both use the absolute pose head (the only supported mode). "
            "The legacy 'final_abs' and 'relative' sources have been removed."
        ),
    )
    parser.add_argument("--symm_mode", type=str, default="none", choices=["none", "continuous", "discrete"])
    parser.add_argument("--symm_axis", type=float, nargs=3, default=None, help="Symmetry axis for continuous mode")
    parser.add_argument(
        "--ar_context_size",
        type=int,
        default=3,
        help="Number of context views per AR chunk (default 3, should match training TRAIN_NUM_CONTEXT_VIEWS)",
    )
    parser.add_argument(
        "--ar_target_size",
        type=int,
        default=3,
        help="Number of target views per AR chunk (default 3, should match training TRAIN_NUM_TARGET_VIEWS)",
    )
    parser.add_argument(
        "--temporal_smooth",
        action="store_true",
        help="Apply causal temporal smoothing on final poses before saving/visualization/context reuse",
    )
    parser.add_argument(
        "--temporal_smooth_rot_alpha",
        type=float,
        default=0.35,
        help="Rotation smoothing strength in [0, 1]; smaller values are smoother",
    )
    parser.add_argument(
        "--temporal_smooth_trans_alpha",
        type=float,
        default=0.30,
        help="Translation smoothing strength in [0, 1]; smaller values are smoother",
    )
    parser.add_argument(
        "--roi_temporal_smooth",
        action="store_true",
        help="Apply causal temporal smoothing on predicted ROI parameters before reusing them in later AR chunks",
    )
    parser.add_argument(
        "--roi_temporal_smooth_type",
        type=str,
        default="adaptive_ema",
        choices=["ema", "adaptive_ema"],
        help="ROI temporal smoothing type",
    )
    parser.add_argument(
        "--roi_temporal_smooth_alpha",
        type=float,
        default=0.40,
        help="Fixed alpha for ROI EMA in [0, 1]; smaller values are smoother",
    )
    parser.add_argument(
        "--roi_temporal_smooth_min_alpha",
        type=float,
        default=0.15,
        help="Minimum alpha for adaptive ROI EMA in [0, 1]",
    )
    parser.add_argument(
        "--roi_temporal_smooth_max_alpha",
        type=float,
        default=0.70,
        help="Maximum alpha for adaptive ROI EMA in [0, 1]",
    )
    parser.add_argument(
        "--mask_postproc",
        type=str,
        default="none",
        choices=["none", "largest_cc", "overlap_cc"],
        help="Post-process auto-mask before ROI extraction/refiner observation",
    )
    parser.add_argument(
        "--mask_prev_dilate_kernel",
        type=int,
        default=11,
        help="Odd kernel size used to dilate previous clean mask for overlap-guided mask selection",
    )
    parser.add_argument(
        "--mask_prev_gate",
        action="store_true",
        help="Intersect current mask with dilated previous clean mask before component filtering when possible",
    )
    parser.add_argument(
        "--mask_post_dilate_kernel",
        type=int,
        default=3,
        help="Odd kernel size used to dilate selected clean mask",
    )
    parser.add_argument(
        "--mask_post_open_kernel",
        type=int,
        default=0,
        help="Odd kernel size used for mask opening before close/dilate; useful for removing thin spurs",
    )
    parser.add_argument(
        "--mask_post_close_kernel",
        type=int,
        default=3,
        help="Odd kernel size used for mask closing after component selection",
    )
    parser.add_argument(
        "--mask_fallback_to_prev",
        action="store_true",
        help="Fallback to previous clean mask when current frame post-processed mask is too unstable",
    )
    parser.add_argument(
        "--external_mask_pad_scale",
        type=float,
        default=None,
        help=(
            "Optional inference-only ROI pad scale used when external masks are available. "
            "If unset, falls back to cfg.INPUT.DZI_PAD_SCALE."
        ),
    )
    parser.add_argument(
        "--context_pose_min_conf",
        type=float,
        default=float(os.environ.get("GDRN_CONTEXT_POSE_MIN_CONF", "0.05")),
        help=(
            "Minimum per-frame geo-prior object confidence required for this frame to be reused "
            "as future AR context pose. Set <=0 to disable confidence-based context gating."
        ),
    )

    args = parser.parse_args()

    if args.roi_mode == "external_bbox":
        raise NotImplementedError(
            "--roi_mode external_bbox is not yet implemented for multiview AR inference. "
            "Use --mask_dir to supply per-frame external masks instead."
        )

    cfg = Config.fromfile(args.config)
    register_datasets_in_cfg(cfg)
    if (args.save_coor_xyz_npy or args.save_pnp_input_debug) and (not bool(cfg.TEST.get("USE_PNP", False))):
        print("[INFO] Enabling cfg.TEST.USE_PNP=True for debug tensor dumping.")
        cfg.TEST.USE_PNP = True
    if args.save_pnp_input_debug:
        cfg.TEST.SAVE_PNP_INTERNALS = True

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

    expected_pnp_in = get_pnp_net_input_channels(cfg.MODEL.CDPN)
    actual_pnp_in = None
    if hasattr(model, "pnp_net") and hasattr(model.pnp_net, "features") and len(model.pnp_net.features) > 0:
        first_layer = model.pnp_net.features[0]
        if hasattr(first_layer, "in_channels"):
            actual_pnp_in = int(first_layer.in_channels)
    if actual_pnp_in is not None and actual_pnp_in != int(expected_pnp_in):
        raise RuntimeError(
            f"PnP input channel mismatch at startup: config expects {int(expected_pnp_in)} "
            f"but model first conv has {actual_pnp_in}. "
            f"Please ensure inference builder uses the same channel formula as training and check config/checkpoint pairing."
        )
    print(f"PnP channels: expected={int(expected_pnp_in)}, model_first_conv_in={actual_pnp_in}")

    print("ROI source: mask-driven inference via forward_infer")
    if args.mask_dir is None:
        print("Mask source: model-predicted mask")
    else:
        print("Mask source: external per-frame mask")

    if args.mask_dir is not None:
        if not osp.isdir(args.mask_dir):
            raise ValueError(f"--mask_dir is not a directory: {args.mask_dir}")
        print(f"Using external masks from {args.mask_dir}")
        print("External masks will drive ROI extraction, mask attention, and refiner observation")

    image_exts = [args.image_ext, args.image_ext.lower(), args.image_ext.upper()]
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(osp.join(args.image_dir, f"*{ext}")))
        image_files.extend(glob.glob(osp.join(args.image_dir, f"*{ext.replace('.', '')}")))
    image_files = sorted(list(set(image_files)))

    if len(image_files) < 1:
        raise ValueError(f"Need at least 1 frame, got {len(image_files)}")

    results_dir = args.output_dir if args.output_dir is not None else osp.join(args.image_dir, "results_vid")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(osp.join(results_dir, "depth"), exist_ok=True)
    os.makedirs(osp.join(results_dir, "mask"), exist_ok=True)
    os.makedirs(osp.join(results_dir, "pointcloud"), exist_ok=True)
    print(f"Found {len(image_files)} frames")
    print(f"Results will be saved to {results_dir}")

    dset_meta = MetadataCatalog.get(args.dataset_name)
    data_ref = ref.__dict__[dset_meta.ref_key]
    model_info = dict(data_ref.get_models_info()[str(args.obj_cls + 1)])
    num_regions = int(cfg.MODEL.CDPN.ROT_HEAD.NUM_REGIONS)
    if num_regions > 1 and hasattr(data_ref, "get_fps_points"):
        try:
            loaded_fps_points = data_ref.get_fps_points()
            obj_key = str(args.obj_cls + 1)
            fps_obj = loaded_fps_points.get(obj_key, None) if isinstance(loaded_fps_points, dict) else None
            fps_points = None
            if isinstance(fps_obj, dict):
                fps_key_center = f"fps{num_regions}_and_center"
                fps_key_plain = f"fps{num_regions}"
                if fps_key_center in fps_obj:
                    fps_points = np.asarray(fps_obj[fps_key_center], dtype=np.float32)
                    if fps_points.ndim == 2 and fps_points.shape[0] == num_regions + 1:
                        fps_points = fps_points[:-1]
                elif fps_key_plain in fps_obj:
                    fps_points = np.asarray(fps_obj[fps_key_plain], dtype=np.float32)
            if fps_points is not None and fps_points.ndim == 2 and fps_points.shape == (num_regions, 3):
                model_info["fps_points"] = fps_points
            else:
                print(
                    f"[WARN] Could not attach valid fps_points for obj={obj_key}, num_regions={num_regions}; "
                    "region debug/recompute will fallback."
                )
        except Exception as e:
            print(
                f"[WARN] Failed to load fps_points for obj={args.obj_cls + 1}, num_regions={num_regions}: {e}"
            )
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
        external_mask = None
        mask_path = None
        if args.mask_dir is not None:
            mask_path = _find_mask_path_for_image(image_name, args.mask_dir)
            if mask_path is None:
                raise ValueError(f"No external mask found for frame {image_name} in {args.mask_dir}")
            external_mask = _load_external_mask(mask_path, image.shape[:2])
        h, w = image.shape[:2]
        bbox = [0.0, 0.0, float(max(w - 1, 1)), float(max(h - 1, 1))]
        obj_cls = args.obj_cls

        view_data, obj_cls = preprocess_single_view(cfg, image, bbox, obj_cls, args.dataset_name)
        if external_mask is not None:
            target_hw = view_data["input_image"].shape[1:]
            view_data["input_obj_mask"] = _load_external_mask(mask_path, target_hw)
            view_data["input_obj_mask_path"] = mask_path
            view_data["roi_source"] = "external_mask"
        else:
            view_data["roi_source"] = "predicted_mask"
        view_data_list.append(view_data)
        image_names.append(image_name)
        obj_cls_list.append(int(obj_cls))

    obj_cls_mv = obj_cls_list[0]
    if any(c != obj_cls_mv for c in obj_cls_list):
        raise ValueError("Inconsistent obj_cls across frames in one clip")

    windows = build_autoregressive_windows(
        len(view_data_list),
        chunk_size=args.ar_context_size + args.ar_target_size,
        context_size=args.ar_context_size,
        target_size=args.ar_target_size,
    )
    if len(windows) == 0:
        raise ValueError(f"Need at least 1 frame for AR inference, got {len(view_data_list)}")
    chunk_size = args.ar_context_size + args.ar_target_size
    print(
        f"Running autoregressive multiview inference: {len(windows)} chunks, "
        f"chunk_size={chunk_size} ({args.ar_context_size}-context+{args.ar_target_size}-target), "
        f"schedule=[first:all-target, next:{args.ar_context_size}-context+{args.ar_target_size}-target]"
    )

    all_results = {}
    pose_results_by_global_idx = {}
    roi_results_by_global_idx = {}
    mask_results_by_global_idx = {}
    x_ref = None
    prev_smoothed_rot = None
    prev_smoothed_trans = None
    prev_smoothed_roi_center = None
    prev_smoothed_scale = None
    prev_smoothed_roi_wh = None
    points_obj_all = []
    colors_all = []
    coor_xyz_missing_warned = False
    pnp_decode_missing_warned = False
    pnp_input_missing_warned = False
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

    for chunk_i, window_spec in enumerate(windows):
        window_global_ids = window_spec["window_global_ids"]
        window_view_data = [view_data_list[i] for i in window_global_ids]
        batch = make_multiview_batch(window_view_data, obj_cls_mv)
        for local_i, gid in enumerate(window_global_ids):
            roi_item = roi_results_by_global_idx.get(int(gid), None)
            if roi_item is None:
                continue
            batch["roi_center"][0, local_i] = torch.as_tensor(roi_item["roi_center"], dtype=batch["roi_center"].dtype)
            batch["scale"][0, local_i] = torch.as_tensor(roi_item["scale"], dtype=batch["scale"].dtype)
            batch["roi_wh"][0, local_i] = torch.as_tensor(roi_item["roi_wh"], dtype=batch["roi_wh"].dtype)
            batch["resize_ratio"][0, local_i] = torch.as_tensor(roi_item["resize_ratio"], dtype=batch["resize_ratio"].dtype)
        target_idx = window_spec["target_local_ids"]
        if len(target_idx) == 0:
            continue
        first_target_global = window_global_ids[target_idx[0]]
        prev_target_global = first_target_global - 1
        prev_clean_mask = mask_results_by_global_idx.get(int(prev_target_global), None)
        if prev_clean_mask is not None:
            batch["prev_target_mask"] = torch.as_tensor(prev_clean_mask[None, None], dtype=torch.float32)

        # Record already-predicted poses for context availability/debug metadata.
        ctx_rot_bt, ctx_trans_bt, ctx_valid = build_window_pose_context(window_global_ids, pose_results_by_global_idx)
        batch["gt_ego_rot"] = ctx_rot_bt
        batch["gt_trans"] = ctx_trans_bt
        # Per-view validity: (1, N) bool tensor so forward_infer can filter out placeholder poses.
        batch["gt_pose_valid"] = torch.as_tensor(ctx_valid[None], dtype=torch.bool)  # [1, N]

        # Determine whether the required context-last frame is available.
        context_last_local = max(min(target_idx) - 1, 0)
        has_context_last = bool(ctx_valid[context_last_local])

        print(
            f"[chunk {chunk_i+1}/{len(windows)}] frames={window_global_ids[0]}..{window_global_ids[-1]}, "
            f"mode={window_spec['mode']}, predict_count={len(target_idx)}"
        )
        out_dict = inference_multiview(
            model,
            batch,
            target_idx=target_idx,
            model_infos=[model_info],
            device=args.device,
            mask_thr=args.mask_thr,
            min_mask_pixels=args.min_mask_pixels,
            mask_postproc=args.mask_postproc,
            mask_prev_dilate_kernel=args.mask_prev_dilate_kernel,
            mask_prev_gate=args.mask_prev_gate,
            mask_post_open_kernel=args.mask_post_open_kernel,
            mask_post_dilate_kernel=args.mask_post_dilate_kernel,
            mask_post_close_kernel=args.mask_post_close_kernel,
            mask_fallback_to_prev=args.mask_fallback_to_prev,
            external_mask_pad_scale=args.external_mask_pad_scale,
        )
        if args.save_coor_xyz_npy and (not coor_xyz_missing_warned):
            has_coor_outputs = all(k in out_dict for k in ("coor_x", "coor_y", "coor_z"))
            if not has_coor_outputs:
                print(
                    "[WARN] --save_coor_xyz_npy enabled but coor outputs are missing in out_dict. "
                    "Set cfg.TEST.USE_PNP=True to dump coor_z/xyz."
                )
                coor_xyz_missing_warned = True

        gp_prior_conf_mean_per_target = None
        gp_prior_amb_mean_per_target = None
        gp_prior_conf_obj_mean_per_target = None
        gp_all_target_fallback = None
        gp_bank_valid_ratio = None
        gp_bank_conf_mean = None
        gp_bank_weight_max_mean = None
        gp_retrieved_gate_mean = None
        if "prior_conf_64" in out_dict:
            prior_conf = out_dict["prior_conf_64"].detach()
            if prior_conf.ndim == 5:
                prior_conf = prior_conf[0]
            if prior_conf.ndim == 4 and prior_conf.shape[0] == len(target_idx):
                gp_prior_conf_mean_per_target = prior_conf.mean(dim=(1, 2, 3)).cpu().numpy()
                if "prior_ambiguity_64" in out_dict:
                    prior_amb = out_dict["prior_ambiguity_64"].detach()
                    if prior_amb.ndim == 5:
                        prior_amb = prior_amb[0]
                    if prior_amb.ndim == 4 and prior_amb.shape[0] == len(target_idx):
                        gp_prior_amb_mean_per_target = prior_amb.mean(dim=(1, 2, 3)).cpu().numpy()
                if "target_view_mask_clean" in out_dict:
                    target_mask_clean = out_dict["target_view_mask_clean"].detach()
                    if target_mask_clean.ndim == 5:
                        target_mask_clean = target_mask_clean[0, :, 0]
                    elif target_mask_clean.ndim == 4:
                        target_mask_clean = target_mask_clean[:, 0]
                    if target_mask_clean.ndim == 3 and target_mask_clean.shape[0] == len(target_idx):
                        mask_ds = torch.nn.functional.interpolate(
                            target_mask_clean.unsqueeze(1),
                            size=prior_conf.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        valid = (mask_ds > 0.5).to(dtype=prior_conf.dtype)
                        gp_prior_conf_obj_mean_per_target = (
                            (prior_conf * valid).sum(dim=(1, 2, 3)) / (valid.sum(dim=(1, 2, 3)) + 1e-6)
                        ).cpu().numpy()
            gp_all_target_fallback = bool(out_dict.get("geo_prior_all_target_fallback", False))
            if "geo_prior_bank_valid_ratio" in out_dict:
                gp_bank_valid_ratio = float(out_dict["geo_prior_bank_valid_ratio"].detach().cpu().item())
            if "geo_prior_bank_conf_mean" in out_dict:
                gp_bank_conf_mean = float(out_dict["geo_prior_bank_conf_mean"].detach().cpu().item())
            if "geo_prior_bank_weight_max_mean" in out_dict:
                gp_bank_weight_max_mean = float(out_dict["geo_prior_bank_weight_max_mean"].detach().cpu().item())
            if "geo_prior_retrieved_gate_mean" in out_dict:
                gp_retrieved_gate_mean = float(out_dict["geo_prior_retrieved_gate_mean"].detach().cpu().item())

        debug_pnp_t_raw = out_dict["pnp_pred_t_raw"].detach().cpu().numpy() if "pnp_pred_t_raw" in out_dict else None
        debug_pnp_rot_raw = out_dict["pnp_pred_rot_raw"].detach().cpu().numpy() if "pnp_pred_rot_raw" in out_dict else None
        if args.save_pnp_decode_debug and (not pnp_decode_missing_warned):
            if debug_pnp_t_raw is None:
                print(
                    "[WARN] --save_pnp_decode_debug enabled but pnp_pred_t_raw is missing in out_dict."
                )
                pnp_decode_missing_warned = True
        if args.save_pnp_input_debug and (not pnp_input_missing_warned):
            has_any_pnp_input_debug = any(
                k in out_dict
                for k in (
                    "dbg_pnp_coor_feat_base",
                    "dbg_pnp_coor_feat_final",
                    "dbg_pnp_region_softmax",
                )
            )
            if not has_any_pnp_input_debug:
                print(
                    "[WARN] --save_pnp_input_debug enabled but PnP input debug tensors are missing in out_dict. "
                    "Set cfg.TEST.SAVE_PNP_INTERNALS=True."
                )
                pnp_input_missing_warned = True

        has_abs_pose = ("abs_rot" in out_dict) and ("abs_trans" in out_dict)
        debug_abs_rot = out_dict["abs_rot"].cpu().numpy() if has_abs_pose else None
        debug_abs_trans = out_dict["abs_trans"].cpu().numpy() if has_abs_pose else None
        debug_roi_cams = out_dict["infer_roi_cams"].cpu().numpy() if "infer_roi_cams" in out_dict else None

        if has_abs_pose:
            pred_rot = debug_abs_rot
            pred_trans = debug_abs_trans
            pose_source_name = "abs_head"
        else:
            pred_rot = out_dict["rot"].cpu().numpy()
            pred_trans = out_dict["trans"].cpu().numpy()
            pose_source_name = "head_fallback"
        if pred_rot.shape[0] != len(target_idx) or pred_trans.shape[0] != len(target_idx):
            raise RuntimeError(
                f"Unexpected output shape: rot={pred_rot.shape}, trans={pred_trans.shape}, "
                f"expected first dim={len(target_idx)}"
            )

        chunk_debug_path = os.environ.get("GDRN_INFER_CHUNK_DEBUG_JSONL", "").strip()
        if chunk_debug_path:
            context_valid_ratio = float(np.mean(ctx_valid.astype(np.float32))) if len(ctx_valid) > 0 else 0.0
            payload = {
                "chunk_index": int(chunk_i),
                "mode": window_spec["mode"],
                "window_global_ids": [int(x) for x in window_global_ids],
                "target_local_ids": [int(x) for x in target_idx],
                "target_global_ids": [int(window_global_ids[x]) for x in target_idx],
                "pose_source": pose_source_name,
                "has_context_last": bool(has_context_last),
                "context_valid_ratio": context_valid_ratio,
                "gp_all_target_fallback": (
                    None if gp_all_target_fallback is None else bool(gp_all_target_fallback)
                ),
                "gp_prior_conf_mean": (
                    None
                    if gp_prior_conf_mean_per_target is None
                    else float(np.mean(gp_prior_conf_mean_per_target))
                ),
                "gp_prior_conf_obj_mean": (
                    None
                    if gp_prior_conf_obj_mean_per_target is None
                    else float(np.mean(gp_prior_conf_obj_mean_per_target))
                ),
                "gp_prior_ambiguity_mean": (
                    None
                    if gp_prior_amb_mean_per_target is None
                    else float(np.mean(gp_prior_amb_mean_per_target))
                ),
                "gp_bank_valid_ratio": gp_bank_valid_ratio,
                "gp_bank_conf_mean": gp_bank_conf_mean,
                "gp_bank_weight_max_mean": gp_bank_weight_max_mean,
                "gp_retrieved_gate_mean": gp_retrieved_gate_mean,
            }
            _maybe_append_jsonl(chunk_debug_path, payload)

        translation_debug_path = os.environ.get("GDRN_TRANSLATION_DEBUG_JSONL", "").strip()

        for out_i, local_frame_i in enumerate(target_idx):
            global_frame_i = window_global_ids[local_frame_i]
            raw_rot = pred_rot[out_i].astype(np.float32)
            raw_trans = pred_trans[out_i].astype(np.float32)
            cur_rot = raw_rot.copy()
            cur_trans = raw_trans.copy()

            if args.temporal_smooth:
                cur_rot, cur_trans = smooth_pose_temporally(
                    cur_rot=cur_rot,
                    cur_trans=cur_trans,
                    prev_rot=prev_smoothed_rot,
                    prev_trans=prev_smoothed_trans,
                    rot_alpha=args.temporal_smooth_rot_alpha,
                    trans_alpha=args.temporal_smooth_trans_alpha,
                )

            if args.symm_mode == "continuous":
                if args.symm_axis is None:
                    raise ValueError("symm_mode=continuous requires --symm_axis")
                cur_rot, x_ref = stabilize_rotation_given_axis_single(cur_rot, np.array(args.symm_axis), x_ref)

            prev_smoothed_rot = cur_rot.copy()
            prev_smoothed_trans = cur_trans.copy()

            image_name = image_names[global_frame_i]
            image_id = osp.splitext(image_name)[0]
            all_results[image_name] = {
                "rotation": cur_rot.tolist(),
                "translation": cur_trans.tolist(),
                "raw_rotation": raw_rot.tolist(),
                "raw_translation": raw_trans.tolist(),
                "source_index": int(global_frame_i),
                "target_mode": window_spec["mode"],
                "chunk_index": int(chunk_i),
                "pose_source": pose_source_name,
                "temporal_smooth": bool(args.temporal_smooth),
                "temporal_smooth_rot_alpha": float(args.temporal_smooth_rot_alpha),
                "temporal_smooth_trans_alpha": float(args.temporal_smooth_trans_alpha),
                "mask_postproc": args.mask_postproc,
                "mask_prev_dilate_kernel": int(args.mask_prev_dilate_kernel),
                "mask_prev_gate": bool(args.mask_prev_gate),
                "mask_post_open_kernel": int(args.mask_post_open_kernel),
                "mask_post_dilate_kernel": int(args.mask_post_dilate_kernel),
                "mask_post_close_kernel": int(args.mask_post_close_kernel),
                "mask_fallback_to_prev": bool(args.mask_fallback_to_prev),
                "external_mask_on_inputs": False,
                "external_mask_pad_scale": (
                    None if args.external_mask_pad_scale is None else float(args.external_mask_pad_scale)
                ),
                "has_context_last": bool(has_context_last),
                "has_abs_pose_output": bool(has_abs_pose),
                "context_pose_min_conf": float(args.context_pose_min_conf),
            }
            if args.save_pnp_decode_debug and debug_pnp_t_raw is not None:
                pnp_t_raw = debug_pnp_t_raw[out_i].astype(np.float32).reshape(-1)
                all_results[image_name]["pnp_pred_t_raw"] = pnp_t_raw.tolist()
                pnp_rot_raw = None
                if debug_pnp_rot_raw is not None:
                    pnp_rot_raw = debug_pnp_rot_raw[out_i].astype(np.float32).reshape(-1)
                    all_results[image_name]["pnp_pred_rot_raw"] = pnp_rot_raw.tolist()
                all_results[image_name].update(
                    save_pnp_decode_debug_npy(
                        image_id=image_id,
                        results_dir=results_dir,
                        pnp_t_raw=pnp_t_raw,
                        pnp_rot_raw=pnp_rot_raw,
                    )
                )

                decode_dbg = {
                    "trans_type": str(cfg.MODEL.CDPN.PNP_NET.TRANS_TYPE),
                    "z_type": str(cfg.MODEL.CDPN.PNP_NET.Z_TYPE),
                }
                if (
                    "infer_resize_ratios" in out_dict
                    and "infer_roi_centers" in out_dict
                    and "infer_roi_whs" in out_dict
                ):
                    raw_resize_ratio_dbg = float(out_dict["infer_resize_ratios"][0, out_i].detach().cpu().item())
                    raw_roi_center_dbg = (
                        out_dict["infer_roi_centers"][0, out_i].detach().cpu().numpy().astype(np.float32).reshape(2)
                    )
                    raw_roi_wh_dbg = (
                        out_dict["infer_roi_whs"][0, out_i].detach().cpu().numpy().astype(np.float32).reshape(2)
                    )
                    roi_cam_dbg = batch["roi_cam"][0, local_frame_i].detach().cpu().numpy().astype(np.float32)
                    decode_dbg.update(
                        {
                            "raw_resize_ratio": raw_resize_ratio_dbg,
                            "raw_roi_center": raw_roi_center_dbg.tolist(),
                            "raw_roi_wh": raw_roi_wh_dbg.tolist(),
                            "raw_roi_cam": roi_cam_dbg.tolist(),
                        }
                    )
                    if str(cfg.MODEL.CDPN.PNP_NET.TRANS_TYPE) == "centroid_z":
                        z_type = str(cfg.MODEL.CDPN.PNP_NET.Z_TYPE).upper()
                        tz_from_raw = None
                        if z_type == "REL":
                            tz_from_raw = float(pnp_t_raw[2] * raw_resize_ratio_dbg)
                        elif z_type == "ABS":
                            tz_from_raw = float(pnp_t_raw[2])
                        if tz_from_raw is not None:
                            cx_abs = float(pnp_t_raw[0] * raw_roi_wh_dbg[0] + raw_roi_center_dbg[0])
                            cy_abs = float(pnp_t_raw[1] * raw_roi_wh_dbg[1] + raw_roi_center_dbg[1])
                            fx = float(roi_cam_dbg[0, 0])
                            fy = float(roi_cam_dbg[1, 1])
                            px = float(roi_cam_dbg[0, 2])
                            py = float(roi_cam_dbg[1, 2])
                            tx_from_raw = float(tz_from_raw * (cx_abs - px) / max(fx, 1e-6))
                            ty_from_raw = float(tz_from_raw * (cy_abs - py) / max(fy, 1e-6))
                            decode_dbg.update(
                                {
                                    "tz_from_pred_t_raw": tz_from_raw,
                                    "tx_from_pred_t_raw": tx_from_raw,
                                    "ty_from_pred_t_raw": ty_from_raw,
                                    "decoded_t_minus_raw_trans": [
                                        float(tx_from_raw - raw_trans[0]),
                                        float(ty_from_raw - raw_trans[1]),
                                        float(tz_from_raw - raw_trans[2]),
                                    ],
                                }
                            )
                all_results[image_name]["pnp_decode_debug"] = decode_dbg
            if debug_abs_rot is not None and debug_abs_trans is not None:
                all_results[image_name]["abs_head_rotation"] = debug_abs_rot[out_i].astype(np.float32).tolist()
                all_results[image_name]["abs_head_translation"] = debug_abs_trans[out_i].astype(np.float32).tolist()
            if "input_obj_mask_path" in view_data_list[global_frame_i]:
                all_results[image_name]["external_mask_path"] = view_data_list[global_frame_i]["input_obj_mask_path"]
            if "roi_source" in view_data_list[global_frame_i]:
                all_results[image_name]["roi_source"] = view_data_list[global_frame_i]["roi_source"]
            if gp_prior_conf_mean_per_target is not None:
                all_results[image_name]["gp_prior_conf_mean"] = float(gp_prior_conf_mean_per_target[out_i])
            if gp_prior_conf_obj_mean_per_target is not None:
                all_results[image_name]["gp_prior_conf_obj_mean"] = float(gp_prior_conf_obj_mean_per_target[out_i])
            if gp_prior_amb_mean_per_target is not None:
                all_results[image_name]["gp_prior_ambiguity_mean"] = float(gp_prior_amb_mean_per_target[out_i])
            if gp_all_target_fallback is not None:
                all_results[image_name]["gp_all_target_fallback"] = bool(gp_all_target_fallback)
            if gp_bank_valid_ratio is not None:
                all_results[image_name]["gp_bank_valid_ratio"] = float(gp_bank_valid_ratio)
            if gp_bank_conf_mean is not None:
                all_results[image_name]["gp_bank_conf_mean"] = float(gp_bank_conf_mean)
            if gp_bank_weight_max_mean is not None:
                all_results[image_name]["gp_bank_weight_max_mean"] = float(gp_bank_weight_max_mean)
            if gp_retrieved_gate_mean is not None:
                all_results[image_name]["gp_retrieved_gate_mean"] = float(gp_retrieved_gate_mean)
            context_pose_valid = True
            if (
                float(args.context_pose_min_conf) > 0.0
                and gp_all_target_fallback is not None
                and (not bool(gp_all_target_fallback))
                and gp_prior_conf_obj_mean_per_target is not None
            ):
                context_pose_valid = bool(float(gp_prior_conf_obj_mean_per_target[out_i]) >= float(args.context_pose_min_conf))
            all_results[image_name]["context_pose_valid"] = bool(context_pose_valid)
            pose_results_by_global_idx[int(global_frame_i)] = {
                "rotation": raw_rot.tolist(),      # raw (unsmoothed) for VI context cue
                "translation": raw_trans.tolist(),  # raw (unsmoothed) for VI context cue
                "context_pose_valid": bool(context_pose_valid),
            }
            if "target_view_mask_clean" in out_dict:
                mask_results_by_global_idx[int(global_frame_i)] = (
                    out_dict["target_view_mask_clean"][0, out_i, 0].detach().cpu().numpy() > 0.5
                ).astype(np.uint8)
            if "infer_roi_centers" in out_dict and "infer_scales" in out_dict and "infer_roi_whs" in out_dict and "infer_resize_ratios" in out_dict:
                raw_roi_center = out_dict["infer_roi_centers"][0, out_i].detach().cpu().numpy().astype(np.float32).reshape(2)
                raw_scale = out_dict["infer_scales"][0, out_i].detach().cpu().numpy().astype(np.float32).reshape(-1)
                raw_roi_wh = out_dict["infer_roi_whs"][0, out_i].detach().cpu().numpy().astype(np.float32).reshape(2)
                roi_center = raw_roi_center.copy()
                scale = raw_scale.copy()
                roi_wh = raw_roi_wh.copy()
                roi_alpha_used = 1.0
                if args.roi_temporal_smooth:
                    roi_center, scale, roi_wh, roi_alpha_used = smooth_roi_temporally(
                        cur_roi_center=roi_center,
                        cur_scale=scale,
                        cur_roi_wh=roi_wh,
                        prev_roi_center=prev_smoothed_roi_center,
                        prev_scale=prev_smoothed_scale,
                        prev_roi_wh=prev_smoothed_roi_wh,
                        smooth_type=args.roi_temporal_smooth_type,
                        alpha=args.roi_temporal_smooth_alpha,
                        min_alpha=args.roi_temporal_smooth_min_alpha,
                        max_alpha=args.roi_temporal_smooth_max_alpha,
                    )
                prev_smoothed_roi_center = roi_center.copy()
                prev_smoothed_scale = scale.copy()
                prev_smoothed_roi_wh = roi_wh.copy()
                resize_ratio = float(cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES / max(float(scale[0]), 1e-6))
                roi_results_by_global_idx[int(global_frame_i)] = {
                    "roi_center": raw_roi_center.tolist(),   # raw (unsmoothed) for VI seed
                    "scale": raw_scale.tolist(),             # raw (unsmoothed) for VI seed
                    "roi_wh": raw_roi_wh.tolist(),
                    "resize_ratio": resize_ratio,
                }
                all_results[image_name].update(
                    {
                        "roi_center": roi_center.tolist(),
                        "scale": scale.tolist(),
                        "roi_wh": roi_wh.tolist(),
                        "resize_ratio": resize_ratio,
                        "raw_roi_center": raw_roi_center.tolist(),
                        "raw_scale": raw_scale.tolist(),
                        "raw_roi_wh": raw_roi_wh.tolist(),
                        "raw_resize_ratio": float(out_dict["infer_resize_ratios"][0, out_i].detach().cpu().item()),
                        "roi_temporal_smooth": bool(args.roi_temporal_smooth),
                        "roi_temporal_smooth_type": args.roi_temporal_smooth_type,
                        "roi_temporal_smooth_alpha": float(args.roi_temporal_smooth_alpha),
                        "roi_temporal_smooth_min_alpha": float(args.roi_temporal_smooth_min_alpha),
                        "roi_temporal_smooth_max_alpha": float(args.roi_temporal_smooth_max_alpha),
                        "roi_temporal_smooth_alpha_used": float(roi_alpha_used),
                    }
                )
            if debug_roi_cams is not None:
                roi_cam = debug_roi_cams[0, out_i].astype(np.float32)
                all_results[image_name]["roi_cam"] = roi_cam.tolist()
            else:
                roi_cam = None

            all_results[image_name].update(
                save_depth_and_mask_multiview(
                    out_dict=out_dict,
                    out_i=out_i,
                    image_id=image_id,
                    results_dir=results_dir,
                )
            )
            if args.save_coor_xyz_npy:
                all_results[image_name].update(
                    save_coor_xyz_npy_multiview(
                        out_dict=out_dict,
                        out_i=out_i,
                        image_id=image_id,
                        results_dir=results_dir,
                    )
                )
            if args.save_pnp_input_debug:
                all_results[image_name].update(
                    save_pnp_input_debug_npy_multiview(
                        out_dict=out_dict,
                        out_i=out_i,
                        image_id=image_id,
                        results_dir=results_dir,
                    )
                )
            if translation_debug_path:
                payload = {
                    "frame": image_name,
                    "frame_index": int(global_frame_i),
                    "chunk_index": int(chunk_i),
                    "target_local_idx": int(out_i),
                    "window_local_idx": int(local_frame_i),
                    "pose_source": pose_source_name,
                    "has_context_last": bool(has_context_last),
                    "chosen_translation": raw_trans.astype(np.float32).tolist(),
                    "chosen_proj_xy": (
                        _project_translation_to_image_xy(raw_trans, roi_cam)
                        if roi_cam is not None else None
                    ),
                    "abs_head_translation": (
                        debug_abs_trans[out_i].astype(np.float32).tolist()
                        if debug_abs_trans is not None else None
                    ),
                    "abs_head_proj_xy": (
                        _project_translation_to_image_xy(debug_abs_trans[out_i], roi_cam)
                        if (roi_cam is not None and debug_abs_trans is not None) else None
                    ),
                    "roi_center": all_results[image_name].get("roi_center"),
                    "raw_roi_center": all_results[image_name].get("raw_roi_center"),
                    "scale": all_results[image_name].get("scale"),
                    "raw_scale": all_results[image_name].get("raw_scale"),
                    "roi_wh": all_results[image_name].get("roi_wh"),
                    "raw_roi_wh": all_results[image_name].get("raw_roi_wh"),
                    "resize_ratio": all_results[image_name].get("resize_ratio"),
                    "raw_resize_ratio": all_results[image_name].get("raw_resize_ratio"),
                    "roi_cam": roi_cam.tolist() if roi_cam is not None else None,
                    "external_mask_path": all_results[image_name].get("external_mask_path"),
                    "mask_path": all_results[image_name].get("mask_path"),
                    "mask_raw_path": all_results[image_name].get("mask_raw_path"),
                }
                _maybe_append_jsonl(translation_debug_path, payload)

            # Accumulate point cloud globally across chunks.
            pred_depth, pred_mask_raw = _get_pred_depth_mask_by_out_idx(out_dict, out_i)
            if pred_depth is not None:
                if pred_mask_raw.min() >= 0.0 and pred_mask_raw.max() <= 1.0:
                    pred_mask = pred_mask_raw
                else:
                    pred_mask = _sigmoid_numpy(pred_mask_raw)
                valid = np.isfinite(pred_depth) & (pred_depth > 0) & (pred_mask > 0.5)
                if np.any(valid):
                    K = batch["roi_cam"][0, local_frame_i].detach().cpu().numpy().astype(np.float32)
                    pts_cam = depth_to_cam_points(pred_depth, K, valid)
                    pts_obj = cam_to_obj(pts_cam.astype(np.float32), cur_rot.astype(np.float32), cur_trans.astype(np.float32))
                    if pts_obj.shape[0] > 0:
                        color_i = np.tile(view_palette[out_i % len(view_palette)][None, :], (pts_obj.shape[0], 1))
                        points_obj_all.append(pts_obj)
                        colors_all.append(color_i)

            if args.save_vis:
                vis_output_path = osp.join(results_dir, f"{image_id}_pred.png")
                if args.symm_mode == "none":
                    visualize_example(
                        K=cfg.TEST_CAM,
                        image=view_data_list[global_frame_i]["original_image"],
                        RT=np.concatenate([cur_rot, cur_trans[..., None]], axis=1),
                        size=model_size,
                        output_path=vis_output_path,
                    )
                else:
                    visualize_symmetric_object(
                        K=cfg.TEST_CAM,
                        image=view_data_list[global_frame_i]["original_image"],
                        RT=np.concatenate([cur_rot, cur_trans[..., None]], axis=1),
                        size=model_size,
                        symmetry_axis="z",
                        output_path=vis_output_path,
                        color=(0, 255, 0),
                        thickness=1,
                    )

    fused_pcd_path = ""
    if len(points_obj_all) > 0:
        points_obj = np.concatenate(points_obj_all, axis=0)
        colors = np.concatenate(colors_all, axis=0)
        max_points = 200000
        if points_obj.shape[0] > max_points:
            keep_idx = np.random.choice(points_obj.shape[0], max_points, replace=False)
            points_obj = points_obj[keep_idx]
            colors = colors[keep_idx]
        fused_pcd_path = osp.join(results_dir, "pointcloud", "pred_depth_fused_obj.ply")
        write_ply_xyzrgb(fused_pcd_path, points_obj.astype(np.float32), colors.astype(np.uint8))

    if fused_pcd_path:
        for image_name in all_results:
            all_results[image_name]["pointcloud_fused_obj_path"] = fused_pcd_path

    results_json_path = osp.join(results_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_json_path}")
    print(f"Predicted frames: {len(all_results)} / {len(image_files)}")


if __name__ == "__main__":
    main()
