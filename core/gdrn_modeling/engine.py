import logging
import os
import os.path as osp
import json
import torch
import torch.distributed as dist
import shutil
import threading
import subprocess
import shlex

import mmcv
import time
import cv2
import numpy as np
from collections import OrderedDict

from detectron2.utils.events import EventStorage
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)

from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.data import MetadataCatalog
from pytorch_lightning.lite import LightningLite  # import LightningLite

from lib.utils.utils import dprint, iprint, get_time_str

from core.utils import solver_utils
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.my_writer import MyCommonMetricPrinter, MyJSONWriter, MyPeriodicWriter, MyTensorboardXWriter
from core.utils.utils import (
    get_emb_show,
    get_emb_show_percentile,
    depth_to_cam_points,
    cam_to_obj,
    write_ply_xyzrgb,
)
from core.utils.data_utils import denormalize_image, crop_resize_by_warp_affine
from .data_loader import build_gdrn_train_loader, build_gdrn_test_loader
from .engine_utils import (
    batch_data,
    batch_data_rand_num_perm,
    get_out_coor,
    get_out_mask,
)
from .gdrn_evaluator import gdrn_inference_on_dataset, GDRN_Evaluator
from .gdrn_custom_evaluator import GDRN_EvaluatorCustom
from .view_interaction_utils import collect_view_interaction_model_kwargs
import ref


logger = logging.getLogger(__name__)

from core.utils.pose_vis_utils import render_pose_vis_frame, write_pose_vis_videos


def _stable_sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _project_translation_to_image_xy(trans, cam, eps=1e-6):
    if trans is None or cam is None:
        return None
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    cam = np.asarray(cam, dtype=np.float64).reshape(3, 3)
    z = float(trans[2])
    if not np.isfinite(z) or abs(z) < eps:
        return None
    fx = float(cam[0, 0])
    fy = float(cam[1, 1])
    px = float(cam[0, 2])
    py = float(cam[1, 2])
    if abs(fx) < eps or abs(fy) < eps:
        return None
    cx = fx * float(trans[0]) / z + px
    cy = fy * float(trans[1]) / z + py
    return [float(cx), float(cy)]


def _count_connected_components(mask_bin, min_area=0):
    mask_u8 = np.asarray(mask_bin, dtype=np.uint8)
    if mask_u8.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_u8.shape}")
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    areas = []
    centers = []
    min_area = int(max(min_area, 0))
    for comp_i in range(1, num_labels):
        area = int(stats[comp_i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        areas.append(area)
        centers.append([float(centroids[comp_i, 0]), float(centroids[comp_i, 1])])
    return len(areas), areas, centers


def _summarize_binary_mask(mask_bin, min_area=0):
    mask_np = np.asarray(mask_bin)
    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_np.shape}")
    mask_u8 = (mask_np > 0).astype(np.uint8)
    num_comp, comp_areas, comp_centers = _count_connected_components(mask_u8, min_area=min_area)
    return {
        "num_components": int(num_comp),
        "component_areas": comp_areas,
        "component_centers": comp_centers,
        "area": int(np.count_nonzero(mask_u8)),
    }


def _storage_latest_scalar(storage, name, default=None):
    try:
        return float(storage.history(name).latest())
    except KeyError:
        return default


def _json_compact(x):
    return json.dumps(x, ensure_ascii=True, separators=(",", ":"))


def _append_bad_case_stats(log_path, lines):
    log_dir = osp.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _write_bad_case_mask_images(
    bad_case_imgs_dir,
    iteration,
    local_i,
    view_i,
    noisy_mask_full,
    noisy_mask_roi,
    gt_mask_full=None,
    gt_mask_roi=None,
):
    os.makedirs(bad_case_imgs_dir, exist_ok=True)
    prefix = f"iter_{iteration:06d}_target_{local_i:02d}_view_{view_i:02d}"
    paths = {
        "noisy_mask_full_png": osp.join(bad_case_imgs_dir, f"{prefix}_noisy_mask_full.png"),
        "noisy_mask_roi_png": osp.join(bad_case_imgs_dir, f"{prefix}_noisy_mask_roi.png"),
    }
    cv2.imwrite(paths["noisy_mask_full_png"], (np.clip(noisy_mask_full, 0.0, 1.0) * 255.0).astype(np.uint8))
    cv2.imwrite(paths["noisy_mask_roi_png"], (np.clip(noisy_mask_roi, 0.0, 1.0) * 255.0).astype(np.uint8))
    if gt_mask_full is not None:
        paths["gt_mask_full_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_gt_mask_full.png")
        cv2.imwrite(paths["gt_mask_full_png"], (np.clip(gt_mask_full, 0.0, 1.0) * 255.0).astype(np.uint8))
    if gt_mask_roi is not None:
        paths["gt_mask_roi_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_gt_mask_roi.png")
        cv2.imwrite(paths["gt_mask_roi_png"], (np.clip(gt_mask_roi, 0.0, 1.0) * 255.0).astype(np.uint8))
    return paths


def _write_bad_case_target_images(
    cfg,
    bad_case_imgs_dir,
    iteration,
    local_i,
    view_i,
    batch,
    out_dict,
    target_idx_list,
    size,
):
    os.makedirs(bad_case_imgs_dir, exist_ok=True)
    prefix = f"iter_{iteration:06d}_target_{local_i:02d}_view_{view_i:02d}"
    paths = {}

    # Full input RGB used by DINOv3 branch.
    if "input_images" in batch:
        input_img = batch["input_images"][0, view_i].detach().cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        input_img = ((input_img * std) + mean) * 255.0
        input_img = np.clip(input_img.transpose(1, 2, 0), 0.0, 255.0).astype(np.uint8)
        paths["input_rgb_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_input_rgb.png")
        cv2.imwrite(paths["input_rgb_png"], cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

    # ROI crop fed to the pose heads.
    if "roi_img" in batch:
        roi_img = batch["roi_img"][0, view_i].detach().cpu().numpy()
        roi_img = denormalize_image(roi_img, cfg).transpose(1, 2, 0)
        roi_img = np.clip(roi_img, 0.0, 255.0).astype(np.uint8)
        paths["roi_rgb_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_roi_rgb.png")
        cv2.imwrite(paths["roi_rgb_png"], cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))

    # Predicted / GT masks and depth on the current target view.
    if (
        torch.is_tensor(out_dict.get("target_view_mask", None))
        and torch.is_tensor(out_dict.get("target_view_depth", None))
        and local_i < out_dict["target_view_mask"].shape[1]
    ):
        pred_mask_raw = out_dict["target_view_mask"][0, local_i, 0].detach().cpu().numpy()
        if pred_mask_raw.min() >= 0.0 and pred_mask_raw.max() <= 1.0:
            pred_mask_prob = pred_mask_raw.astype(np.float32)
        else:
            pred_mask_prob = _stable_sigmoid_np(pred_mask_raw)
        pred_mask_bin = pred_mask_prob > 0.5
        paths["pred_mask_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_pred_mask.png")
        cv2.imwrite(paths["pred_mask_png"], (np.clip(pred_mask_prob, 0.0, 1.0) * 255.0).astype(np.uint8))

        pred_depth = out_dict["target_view_depth"][0, local_i, 0].detach().cpu().numpy()
        pred_depth_valid = pred_mask_bin & np.isfinite(pred_depth) & (pred_depth > 0)

        gt_depth = None
        gt_depth_valid = None
        if "input_depths" in batch:
            gt_depth = batch["input_depths"][0, view_i].detach().cpu().numpy()
            gt_depth_valid = np.isfinite(gt_depth) & (gt_depth > 0)
        if "input_obj_masks" in batch:
            gt_mask = batch["input_obj_masks"][0, view_i].detach().cpu().numpy()
            paths["gt_mask_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_gt_mask.png")
            cv2.imwrite(paths["gt_mask_png"], (np.clip(gt_mask, 0.0, 1.0) * 255.0).astype(np.uint8))
            if gt_depth_valid is not None:
                gt_depth_valid = gt_depth_valid & (gt_mask > 0.5)

        if np.count_nonzero(pred_depth_valid) > 10:
            depth_min = float(np.percentile(pred_depth[pred_depth_valid], 2))
            depth_max = float(np.percentile(pred_depth[pred_depth_valid], 98))
        elif gt_depth is not None and np.any(gt_depth_valid):
            depth_min = float(np.percentile(gt_depth[gt_depth_valid], 2))
            depth_max = float(np.percentile(gt_depth[gt_depth_valid], 98))
        else:
            finite = np.isfinite(pred_depth)
            if np.any(finite):
                depth_min = float(np.percentile(pred_depth[finite], 2))
                depth_max = float(np.percentile(pred_depth[finite], 98))
            else:
                depth_min, depth_max = 0.0, 1.0
        if abs(depth_max - depth_min) < 1e-8:
            depth_max = depth_min + 1e-8

        pred_depth_gray = np.zeros_like(pred_depth, dtype=np.uint8)
        if np.any(pred_depth_valid):
            pred_depth_norm = np.clip((pred_depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
            pred_depth_gray[pred_depth_valid] = (pred_depth_norm[pred_depth_valid] * 255.0).astype(np.uint8)
        pred_depth_vis = cv2.applyColorMap(pred_depth_gray, cv2.COLORMAP_JET)
        pred_depth_vis[~pred_mask_bin] = 0
        paths["pred_depth_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_pred_depth.png")
        cv2.imwrite(paths["pred_depth_png"], pred_depth_vis)

        if gt_depth is not None and gt_depth_valid is not None:
            gt_depth_gray = np.zeros_like(pred_depth_gray)
            if np.any(gt_depth_valid):
                gt_depth_norm = np.clip((gt_depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
                gt_depth_gray[gt_depth_valid] = (gt_depth_norm[gt_depth_valid] * 255.0).astype(np.uint8)
            gt_depth_vis = cv2.applyColorMap(gt_depth_gray, cv2.COLORMAP_JET)
            gt_depth_vis[~gt_depth_valid] = 0
            paths["gt_depth_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_gt_depth.png")
            cv2.imwrite(paths["gt_depth_png"], gt_depth_vis)

            err_valid = pred_depth_valid & gt_depth_valid
            depth_err_gray = np.zeros_like(pred_depth_gray)
            if np.any(err_valid):
                depth_err = np.abs(pred_depth - gt_depth)
                depth_err_norm = np.clip(depth_err / max(depth_max - depth_min, 1e-8), 0.0, 1.0)
                depth_err_gray[err_valid] = (depth_err_norm[err_valid] * 255.0).astype(np.uint8)
            depth_err_vis = cv2.applyColorMap(depth_err_gray, cv2.COLORMAP_HOT)
            depth_err_vis[~err_valid] = 0
            paths["depth_err_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_depth_err.png")
            cv2.imwrite(paths["depth_err_png"], depth_err_vis)

    if size is not None and "ego_rot" in batch and "trans" in batch and "roi_cam" in batch:
        pose_vis_frames = []
        gt_rot = batch["ego_rot"][0, view_i].detach().cpu().numpy()
        gt_trans = batch["trans"][0, view_i].detach().cpu().numpy()
        if gt_rot.ndim == 3:
            gt_rot = gt_rot[0]
        if gt_trans.ndim > 1:
            gt_trans = gt_trans.ravel()
        gt_frame = render_pose_vis_frame(batch, view_i, size, gt_rot, gt_trans, "gt")
        paths["gt_pose_vis_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_gt_pose_vis.png")
        cv2.imwrite(paths["gt_pose_vis_png"], gt_frame)
        pose_vis_frames.append(gt_frame)

        if torch.is_tensor(out_dict.get("rot", None)) and torch.is_tensor(out_dict.get("trans", None)):
            if local_i < out_dict["rot"].shape[0] and local_i < out_dict["trans"].shape[0]:
                abs_frame = render_pose_vis_frame(
                    batch,
                    view_i,
                    size,
                    out_dict["rot"][local_i].detach().cpu().numpy(),
                    out_dict["trans"][local_i].detach().cpu().numpy(),
                    "abs_head",
                )
                paths["abs_head_pose_vis_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_abs_head_pose_vis.png")
                cv2.imwrite(paths["abs_head_pose_vis_png"], abs_frame)
                pose_vis_frames.append(abs_frame)

        if len(pose_vis_frames) >= 2:
            paths["pose_compare_png"] = osp.join(bad_case_imgs_dir, f"{prefix}_pose_compare.png")
            cv2.imwrite(paths["pose_compare_png"], cv2.hconcat(pose_vis_frames))

    return paths


def _maybe_log_bad_case_stats(
    cfg,
    storage,
    batch,
    out_dict,
    loss_dict,
    iteration,
    epoch,
    target_idx,
    pose_vis_enabled,
    pose_vis_interval,
    pose_vis_output_dir,
    bad_case_imgs_dir,
    size,
):
    train_cfg = cfg.TRAIN
    enabled = bool(train_cfg.get("BAD_CASE_STATS_ENABLED", True))
    if not enabled:
        return None

    target_idx_list = [int(v) for v in target_idx] if isinstance(target_idx, (list, tuple)) else None
    if target_idx_list is None:
        if isinstance(target_idx, np.ndarray):
            target_idx_list = [int(v) for v in target_idx.reshape(-1).tolist()]
        elif torch.is_tensor(target_idx):
            target_idx_list = [int(v) for v in target_idx.reshape(-1).tolist()]
        else:
            target_idx_list = [int(target_idx)]
    if len(target_idx_list) == 0:
        return None

    target_num = len(target_idx_list)
    gt_trans_bt = out_dict.get("gt_trans", None)
    pred_abs_trans = out_dict.get("trans", None)
    if gt_trans_bt is None or pred_abs_trans is None:
        return None
    if (not torch.is_tensor(gt_trans_bt)) or (not torch.is_tensor(pred_abs_trans)):
        return None
    if gt_trans_bt.dim() < 3 or pred_abs_trans.shape[0] < target_num:
        return None

    min_iter = int(train_cfg.get("BAD_CASE_MIN_ITER", 1000))
    if iteration < min_iter:
        return None

    abs_thresh_cm = float(train_cfg.get("BAD_CASE_T_CM_THRESHOLD", 5.0))
    comp_thresh = int(train_cfg.get("BAD_CASE_MAX_MASK_COMPONENTS", 1))
    min_comp_area = int(train_cfg.get("BAD_CASE_MIN_COMPONENT_AREA", 64))

    # Training `out_dict["gt_trans"]` is already sliced to target views inside
    # the model forward, so do not index it again with the original window-local
    # ids (for example [3,4,5]) or we will hit OOB on the target dimension.
    gt_trans = gt_trans_bt[0].detach().cpu().numpy().astype(np.float32)
    abs_trans = pred_abs_trans[:target_num].detach().cpu().numpy().astype(np.float32)
    abs_err = abs_trans - gt_trans
    abs_err_cm = np.linalg.norm(abs_err, axis=-1) * 100.0
    abs_err_xy_cm = np.linalg.norm(abs_err[:, :2], axis=-1) * 100.0
    abs_err_z_cm = np.abs(abs_err[:, 2]) * 100.0

    context_trans = None
    if torch.is_tensor(out_dict.get("context_trans", None)) and out_dict["context_trans"].shape[0] >= target_num:
        context_trans = out_dict["context_trans"][:target_num].detach().cpu().numpy().astype(np.float32)

    coarse_t_raw = None
    if torch.is_tensor(out_dict.get("coarse_t_raw", None)) and out_dict["coarse_t_raw"].shape[0] >= target_num:
        coarse_t_raw = out_dict["coarse_t_raw"][:target_num].detach().cpu().numpy().astype(np.float32)

    target_mask = out_dict.get("target_view_mask", None)
    pred_mask_stats = []
    max_mask_components = 0
    if torch.is_tensor(target_mask) and target_mask.dim() >= 5 and target_mask.shape[1] >= target_num:
        pred_mask_raw = target_mask[0, :target_num, 0].detach().cpu().numpy()
        for local_i in range(target_num):
            pred_mask_i_raw = pred_mask_raw[local_i]
            if pred_mask_i_raw.min() >= 0.0 and pred_mask_i_raw.max() <= 1.0:
                pred_mask_prob = pred_mask_i_raw
            else:
                pred_mask_prob = _stable_sigmoid_np(pred_mask_i_raw)
            pred_mask_bin = (pred_mask_prob > 0.5).astype(np.uint8)
            num_comp, comp_areas, comp_centers = _count_connected_components(pred_mask_bin, min_area=min_comp_area)
            max_mask_components = max(max_mask_components, num_comp)

            gt_mask_area = None
            if "input_obj_masks" in batch:
                gt_mask_i = batch["input_obj_masks"][0, target_idx_list[local_i]].detach().cpu().numpy()
                gt_mask_area = int(np.count_nonzero(gt_mask_i > 0.5))
            pred_depth_valid = None
            if "target_view_depth" in out_dict and torch.is_tensor(out_dict["target_view_depth"]):
                pred_depth_i = out_dict["target_view_depth"][0, local_i, 0].detach().cpu().numpy()
                pred_depth_valid = int(np.count_nonzero(pred_mask_bin.astype(bool) & np.isfinite(pred_depth_i) & (pred_depth_i > 0)))

            pred_mask_stats.append(
                {
                    "num_components": int(num_comp),
                    "component_areas": comp_areas,
                    "component_centers": comp_centers,
                    "pred_mask_area": int(np.count_nonzero(pred_mask_bin)),
                    "gt_mask_area": gt_mask_area,
                    "pred_depth_valid_px": pred_depth_valid,
                }
            )
    else:
        pred_mask_stats = [{} for _ in range(target_num)]

    noisy_mask_stats = []
    noisy_mask_image_paths = []
    target_image_paths = []
    if "noisy_obj_masks" in batch:
        out_res = int(cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES)
        noisy_masks_bt = batch["noisy_obj_masks"][0].detach().cpu().numpy()
        gt_masks_bt = (
            batch["input_obj_masks"][0].detach().cpu().numpy()
            if "input_obj_masks" in batch
            else None
        )
        for local_i, view_i in enumerate(target_idx_list):
            noisy_mask_full = noisy_masks_bt[view_i]
            noisy_mask_full_bin = noisy_mask_full > 0.5
            noisy_full_stats = _summarize_binary_mask(noisy_mask_full_bin, min_area=min_comp_area)

            gt_mask_full_bin = None
            if gt_masks_bt is not None:
                gt_mask_full_bin = gt_masks_bt[view_i] > 0.5
            diff_px = (
                int(np.count_nonzero(noisy_mask_full_bin != gt_mask_full_bin))
                if gt_mask_full_bin is not None
                else None
            )
            changed = bool(diff_px is not None and diff_px > 0)

            roi_center_np = batch["roi_center"][0, view_i].detach().cpu().numpy().astype(np.float32)
            scale_np = float(batch["scale"][0, view_i].detach().cpu().numpy().reshape(-1)[0])
            noisy_mask_roi = crop_resize_by_warp_affine(
                noisy_mask_full.astype(np.float32),
                roi_center_np,
                scale_np,
                out_res,
                interpolation=cv2.INTER_NEAREST,
            )
            if noisy_mask_roi.ndim == 3:
                noisy_mask_roi = noisy_mask_roi[..., 0]
            noisy_roi_bin = noisy_mask_roi > 0.5
            noisy_roi_stats = _summarize_binary_mask(noisy_roi_bin, min_area=max(min_comp_area // 4, 1))
            gt_mask_roi = None
            if gt_mask_full_bin is not None:
                gt_mask_roi = crop_resize_by_warp_affine(
                    gt_mask_full_bin.astype(np.float32),
                    roi_center_np,
                    scale_np,
                    out_res,
                    interpolation=cv2.INTER_NEAREST,
                )
                if gt_mask_roi.ndim == 3:
                    gt_mask_roi = gt_mask_roi[..., 0]

            image_paths = _write_bad_case_mask_images(
                bad_case_imgs_dir=bad_case_imgs_dir,
                iteration=iteration,
                local_i=local_i,
                view_i=view_i,
                noisy_mask_full=noisy_mask_full.astype(np.float32),
                noisy_mask_roi=noisy_mask_roi.astype(np.float32),
                gt_mask_full=gt_mask_full_bin.astype(np.float32) if gt_mask_full_bin is not None else None,
                gt_mask_roi=gt_mask_roi.astype(np.float32) if gt_mask_roi is not None else None,
            )

            noisy_mask_stats.append(
                {
                    "changed_vs_gt": changed,
                    "diff_px_vs_gt": diff_px,
                    "full_mask": noisy_full_stats,
                    "roi_attention_mask": noisy_roi_stats,
                }
            )
            noisy_mask_image_paths.append(image_paths)
    else:
        noisy_mask_stats = [None for _ in range(target_num)]
        noisy_mask_image_paths = [None for _ in range(target_num)]

    for local_i, view_i in enumerate(target_idx_list):
        target_image_paths.append(
            _write_bad_case_target_images(
                cfg=cfg,
                bad_case_imgs_dir=bad_case_imgs_dir,
                iteration=iteration,
                local_i=local_i,
                view_i=view_i,
                batch=batch,
                out_dict=out_dict,
                target_idx_list=target_idx_list,
                size=size,
            )
        )

    reasons = []
    max_abs_err_cm = float(np.max(abs_err_cm)) if abs_err_cm.size > 0 else 0.0
    if max_abs_err_cm >= abs_thresh_cm:
        reasons.append(f"abs_t_err_cm={max_abs_err_cm:.3f}>={abs_thresh_cm:.3f}")
    if max_mask_components > comp_thresh:
        reasons.append(f"mask_components={max_mask_components}>{comp_thresh}")
    noisy_changed_count = sum(
        1 for item in noisy_mask_stats
        if isinstance(item, dict) and bool(item.get("changed_vs_gt", False))
    )
    if not reasons:
        return None

    file_names = batch.get("file_name", None)
    scene_im_ids = batch.get("scene_im_id", None)
    file_names_0 = file_names[0] if isinstance(file_names, list) and len(file_names) > 0 else None
    scene_im_ids_0 = scene_im_ids[0] if isinstance(scene_im_ids, list) and len(scene_im_ids) > 0 else None

    vis_saved = bool(pose_vis_enabled and (iteration % max(int(pose_vis_interval), 1) == 0))
    vis_paths = {
        "gt_pose_png": osp.join(pose_vis_output_dir, f"gt_{iteration:06d}.png"),
        "pred_pose_png": osp.join(pose_vis_output_dir, f"pred_{iteration:06d}.png"),
        "pred_mask_png": osp.join(pose_vis_output_dir, f"pred_mask_{iteration:06d}.png"),
        "gt_mask_png": osp.join(pose_vis_output_dir, f"gt_mask_{iteration:06d}.png"),
        "pred_depth_png": osp.join(pose_vis_output_dir, f"pred_depth_{iteration:06d}.png"),
        "gt_depth_png": osp.join(pose_vis_output_dir, f"gt_depth_{iteration:06d}.png"),
        "depth_err_png": osp.join(pose_vis_output_dir, f"depth_err_{iteration:06d}.png"),
        "pred_fused_obj_ply": osp.join(pose_vis_output_dir, f"pred_depth_fused_obj_{iteration:06d}.ply"),
        "gt_fused_obj_ply": osp.join(pose_vis_output_dir, f"gt_depth_fused_obj_{iteration:06d}.ply"),
        "pred_pose_abs_head_mp4": osp.join(pose_vis_output_dir, f"pred_pose_abs_head_all_views_{iteration:06d}.mp4"),
        "bad_case_imgs_dir": bad_case_imgs_dir,
    }

    tracked_loss_names = [
        "loss_PM_RT",
        "loss_mask",
        "loss_obj_mask",
        "loss_centroid",
        "loss_z",
        "loss_trans_xy",
        "loss_trans_z",
    ]
    loss_snapshot = {}
    for name in tracked_loss_names:
        if name in loss_dict:
            cur_val = loss_dict[name]
            if torch.is_tensor(cur_val):
                cur_val = float(cur_val.detach().item())
            else:
                cur_val = float(cur_val)
            loss_snapshot[name] = cur_val

    vis_snapshot_names = [
        "vis/error_R",
        "vis/error_t",
        "vis/error_t_xy",
        "vis/error_t_z",
        "vis/error_R_context",
        "vis/error_t_context",
        "vis/error_t_context_xy",
        "vis/error_t_context_z",
        "vis/tx_gt",
        "vis/ty_gt",
        "vis/tz_gt",
        "vis/tx_pred",
        "vis/ty_pred",
        "vis/tz_pred",
    ]
    vis_snapshot = {
        name: _storage_latest_scalar(storage, name, default=None)
        for name in vis_snapshot_names
    }

    lines = []
    lines.append("=" * 120)
    lines.append(
        f"iter={iteration:06d} epoch={epoch} reasons={';'.join(reasons)}"
    )
    lines.append(
        f"window=[{int(batch.get('window_start_idx', -1))},{int(batch.get('window_end_idx', -1))}) target_idx={target_idx_list} "
        f"target_num={target_num} obj_cls={int(batch['roi_cls'][0].detach().item()) + 1}"
    )
    lines.append(
        f"vis_saved={vis_saved} vis_dir={pose_vis_output_dir} batch_file_targets={_json_compact(file_names_0 if file_names_0 is not None else [])}"
    )
    lines.append(f"noisy_mask_changed_targets={int(noisy_changed_count)}")
    lines.append(f"loss_snapshot={_json_compact(loss_snapshot)}")
    lines.append(f"vis_snapshot={_json_compact(vis_snapshot)}")

    for local_i, view_i in enumerate(target_idx_list):
        roi_center = batch["roi_center"][0, view_i].detach().cpu().numpy().astype(np.float32).tolist()
        roi_wh = batch["roi_wh"][0, view_i].detach().cpu().numpy().astype(np.float32).tolist()
        scale = batch["scale"][0, view_i].detach().cpu().numpy().astype(np.float32).reshape(-1).tolist() if "scale" in batch else None
        resize_ratio = float(batch["resize_ratio"][0, view_i].detach().cpu().item()) if "resize_ratio" in batch else None
        roi_cam = batch["roi_cam"][0, view_i].detach().cpu().numpy().astype(np.float32)
        entry = {
            "target_local_idx": int(local_i),
            "window_local_idx": int(view_i),
            "file_name": file_names_0[view_i] if file_names_0 is not None and view_i < len(file_names_0) else None,
            "scene_im_id": scene_im_ids_0[view_i] if scene_im_ids_0 is not None and view_i < len(scene_im_ids_0) else None,
            "gt_translation": gt_trans[local_i].tolist(),
            "abs_head_translation": abs_trans[local_i].tolist(),
            "abs_head_t_raw": coarse_t_raw[local_i].tolist() if coarse_t_raw is not None else None,
            "abs_head_proj_xy": _project_translation_to_image_xy(abs_trans[local_i], roi_cam),
            "gt_proj_xy": _project_translation_to_image_xy(gt_trans[local_i], roi_cam),
            "abs_t_err_cm": float(abs_err_cm[local_i]),
            "abs_t_xy_err_cm": float(abs_err_xy_cm[local_i]),
            "abs_t_z_err_cm": float(abs_err_z_cm[local_i]),
            "context_translation": context_trans[local_i].tolist() if context_trans is not None else None,
            "context_proj_xy": _project_translation_to_image_xy(context_trans[local_i], roi_cam) if context_trans is not None else None,
            "roi_center": roi_center,
            "roi_wh": roi_wh,
            "scale": scale,
            "resize_ratio": resize_ratio,
            "roi_cam": roi_cam.tolist(),
            "mask_stats": pred_mask_stats[local_i] if local_i < len(pred_mask_stats) else None,
            "noisy_mask_stats": noisy_mask_stats[local_i] if local_i < len(noisy_mask_stats) else None,
            "noisy_mask_image_paths": (
                noisy_mask_image_paths[local_i] if local_i < len(noisy_mask_image_paths) else None
            ),
            "target_image_paths": (
                target_image_paths[local_i] if local_i < len(target_image_paths) else None
            ),
        }
        lines.append(f"target[{local_i}]={_json_compact(entry)}")

    if vis_saved:
        lines.append(f"artifacts={_json_compact(vis_paths)}")

    return lines


def _sample_shared_bool(prob, device, enabled=True):
    prob = float(min(max(prob, 0.0), 1.0))
    if not enabled or prob <= 0.0:
        return False
    if prob >= 1.0:
        return True

    if dist.is_available() and dist.is_initialized():
        sample = torch.zeros(1, device=device, dtype=torch.int64)
        if dist.get_rank() == 0:
            sample.fill_(1 if np.random.rand() < prob else 0)
        dist.broadcast(sample, src=0)
        return bool(sample.item())
    return bool(np.random.rand() < prob)



def _parse_ckpt_sync_cfg(cfg):
    sync_cfg = cfg.get("CKPT_SYNC", {})
    enabled = bool(sync_cfg.get("ENABLED", False))
    remote_dir = str(sync_cfg.get("REMOTE_DIR", "")).strip()
    if enabled and not remote_dir:
        logger.warning("CKPT_SYNC is enabled but REMOTE_DIR is empty. Disable checkpoint sync.")
        enabled = False
    return {
        "enabled": enabled,
        "remote_dir": remote_dir.rstrip("/"),
        "cmd_template": str(sync_cfg.get("CMD_TEMPLATE", "ossutil cp -f {local_path} {remote_path}")),
        "delete_local": bool(sync_cfg.get("DELETE_LOCAL_AFTER_SYNC", False)),
        "retry_times": int(sync_cfg.get("RETRY_TIMES", 3)),
        "retry_interval_sec": int(sync_cfg.get("RETRY_INTERVAL_SEC", 15)),
        "timeout_sec": int(sync_cfg.get("TIMEOUT_SEC", 1800)),
    }


def _sync_ckpt_to_remote(ckpt_path, sync_cfg):
    filename = osp.basename(ckpt_path)
    remote_path = "{}/{}".format(sync_cfg["remote_dir"], filename)
    cmd = sync_cfg["cmd_template"].format(
        local_path=shlex.quote(ckpt_path),
        remote_path=shlex.quote(remote_path),
        filename=shlex.quote(filename),
        remote_dir=shlex.quote(sync_cfg["remote_dir"]),
    )

    for attempt in range(1, sync_cfg["retry_times"] + 1):
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=sync_cfg["timeout_sec"],
            )
            if result.returncode == 0:
                logger.info(f"Checkpoint synced to {remote_path}")
                return True
            logger.error(
                "Checkpoint sync failed (attempt %d/%d), cmd: %s, stderr: %s",
                attempt,
                sync_cfg["retry_times"],
                cmd,
                (result.stderr or "").strip()[-500:],
            )
        except Exception as e:
            logger.error(
                "Checkpoint sync exception (attempt %d/%d) for %s: %s",
                attempt,
                sync_cfg["retry_times"],
                ckpt_path,
                e,
            )
        if attempt < sync_cfg["retry_times"]:
            time.sleep(sync_cfg["retry_interval_sec"])
    return False

class GDRN_Lite(LightningLite):
    def get_evaluator(self, cfg, dataset_name, output_folder=None):
        """Create evaluator(s) for a given dataset.

        This uses the special metadata "evaluator_type" associated with
        each builtin dataset. For your own dataset, you can simply
        create an evaluator manually in your script and do not have to
        worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = osp.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= self.global_rank
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= self.global_rank
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)

        _distributed = self.world_size > 1
        dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        train_obj_names = dataset_meta.objs
        if evaluator_type == "bop":
            if cfg.VAL.get("USE_BOP", False):
                return GDRN_Evaluator(
                    cfg, dataset_name, distributed=_distributed, output_dir=output_folder, train_objs=train_obj_names
                )
            else:
                return GDRN_EvaluatorCustom(
                    cfg, dataset_name, distributed=_distributed, output_dir=output_folder, train_objs=train_obj_names
                )

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    def get_tbx_event_writer(self, out_dir, backup=False):
        tb_logdir = osp.join(out_dir, "tb")
        mmcv.mkdir_or_exist(tb_logdir)
        if backup and self.is_global_zero:
            old_tb_logdir = osp.join(out_dir, "tb_old")
            mmcv.mkdir_or_exist(old_tb_logdir)
            os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))

        tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="tensorboardX")
        return tbx_event_writer

    def do_test(self, cfg, model, epoch=None, iteration=None):
        results = OrderedDict()
        model_name = osp.basename(cfg.MODEL.WEIGHTS).split(".")[0]
        for dataset_name in cfg.DATASETS.TEST:
            if epoch is not None and iteration is not None:
                evaluator = self.get_evaluator(
                    cfg,
                    dataset_name,
                    osp.join(cfg.OUTPUT_DIR, f"inference_epoch_{epoch}_iter_{iteration}", dataset_name),
                )
            else:
                evaluator = self.get_evaluator(
                    cfg, dataset_name, osp.join(cfg.OUTPUT_DIR, f"inference_{model_name}", dataset_name)
                )
            data_loader = build_gdrn_test_loader(cfg, dataset_name, train_objs=evaluator.train_objs)
            data_loader = self.setup_dataloaders(data_loader, replace_sampler=False, move_to_device=False)
            results_i = gdrn_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=cfg.TEST.AMP_TEST)
            results[dataset_name] = results_i

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def do_train(self, cfg, args, model, optimizer, resume=False):
        model.train()

        # some basic settings =========================
        dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        data_ref = ref.__dict__[dataset_meta.ref_key]
        obj_names = dataset_meta.objs

        # load data ===================================
        train_dset_names = cfg.DATASETS.TRAIN
        data_loader = build_gdrn_train_loader(cfg, train_dset_names)
        data_loader_iter = iter(data_loader)

        # load 2nd train dataloader if needed
        train_2_dset_names = cfg.DATASETS.get("TRAIN2", ())
        train_2_ratio = cfg.DATASETS.get("TRAIN2_RATIO", 0.0)
        if train_2_ratio > 0.0 and len(train_2_dset_names) > 0:
            data_loader_2 = build_gdrn_train_loader(cfg, train_2_dset_names)
            data_loader_2_iter = iter(data_loader_2)
        else:
            data_loader_2 = None
            data_loader_2_iter = None

        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        if isinstance(data_loader, AspectRatioGroupedDataset):
            dataset_len = len(data_loader.dataset.dataset)
            if data_loader_2 is not None:
                dataset_len += len(data_loader_2.dataset.dataset)
            iters_per_epoch = dataset_len // images_per_batch
        else:
            dataset_len = len(data_loader.dataset)
            if data_loader_2 is not None:
                dataset_len += len(data_loader_2.dataset)
            iters_per_epoch = dataset_len // images_per_batch
        max_iter = cfg.SOLVER.TOTAL_EPOCHS * iters_per_epoch
        dprint("images_per_batch: ", images_per_batch)
        dprint("dataset length: ", dataset_len)
        dprint("iters per epoch: ", iters_per_epoch)
        dprint("total iters: ", max_iter)

        data_loader = self.setup_dataloaders(data_loader, replace_sampler=False, move_to_device=False)
        if data_loader_2 is not None:
            data_loader_2 = self.setup_dataloaders(data_loader_2, replace_sampler=False, move_to_device=False)

        scheduler = solver_utils.build_lr_scheduler(cfg, optimizer, total_iters=max_iter)

        # resume or load model ===================================
        extra_ckpt_dict = dict(
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if hasattr(self._precision_plugin, "scaler"):
            extra_ckpt_dict["gradscaler"] = self._precision_plugin.scaler

        checkpointer = MyCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=self.is_global_zero,
            **extra_ckpt_dict,
        )
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

        if cfg.SOLVER.CHECKPOINT_BY_EPOCH:
            ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch
        else:
            ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, ckpt_period, max_iter=max_iter, max_to_keep=cfg.SOLVER.MAX_TO_KEEP
        )
        ckpt_sync_cfg = _parse_ckpt_sync_cfg(cfg)

        # build writers ==============================================
        tbx_event_writer = self.get_tbx_event_writer(cfg.OUTPUT_DIR, backup=not cfg.get("RESUME", False))
        tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
        writers = (
            [MyCommonMetricPrinter(max_iter), MyJSONWriter(osp.join(cfg.OUTPUT_DIR, "metrics.json")), tbx_event_writer]
            if self.is_global_zero
            else []
        )
        pose_vis_enabled = bool(cfg.TRAIN.get("VIS_POSE_VIDEO", False))
        pose_vis_interval = max(int(cfg.TRAIN.get("VIS_POSE_VIDEO_INTERVAL", 200)), 1)
        pose_vis_output_dir = str(cfg.TRAIN.get("VIS_POSE_VIDEO_OUTPUT_DIR", "/mnt/afs/TransparentObjectPose/debug")).strip()
        bad_case_stats_enabled = bool(cfg.TRAIN.get("BAD_CASE_STATS_ENABLED", True))
        bad_case_stats_path = str(
            cfg.TRAIN.get("BAD_CASE_STATS_PATH", osp.join(pose_vis_output_dir, "bad_cases_stats.txt"))
        ).strip()
        bad_case_imgs_dir = str(
            cfg.TRAIN.get("BAD_CASE_IMGS_DIR", osp.join(pose_vis_output_dir, "bad_cases_imgs"))
        ).strip()
        bad_case_stats_cooldown = max(int(cfg.TRAIN.get("BAD_CASE_COOLDOWN_ITERS", 20)), 0)
        bad_case_stats_max_logs = max(int(cfg.TRAIN.get("BAD_CASE_MAX_LOGS", 500)), 1)
        bad_case_last_logged_iter = -10 ** 9
        bad_case_logged_count = 0
        abs_pose_debug_interval = max(int(cfg.TRAIN.get("ABS_POSE_DEBUG_INTERVAL", 10)), 0)
        pose_visualize_example = None
        if pose_vis_enabled and self.is_global_zero:
            mmcv.mkdir_or_exist(pose_vis_output_dir)
            import sys

            tools_path = "/mnt/afs/TransparentObjectPose/tools"
            if tools_path not in sys.path:
                sys.path.append(tools_path)
            from visualize_3d_bbox import visualize_example as pose_visualize_example

        # compared to "train_net.py", we do not support accurate timing and
        # precise BN here, because they are not trivial to implement
        logger.info("Starting training from iteration {}".format(start_iter))
        phase_profile_enabled = str(os.environ.get("GDRN_PROFILE_PHASES", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        phase_profile_sync = str(os.environ.get("GDRN_PROFILE_PHASES_SYNC", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        phase_profile_print_every = max(int(os.environ.get("GDRN_PROFILE_PHASES_PRINT_EVERY", "1")), 1)

        def _phase_sync():
            if phase_profile_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

        def _phase_begin(active):
            if not active:
                return None
            _phase_sync()
            return time.perf_counter()

        def _phase_end(active, phase_stats, key, start_t):
            if not active or start_t is None:
                return
            _phase_sync()
            phase_stats[key] = phase_stats.get(key, 0.0) + (time.perf_counter() - start_t)

        iter_time = None
        with EventStorage(start_iter) as storage:
            # torch.autograd.set_detect_anomaly(True)
            for iteration in range(start_iter, max_iter):
                _session_iter = iteration - start_iter
                profile_this_iter = (
                    phase_profile_enabled
                    and self.is_global_zero
                    and ((iteration + 1) % phase_profile_print_every == 0)
                )
                phase_stats = OrderedDict() if profile_this_iter else None
                iter_profile_t0 = _phase_begin(profile_this_iter)
                storage.iter = iteration
                epoch = iteration // dataset_len + 1

                data_fetch_t0 = _phase_begin(profile_this_iter)
                if np.random.rand() < train_2_ratio:
                    data = next(data_loader_2_iter)
                else:
                    data = next(data_loader_iter)
                _phase_end(profile_this_iter, phase_stats, "data_fetch", data_fetch_t0)

                if iter_time is not None:
                    storage.put_scalar("time", time.perf_counter() - iter_time)
                iter_time = time.perf_counter()

                # # forward ============================================================
                # print(f"data: {data[0]['scene_im_id']}, {data[0]['file_name']}, {data[0]['trans'].detach().cpu().numpy()}")
                # import pdb; pdb.set_trace()
                # batch = batch_data(cfg, data)
                # Periodically clear CUDA cache to avoid OOM from memory fragmentation,
                # but not every iteration (that would hurt throughput and cause NCCL timeouts).
                if iteration % 100 == 0:
                    torch.cuda.empty_cache()
                batch_prep_t0 = _phase_begin(profile_this_iter)
                batch = batch_data_rand_num_perm(cfg, data)
                use_view_time_layout = batch["roi_img"].dim() == 6
                if use_view_time_layout:
                    num_times = int(batch["roi_img"].shape[2])
                    num_target_views = num_times
                else:
                    num_times = int(batch["roi_img"].shape[1])
                    num_target_views = num_times
                num_context_views = int(cfg.MODEL.CDPN.PNP_NET.get("TRAIN_NUM_CONTEXT_VIEWS", 3))
                num_target_views_cfg = int(cfg.MODEL.CDPN.PNP_NET.get("TRAIN_NUM_TARGET_VIEWS", 3))
                context_then_target_prob = float(cfg.MODEL.CDPN.PNP_NET.get("TRAIN_CONTEXT_THEN_TARGET_PROB", 0.5))
                context_then_target_prob = min(max(context_then_target_prob, 0.0), 1.0)
                # Two training modes for autoregressive setup:
                # mode A: all views are target views
                # mode B: first num_context_views are context, last num_target_views_cfg are target
                use_context_then_target = _sample_shared_bool(
                    context_then_target_prob,
                    device=batch["roi_img"].device,
                    enabled=(num_target_views >= num_context_views + num_target_views_cfg),
                )
                if use_context_then_target:
                    context_num = min(num_context_views, max(0, num_target_views - 1))
                    target_end = min(context_num + num_target_views_cfg, num_target_views)
                    target_idx = list(range(context_num, target_end))
                    if len(target_idx) == 0:
                        target_idx = [num_target_views - 1]
                else:
                    target_idx = list(range(0, num_target_views))
                target_time_idx = target_idx if use_view_time_layout else None
                _phase_end(profile_this_iter, phase_stats, "batch_prep", batch_prep_t0)

                forward_t0 = _phase_begin(profile_this_iter)
                vi_model_kwargs, _ = collect_view_interaction_model_kwargs(batch)
                out_dict, loss_dict = model(
                    batch["roi_img"],
                    gt_xyz=batch.get("roi_xyz", None),
                    gt_xyz_bin=batch.get("roi_xyz_bin", None),
                    gt_mask_trunc=batch["roi_mask_trunc"],
                    gt_mask_visib=batch["roi_mask_visib"],
                    gt_mask_obj=batch["roi_mask_obj"],
                    gt_region=batch.get("roi_region", None),
                    gt_allo_quat=batch.get("allo_quat", None),
                    gt_ego_quat=batch.get("ego_quat", None),
                    gt_allo_rot6d=batch.get("allo_rot6d", None),
                    gt_ego_rot6d=batch.get("ego_rot6d", None),
                    gt_trans_ratio=batch["roi_trans_ratio"],
                    gt_points=batch.get("model_points", None),
                    sym_infos=batch.get("sym_info", None),
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    scales=batch["scale"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_extents=batch.get("roi_extent", None),
                    input_images=batch["input_images"],
                    noisy_obj_masks=batch.get("noisy_obj_masks", None),
                    target_idx=target_idx,
                    target_time_idx=target_time_idx,
                    do_loss=True,
                    **vi_model_kwargs,
                )
                _phase_end(profile_this_iter, phase_stats, "forward", forward_t0)
                
                # operate only on rank 0
                pose_vis_t0 = _phase_begin(profile_this_iter)
                if self.is_global_zero and pose_vis_enabled and iteration % pose_vis_interval == 0:
                    # get GT rot and trans from out_dict
                    gt_rot, gt_trans = out_dict["gt_rot"][0], out_dict["gt_trans"][0]
                    # In multiview mode, pick the first selected view for visualization.
                    if gt_rot.dim() == 3:
                        gt_rot = gt_rot[0]
                        gt_trans = gt_trans[0]
                    gt_RT = np.concatenate([gt_rot.detach().cpu().numpy(), gt_trans[..., None].detach().cpu().numpy()], axis=1)
                    
                    # get pred rot and trans from out_dict
                    pred_rot, pred_trans = out_dict["rot"][0], out_dict["trans"][0]
                    if pred_rot.dim() == 3:
                        pred_rot = pred_rot[0]
                        pred_trans = pred_trans[0]
                    pred_RT = np.concatenate([pred_rot.detach().cpu().numpy(), pred_trans[..., None].detach().cpu().numpy()], axis=1)
                    
                    # get input image from batch
                    if isinstance(target_idx, (list, tuple)):
                        vis_target_idx = target_idx[0]
                    elif isinstance(target_idx, np.ndarray) and target_idx.ndim > 0:
                        vis_target_idx = int(target_idx.reshape(-1)[0])
                    elif torch.is_tensor(target_idx) and target_idx.numel() > 1:
                        vis_target_idx = int(target_idx.reshape(-1)[0].item())
                    else:
                        vis_target_idx = target_idx
                    input_images = batch["input_images"][0, vis_target_idx] # (C, H, W)
                    
                    mean = torch.tensor([0.485, 0.456, 0.406], dtype=input_images.dtype).to(input_images.device)
                    std = torch.tensor([0.229, 0.224, 0.225], dtype=input_images.dtype).to(input_images.device)
                    mean = mean.unsqueeze(-1).unsqueeze(-1)
                    std = std.unsqueeze(-1).unsqueeze(-1)
                    input_images = (input_images * std + mean) * 255.0
                    input_images = input_images.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    # visualize the input image, GT rot, GT trans, pred rot, pred trans
                    obj_cls = batch["roi_cls"][0].detach().item() + 1 # since the obj_cls is 0-based index
                    # obj_cls_name = data_ref.id2obj[obj_cls]
                    model_info = data_ref.get_models_info()[str(obj_cls)]
                    mins = [model_info["min_x"] / 1000.0, model_info["min_y"] / 1000.0, model_info["min_z"] / 1000.0]
                    sizes = [model_info["size_x"] / 1000.0, model_info["size_y"] / 1000.0, model_info["size_z"] / 1000.0]
                    size = mins + sizes
                    pose_visualize_example(
                        K=None,
                        image=input_images,
                        RT=gt_RT,
                        size=size,
                        output_path=osp.join(pose_vis_output_dir, f"gt_{iteration:06d}.png"),
                    )
                    pose_visualize_example(
                        K=None,
                        image=input_images,
                        RT=pred_RT,
                        size=size,
                        output_path=osp.join(pose_vis_output_dir, f"pred_{iteration:06d}.png"),
                    )

                    # visualize depth/mask prediction for selected target view
                    if "target_view_depth" in out_dict and "target_view_mask" in out_dict:
                        pred_depth = out_dict["target_view_depth"][0, 0, 0].detach().cpu().numpy()
                        pred_mask_raw = out_dict["target_view_mask"][0, 0, 0].detach().cpu().numpy()
                        # Avoid double-sigmoid when mask head already outputs probabilities.
                        if pred_mask_raw.min() >= 0.0 and pred_mask_raw.max() <= 1.0:
                            pred_mask = pred_mask_raw
                        else:
                            pred_mask = _stable_sigmoid_np(pred_mask_raw)

                        pred_mask_bin = pred_mask > 0.5
                        pred_depth_finite = np.isfinite(pred_depth)
                        pred_depth_valid = pred_mask_bin & pred_depth_finite

                        gt_depth = None
                        gt_depth_valid = None
                        if "input_depths" in batch:
                            gt_depth = batch["input_depths"][0, vis_target_idx].detach().cpu().numpy()
                            gt_depth_valid = (gt_depth > 0) & np.isfinite(gt_depth)

                        if np.count_nonzero(pred_depth_valid) > 10:
                            depth_min = float(np.percentile(pred_depth[pred_depth_valid], 2))
                            depth_max = float(np.percentile(pred_depth[pred_depth_valid], 98))
                        elif gt_depth is not None and np.any(gt_depth_valid):
                            depth_min = float(np.percentile(gt_depth[gt_depth_valid], 2))
                            depth_max = float(np.percentile(gt_depth[gt_depth_valid], 98))
                        else:
                            fallback_valid = pred_depth_finite
                            depth_min = float(np.percentile(pred_depth[fallback_valid], 2))
                            depth_max = float(np.percentile(pred_depth[fallback_valid], 98))

                        if abs(depth_max - depth_min) < 1e-8:
                            depth_max = depth_min + 1e-8

                        pred_depth_norm = np.clip((pred_depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
                        pred_depth_gray = np.zeros_like(pred_depth_norm, dtype=np.uint8)
                        pred_depth_gray[pred_depth_valid] = (pred_depth_norm[pred_depth_valid] * 255.0).astype(np.uint8)

                        pred_depth_vis = cv2.applyColorMap(pred_depth_gray, cv2.COLORMAP_JET)
                        pred_depth_vis[~pred_mask_bin] = 0
                        cv2.imwrite(osp.join(pose_vis_output_dir, f"pred_depth_{iteration:06d}.png"), pred_depth_vis)

                        pred_mask_vis = (np.clip(pred_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                        cv2.imwrite(osp.join(pose_vis_output_dir, f"pred_mask_{iteration:06d}.png"), pred_mask_vis)

                        if gt_depth is not None:
                            gt_depth_gray = np.zeros_like(pred_depth_gray)
                            if np.any(gt_depth_valid):
                                gt_depth_norm = np.clip((gt_depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
                                gt_depth_gray[gt_depth_valid] = (gt_depth_norm[gt_depth_valid] * 255.0).astype(np.uint8)
                            gt_depth_vis = cv2.applyColorMap(gt_depth_gray, cv2.COLORMAP_JET)
                            gt_depth_vis[~gt_depth_valid] = 0
                            cv2.imwrite(osp.join(pose_vis_output_dir, f"gt_depth_{iteration:06d}.png"), gt_depth_vis)

                            depth_err = np.abs(pred_depth - gt_depth)
                            depth_err_gray = np.zeros_like(pred_depth_gray)
                            err_valid = pred_depth_valid & gt_depth_valid
                            if np.any(err_valid):
                                depth_err_norm = np.clip(depth_err / max(depth_max - depth_min, 1e-8), 0.0, 1.0)
                                depth_err_gray[err_valid] = (depth_err_norm[err_valid] * 255.0).astype(np.uint8)
                            depth_err_vis = cv2.applyColorMap(depth_err_gray, cv2.COLORMAP_HOT)
                            depth_err_vis[~err_valid] = 0
                            cv2.imwrite(osp.join(pose_vis_output_dir, f"depth_err_{iteration:06d}.png"), depth_err_vis)

                        if "input_obj_masks" in batch:
                            gt_mask = batch["input_obj_masks"][0, vis_target_idx].detach().cpu().numpy()
                            gt_mask_vis = (np.clip(gt_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                            cv2.imwrite(osp.join(pose_vis_output_dir, f"gt_mask_{iteration:06d}.png"), gt_mask_vis)

                        # Fuse multiview depth into a single object-frame point cloud for consistency check.
                        if all(k in batch for k in ["roi_cam", "ego_rot", "trans"]):
                            pred_depth_views = out_dict["target_view_depth"][0].detach().cpu().numpy()  # (T,1,H,W)
                            pred_mask_views_raw = out_dict["target_view_mask"][0].detach().cpu().numpy()  # (T,1,H,W)
                            if isinstance(target_idx, (list, tuple)):
                                target_view_ids = [int(v) for v in target_idx]
                            elif isinstance(target_idx, np.ndarray):
                                target_view_ids = [int(v) for v in target_idx.reshape(-1).tolist()]
                            elif torch.is_tensor(target_idx):
                                target_view_ids = [int(v) for v in target_idx.reshape(-1).tolist()]
                            else:
                                target_view_ids = [int(target_idx)]

                            num_pred_views = min(len(target_view_ids), pred_depth_views.shape[0])
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
                            pred_points_obj_all, pred_colors_all = [], []
                            gt_points_obj_all, gt_colors_all = [], []
                            for local_i in range(num_pred_views):
                                view_i = target_view_ids[local_i]
                                if view_i < 0 or view_i >= batch["roi_cam"].shape[1]:
                                    continue
                                K_i = batch["roi_cam"][0, view_i].detach().cpu().numpy().astype(np.float32)
                                R_i = batch["ego_rot"][0, view_i].detach().cpu().numpy().astype(np.float32)
                                t_i = batch["trans"][0, view_i].detach().cpu().numpy().astype(np.float32)

                                pred_depth_i = pred_depth_views[local_i, 0]
                                pred_mask_i_raw = pred_mask_views_raw[local_i, 0]
                                if pred_mask_i_raw.min() >= 0.0 and pred_mask_i_raw.max() <= 1.0:
                                    pred_mask_i = pred_mask_i_raw
                                else:
                                    pred_mask_i = _stable_sigmoid_np(pred_mask_i_raw)
                                pred_valid_i = (pred_mask_i > 0.5) & np.isfinite(pred_depth_i) & (pred_depth_i > 0)
                                pred_pts_cam = depth_to_cam_points(pred_depth_i, K_i, pred_valid_i)
                                pred_pts_obj = cam_to_obj(pred_pts_cam, R_i, t_i)
                                if pred_pts_obj.shape[0] > 0:
                                    pred_points_obj_all.append(pred_pts_obj)
                                    color_i = np.tile(
                                        view_palette[local_i % len(view_palette)][None, :],
                                        (pred_pts_obj.shape[0], 1),
                                    )
                                    pred_colors_all.append(color_i)

                                if "input_depths" in batch:
                                    gt_depth_i = batch["input_depths"][0, view_i].detach().cpu().numpy()
                                    if "input_obj_masks" in batch:
                                        gt_mask_i = batch["input_obj_masks"][0, view_i].detach().cpu().numpy() > 0.5
                                    else:
                                        gt_mask_i = np.ones_like(gt_depth_i, dtype=bool)
                                    gt_valid_i = gt_mask_i & np.isfinite(gt_depth_i) & (gt_depth_i > 0)
                                    gt_pts_cam = depth_to_cam_points(gt_depth_i, K_i, gt_valid_i)
                                    gt_pts_obj = cam_to_obj(gt_pts_cam, R_i, t_i)
                                    if gt_pts_obj.shape[0] > 0:
                                        gt_points_obj_all.append(gt_pts_obj)
                                        color_i = np.tile(
                                            view_palette[local_i % len(view_palette)][None, :],
                                            (gt_pts_obj.shape[0], 1),
                                        )
                                        gt_colors_all.append(color_i)

                            if pred_points_obj_all:
                                pred_points_obj = np.concatenate(pred_points_obj_all, axis=0)
                                pred_colors = np.concatenate(pred_colors_all, axis=0)
                                if pred_points_obj.shape[0] > 200000:
                                    keep_idx = np.random.choice(pred_points_obj.shape[0], 200000, replace=False)
                                    pred_points_obj = pred_points_obj[keep_idx]
                                    pred_colors = pred_colors[keep_idx]
                                write_ply_xyzrgb(
                                    osp.join(pose_vis_output_dir, f"pred_depth_fused_obj_{iteration:06d}.ply"),
                                    pred_points_obj,
                                    pred_colors,
                                )

                            if gt_points_obj_all:
                                gt_points_obj = np.concatenate(gt_points_obj_all, axis=0)
                                gt_colors = np.concatenate(gt_colors_all, axis=0)
                                if gt_points_obj.shape[0] > 200000:
                                    keep_idx = np.random.choice(gt_points_obj.shape[0], 200000, replace=False)
                                    gt_points_obj = gt_points_obj[keep_idx]
                                    gt_colors = gt_colors[keep_idx]
                                write_ply_xyzrgb(
                                    osp.join(pose_vis_output_dir, f"gt_depth_fused_obj_{iteration:06d}.ply"),
                                    gt_points_obj,
                                    gt_colors,
                                )

                        # Pose prediction videos show GT on context views and
                        # abs-head predictions on target views.
                    if all(k in batch for k in ["roi_cam", "ego_rot", "trans"]) and batch["input_images"].shape[1] > 1:
                        if isinstance(target_idx, (list, tuple)):
                            vis_target_num = len(target_idx)
                            has_context_views = min(int(v) for v in target_idx) > 0 if len(target_idx) > 0 else False
                        elif isinstance(target_idx, np.ndarray):
                            vis_target_num = int(target_idx.size) if target_idx.ndim > 0 else 1
                            has_context_views = int(target_idx.reshape(-1)[0]) > 0 if vis_target_num > 0 else False
                        elif torch.is_tensor(target_idx):
                            vis_target_num = int(target_idx.numel()) if target_idx.numel() > 1 else 1
                            has_context_views = int(target_idx.reshape(-1)[0].item()) > 0 if vis_target_num > 0 else False
                        else:
                            vis_target_num = 1
                            has_context_views = int(target_idx) > 0

                        def _make_vis_pose_dict(rot_key, trans_key):
                            if rot_key not in out_dict or trans_key not in out_dict:
                                return None
                            return {
                                "rot": out_dict[rot_key][:vis_target_num].unsqueeze(0).detach(),
                                "trans": out_dict[trans_key][:vis_target_num].unsqueeze(0).detach(),
                            }

                        abs_head_vis = _make_vis_pose_dict("rot", "trans")
                        if abs_head_vis is not None:
                            write_pose_vis_videos(
                                batch,
                                abs_head_vis,
                                target_idx,
                                data_ref,
                                obj_cls,
                                size,
                                iteration,
                                pose_vis_output_dir,
                                fps=5,
                                output_prefix="pred_pose_abs_head",
                                pred_label="abs_head",
                            )

                        if has_context_views:
                            pass  # rel/final fusion visualization removed

                _phase_end(profile_this_iter, phase_stats, "pose_vis_debug", pose_vis_t0)

                bad_case_t0 = _phase_begin(profile_this_iter and bad_case_stats_enabled)
                if (
                    bad_case_stats_enabled
                    and self.is_global_zero
                    and bad_case_logged_count < bad_case_stats_max_logs
                    and (iteration - bad_case_last_logged_iter) >= bad_case_stats_cooldown
                ):
                    bad_case_size = None
                    try:
                        obj_cls = int(batch["roi_cls"][0].detach().item()) + 1
                        model_info = data_ref.get_models_info()[str(obj_cls)]
                        mins = [model_info["min_x"] / 1000.0, model_info["min_y"] / 1000.0, model_info["min_z"] / 1000.0]
                        sizes = [model_info["size_x"] / 1000.0, model_info["size_y"] / 1000.0, model_info["size_z"] / 1000.0]
                        bad_case_size = mins + sizes
                    except Exception:
                        bad_case_size = None
                    bad_case_lines = _maybe_log_bad_case_stats(
                        cfg=cfg,
                        storage=storage,
                        batch=batch,
                        out_dict=out_dict,
                        loss_dict=loss_dict,
                        iteration=iteration,
                        epoch=epoch,
                        target_idx=target_idx,
                        pose_vis_enabled=pose_vis_enabled,
                        pose_vis_interval=pose_vis_interval,
                        pose_vis_output_dir=pose_vis_output_dir,
                        bad_case_imgs_dir=bad_case_imgs_dir,
                        size=bad_case_size,
                    )
                    if bad_case_lines is not None:
                        _append_bad_case_stats(bad_case_stats_path, bad_case_lines)
                        bad_case_last_logged_iter = iteration
                        bad_case_logged_count += 1
                _phase_end(profile_this_iter and bad_case_stats_enabled, phase_stats, "bad_case_stats", bad_case_t0)

                # Keep zero-weight temporal loss fully disabled (not added/logged).
                loss_reduce_t0 = _phase_begin(profile_this_iter)
                if float(cfg.MODEL.CDPN.PNP_NET.get("TEMPORAL_ADDS_LW", 0.0)) <= 0.0:
                    loss_dict.pop("loss_temporal_adds", None)

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if self.is_global_zero:
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                _phase_end(profile_this_iter, phase_stats, "loss_reduce_log", loss_reduce_t0)

                zero_grad_t0 = _phase_begin(profile_this_iter)
                optimizer.zero_grad(set_to_none=True)
                _phase_end(profile_this_iter, phase_stats, "zero_grad", zero_grad_t0)

                backward_t0 = _phase_begin(profile_this_iter)
                self.backward(losses)
                _phase_end(profile_this_iter, phase_stats, "backward", backward_t0)

                optimizer_step_t0 = _phase_begin(profile_this_iter)
                optimizer.step()
                _phase_end(profile_this_iter, phase_stats, "optimizer_step", optimizer_step_t0)

                scheduler_t0 = _phase_begin(profile_this_iter)
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()
                _phase_end(profile_this_iter, phase_stats, "scheduler", scheduler_t0)

                abs_pose_debug_t0 = _phase_begin(profile_this_iter)
                if self.is_global_zero and abs_pose_debug_interval > 0 and (iteration + 1) % abs_pose_debug_interval == 0:
                    debug_names = [
                        "vis/error_R",
                        "vis/error_R_sym",
                        "vis/error_t",
                        "vis/error_t_sym",
                        "vis/error_t_xy",
                        "vis/error_t_z",
                        "vis/tx_pred",
                        "vis/ty_pred",
                        "vis/tz_pred",
                        "vis/tx_net",
                        "vis/ty_net",
                        "vis/tz_net",
                        "vis/tx_gt",
                        "vis/ty_gt",
                        "vis/tz_gt",
                        "vis/geo_prior_active",
                        "vis/geo_prior_dropped",
                        "vis/geo_prior_has_bank",
                        "vis/geo_prior_bank_valid_ratio",
                        "vis/prior_conf_mean",
                        "vis/prior_conf_obj_mean",
                        "vis/prior_ambiguity_mean",
                        "vis/prior_xyz_abs_err",
                        "vis/loss_rel_geo_sym_raw",
                        "vis/loss_rel_trans_raw",
                    ]
                    debug_vals = {}
                    for name in debug_names:
                        try:
                            debug_vals[name] = float(storage.history(name).latest())
                        except KeyError:
                            continue
                    if debug_vals:
                        loss_name_pairs = [
                            ("loss_PM_RT", "pm_rt"),
                            ("loss_rot", "rot"),
                            ("loss_centroid", "cent"),
                            ("loss_z", "z"),
                            ("loss_trans_xy", "txy"),
                            ("loss_trans_z", "tz"),
                            ("loss_bind", "bind"),
                            ("loss_mask", "mask"),
                            ("loss_region", "region"),
                            ("loss_obj_mask", "obj_mask"),
                            ("loss_dp_reg", "dp_reg"),
                            ("loss_dp_gd", "dp_gd"),
                            ("loss_dp_3d", "dp_3d"),
                            ("loss_pose_flow", "pose_flow"),
                            ("loss_xyz_prior", "xyz_prior"),
                            ("loss_prior_conf", "prior_conf"),
                            ("loss_prior_ambiguity", "prior_amb"),
                            ("loss_rel_geo_sym", "rel_geo"),
                            ("loss_rel_trans", "rel_t"),
                        ]
                        loss_summary = " ".join(
                            f"{label}={loss_dict_reduced[name]:.4f}"
                            for name, label in loss_name_pairs
                            if name in loss_dict_reduced
                        )
                        logger.info(
                            "abs_pose_debug iter=%d R=%.4f R_sym=%.4f t_cm=%.4f t_sym_cm=%.4f t_xy_cm=%.4f t_z_cm=%.4f pred_t=(%.4f,%.4f,%.4f) net_t=(%.4f,%.4f,%.4f) gt_t=(%.4f,%.4f,%.4f) gp(active=%.0f drop=%.0f bank=%.0f ratio=%.3f conf=%.3f conf_obj=%.3f amb=%.3f xyzerr=%.4f) rel_raw=(%.4f,%.4f) losses[%s]",
                            iteration,
                            debug_vals.get("vis/error_R", 0.0),
                            debug_vals.get("vis/error_R_sym", 0.0),
                            debug_vals.get("vis/error_t", 0.0),
                            debug_vals.get("vis/error_t_sym", 0.0),
                            debug_vals.get("vis/error_t_xy", 0.0),
                            debug_vals.get("vis/error_t_z", 0.0),
                            debug_vals.get("vis/tx_pred", 0.0),
                            debug_vals.get("vis/ty_pred", 0.0),
                            debug_vals.get("vis/tz_pred", 0.0),
                            debug_vals.get("vis/tx_net", 0.0),
                            debug_vals.get("vis/ty_net", 0.0),
                            debug_vals.get("vis/tz_net", 0.0),
                            debug_vals.get("vis/tx_gt", 0.0),
                            debug_vals.get("vis/ty_gt", 0.0),
                            debug_vals.get("vis/tz_gt", 0.0),
                            debug_vals.get("vis/geo_prior_active", 0.0),
                            debug_vals.get("vis/geo_prior_dropped", 0.0),
                            debug_vals.get("vis/geo_prior_has_bank", 0.0),
                            debug_vals.get("vis/geo_prior_bank_valid_ratio", 0.0),
                            debug_vals.get("vis/prior_conf_mean", 0.0),
                            debug_vals.get("vis/prior_conf_obj_mean", 0.0),
                            debug_vals.get("vis/prior_ambiguity_mean", 0.0),
                            debug_vals.get("vis/prior_xyz_abs_err", 0.0),
                            debug_vals.get("vis/loss_rel_geo_sym_raw", 0.0),
                            debug_vals.get("vis/loss_rel_trans_raw", 0.0),
                            loss_summary,
                        )
                _phase_end(profile_this_iter, phase_stats, "abs_pose_debug", abs_pose_debug_t0)

                eval_t0 = _phase_begin(profile_this_iter)
                if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
                ):
                    self.do_test(cfg, model, epoch=epoch, iteration=iteration)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    self.barrier()
                _phase_end(profile_this_iter, phase_stats, "eval", eval_t0)

                writer_vis_t0 = _phase_begin(profile_this_iter)
                if iteration - start_iter > 5 and (
                    (iteration + 1) % cfg.TRAIN.PRINT_FREQ == 0 or iteration == max_iter - 1 or iteration < 100
                ):
                    for writer in writers:
                        writer.write()
                    # visualize some images ========================================
                    if cfg.TRAIN.VIS_IMG:
                        with torch.no_grad():
                            batch_idx = 0
                            if isinstance(target_idx, (list, tuple)):
                                target_idx_list = [int(v) for v in target_idx]
                            elif isinstance(target_idx, np.ndarray):
                                target_idx_list = [int(v) for v in target_idx.reshape(-1).tolist()]
                            elif torch.is_tensor(target_idx):
                                target_idx_list = [int(v) for v in target_idx.reshape(-1).tolist()]
                            else:
                                target_idx_list = [int(target_idx)]
                            if len(target_idx_list) == 0:
                                target_idx_list = [0]

                            # Keep pred/gt visualization on the same target view.
                            vis_target_local = 0
                            vis_target_global = target_idx_list[vis_target_local]
                            vis_target_num = max(len(target_idx_list), 1)

                            roi_img_vis = batch["roi_img"][batch_idx, vis_target_global].cpu().numpy()
                            roi_img_vis = denormalize_image(roi_img_vis, cfg).transpose(1, 2, 0).astype("uint8")
                            tbx_writer.add_image("input_images", roi_img_vis, iteration, dataformats="HWC")

                            out_coor_x = out_dict["coor_x"].detach()
                            out_coor_y = out_dict["coor_y"].detach()
                            out_coor_z = out_dict["coor_z"].detach()
                            out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
                            pred_vis_idx = min(batch_idx * vis_target_num + vis_target_local, out_xyz.shape[0] - 1)

                            out_xyz_vis = out_xyz[pred_vis_idx].cpu().numpy().transpose(1, 2, 0)
                            # out_xyz_vis = get_emb_show(out_xyz_vis)
                            out_xyz_vis = get_emb_show_percentile(out_xyz_vis, lower_percentile=5, upper_percentile=95)
                            tbx_writer.add_image("out_xyz", out_xyz_vis, iteration, dataformats="HWC")
                            mmcv.mkdir_or_exist(pose_vis_output_dir)
                            out_xyz_vis_u8 = (np.clip(out_xyz_vis, 0.0, 1.0) * 255.0).astype(np.uint8)
                            cv2.imwrite(osp.join(pose_vis_output_dir, f"pred_xyz_{iteration:06d}.png"), out_xyz_vis_u8)

                            gt_xyz_vis = batch["roi_xyz"][batch_idx, vis_target_global].cpu().numpy().transpose(1, 2, 0)
                            # gt_xyz_vis = get_emb_show(gt_xyz_vis)
                            gt_xyz_vis = get_emb_show_percentile(gt_xyz_vis, lower_percentile=5, upper_percentile=95)
                            tbx_writer.add_image("gt_xyz", gt_xyz_vis, iteration, dataformats="HWC")
                            gt_xyz_vis_u8 = (np.clip(gt_xyz_vis, 0.0, 1.0) * 255.0).astype(np.uint8)
                            cv2.imwrite(osp.join(pose_vis_output_dir, f"gt_xyz_{iteration:06d}.png"), gt_xyz_vis_u8)

                            out_mask = out_dict["mask"].detach()
                            out_mask = get_out_mask(cfg, out_mask)
                            out_mask_vis = out_mask[pred_vis_idx, 0].cpu().numpy()
                            out_mask_vis = out_mask_vis[..., None]
                            tbx_writer.add_image("out_mask", out_mask_vis, iteration, dataformats="HWC")

                            gt_mask_vis = batch["roi_mask_obj"][batch_idx, vis_target_global].detach().cpu().numpy()
                            gt_mask_vis = gt_mask_vis[..., None]
                            tbx_writer.add_image("gt_mask", gt_mask_vis, iteration, dataformats="HWC")
                            
                        # dprint("Vis tensorboard writer done.")
                _phase_end(profile_this_iter, phase_stats, "writer_vis", writer_vis_t0)

                ckpt_t0 = _phase_begin(profile_this_iter)
                if (iteration + 1) % periodic_checkpointer.period == 0 or (
                    periodic_checkpointer.max_iter is not None and (iteration + 1) >= periodic_checkpointer.max_iter
                ):
                    if hasattr(optimizer, "consolidate_state_dict"):  # for ddp_sharded
                        optimizer.consolidate_state_dict()
                    # Barrier to ensure all processes are synchronized before checkpoint saving
                    self.barrier()

                    if self.is_global_zero:
                        # Save to local /tmp first, then copy to (potentially slow) network FS
                        # in a background thread to avoid NCCL collective timeout.
                        local_ckpt_dir = "/tmp/gdrn_ckpt_tmp"
                        os.makedirs(local_ckpt_dir, exist_ok=True)

                        original_save_dir = checkpointer.save_dir
                        checkpointer.save_dir = local_ckpt_dir
                        try:
                            periodic_checkpointer.step(iteration, epoch=epoch)
                        finally:
                            checkpointer.save_dir = original_save_dir

                        # Move all newly saved files to the real output dir in a background thread
                        files_to_move = [
                            f for f in os.listdir(local_ckpt_dir) if f.endswith(".pth")
                        ]
                        if files_to_move:
                            def _async_move_ckpt(src_dir, dst_dir, filenames):
                                for fname in filenames:
                                    src = osp.join(src_dir, fname)
                                    dst = osp.join(dst_dir, fname)
                                    try:
                                        shutil.move(src, dst)
                                        logger.info(f"Checkpoint moved to {dst}")
                                        if ckpt_sync_cfg["enabled"]:
                                            upload_ok = _sync_ckpt_to_remote(dst, ckpt_sync_cfg)
                                            if upload_ok and ckpt_sync_cfg["delete_local"]:
                                                try:
                                                    os.remove(dst)
                                                    logger.info(f"Removed local checkpoint after sync: {dst}")
                                                except Exception as rm_err:
                                                    logger.error(
                                                        "Failed to remove local checkpoint %s after sync: %s",
                                                        dst,
                                                        rm_err,
                                                    )
                                    except Exception as e:
                                        logger.error(f"Failed to move checkpoint {src} -> {dst}: {e}")

                            t = threading.Thread(
                                target=_async_move_ckpt,
                                args=(local_ckpt_dir, original_save_dir, files_to_move),
                                daemon=True,
                            )
                            t.start()
                    # Quick barrier since save was to fast local disk
                    self.barrier()
                _phase_end(profile_this_iter, phase_stats, "checkpoint", ckpt_t0)

                _phase_end(profile_this_iter, phase_stats, "iter_total", iter_profile_t0)
                if profile_this_iter and phase_stats:
                    total_t = max(float(phase_stats.get("iter_total", 0.0)), 1e-8)
                    logger.info(
                        "[GDRN_PHASE] iter=%d summary_start entries=%d",
                        iteration,
                        len(phase_stats),
                    )
                    for key, value in sorted(phase_stats.items(), key=lambda kv: kv[1], reverse=True):
                        logger.info(
                            "[GDRN_PHASE] %s total=%.4fs ratio=%.2f%%",
                            key,
                            value,
                            100.0 * value / total_t,
                        )
                    logger.info("[GDRN_PHASE] iter=%d summary_end", iteration)
