#!/usr/bin/env python3
import argparse
import csv
import json
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from detectron2.utils.events import EventStorage
from mmcv import Config

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.normpath(osp.join(cur_dir, "..")))

from core.gdrn_modeling.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine_utils import batch_data_rand_num_perm
from core.utils.my_checkpoint import MyCheckpointer
from inference_batch import build_model_only


def summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def safe_corr(a: np.ndarray, b: np.ndarray):
    if a.size == 0 or b.size == 0:
        return None
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def best_scale(pred: np.ndarray, gt: np.ndarray):
    denom = float((pred * pred).sum())
    if abs(denom) < 1e-12:
        return None
    return float((pred * gt).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="labsim_train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATASETS.TRAIN = (args.dataset_name,)
    register_datasets_in_cfg(cfg)

    model = build_model_only(cfg, device=args.device)
    MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    data_loader = build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)
    data_iter = iter(data_loader)

    rows = []
    for batch_idx in range(args.num_batches):
        data = next(data_iter)
        batch = batch_data_rand_num_perm(cfg, data, device=args.device, phase="train")
        target_idx = list(range(batch["roi_img"].shape[1]))

        with torch.no_grad():
            with EventStorage(batch_idx):
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
                    gt_ego_rot=batch.get("ego_rot", None),
                    gt_trans=batch.get("trans", None),
                    gt_trans_ratio=batch["roi_trans_ratio"],
                    gt_points=batch.get("model_points", None),
                    sym_infos=batch.get("sym_info", None),
                    model_infos=batch.get("model_info", None),
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    scales=batch["scale"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_extents=batch.get("roi_extent", None),
                    input_images=batch["input_images"],
                    input_depths=batch.get("input_depths", None),
                    input_obj_masks=batch.get("input_obj_masks", None),
                    noisy_obj_masks=batch.get("noisy_obj_masks", None),
                    target_idx=target_idx,
                    do_loss=True,
                )

        pred_trans = out_dict["trans"].detach().cpu().numpy()
        pred_t_raw = out_dict["coarse_t_raw"].detach().cpu().numpy() if "coarse_t_raw" in out_dict else None

        gt_trans = batch["trans"][:, target_idx].reshape(-1, 3).detach().cpu().numpy()
        gt_trans_ratio = batch["roi_trans_ratio"][:, target_idx].reshape(-1, 3).detach().cpu().numpy()
        resize_ratio = batch["resize_ratio"][:, target_idx].reshape(-1).detach().cpu().numpy()
        roi_center = batch["roi_center"][:, target_idx].reshape(-1, 2).detach().cpu().numpy()
        roi_wh = batch["roi_wh"][:, target_idx].reshape(-1, 2).detach().cpu().numpy()

        for sample_idx in range(gt_trans.shape[0]):
            row = {
                "batch_idx": batch_idx,
                "sample_idx": sample_idx,
                "pred_tx": float(pred_trans[sample_idx, 0]),
                "pred_ty": float(pred_trans[sample_idx, 1]),
                "pred_tz": float(pred_trans[sample_idx, 2]),
                "gt_tx": float(gt_trans[sample_idx, 0]),
                "gt_ty": float(gt_trans[sample_idx, 1]),
                "gt_tz": float(gt_trans[sample_idx, 2]),
                "gt_ratio_tx": float(gt_trans_ratio[sample_idx, 0]),
                "gt_ratio_ty": float(gt_trans_ratio[sample_idx, 1]),
                "gt_ratio_tz": float(gt_trans_ratio[sample_idx, 2]),
                "resize_ratio": float(resize_ratio[sample_idx]),
                "roi_center_x": float(roi_center[sample_idx, 0]),
                "roi_center_y": float(roi_center[sample_idx, 1]),
                "roi_w": float(roi_wh[sample_idx, 0]),
                "roi_h": float(roi_wh[sample_idx, 1]),
                "loss_trans_xy": float(loss_dict.get("loss_trans_xy", torch.tensor(0.0)).detach().cpu().item()),
                "loss_trans_z": float(loss_dict.get("loss_trans_z", torch.tensor(0.0)).detach().cpu().item()),
                "loss_z": float(loss_dict.get("loss_z", torch.tensor(0.0)).detach().cpu().item()),
                "loss_centroid": float(loss_dict.get("loss_centroid", torch.tensor(0.0)).detach().cpu().item()),
            }
            if pred_t_raw is not None:
                row["pred_raw_tx"] = float(pred_t_raw[sample_idx, 0])
                row["pred_raw_ty"] = float(pred_t_raw[sample_idx, 1])
                row["pred_raw_tz"] = float(pred_t_raw[sample_idx, 2])
            else:
                row["pred_raw_tx"] = np.nan
                row["pred_raw_ty"] = np.nan
                row["pred_raw_tz"] = np.nan
            rows.append(row)

    pred_tz = np.asarray([r["pred_tz"] for r in rows], dtype=np.float64)
    gt_tz = np.asarray([r["gt_tz"] for r in rows], dtype=np.float64)
    gt_ratio_tz = np.asarray([r["gt_ratio_tz"] for r in rows], dtype=np.float64)
    pred_raw_tz = np.asarray([r["pred_raw_tz"] for r in rows], dtype=np.float64)

    summary = {
        "config": osp.abspath(args.config),
        "weights": osp.abspath(args.weights),
        "dataset_name": args.dataset_name,
        "num_rows": len(rows),
        "pred_tz": summarize(pred_tz),
        "gt_tz": summarize(gt_tz),
        "gt_ratio_tz": summarize(gt_ratio_tz),
        "pred_raw_tz": summarize(pred_raw_tz[np.isfinite(pred_raw_tz)]),
        "corr_pred_tz_vs_gt_tz": safe_corr(pred_tz, gt_tz),
        "corr_pred_raw_tz_vs_gt_ratio_tz": safe_corr(pred_raw_tz[np.isfinite(pred_raw_tz)], gt_ratio_tz[np.isfinite(pred_raw_tz)]),
        "best_scale_pred_tz_to_gt_tz": best_scale(pred_tz, gt_tz),
        "best_scale_pred_raw_tz_to_gt_ratio_tz": best_scale(
            pred_raw_tz[np.isfinite(pred_raw_tz)],
            gt_ratio_tz[np.isfinite(pred_raw_tz)],
        ),
    }

    csv_path = osp.join(args.output_dir, "samples.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = osp.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
