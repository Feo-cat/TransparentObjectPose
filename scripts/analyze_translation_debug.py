#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import os.path as osp
from typing import Dict, List, Optional

import numpy as np


BRANCHES = {
    "chosen": "translation",
    "abs_head": "abs_head_translation",
    "context": "context_translation",
}


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def frame_index_from_name(name: str) -> int:
    return int(osp.splitext(osp.basename(name))[0])


def load_gt_translations(gt_dir: str, frame_names: List[str]) -> Dict[str, np.ndarray]:
    gt = {}
    for frame_name in frame_names:
        frame_idx = frame_index_from_name(frame_name)
        gt_path = osp.join(gt_dir, f"{frame_idx:06d}.npz")
        pose = np.load(gt_path, allow_pickle=True)["obj2cam_poses"]
        gt[frame_name] = np.asarray(pose[:3, 3], dtype=np.float64)
    return gt


def load_camera(meta_json: Optional[str], cam_values: Optional[List[float]]) -> Optional[np.ndarray]:
    if meta_json:
        meta = load_json(meta_json)
        cam = meta.get("camera_matrix")
        if cam is None:
            raise ValueError(f"camera_matrix missing in {meta_json}")
        return np.asarray(cam, dtype=np.float64).reshape(3, 3)
    if cam_values:
        if len(cam_values) != 9:
            raise ValueError("--cam expects 9 numbers")
        return np.asarray(cam_values, dtype=np.float64).reshape(3, 3)
    return None


def safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


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


def best_scale(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    denom = float((pred * pred).sum())
    if abs(denom) < 1e-12:
        return None
    return float((pred * gt).sum() / denom)


def project_center(trans: np.ndarray, cam: np.ndarray) -> Optional[np.ndarray]:
    z = float(trans[2])
    if abs(z) < 1e-12:
        return None
    fx = float(cam[0, 0])
    fy = float(cam[1, 1])
    px = float(cam[0, 2])
    py = float(cam[1, 2])
    return np.asarray([fx * float(trans[0]) / z + px, fy * float(trans[1]) / z + py], dtype=np.float64)


def branch_translation(item: dict, branch: str) -> Optional[np.ndarray]:
    key = BRANCHES[branch]
    if key not in item:
        return None
    return np.asarray(item[key], dtype=np.float64)


def branch_summary(
    frame_names: List[str],
    data: dict,
    gt_trans: Dict[str, np.ndarray],
    cam: Optional[np.ndarray],
) -> dict:
    summary = {}
    for branch in BRANCHES:
        series = []
        for frame in frame_names:
            trans = branch_translation(data[frame], branch)
            if trans is not None:
                series.append((frame, trans))
        if len(series) != len(frame_names):
            continue

        pred = np.stack([t for _, t in series], axis=0)
        gt = np.stack([gt_trans[name] for name, _ in series], axis=0)

        branch_info = {
            "best_global_scale": best_scale(pred.reshape(-1), gt.reshape(-1)),
            "level_rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
            "level_abs_mean": float(np.mean(np.abs(pred - gt))),
            "norm_pred": summarize(np.linalg.norm(pred, axis=1)),
            "norm_gt": summarize(np.linalg.norm(gt, axis=1)),
            "per_axis": {},
            "delta": {},
        }

        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            p = pred[:, axis_idx]
            g = gt[:, axis_idx]
            s = best_scale(p, g)
            if s is None:
                err = p - g
            else:
                err = s * p - g
            branch_info["per_axis"][axis_name] = {
                "corr": safe_corr(p, g),
                "best_scale": s,
                "rmse_after_scale": float(np.sqrt(np.mean(err ** 2))),
                "mean_abs_after_scale": float(np.mean(np.abs(err))),
                "pred": summarize(p),
                "gt": summarize(g),
            }

        if pred.shape[0] > 1:
            pred_delta = np.diff(pred, axis=0)
            gt_delta = np.diff(gt, axis=0)
            branch_info["delta"]["norm_pred"] = summarize(np.linalg.norm(pred_delta, axis=1))
            branch_info["delta"]["norm_gt"] = summarize(np.linalg.norm(gt_delta, axis=1))
            branch_info["delta"]["norm_corr"] = safe_corr(
                np.linalg.norm(pred_delta, axis=1),
                np.linalg.norm(gt_delta, axis=1),
            )
            branch_info["delta"]["per_axis"] = {}
            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                p = pred_delta[:, axis_idx]
                g = gt_delta[:, axis_idx]
                branch_info["delta"]["per_axis"][axis_name] = {
                    "corr": safe_corr(p, g),
                    "pred": summarize(p),
                    "gt": summarize(g),
                }
            top_z = np.argsort(-np.abs(pred_delta[:, 2]))[:20]
            branch_info["top_z_jumps"] = [
                {
                    "prev_frame": frame_names[int(i)],
                    "frame": frame_names[int(i) + 1],
                    "pred_delta_z": float(pred_delta[int(i), 2]),
                    "gt_delta_z": float(gt_delta[int(i), 2]),
                    "pred_delta_y": float(pred_delta[int(i), 1]),
                    "gt_delta_y": float(gt_delta[int(i), 1]),
                }
                for i in top_z
            ]

        if cam is not None:
            pred_uv = []
            gt_uv = []
            for frame, trans in series:
                pred_xy = project_center(trans, cam)
                gt_xy = project_center(gt_trans[frame], cam)
                if pred_xy is None or gt_xy is None:
                    continue
                pred_uv.append(pred_xy)
                gt_uv.append(gt_xy)
            if pred_uv:
                pred_uv = np.stack(pred_uv, axis=0)
                gt_uv = np.stack(gt_uv, axis=0)
                uv_err = np.linalg.norm(pred_uv - gt_uv, axis=1)
                branch_info["projection"] = {
                    "pixel_err": summarize(uv_err),
                    "x_err": summarize(np.abs(pred_uv[:, 0] - gt_uv[:, 0])),
                    "y_err": summarize(np.abs(pred_uv[:, 1] - gt_uv[:, 1])),
                }

        summary[branch] = branch_info
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--meta-json", default="")
    parser.add_argument("--cam", type=float, nargs="*")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_json(args.results_json)
    frame_names = sorted(data.keys())
    gt_trans = load_gt_translations(args.gt_dir, frame_names)
    cam = load_camera(args.meta_json or None, args.cam)

    per_frame_rows = []
    for idx, frame in enumerate(frame_names):
        item = data[frame]
        gt = gt_trans[frame]
        row = {
            "frame": frame,
            "frame_idx": idx,
            "chunk_index": int(item.get("chunk_index", -1)),
            "resize_ratio": float(item.get("resize_ratio", math.nan)),
            "raw_resize_ratio": float(item.get("raw_resize_ratio", math.nan)),
            "gt_tx": float(gt[0]),
            "gt_ty": float(gt[1]),
            "gt_tz": float(gt[2]),
        }
        resize_ratio = float(item.get("resize_ratio", math.nan))
        if math.isfinite(resize_ratio) and abs(resize_ratio) > 1e-12:
            row["gt_z_ratio"] = float(gt[2] / resize_ratio)
        else:
            row["gt_z_ratio"] = math.nan

        for branch, key in BRANCHES.items():
            trans = branch_translation(item, branch)
            if trans is not None:
                row[f"{branch}_tx"] = float(trans[0])
                row[f"{branch}_ty"] = float(trans[1])
                row[f"{branch}_tz"] = float(trans[2])
                row[f"{branch}_err"] = float(np.linalg.norm(trans - gt))
            else:
                row[f"{branch}_tx"] = math.nan
                row[f"{branch}_ty"] = math.nan
                row[f"{branch}_tz"] = math.nan
                row[f"{branch}_err"] = math.nan

        for key in ["abs_head_t_raw"]:
            raw = item.get(key)
            prefix = key.replace("_t_raw", "")
            if raw is not None:
                raw = np.asarray(raw, dtype=np.float64)
                row[f"{prefix}_raw_tx"] = float(raw[0])
                row[f"{prefix}_raw_ty"] = float(raw[1])
                row[f"{prefix}_raw_tz"] = float(raw[2])
            else:
                row[f"{prefix}_raw_tx"] = math.nan
                row[f"{prefix}_raw_ty"] = math.nan
                row[f"{prefix}_raw_tz"] = math.nan

        if cam is not None:
            gt_uv = project_center(gt, cam)
            row["gt_cx"] = float(gt_uv[0]) if gt_uv is not None else math.nan
            row["gt_cy"] = float(gt_uv[1]) if gt_uv is not None else math.nan
            for branch in BRANCHES:
                trans = branch_translation(item, branch)
                uv = project_center(trans, cam) if trans is not None else None
                row[f"{branch}_cx"] = float(uv[0]) if uv is not None else math.nan
                row[f"{branch}_cy"] = float(uv[1]) if uv is not None else math.nan
        per_frame_rows.append(row)

    summary = {
        "results_json": osp.abspath(args.results_json),
        "gt_dir": osp.abspath(args.gt_dir),
        "meta_json": osp.abspath(args.meta_json) if args.meta_json else None,
        "frame_count": len(frame_names),
        "branches": branch_summary(frame_names, data, gt_trans, cam),
    }

    gt_arr = np.stack([gt_trans[f] for f in frame_names], axis=0)
    chosen = np.stack([np.asarray(data[f]["translation"], dtype=np.float64) for f in frame_names], axis=0)
    resize_ratio_arr = np.asarray([float(data[f].get("resize_ratio", math.nan)) for f in frame_names], dtype=np.float64)
    gt_z_ratio = np.where(np.abs(resize_ratio_arr) > 1e-12, gt_arr[:, 2] / resize_ratio_arr, np.nan)

    raw_stats = {}
    for key in ["abs_head_t_raw"]:
        vals = []
        valid_gt = []
        for frame in frame_names:
            raw = data[frame].get(key)
            rr = float(data[frame].get("resize_ratio", math.nan))
            if raw is None or (not math.isfinite(rr)) or abs(rr) < 1e-12:
                continue
            vals.append(float(raw[2]))
            valid_gt.append(float(gt_trans[frame][2] / rr))
        if vals:
            vals_arr = np.asarray(vals, dtype=np.float64)
            gt_arr_zr = np.asarray(valid_gt, dtype=np.float64)
            scale = best_scale(vals_arr, gt_arr_zr)
            raw_stats[key] = {
                "pred_raw_z": summarize(vals_arr),
                "gt_z_ratio": summarize(gt_arr_zr),
                "corr": safe_corr(vals_arr, gt_arr_zr),
                "best_scale_to_gt_z_ratio": scale,
            }
    summary["raw_z_vs_gt_z_ratio"] = raw_stats
    summary["chosen_vs_gt_global_scale"] = best_scale(chosen.reshape(-1), gt_arr.reshape(-1))
    summary["gt_z_ratio"] = summarize(gt_z_ratio[np.isfinite(gt_z_ratio)])

    csv_path = osp.join(args.output_dir, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    summary_path = osp.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
