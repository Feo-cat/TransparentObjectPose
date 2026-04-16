#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def rotation_angle_deg(r1: np.ndarray, r2: np.ndarray) -> float:
    rel = np.asarray(r1, dtype=np.float64) @ np.asarray(r2, dtype=np.float64).T
    val = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(val))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def angle_between_vec_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = unit(v1)
    v2 = unit(v2)
    val = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(val))


def frame_index_from_name(name: str) -> int:
    return int(osp.splitext(osp.basename(name))[0])


def load_gt_rotations(gt_dir: str, frame_names: List[str]) -> Dict[str, np.ndarray]:
    gt = {}
    for frame_name in frame_names:
        frame_idx = frame_index_from_name(frame_name)
        gt_path = osp.join(gt_dir, f"{frame_idx:06d}.npz")
        data = np.load(gt_path, allow_pickle=True)
        pose = data["obj2cam_poses"]
        gt[frame_name] = np.asarray(pose[:3, :3], dtype=np.float64)
    return gt


def mask_box(mask_path: str) -> Optional[np.ndarray]:
    arr = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max(), float(xs.mean()), float(ys.mean())],
        dtype=np.float64,
    )


def mask_iou(path_a: str, path_b: str) -> float:
    a = np.array(Image.open(path_a).convert("L")) > 0
    b = np.array(Image.open(path_b).convert("L")) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / max(float(union), 1.0)


def compute_spin_delta_deg(r1: np.ndarray, r2: np.ndarray, axis_local: np.ndarray) -> float:
    # Measure residual relative rotation around the object axis after removing axis direction change.
    a1 = unit(np.asarray(r1) @ axis_local)
    a2 = unit(np.asarray(r2) @ axis_local)
    axis_delta = angle_between_vec_deg(a1, a2)
    full_delta = rotation_angle_deg(r1, r2)
    spin_sq = max(full_delta * full_delta - axis_delta * axis_delta, 0.0)
    return math.sqrt(spin_sq)


def branch_rotation(item: dict, branch: str) -> Optional[np.ndarray]:
    key = {
        "chosen": "rotation",
        "raw": "raw_rotation",
        "abs_head": "abs_head_rotation",
    }[branch]
    if key not in item:
        return None
    return np.asarray(item[key], dtype=np.float64)


def summarize(values: List[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--axis", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_json(args.results_json)
    frame_names = sorted(data.keys())
    gt_rots = load_gt_rotations(args.gt_dir, frame_names)
    axis_local = unit(np.asarray(args.axis, dtype=np.float64))

    branches = ["chosen", "raw", "abs_head"]
    per_frame_rows = []
    summary = {
        "results_json": osp.abspath(args.results_json),
        "gt_dir": osp.abspath(args.gt_dir),
        "frame_count": len(frame_names),
        "branches": {},
        "mask": {},
        "worst_windows": [],
    }

    branch_series = {b: [] for b in branches}
    for name in frame_names:
        item = data[name]
        for branch in branches:
            rot = branch_rotation(item, branch)
            if rot is not None:
                branch_series[branch].append((name, rot))

    for branch in branches:
        if len(branch_series[branch]) != len(frame_names):
            continue
        axis_err = []
        full_delta = []
        axis_delta = []
        spin_delta = []
        top_jumps = []
        for idx, (name, rot) in enumerate(branch_series[branch]):
            gt_rot = gt_rots[name]
            axis_err_val = angle_between_vec_deg(rot @ axis_local, gt_rot @ axis_local)
            axis_err.append(axis_err_val)
            row = {
                "frame": name,
                "branch": branch,
                "chunk_index": int(data[name].get("chunk_index", -1)),
                "axis_err_deg": axis_err_val,
            }
            if idx > 0:
                prev_name, prev_rot = branch_series[branch][idx - 1]
                full_delta_val = rotation_angle_deg(rot, prev_rot)
                axis_delta_val = angle_between_vec_deg(rot @ axis_local, prev_rot @ axis_local)
                spin_delta_val = compute_spin_delta_deg(rot, prev_rot, axis_local)
                full_delta.append(full_delta_val)
                axis_delta.append(axis_delta_val)
                spin_delta.append(spin_delta_val)
                row.update(
                    {
                        "prev_frame": prev_name,
                        "full_delta_deg": full_delta_val,
                        "axis_delta_deg": axis_delta_val,
                        "spin_delta_deg": spin_delta_val,
                    }
                )
                top_jumps.append(
                    {
                        "prev_frame": prev_name,
                        "frame": name,
                        "chunk_index": int(data[name].get("chunk_index", -1)),
                        "full_delta_deg": full_delta_val,
                        "axis_delta_deg": axis_delta_val,
                        "spin_delta_deg": spin_delta_val,
                    }
                )
            per_frame_rows.append(row)
        summary["branches"][branch] = {
            "axis_err_deg": summarize(axis_err),
            "full_delta_deg": summarize(full_delta),
            "axis_delta_deg": summarize(axis_delta),
            "spin_delta_deg": summarize(spin_delta),
            "top_full_delta_jumps": sorted(top_jumps, key=lambda x: x["full_delta_deg"], reverse=True)[:20],
            "top_axis_err_frames": sorted(
                [
                    {
                        "frame": name,
                        "chunk_index": int(data[name].get("chunk_index", -1)),
                        "axis_err_deg": angle_between_vec_deg(rot @ axis_local, gt_rots[name] @ axis_local),
                    }
                    for name, rot in branch_series[branch]
                ],
                key=lambda x: x["axis_err_deg"],
                reverse=True,
            )[:20],
        }

    ext_clean_iou = []
    clean_raw_center_delta = []
    ext_clean_center_delta = []
    for name in frame_names:
        item = data[name]
        if "external_mask_path" in item and "mask_path" in item and "mask_raw_path" in item:
            ext_clean_iou.append(mask_iou(item["external_mask_path"], item["mask_path"]))
            ext_box = mask_box(item["external_mask_path"])
            clean_box = mask_box(item["mask_path"])
            raw_box = mask_box(item["mask_raw_path"])
            if ext_box is not None and clean_box is not None:
                ext_clean_center_delta.append(float(np.linalg.norm(ext_box[4:6] - clean_box[4:6])))
            if clean_box is not None and raw_box is not None:
                clean_raw_center_delta.append(float(np.linalg.norm(clean_box[4:6] - raw_box[4:6])))

    summary["mask"] = {
        "external_clean_iou": summarize(ext_clean_iou),
        "external_clean_center_delta": summarize(ext_clean_center_delta),
        "clean_raw_center_delta": summarize(clean_raw_center_delta),
    }

    if "abs_head" in summary["branches"]:
        jumps = summary["branches"]["abs_head"]["top_full_delta_jumps"]
        summary["worst_windows"] = jumps[:12]
    elif "chosen" in summary["branches"]:
        jumps = summary["branches"]["chosen"]["top_full_delta_jumps"]
        summary["worst_windows"] = jumps[:12]

    summary_path = osp.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = osp.join(args.output_dir, "per_frame_metrics.csv")
    fieldnames = sorted({k for row in per_frame_rows for k in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_frame_rows:
            writer.writerow(row)

    print(f"Saved summary to {summary_path}")
    print(f"Saved per-frame metrics to {csv_path}")


if __name__ == "__main__":
    main()
