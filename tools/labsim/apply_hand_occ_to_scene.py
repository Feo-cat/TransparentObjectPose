#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import os.path as osp
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


CLIP_CANVAS_W = 854.0
CLIP_CANVAS_H = 480.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply VISOR hand clips to labsim scene and export to a new folder."
    )
    parser.add_argument(
        "--src-scene-root",
        default="/mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002",
        help="Source labsim scene folder.",
    )
    parser.add_argument(
        "--dst-scene-root",
        default="/mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002_hand_occ",
        help="Destination folder for augmented scene.",
    )
    parser.add_argument(
        "--hand-manifest",
        default="/mnt/afs/TransparentObjectPose/debug/visor_hand_clips_batch1/manifest.json",
        help="Manifest of extracted hand clips.",
    )
    parser.add_argument(
        "--hand-pair-manifest",
        default="/mnt/afs/TransparentObjectPose/debug/visor_hand_pair_clips_batch1/manifest.json",
        help="Manifest of extracted pair hand clips (both hands in one patch).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=204,
        help="Max number of frames to generate (sorted by key).",
    )
    parser.add_argument(
        "--max-folders",
        type=int,
        default=None,
        help="Optional number of source folders/sequences to export. Overrides --max-frames when set.",
    )
    parser.add_argument(
        "--folder-selection",
        choices=["first", "random"],
        default="first",
        help="How to select source folders when --max-folders is set.",
    )
    parser.add_argument(
        "--folder-offset",
        type=int,
        default=0,
        help="Start offset into sorted source folders when --max-folders is set.",
    )
    parser.add_argument(
        "--min-visible-ratio",
        type=float,
        default=0.6,
        help="Minimum visible ratio of pasted hand patch in image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260401,
        help="Random seed for reproducible placement.",
    )
    parser.add_argument(
        "--double-hand-prob",
        type=float,
        default=0.35,
        help="Probability of using a paired two-hand clip in one target folder.",
    )
    parser.add_argument(
        "--allow-mixed-two-hand",
        action="store_true",
        help="Allow combining two unrelated single-hand clips when a pair clip is unavailable.",
    )
    parser.add_argument(
        "--target-height-ratio-min",
        type=float,
        default=0.18,
        help="Minimum pasted hand height ratio to image height (480).",
    )
    parser.add_argument(
        "--target-height-ratio-max",
        type=float,
        default=0.38,
        help="Maximum pasted hand height ratio to image height (480).",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=2.2,
        help="Upper bound for scale to avoid unrealistically huge pasted hands.",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.35,
        help="Lower bound for scale to avoid too tiny pasted hands.",
    )
    parser.add_argument(
        "--placement-mode",
        choices=["random", "object_biased"],
        default="object_biased",
        help="Placement strategy for pasted hand clips.",
    )
    parser.add_argument(
        "--placement-trials",
        type=int,
        default=80,
        help="Number of candidate placements to score in object_biased mode.",
    )
    parser.add_argument(
        "--object-jitter",
        type=float,
        default=0.45,
        help="Relative placement jitter around the object center in object_biased mode.",
    )
    parser.add_argument(
        "--placement-bbox-source",
        choices=["bbox_obj", "bbox_visib"],
        default="bbox_obj",
        help="Which bbox from scene_gt_info to use for object_biased placement.",
    )
    parser.add_argument(
        "--min-object-occlusion-ratio",
        type=float,
        default=0.03,
        help="Minimum average object occlusion ratio per folder to accept a placement.",
    )
    parser.add_argument(
        "--min-frame-object-occlusion-ratio",
        type=float,
        default=0.0,
        help="Minimum per-frame object occlusion ratio required for every frame in the chosen clip.",
    )
    parser.add_argument(
        "--max-object-occlusion-ratio",
        type=float,
        default=0.20,
        help="Maximum per-frame object occlusion ratio allowed for a placement.",
    )
    parser.add_argument(
        "--target-object-occlusion-ratio",
        type=float,
        default=0.12,
        help="Target average object occlusion ratio used to rank valid placements.",
    )
    parser.add_argument(
        "--scale-attempts",
        type=int,
        default=8,
        help="Number of different scales to try per sampled hand clip.",
    )
    parser.add_argument(
        "--clip-attempts-per-folder",
        type=int,
        default=12,
        help="Number of clip candidates to try when searching a valid placement for a folder.",
    )
    parser.add_argument(
        "--min-hand-height-px",
        type=int,
        default=80,
        help="Minimum resized hand patch height in pixels.",
    )
    parser.add_argument(
        "--max-hand-height-px",
        type=int,
        default=1000000,
        help="Maximum resized hand patch height in pixels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it already exists.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def key_to_folder_frame(key: str) -> Tuple[str, str]:
    folder, frame = key.split("_")
    return folder, frame


def xywh_to_xyxy(box: List[int]) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return int(x), int(y), int(x + w), int(y + h)


def bbox_area_xywh(box: List[int]) -> int:
    return max(0, int(box[2])) * max(0, int(box[3]))


def bbox_from_binary(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def visible_ratio(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> float:
    ix1 = max(0, x)
    iy1 = max(0, y)
    ix2 = min(image_w, x + w)
    iy2 = min(image_h, y + h)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    denom = max(1, w * h)
    return float(inter) / float(denom)


def intersection_area_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def choose_base_xy(
    rng: random.Random,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
    min_ratio: float,
    max_trials: int = 100,
) -> Tuple[int, int]:
    low_x = -int(0.35 * w)
    high_x = int(image_w - 0.65 * w)
    low_y = -int(0.35 * h)
    high_y = int(image_h - 0.65 * h)
    for _ in range(max_trials):
        x = rng.randint(low_x, high_x)
        y = rng.randint(low_y, high_y)
        if visible_ratio(x, y, w, h, image_w, image_h) >= min_ratio:
            return x, y
    x = int((image_w - w) / 2)
    y = int((image_h - h) / 2)
    return x, y


def choose_selected_keys(
    rng: random.Random,
    scene_gt: Dict[str, List[Dict]],
    max_frames: int,
    max_folders: int | None,
    folder_selection: str,
    folder_offset: int,
) -> List[str]:
    all_keys = sorted(scene_gt.keys())
    if max_folders is None:
        return all_keys[: min(max_frames, len(all_keys))]

    folder_to_frames: Dict[str, List[str]] = {}
    for key in all_keys:
        folder, frame = key_to_folder_frame(key)
        folder_to_frames.setdefault(folder, []).append(frame)

    all_folders = sorted(folder_to_frames.keys())
    start = max(0, min(folder_offset, len(all_folders)))
    candidate_folders = all_folders[start:]
    if not candidate_folders:
        return []

    count = min(max_folders, len(candidate_folders))
    if folder_selection == "random":
        selected_folders = sorted(rng.sample(candidate_folders, count))
    else:
        selected_folders = candidate_folders[:count]

    selected = []
    for folder in selected_folders:
        for frame in sorted(folder_to_frames[folder]):
            selected.append(f"{folder}_{frame}")
    return selected


def choose_primary_bbox(info_list: List[Dict], bbox_key: str) -> List[int]:
    if not info_list:
        return [0, 0, 0, 0]
    best = max(info_list, key=lambda item: int(item.get("px_count_visib", 0)))
    box = best.get(bbox_key) or best.get("bbox_obj") or best.get("bbox_visib") or [0, 0, 0, 0]
    return [int(v) for v in box]


def build_folder_object_boxes(
    folder: str,
    frames: List[str],
    scene_gt_info: Dict[str, List[Dict]],
    bbox_key: str,
) -> List[List[int]]:
    return [choose_primary_bbox(scene_gt_info.get(f"{folder}_{frame}", []), bbox_key) for frame in frames]


def load_folder_object_track(
    src_root: str,
    folder: str,
    frames: List[str],
    scene_gt_info: Dict[str, List[Dict]],
    bbox_key: str,
) -> List[Dict]:
    track = []
    for frame in frames:
        key = f"{folder}_{frame}"
        infos = scene_gt_info.get(key, [])
        bbox = choose_primary_bbox(infos, bbox_key)
        mask_union = np.zeros((480, 640), dtype=bool)
        for anno_i, _info in enumerate(infos):
            mask_path = osp.join(src_root, "mask_visib", folder, f"{frame}_{anno_i:06d}.png")
            if not osp.exists(mask_path):
                continue
            mask_union |= np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        area = int(mask_union.sum())
        if area <= 0:
            area = max(1, int(sum(int(info.get("px_count_visib", 0)) for info in infos)))
        track.append(
            {
                "bbox": bbox,
                "mask": mask_union,
                "area": max(1, area),
            }
        )
    return track


def resize_patch_rgba(patch_path: str, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    patch = Image.open(patch_path).convert("RGBA")
    pw, ph = patch.size
    sw = max(1, int(round(pw * scale)))
    sh = max(1, int(round(ph * scale)))
    patch = patch.resize((sw, sh), Image.Resampling.LANCZOS)
    patch_u8 = np.array(patch, dtype=np.uint8)
    alpha = patch_u8[..., 3] > 0
    return patch_u8, alpha


def prepare_clip_frames_for_scale(clip: Dict, scale: float, sx: float, sy: float) -> List[Dict]:
    clip_frames = clip["meta"]["frames"]
    if not clip_frames:
        return []
    bbox0 = clip_frames[0]["bbox_xyxy"]
    prepared = []
    for frame_clip in clip_frames:
        patch_path = osp.join(clip["clip_dir"], frame_clip["patch_file"])
        patch_u8, alpha = resize_patch_rgba(patch_path, scale)
        b = frame_clip["bbox_xyxy"]
        prepared.append(
            {
                "rgba_u8": patch_u8,
                "alpha": alpha,
                "dx": int(round((b[0] - bbox0[0]) * sx)),
                "dy": int(round((b[1] - bbox0[1]) * sy)),
                "w": int(patch_u8.shape[1]),
                "h": int(patch_u8.shape[0]),
                "frame_index": int(frame_clip["index"]),
            }
        )
    return prepared


def compute_mask_overlap(alpha_mask: np.ndarray, x: int, y: int, object_mask: np.ndarray) -> int:
    h, w = alpha_mask.shape
    img_h, img_w = object_mask.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x1 >= x2 or y1 >= y2:
        return 0

    px1 = x1 - x
    py1 = y1 - y
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)
    alpha_crop = alpha_mask[py1:py2, px1:px2]
    obj_crop = object_mask[y1:y2, x1:x2]
    return int(np.logical_and(alpha_crop, obj_crop).sum())


def choose_candidate_edges(preferred_edge: str | None, object_track: List[Dict], image_w: int, image_h: int) -> List[str]:
    valid_boxes = [item["bbox"] for item in object_track if bbox_area_xywh(item["bbox"]) > 0]
    if not valid_boxes:
        return [preferred_edge] if preferred_edge is not None else ["bottom"]

    centers_x = [b[0] + 0.5 * b[2] for b in valid_boxes]
    centers_y = [b[1] + 0.5 * b[3] for b in valid_boxes]
    ref_cx = float(np.median(centers_x))
    ref_cy = float(np.median(centers_y))
    obj_edge_dists = {
        "left": ref_cx,
        "right": image_w - ref_cx,
        "top": ref_cy,
        "bottom": image_h - ref_cy,
    }
    ranked_edges = sorted(obj_edge_dists.keys(), key=lambda edge: obj_edge_dists[edge])
    candidate_edges = []
    if preferred_edge is not None:
        candidate_edges.append(preferred_edge)
    candidate_edges.extend(ranked_edges[:2])
    return list(dict.fromkeys(candidate_edges))


def sample_edge_candidate(
    rng: random.Random,
    edge: str,
    ref_box: List[int],
    patch_w: int,
    patch_h: int,
    image_w: int,
    image_h: int,
    jitter: float,
) -> Tuple[int, int]:
    x, y, w, h = [float(v) for v in ref_box]
    cx = x + 0.5 * w
    jitter_x = jitter * max(w, 0.4 * patch_w)
    jitter_y = jitter * max(h, 0.4 * patch_h)

    if edge == "left":
        base_x = int(round(rng.uniform(-0.25 * patch_w, 0.05 * patch_w)))
        touch_y = rng.uniform(y + 0.2 * h - jitter_y, y + 0.85 * h + jitter_y)
        base_y = int(round(touch_y - rng.uniform(0.3, 0.7) * patch_h))
    elif edge == "right":
        base_x = int(round(rng.uniform(image_w - patch_w - 0.05 * patch_w, image_w - 0.75 * patch_w)))
        touch_y = rng.uniform(y + 0.2 * h - jitter_y, y + 0.85 * h + jitter_y)
        base_y = int(round(touch_y - rng.uniform(0.3, 0.7) * patch_h))
    elif edge == "top":
        touch_x = rng.uniform(cx - jitter_x, cx + jitter_x)
        base_x = int(round(touch_x - rng.uniform(0.3, 0.7) * patch_w))
        touch_y = rng.uniform(y + 0.05 * h - jitter_y, y + 0.45 * h + jitter_y)
        base_y = int(round(touch_y - rng.uniform(0.05, 0.35) * patch_h))
    else:
        touch_x = rng.uniform(cx - jitter_x, cx + jitter_x)
        base_x = int(round(touch_x - rng.uniform(0.3, 0.7) * patch_w))
        # Bottom-entry hands should more often interact with the lower half
        # of the object, instead of always crossing the object's top.
        touch_y = rng.uniform(y + 0.45 * h - jitter_y, y + 0.95 * h + jitter_y)
        base_y = int(round(touch_y - rng.uniform(0.65, 0.95) * patch_h))

    return base_x, base_y


def evaluate_placement(
    prepared_frames: List[Dict],
    object_track: List[Dict],
    base_x: int,
    base_y: int,
    min_visible_ratio: float,
    min_frame_occ_ratio: float,
    min_occ_ratio: float,
    max_occ_ratio: float,
    target_occ_ratio: float,
    image_w: int,
    image_h: int,
) -> Dict | None:
    ratios = []
    overlaps = []
    for frame_idx, obj in enumerate(object_track):
        prepared = prepared_frames[frame_idx % len(prepared_frames)]
        x = base_x + prepared["dx"]
        y = base_y + prepared["dy"]
        if visible_ratio(x, y, prepared["w"], prepared["h"], image_w, image_h) < min_visible_ratio:
            return None
        overlap = compute_mask_overlap(prepared["alpha"], x, y, obj["mask"])
        ratio = float(overlap) / float(max(1, obj["area"]))
        if ratio < min_frame_occ_ratio:
            return None
        if ratio > max_occ_ratio:
            return None
        ratios.append(ratio)
        overlaps.append(overlap)

    avg_ratio = float(np.mean(ratios)) if ratios else 0.0
    max_ratio = float(np.max(ratios)) if ratios else 0.0
    total_overlap = int(sum(overlaps))
    if avg_ratio < min_occ_ratio or total_overlap <= 0:
        return None

    score = -abs(avg_ratio - target_occ_ratio) * 100000.0 + float(total_overlap)
    return {
        "score": score,
        "avg_ratio": avg_ratio,
        "max_ratio": max_ratio,
        "total_overlap": total_overlap,
        "ratios": ratios,
    }


def find_valid_base_xy(
    rng: random.Random,
    prepared_frames: List[Dict],
    object_track: List[Dict],
    min_visible_ratio: float,
    min_frame_occ_ratio: float,
    min_occ_ratio: float,
    max_occ_ratio: float,
    target_occ_ratio: float,
    image_w: int,
    image_h: int,
    trials: int,
    jitter: float,
    preferred_edge: str | None,
) -> Tuple[int, int, Dict] | Tuple[None, None, None]:
    if not prepared_frames:
        return None, None, None

    ref_box = next((item["bbox"] for item in object_track if bbox_area_xywh(item["bbox"]) > 0), [0, 0, 0, 0])
    if bbox_area_xywh(ref_box) <= 0:
        return None, None, None

    patch_w = prepared_frames[0]["w"]
    patch_h = prepared_frames[0]["h"]
    candidate_edges = choose_candidate_edges(preferred_edge, object_track, image_w, image_h)
    best = None
    trials_per_edge = max(1, int(np.ceil(float(max(1, trials)) / float(max(1, len(candidate_edges))))))
    for edge in candidate_edges:
        for _ in range(trials_per_edge):
            base_x, base_y = sample_edge_candidate(
                rng=rng,
                edge=edge,
                ref_box=ref_box,
                patch_w=patch_w,
                patch_h=patch_h,
                image_w=image_w,
                image_h=image_h,
                jitter=jitter,
            )
            metrics = evaluate_placement(
                prepared_frames=prepared_frames,
                object_track=object_track,
                base_x=base_x,
                base_y=base_y,
                min_visible_ratio=min_visible_ratio,
                min_frame_occ_ratio=min_frame_occ_ratio,
                min_occ_ratio=min_occ_ratio,
                max_occ_ratio=max_occ_ratio,
                target_occ_ratio=target_occ_ratio,
                image_w=image_w,
                image_h=image_h,
            )
            if metrics is None:
                continue
            if (best is None) or (metrics["score"] > best["metrics"]["score"]):
                best = {
                    "base_x": base_x,
                    "base_y": base_y,
                    "metrics": metrics,
                }

    if best is None:
        return None, None, None
    return best["base_x"], best["base_y"], best["metrics"]


def infer_clip_entry_edge(clip_frames: List[Dict]) -> str:
    if not clip_frames:
        return "bottom"

    first = clip_frames[0]["bbox_xyxy"]
    x1, y1, x2, y2 = [int(v) for v in first]
    tol = 2
    touched = []
    if x1 <= tol:
        touched.append(("left", y2 - y1))
    if x2 >= int(CLIP_CANVAS_W) - tol:
        touched.append(("right", y2 - y1))
    if y1 <= tol:
        touched.append(("top", x2 - x1))
    if y2 >= int(CLIP_CANVAS_H) - tol:
        touched.append(("bottom", x2 - x1))
    if touched:
        return max(touched, key=lambda item: item[1])[0]

    dists = {
        "left": x1,
        "right": int(CLIP_CANVAS_W) - x2,
        "top": y1,
        "bottom": int(CLIP_CANVAS_H) - y2,
    }
    return min(dists, key=dists.get)


def find_hand_state_for_clip(
    rng: random.Random,
    clip: Dict,
    object_track: List[Dict],
    args,
    sx: float,
    sy: float,
) -> Dict | None:
    clip_frames = clip["meta"]["frames"]
    clip_len = len(clip_frames)
    if clip_len == 0:
        return None

    first_patch_path = osp.join(clip["clip_dir"], clip_frames[0]["patch_file"])
    first_patch = Image.open(first_patch_path).convert("RGBA")
    _fp_w, fp_h = first_patch.size
    entry_edge = infer_clip_entry_edge(clip_frames)

    for _ in range(max(1, args.scale_attempts)):
        scale = sample_scale(
            rng=rng,
            patch_h=fp_h,
            target_h_min=args.target_height_ratio_min,
            target_h_max=args.target_height_ratio_max,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
        prepared_frames = prepare_clip_frames_for_scale(clip, scale, sx, sy)
        if not prepared_frames:
            continue
        if min(frame["h"] for frame in prepared_frames) < args.min_hand_height_px:
            continue
        if max(frame["h"] for frame in prepared_frames) > args.max_hand_height_px:
            continue
        base_x, base_y, metrics = find_valid_base_xy(
            rng=rng,
            prepared_frames=prepared_frames,
            object_track=object_track,
            min_visible_ratio=args.min_visible_ratio,
            min_frame_occ_ratio=args.min_frame_object_occlusion_ratio,
            min_occ_ratio=args.min_object_occlusion_ratio,
            max_occ_ratio=args.max_object_occlusion_ratio,
            target_occ_ratio=args.target_object_occlusion_ratio,
            image_w=640,
            image_h=480,
            trials=args.placement_trials,
            jitter=args.object_jitter,
            preferred_edge=entry_edge,
        )
        if metrics is None:
            continue
        return {
            "clip": clip,
            "clip_meta": clip["meta"],
            "clip_frames": clip_frames,
            "clip_len": clip_len,
            "scale": scale,
            "base_x": base_x,
            "base_y": base_y,
            "bbox0": clip_frames[0]["bbox_xyxy"],
            "entry_edge": entry_edge,
            "prepared_frames": prepared_frames,
            "placement_metrics": metrics,
        }
    return None


def overlay_rgba_on_rgb(
    rgb_u8: np.ndarray, patch_rgba_u8: np.ndarray, x: int, y: int
) -> Tuple[np.ndarray, np.ndarray]:
    out = rgb_u8.copy()
    hand_mask = np.zeros((rgb_u8.shape[0], rgb_u8.shape[1]), dtype=np.uint8)
    h_img, w_img = out.shape[:2]
    h_patch, w_patch = patch_rgba_u8.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w_patch)
    y2 = min(h_img, y + h_patch)

    if x1 >= x2 or y1 >= y2:
        return out, hand_mask

    px1 = x1 - x
    py1 = y1 - y
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)

    patch_crop = patch_rgba_u8[py1:py2, px1:px2, :]
    alpha = (patch_crop[..., 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
    fg = patch_crop[..., :3].astype(np.float32)
    bg = out[y1:y2, x1:x2, :].astype(np.float32)
    out[y1:y2, x1:x2, :] = np.round(alpha * fg + (1.0 - alpha) * bg).astype(np.uint8)
    hand_mask[y1:y2, x1:x2] = (patch_crop[..., 3] > 0).astype(np.uint8)
    return out, hand_mask


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_hand_clips(manifest_path: str) -> List[Dict]:
    manifest = load_json(manifest_path)
    clips = []
    for item in manifest:
        meta_path = osp.join(item["clip_dir"], "meta.json")
        meta = load_json(meta_path)
        clips.append(
            {
                "clip_dir": item["clip_dir"],
                "meta": meta,
            }
        )
    if not clips:
        raise RuntimeError("No hand clips found in manifest.")
    return clips


def load_optional_hand_clips(manifest_path: str) -> List[Dict]:
    if not manifest_path or (not osp.exists(manifest_path)):
        return []
    manifest = load_json(manifest_path)
    if not manifest:
        return []
    return load_hand_clips(manifest_path)


def sample_hand_clips_for_folder(rng: random.Random, clips: List[Dict], two_hand_prob: float) -> List[Dict]:
    if len(clips) == 1:
        return [clips[0]]

    use_two = rng.random() < two_hand_prob
    if not use_two:
        return [rng.choice(clips)]

    left = [c for c in clips if c["meta"].get("hand_name") == "left hand"]
    right = [c for c in clips if c["meta"].get("hand_name") == "right hand"]
    if left and right:
        return [rng.choice(left), rng.choice(right)]

    a, b = rng.sample(clips, 2)
    return [a, b]


def sample_scale(
    rng: random.Random,
    patch_h: int,
    target_h_min: float,
    target_h_max: float,
    min_scale: float,
    max_scale: float,
) -> float:
    target_h = int(rng.uniform(target_h_min, target_h_max) * 480.0)
    scale = max(0.1, float(target_h) / float(max(1, patch_h)))
    scale = max(min_scale, min(max_scale, scale))
    return scale


def copy_root_bbox_jsons(src_root: str, dst_root: str):
    for fn in os.listdir(src_root):
        if fn.endswith("_bbox.json"):
            shutil.copy2(osp.join(src_root, fn), osp.join(dst_root, fn))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    src_root = args.src_scene_root
    dst_root = args.dst_scene_root

    if not osp.isdir(src_root):
        raise FileNotFoundError(f"Source scene not found: {src_root}")

    if osp.exists(dst_root):
        if args.overwrite:
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(f"Destination exists: {dst_root}. Use --overwrite to replace.")
    ensure_dir(dst_root)

    scene_gt = load_json(osp.join(src_root, "scene_gt.json"))
    scene_gt_info = load_json(osp.join(src_root, "scene_gt_info.json"))
    scene_camera = load_json(osp.join(src_root, "scene_camera.json"))
    sel_keys = choose_selected_keys(
        rng=rng,
        scene_gt=scene_gt,
        max_frames=args.max_frames,
        max_folders=args.max_folders,
        folder_selection=args.folder_selection,
        folder_offset=args.folder_offset,
    )
    if not sel_keys:
        raise RuntimeError("No frame keys selected.")

    clips = load_hand_clips(args.hand_manifest)
    pair_clips = load_optional_hand_clips(args.hand_pair_manifest)
    print(f"Loaded {len(clips)} hand clips")
    print(f"Loaded {len(pair_clips)} pair clips")
    print(f"Selected {len(sel_keys)} frame keys from source scene")
    if args.max_folders is not None:
        print(
            f"Selected {len({key_to_folder_frame(k)[0] for k in sel_keys})} folders "
            f"using mode={args.folder_selection}"
        )

    sel_by_folder: Dict[str, List[str]] = {}
    for key in sel_keys:
        folder, frame = key_to_folder_frame(key)
        sel_by_folder.setdefault(folder, []).append(frame)
    for folder in sel_by_folder:
        sel_by_folder[folder] = sorted(sel_by_folder[folder])

    out_scene_gt = {}
    out_scene_gt_info = {}
    out_scene_camera = {}
    aug_manifest = []
    stats = {
        "total_frames": 0,
        "frames_with_overlap": 0,
        "frames_without_overlap": 0,
        "total_occluded_pixels": 0,
        "folders_single_hand": 0,
        "folders_double_hand": 0,
        "folders_double_hand_pair": 0,
        "folders_double_hand_split": 0,
        "folders_requested_double_but_missing_pair": 0,
    }

    for sub in ["rgb", "depth", "mask", "mask_visib", "npz"]:
        ensure_dir(osp.join(dst_root, sub))

    for folder in sorted(sel_by_folder.keys()):
        use_double = rng.random() < args.double_hand_prob
        want_pair = use_double and (len(pair_clips) > 0)
        object_track = load_folder_object_track(
            src_root=src_root,
            folder=folder,
            frames=sel_by_folder[folder],
            scene_gt_info=scene_gt_info,
            bbox_key=args.placement_bbox_source,
        )

        hand_states = []
        if want_pair:
            for _ in range(max(1, args.clip_attempts_per_folder)):
                state = find_hand_state_for_clip(
                    rng=rng,
                    clip=rng.choice(pair_clips),
                    object_track=object_track,
                    args=args,
                    sx=640.0 / CLIP_CANVAS_W,
                    sy=480.0 / CLIP_CANVAS_H,
                )
                if state is not None:
                    hand_states = [state]
                    break
        elif use_double and args.allow_mixed_two_hand:
            selected_states = []
            for desired_name in ("left hand", "right hand"):
                pool = [c for c in clips if c["meta"].get("hand_name") == desired_name]
                if not pool:
                    selected_states = []
                    break
                state = None
                for _ in range(max(1, args.clip_attempts_per_folder)):
                    state = find_hand_state_for_clip(
                        rng=rng,
                        clip=rng.choice(pool),
                        object_track=object_track,
                        args=args,
                        sx=640.0 / CLIP_CANVAS_W,
                        sy=480.0 / CLIP_CANVAS_H,
                    )
                    if state is not None:
                        break
                if state is None:
                    selected_states = []
                    break
                selected_states.append(state)
            hand_states = selected_states
        else:
            if use_double and not pair_clips:
                stats["folders_requested_double_but_missing_pair"] += 1
            for _ in range(max(1, args.clip_attempts_per_folder)):
                state = find_hand_state_for_clip(
                    rng=rng,
                    clip=rng.choice(clips),
                    object_track=object_track,
                    args=args,
                    sx=640.0 / CLIP_CANVAS_W,
                    sy=480.0 / CLIP_CANVAS_H,
                )
                if state is not None:
                    hand_states = [state]
                    break

        if not hand_states:
            continue

        if want_pair and len(hand_states) == 1:
            stats["folders_double_hand"] += 1
            stats["folders_double_hand_pair"] += 1
        elif len(hand_states) == 1:
            stats["folders_single_hand"] += 1
        else:
            stats["folders_double_hand"] += 1
            stats["folders_double_hand_split"] += 1

        sx = 640.0 / CLIP_CANVAS_W
        sy = 480.0 / CLIP_CANVAS_H

        for idx, frame in enumerate(sel_by_folder[folder]):
            key = f"{folder}_{frame}"
            src_rgb_path = osp.join(src_root, "rgb", folder, f"{frame}.png")
            src_rgb = np.array(Image.open(src_rgb_path).convert("RGB"), dtype=np.uint8)
            out_rgb = src_rgb.copy()
            hand_mask = np.zeros((src_rgb.shape[0], src_rgb.shape[1]), dtype=np.uint8)
            frame_sources = []

            for state in hand_states:
                prepared = state["prepared_frames"][idx % len(state["prepared_frames"])]
                patch_u8 = prepared["rgba_u8"]
                x = state["base_x"] + prepared["dx"]
                y = state["base_y"] + prepared["dy"]

                out_rgb, hand_mask_i = overlay_rgba_on_rgb(out_rgb, patch_u8, x, y)
                hand_mask = np.maximum(hand_mask, hand_mask_i)
                frame_sources.append(
                    {
                        "source_clip_video": state["clip_meta"]["video_id"],
                        "source_clip_interp": state["clip_meta"]["interpolation"],
                        "source_hand_name": state["clip_meta"]["hand_name"],
                        "source_clip_frame_index": int(prepared["frame_index"]),
                        "source_entry_edge": state["entry_edge"],
                        "paste_xy": [int(x), int(y)],
                        "scale": float(state["scale"]),
                        "avg_occlusion_ratio": float(state["placement_metrics"]["avg_ratio"]),
                        "max_occlusion_ratio": float(state["placement_metrics"]["max_ratio"]),
                    }
                )

            out_rgb_path = osp.join(dst_root, "rgb", folder, f"{frame}.png")
            ensure_dir(osp.dirname(out_rgb_path))
            Image.fromarray(out_rgb).save(out_rgb_path)

            src_depth = osp.join(src_root, "depth", folder, f"{frame}.npy")
            dst_depth = osp.join(dst_root, "depth", folder, f"{frame}.npy")
            ensure_dir(osp.dirname(dst_depth))
            shutil.copy2(src_depth, dst_depth)

            src_npz = osp.join(src_root, "npz", folder, f"{frame}.npz")
            dst_npz = osp.join(dst_root, "npz", folder, f"{frame}.npz")
            ensure_dir(osp.dirname(dst_npz))
            shutil.copy2(src_npz, dst_npz)

            annos = scene_gt[key]
            infos = copy.deepcopy(scene_gt_info[key])
            out_infos = []
            frame_overlap_any = False
            frame_occluded_pixels = 0

            for anno_i, info in enumerate(infos):
                src_mask_path = osp.join(src_root, "mask", folder, f"{frame}_{anno_i:06d}.png")
                dst_mask_path = osp.join(dst_root, "mask", folder, f"{frame}_{anno_i:06d}.png")
                ensure_dir(osp.dirname(dst_mask_path))
                shutil.copy2(src_mask_path, dst_mask_path)

                src_mv_path = osp.join(src_root, "mask_visib", folder, f"{frame}_{anno_i:06d}.png")
                old_mv = np.array(Image.open(src_mv_path).convert("L"), dtype=np.uint8)
                old_fg = (old_mv > 0).astype(np.uint8)
                overlap = (old_fg > 0) & (hand_mask > 0)
                overlap_pixels = int(overlap.sum())
                frame_occluded_pixels += overlap_pixels
                if overlap_pixels > 0:
                    frame_overlap_any = True
                    new_fg = old_fg.copy()
                    new_fg[overlap] = 0
                else:
                    new_fg = old_fg

                new_mv = (new_fg * 255).astype(np.uint8)
                dst_mv_path = osp.join(dst_root, "mask_visib", folder, f"{frame}_{anno_i:06d}.png")
                ensure_dir(osp.dirname(dst_mv_path))
                Image.fromarray(new_mv).save(dst_mv_path)

                updated = copy.deepcopy(info)
                px_count_visib = int(new_fg.sum())
                updated["px_count_visib"] = px_count_visib
                updated["bbox_visib"] = bbox_from_binary(new_fg)
                px_all = max(1, int(updated.get("px_count_all", 1)))
                updated["visib_fract"] = float(px_count_visib) / float(px_all)
                out_infos.append(updated)

            out_scene_gt[key] = copy.deepcopy(annos)
            out_scene_gt_info[key] = out_infos
            out_scene_camera[key] = copy.deepcopy(scene_camera[key])

            stats["total_frames"] += 1
            stats["total_occluded_pixels"] += frame_occluded_pixels
            if frame_overlap_any:
                stats["frames_with_overlap"] += 1
            else:
                stats["frames_without_overlap"] += 1

            aug_manifest.append(
                {
                    "key": key,
                    "folder": folder,
                    "frame": frame,
                    "num_hands": int(
                        sum(2 if s.get("source_hand_name") == "both hands" else 1 for s in frame_sources)
                    ),
                    "num_sources": len(frame_sources),
                    "sources": frame_sources,
                    "did_overlap_object": bool(frame_overlap_any),
                    "occluded_pixels": int(frame_occluded_pixels),
                }
            )

    dump_json(out_scene_gt, osp.join(dst_root, "scene_gt.json"))
    dump_json(out_scene_gt_info, osp.join(dst_root, "scene_gt_info.json"))
    dump_json(out_scene_camera, osp.join(dst_root, "scene_camera.json"))
    dump_json(aug_manifest, osp.join(dst_root, "hand_occ_manifest.json"))
    dump_json(stats, osp.join(dst_root, "hand_occ_stats.json"))
    copy_root_bbox_jsons(src_root, dst_root)

    print("Done.")
    print(f"Output: {dst_root}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
