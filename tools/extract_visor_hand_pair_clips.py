#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import os.path as osp
import re
import tarfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract paired (left+right) hand clips from VISOR dense annotations.")
    parser.add_argument("--visor-root", default="/mnt/afs/datasets/2v6cgv1x04ol22qp9rm9x2j6a7")
    parser.add_argument("--epic-root", default="/mnt/afs/datasets/EPIC-KITCHENS/EPIC-KITCHENS")
    parser.add_argument(
        "--output-root",
        default="/mnt/afs/TransparentObjectPose/debug/visor_hand_pair_clips_batch1",
    )
    parser.add_argument(
        "--videos",
        default="P01_01,P02_01,P03_03,P04_02,P06_01",
        help="Comma separated VISOR video IDs.",
    )
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--max-per-video", type=int, default=1)
    parser.add_argument("--min-frames-per-clip", type=int, default=6)
    parser.add_argument("--max-frames-per-clip", type=int, default=16)
    parser.add_argument("--min-mask-pixels-per-hand", type=int, default=900)
    parser.add_argument("--dense-size", default="854x480")
    return parser.parse_args()


def parse_frame_number(name: str) -> int:
    m = FRAME_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse frame number from {name}")
    return int(m.group(1))


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def stream_video_annotations_from_zip(zip_path: str):
    decoder = json.JSONDecoder()
    with zipfile.ZipFile(zip_path, "r") as zf:
        inner = zf.namelist()[0]
        with zf.open(inner, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            buf = ""
            found = False
            while not found:
                chunk = text.read(1 << 20)
                if not chunk:
                    raise RuntimeError(f"video_annotations not found in {zip_path}")
                buf += chunk
                k = buf.find('"video_annotations"')
                if k >= 0:
                    p = buf.find("[", k)
                    if p >= 0:
                        buf = buf[p + 1 :]
                        found = True

            while True:
                buf = buf.lstrip()
                if buf.startswith(","):
                    buf = buf[1:]
                    continue
                if buf.startswith("]"):
                    return
                try:
                    obj, idx = decoder.raw_decode(buf)
                    yield obj
                    buf = buf[idx:]
                except json.JSONDecodeError:
                    chunk = text.read(1 << 20)
                    if not chunk:
                        raise
                    buf += chunk


def compute_dense_offset(image_info: dict, mapping: dict):
    s = image_info.get("interpolation_start_frame", "").replace(".png", ".jpg")
    e = image_info.get("interpolation_end_frame", "").replace(".png", ".jpg")
    if s not in mapping or e not in mapping:
        return None
    so = parse_frame_number(mapping[s]) - parse_frame_number(s)
    eo = parse_frame_number(mapping[e]) - parse_frame_number(e)
    if so != eo:
        return None
    return so


def polygon_to_mask(size, poly):
    mask = Image.new("L", size, 0)
    if len(poly) >= 2:
        ImageDraw.Draw(mask).polygon(poly, fill=255)
    return mask


def build_tar_index(tar_path: str):
    with tarfile.open(tar_path, "r") as tf:
        return {m.name: m for m in tf if m.isfile()}


def first_valid_polygon(segments):
    for seg in segments or []:
        poly = []
        for p in seg:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x = float(p[0])
                y = float(p[1])
            except (TypeError, ValueError):
                continue
            poly.append((x, y))
        if len(poly) >= 3:
            return poly
    return None


def save_pair_clip(output_root: Path, clip_idx: int, clip: dict):
    clip_name = (
        f"pair_clip_{clip_idx:03d}_{clip['video_id']}_{clip['interpolation']}_both_hands"
    )
    clip_dir = output_root / sanitize_name(clip_name)
    clip_dir.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    for i, fr in enumerate(clip["frames"]):
        fn = f"{i:04d}.png"
        fr["patch"].save(clip_dir / fn)
        x1, y1, x2, y2 = fr["bbox"]
        frames_meta.append(
            {
                "index": i,
                "visor_frame": fr["visor_frame"],
                "epic_frame": fr["epic_frame"],
                "patch_file": fn,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "mask_pixels": int(fr["mask_pixels"]),
                "left_mask_pixels": int(fr["left_mask_pixels"]),
                "right_mask_pixels": int(fr["right_mask_pixels"]),
                "annotation_type_left": int(fr["annotation_type_left"]),
                "annotation_type_right": int(fr["annotation_type_right"]),
            }
        )

    meta = {
        "video_id": clip["video_id"],
        "interpolation": clip["interpolation"],
        "hand_name": "both hands",
        "frame_count": len(frames_meta),
        "offset": int(clip["offset"]),
        "frames": frames_meta,
    }
    with open(clip_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return clip_dir, meta


def collect_pair_clips_for_video(video_id: str, args, frame_mapping: dict):
    dense_zip = osp.join(
        args.visor_root, "Interpolations-DenseAnnotations", "train", f"{video_id}_interpolations.zip"
    )
    tar_path = osp.join(args.epic_root, video_id[:3], "rgb_frames", f"{video_id}.tar")
    if not osp.exists(dense_zip) or not osp.exists(tar_path):
        return []

    w, h = map(int, args.dense_size.lower().split("x"))
    canvas = (w, h)
    tar_index = build_tar_index(tar_path)
    clips = []
    state = None

    def flush_state():
        nonlocal state
        if state is None:
            return
        if len(state["frames"]) >= args.min_frames_per_clip:
            clips.append(
                {
                    "video_id": video_id,
                    "interpolation": state["interpolation"],
                    "offset": state["offset"],
                    "frames": state["frames"][: args.max_frames_per_clip],
                }
            )
        state = None

    with tarfile.open(tar_path, "r") as tf:
        for item in stream_video_annotations_from_zip(dense_zip):
            img_info = item["image"]
            interp = img_info.get("interpolation")
            if state is None or interp != state["interpolation"]:
                flush_state()
                if len(clips) >= args.max_per_video:
                    break
                offset = compute_dense_offset(img_info, frame_mapping)
                state = {"interpolation": interp, "offset": offset, "frames": []}

            if state["offset"] is None:
                continue

            left_ann = None
            right_ann = None
            left_poly = None
            right_poly = None
            for ann in item.get("annotations", []):
                n = ann.get("name", "")
                if n == "left hand" and ann.get("segments"):
                    poly = first_valid_polygon(ann.get("segments", []))
                    if poly is None:
                        continue
                    left_ann = ann
                    left_poly = poly
                elif n == "right hand" and ann.get("segments"):
                    poly = first_valid_polygon(ann.get("segments", []))
                    if poly is None:
                        continue
                    right_ann = ann
                    right_poly = poly
            if left_ann is None or right_ann is None:
                continue

            left_mask = polygon_to_mask(canvas, left_poly)
            right_mask = polygon_to_mask(canvas, right_poly)
            left_px = int(sum(left_mask.getdata()) / 255)
            right_px = int(sum(right_mask.getdata()) / 255)
            if left_px < args.min_mask_pixels_per_hand or right_px < args.min_mask_pixels_per_hand:
                continue

            visor_name = img_info["name"].replace(".png", ".jpg")
            if visor_name not in frame_mapping:
                continue
            epic_frame = frame_mapping[visor_name]
            member_name = f"./{epic_frame}"
            member = tar_index.get(member_name)
            if member is None:
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            image = Image.open(io.BytesIO(fobj.read())).convert("RGBA")
            if image.size != canvas:
                image = image.resize(canvas, Image.Resampling.BILINEAR)

            union_mask = Image.new("L", canvas, 0)
            union_mask.paste(left_mask, mask=left_mask)
            union_mask.paste(right_mask, mask=right_mask)
            bbox = union_mask.getbbox()
            if bbox is None:
                continue
            rgba = image.copy()
            rgba.putalpha(union_mask)
            patch = rgba.crop(bbox)
            mask_pixels = int(sum(union_mask.getdata()) / 255)

            state["frames"].append(
                {
                    "visor_frame": img_info["name"],
                    "epic_frame": epic_frame,
                    "bbox": bbox,
                    "mask_pixels": mask_pixels,
                    "left_mask_pixels": left_px,
                    "right_mask_pixels": right_px,
                    "annotation_type_left": left_ann.get("type", 0),
                    "annotation_type_right": right_ann.get("type", 0),
                    "patch": patch,
                }
            )

    flush_state()
    return clips[: args.max_per_video]


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(osp.join(args.visor_root, "frame_mapping.json"), "r", encoding="utf-8") as f:
        mapping = json.load(f)

    videos = [x.strip() for x in args.videos.split(",") if x.strip()]
    clip_idx = 0
    manifest = []

    for vid in videos:
        if clip_idx >= args.max_clips:
            break
        if vid not in mapping:
            print(f"[skip] {vid}: no frame_mapping")
            continue
        print(f"[scan] {vid}")
        clips = collect_pair_clips_for_video(vid, args, mapping[vid])
        if not clips:
            print(f"[skip] {vid}: no valid pair clip")
            continue
        for c in clips:
            if clip_idx >= args.max_clips:
                break
            clip_dir, meta = save_pair_clip(output_root, clip_idx, c)
            manifest.append({"clip_dir": str(clip_dir), **meta})
            print(f"[saved] pair_clip_{clip_idx:03d} {c['video_id']} {c['interpolation']} frames={len(c['frames'])}")
            clip_idx += 1

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(manifest)} pair clips to {output_root}")


if __name__ == "__main__":
    main()
