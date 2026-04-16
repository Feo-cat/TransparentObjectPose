#!/usr/bin/env python3

import argparse
import io
import json
import os
import re
import tarfile
import zipfile
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw


FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract first-batch VISOR hand clips as cropped PNG+alpha sequences.")
    parser.add_argument(
        "--visor-root",
        default="/mnt/afs/datasets/2v6cgv1x04ol22qp9rm9x2j6a7",
        help="Root of the VISOR dataset release.",
    )
    parser.add_argument(
        "--epic-root",
        default="/mnt/afs/datasets/EPIC-KITCHENS/EPIC-KITCHENS",
        help="Root of downloaded EPIC-KITCHENS rgb_frames tar files.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/afs/TransparentObjectPose/debug/visor_hand_clips_batch1",
        help="Output directory for extracted clips.",
    )
    parser.add_argument(
        "--videos",
        default="P01_01,P02_01,P03_03,P04_02,P06_01",
        help="Comma separated VISOR/EPIC video ids.",
    )
    parser.add_argument("--max-clips", type=int, default=5, help="Total number of clips to export.")
    parser.add_argument("--max-per-video", type=int, default=1, help="Maximum clips to export per video.")
    parser.add_argument("--max-frames-per-clip", type=int, default=16, help="Maximum frames per exported clip.")
    parser.add_argument("--min-frames-per-clip", type=int, default=6, help="Minimum frames required for a valid clip.")
    parser.add_argument("--min-mask-pixels", type=int, default=1500, help="Minimum hand mask area in pixels.")
    parser.add_argument(
        "--dense-size",
        default="854x480",
        help="Dense VISOR annotation size, formatted as WIDTHxHEIGHT.",
    )
    return parser.parse_args()


def parse_frame_number(name):
    match = FRAME_RE.search(name)
    if not match:
        raise ValueError(f"Cannot parse frame number from {name}")
    return int(match.group(1))


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def sanitize_name(name):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())


def stream_video_annotations_from_zip(zip_path):
    decoder = json.JSONDecoder()

    with zipfile.ZipFile(zip_path, "r") as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            buffer = ""
            found_array = False

            while not found_array:
                chunk = text.read(1 << 20)
                if not chunk:
                    raise RuntimeError(f"video_annotations array not found in {zip_path}")
                buffer += chunk
                key_pos = buffer.find('"video_annotations"')
                if key_pos >= 0:
                    array_pos = buffer.find("[", key_pos)
                    if array_pos >= 0:
                        buffer = buffer[array_pos + 1 :]
                        found_array = True

            while True:
                while True:
                    stripped = buffer.lstrip()
                    leading_ws = len(buffer) - len(stripped)
                    buffer = stripped
                    if buffer.startswith(","):
                        buffer = buffer[1:]
                        continue
                    if buffer.startswith("]"):
                        return
                    break

                try:
                    obj, idx = decoder.raw_decode(buffer)
                    yield obj
                    buffer = buffer[idx:]
                except json.JSONDecodeError:
                    chunk = text.read(1 << 20)
                    if not chunk:
                        raise
                    buffer += chunk


def compute_dense_offset(image_info, mapping):
    start_name = image_info.get("interpolation_start_frame", "").replace(".png", ".jpg")
    end_name = image_info.get("interpolation_end_frame", "").replace(".png", ".jpg")
    if start_name not in mapping or end_name not in mapping:
        return None

    start_offset = parse_frame_number(mapping[start_name]) - parse_frame_number(start_name)
    end_offset = parse_frame_number(mapping[end_name]) - parse_frame_number(end_name)
    if start_offset != end_offset:
        return None
    return start_offset


def render_cropped_rgba(image_bytes, polygon_points, dense_size):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    if image.size != dense_size:
        image = image.resize(dense_size, Image.Resampling.BILINEAR)

    mask = Image.new("L", dense_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, fill=255)

    bbox = mask.getbbox()
    if bbox is None:
        return None

    rgba = image.copy()
    rgba.putalpha(mask)
    return rgba.crop(bbox), bbox, mask


def save_clip(output_root, clip_index, clip, epic_tar_path, dense_size):
    clip_dir_name = f"clip_{clip_index:03d}_{clip['video_id']}_{clip['interpolation']}_{sanitize_name(clip['hand_name'])}"
    clip_dir = output_root / clip_dir_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    with tarfile.open(epic_tar_path, "r") as tf:
        members = {member.name: member for member in tf if member.isfile()}

        for frame_idx, frame in enumerate(clip["frames"]):
            member = members.get(frame["epic_member"])
            if member is None:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            image_bytes = extracted.read()
            rendered = render_cropped_rgba(image_bytes, frame["polygon"], dense_size)
            if rendered is None:
                continue
            patch, bbox, mask = rendered
            mask_pixels = int(sum(mask.getdata()) / 255)
            if mask_pixels <= 0:
                continue

            filename = f"{frame_idx:04d}.png"
            patch.save(clip_dir / filename)
            frames_meta.append(
                {
                    "index": frame_idx,
                    "visor_frame": frame["visor_frame"],
                    "epic_frame": os.path.basename(frame["epic_member"]),
                    "patch_file": filename,
                    "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "mask_pixels": int(mask_pixels),
                    "annotation_type": int(frame["annotation_type"]),
                }
            )

    clip_meta = {
        "video_id": clip["video_id"],
        "interpolation": clip["interpolation"],
        "hand_name": clip["hand_name"],
        "frame_count": len(frames_meta),
        "offset": int(clip["offset"]),
        "frames": frames_meta,
    }
    with open(clip_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(clip_meta, f, ensure_ascii=False, indent=2)
    return clip_dir, clip_meta


def finalize_interpolation(interpolation_state, clip_candidates, args, video_id):
    if interpolation_state is None:
        return

    for hand_name, frames in interpolation_state["hands"].items():
        if len(frames) < args.min_frames_per_clip:
            continue
        clip_candidates.append(
            {
                "video_id": video_id,
                "interpolation": interpolation_state["interpolation"],
                "hand_name": hand_name,
                "offset": interpolation_state["offset"],
                "frames": frames[: args.max_frames_per_clip],
            }
        )


def collect_first_clips_for_video(video_id, args, mapping):
    dense_zip_path = os.path.join(
        args.visor_root,
        "Interpolations-DenseAnnotations",
        "train",
        f"{video_id}_interpolations.zip",
    )
    epic_tar_path = os.path.join(
        args.epic_root,
        video_id[:3],
        "rgb_frames",
        f"{video_id}.tar",
    )

    dense_w, dense_h = map(int, args.dense_size.lower().split("x"))
    dense_size = (dense_w, dense_h)
    clip_candidates = []
    interpolation_state = None

    for item in stream_video_annotations_from_zip(dense_zip_path):
        image_info = item["image"]
        interpolation = image_info.get("interpolation")
        if interpolation_state is None or interpolation != interpolation_state["interpolation"]:
            finalize_interpolation(interpolation_state, clip_candidates, args, video_id)
            if len(clip_candidates) >= args.max_per_video:
                break
            offset = compute_dense_offset(image_info, mapping)
            interpolation_state = {
                "interpolation": interpolation,
                "offset": offset,
                "hands": OrderedDict(),
            }

        if interpolation_state["offset"] is None:
            continue

        visor_frame_num = parse_frame_number(image_info["name"])
        epic_frame_num = visor_frame_num + interpolation_state["offset"]
        epic_member_name = f"./frame_{epic_frame_num:010d}.jpg"

        hand_annotations = []
        for ann in item.get("annotations", []):
            hand_name = ann.get("name", "")
            if hand_name not in ("left hand", "right hand"):
                continue
            segments = ann.get("segments", [])
            if not segments:
                continue
            polygon = [tuple(map(float, p)) for p in segments[0]]
            if polygon_area(polygon) < args.min_mask_pixels:
                continue
            hand_annotations.append((hand_name, ann, polygon))

        if not hand_annotations:
            continue

        for hand_name, ann, polygon in hand_annotations:
            frames = interpolation_state["hands"].setdefault(hand_name, [])
            frames.append(
                {
                    "visor_frame": image_info["name"],
                    "epic_member": epic_member_name,
                    "annotation_type": ann.get("type", 0),
                    "polygon": polygon,
                }
            )

    finalize_interpolation(interpolation_state, clip_candidates, args, video_id)
    return clip_candidates[: args.max_per_video]


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.visor_root, "frame_mapping.json"), "r", encoding="utf-8") as f:
        frame_mapping = json.load(f)

    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    all_clip_meta = []
    clip_index = 0

    for video_id in videos:
        if clip_index >= args.max_clips:
            break
        if video_id not in frame_mapping:
            print(f"[skip] {video_id}: no frame mapping found")
            continue
        print(f"[scan] {video_id}")
        clips = collect_first_clips_for_video(video_id, args, frame_mapping[video_id])
        if not clips:
            print(f"[skip] {video_id}: no qualifying hand clip found")
            continue
        dense_w, dense_h = map(int, args.dense_size.lower().split("x"))
        dense_size = (dense_w, dense_h)
        epic_tar_path = os.path.join(
            args.epic_root,
            video_id[:3],
            "rgb_frames",
            f"{video_id}.tar",
        )
        for clip in clips:
            if clip_index >= args.max_clips:
                break
            clip_dir, clip_meta = save_clip(output_root, clip_index, clip, epic_tar_path, dense_size)
            all_clip_meta.append({"clip_dir": str(clip_dir), **clip_meta})
            print(
                f"[saved] clip_{clip_index:03d} {clip['video_id']} {clip['interpolation']} "
                f"{clip['hand_name']} frames={len(clip['frames'])}"
            )
            clip_index += 1

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_clip_meta, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(all_clip_meta)} clips to {output_root}")


if __name__ == "__main__":
    main()
