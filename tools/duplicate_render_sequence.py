#!/usr/bin/env python3
"""Duplicate a rendered frame sequence by repeating each source frame.

Example:
  python tools/duplicate_render_sequence.py \
    --input-dir /mnt/afs/bpy_rendering/foo \
    --output-dir /mnt/afs/bpy_rendering/foo_512frames \
    --repeat 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


FILE_PATTERNS = [
    (re.compile(r"^(\d+)\.png$"), ".png"),
    (re.compile(r"^(\d+)_mask\.png$"), "_mask.png"),
    (re.compile(r"^(\d+)_mask_visib\.png$"), "_mask_visib.png"),
    (re.compile(r"^(\d+)_depth\.exr$"), "_depth.exr"),
    (re.compile(r"^(\d+)\.npz$"), ".npz"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2, help="Repeat each source frame this many times.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_groups(input_dir: Path) -> tuple[dict[str, dict[int, Path]], list[int], int]:
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    index_width = 0
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        for regex, suffix in FILE_PATTERNS:
            match = regex.fullmatch(path.name)
            if match is None:
                continue
            frame_idx_str = match.group(1)
            frame_idx = int(frame_idx_str)
            groups[suffix][frame_idx] = path
            index_width = max(index_width, len(frame_idx_str))
            break
    if ".png" not in groups or not groups[".png"]:
        raise ValueError(f"No RGB frames found in {input_dir}")
    rgb_indices = sorted(groups[".png"].keys())
    expected = list(range(rgb_indices[0], rgb_indices[0] + len(rgb_indices)))
    if rgb_indices != expected:
        raise ValueError(f"RGB indices are not consecutive in {input_dir}")
    for suffix, mapping in groups.items():
        if suffix == ".png":
            continue
        if sorted(mapping.keys()) != rgb_indices:
            raise ValueError(f"Indices for {suffix} do not match RGB frames in {input_dir}")
    return groups, rgb_indices, index_width


def format_name(index: int, width: int, suffix: str) -> str:
    return f"{index:0{width}d}{suffix}"


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    repeat = int(args.repeat)

    if repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    groups, rgb_indices, index_width = collect_groups(input_dir)
    total_frames = len(rgb_indices) * repeat
    index_width = max(index_width, len(str(total_frames - 1)))

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for src_pos, src_idx in enumerate(rgb_indices):
        for rep in range(repeat):
            dst_idx = src_pos * repeat + rep
            for suffix, mapping in groups.items():
                src_path = mapping[src_idx]
                dst_path = output_dir / format_name(dst_idx, index_width, suffix)
                shutil.copy2(src_path, dst_path)

    meta = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "repeat": repeat,
        "source_frame_count": len(rgb_indices),
        "output_frame_count": total_frames,
        "copied_suffixes": sorted(groups.keys()),
    }
    (output_dir / "duplicate_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
