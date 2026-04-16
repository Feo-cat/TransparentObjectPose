#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import shutil
from pathlib import Path

import numpy as np


LINK_DIRS = [
    "frames",
    "masks_frame1based",
    "masks_visib_frame1based",
    "infer_external_mask",
]


def ensure_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src, dst)


def scale_pose_npz(src_path: Path, dst_path: Path, scale: float) -> None:
    with np.load(src_path, allow_pickle=False) as data:
        new_dict = {}
        for key in data.files:
            value = data[key]
            if key == "obj2cam_poses":
                pose = np.array(value, copy=True)
                pose[:3, 3] = pose[:3, 3] / scale
                new_dict[key] = pose
            elif key == "RT":
                rt = np.array(value, copy=True)
                if rt.shape[-2:] == (4, 4):
                    rt[:3, 3] = rt[:3, 3] / scale
                elif rt.shape[-2:] == (3, 4):
                    rt[:3, 3] = rt[:3, 3] / scale
                new_dict[key] = rt
            elif key == "distance":
                new_dict[key] = np.array(value, copy=True) / scale
            else:
                new_dict[key] = np.array(value, copy=True)
    np.savez(dst_path, **new_dict)


def maybe_scale_depth_dir(src_dir: Path, dst_dir: Path, scale: float) -> bool:
    if not src_dir.exists():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    scaled_any = False
    for src_path in sorted(src_dir.iterdir()):
        if src_path.is_dir():
            sub_dst = dst_dir / src_path.name
            sub_dst.mkdir(parents=True, exist_ok=True)
            for nested in sorted(src_path.iterdir()):
                if nested.suffix.lower() != ".npy":
                    shutil.copy2(nested, sub_dst / nested.name)
                    continue
                depth = np.load(nested).astype(np.float32) / scale
                np.save(sub_dst / nested.name, depth)
                scaled_any = True
        else:
            if src_path.suffix.lower() != ".npy":
                shutil.copy2(src_path, dst_dir / src_path.name)
                continue
            depth = np.load(src_path).astype(np.float32) / scale
            np.save(dst_dir / src_path.name, depth)
            scaled_any = True
    return scaled_any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument("--global-scale", type=float, default=20.0)
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    scale = float(args.global_scale)
    if scale <= 0:
        raise ValueError(f"--global-scale must be > 0, got {scale}")

    ensure_clean_dir(dst_dir)

    for rel_dir in LINK_DIRS:
        src = src_dir / rel_dir
        if src.exists():
            safe_symlink(src, dst_dir / rel_dir)

    src_pose_dir = src_dir / "poses_npz_frame1based"
    dst_pose_dir = dst_dir / "poses_npz_frame1based"
    dst_pose_dir.mkdir(parents=True, exist_ok=True)
    pose_count = 0
    for src_path in sorted(src_pose_dir.glob("*.npz")):
        scale_pose_npz(src_path, dst_pose_dir / src_path.name, scale)
        pose_count += 1

    depth_scaled = maybe_scale_depth_dir(src_dir / "depth", dst_dir / "depth", scale)

    src_meta_path = src_dir / "meta.json"
    if src_meta_path.exists():
        meta = json.loads(src_meta_path.read_text())
        meta["processed_dir"] = str(dst_dir)
        meta["poses_npz_dir"] = str(dst_pose_dir)
        if "source_metadata_example" in meta and isinstance(meta["source_metadata_example"], dict):
            example = dict(meta["source_metadata_example"])
            if "distance" in example:
                example["distance"] = float(example["distance"]) / scale
            meta["source_metadata_example"] = example
        meta["global_scale_applied_for_gt"] = scale
        meta["gt_coordinate_note"] = "obj2cam translation and depth values are divided by global_scale to match labsim training coordinates"
        if depth_scaled:
            meta["depth_dir"] = str(dst_dir / "depth")
        (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    summary = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "global_scale": scale,
        "pose_npz_count": pose_count,
        "depth_scaled": depth_scaled,
    }
    (dst_dir / "scale_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
