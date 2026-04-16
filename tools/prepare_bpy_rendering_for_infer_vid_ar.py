#!/usr/bin/env python3
"""Prepare Blender-rendered frame folders for direct image-folder AR inference.

Input scene folders are expected to look like:

  000.png
  000_mask.png
  000_mask_visib.png
  000.npz

For every scene folder this script creates a sibling folder named
<scene>_processed containing:

  frames/000001.png ...
  masks_frame1based/000001.png ...
  masks_visib_frame1based/000001.png ...
  meta.json
  run_infer.sh
  run_infer_auto_mask.sh
  run_infer_external_mask.sh

The prepared frames are converted to 3-channel RGB PNGs so they can be passed
directly to inference without first encoding a video.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


RGB_FRAME_RE = re.compile(r"^(\d+)\.png$")
MASK_FRAME_RE = re.compile(r"^(\d+)_mask\.png$")
MASK_VISIB_FRAME_RE = re.compile(r"^(\d+)_mask_visib\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/mnt/afs/bpy_rendering"),
        help="Root directory that contains rendered scene subfolders.",
    )
    parser.add_argument(
        "--processed-suffix",
        type=str,
        default="_processed",
        help="Suffix appended to each scene directory name for prepared output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed folders.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Optional scene folder name to prepare. Can be passed multiple times.",
    )
    return parser.parse_args()


def find_scene_dirs(input_root: Path, scene_names: list[str] | None, processed_suffix: str) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    scene_dirs = sorted(
        [
            path
            for path in input_root.iterdir()
            if path.is_dir() and not path.name.endswith(processed_suffix)
        ]
    )
    if scene_names:
        wanted = set(scene_names)
        scene_dirs = [path for path in scene_dirs if path.name in wanted]
        missing = sorted(wanted - {path.name for path in scene_dirs})
        if missing:
            raise FileNotFoundError(f"Scene folders not found: {missing}")
    if not scene_dirs:
        raise ValueError(f"No source scene folders found under {input_root}")
    return scene_dirs


def sorted_matching_files(scene_dir: Path, pattern: re.Pattern[str]) -> list[Path]:
    matched: list[tuple[int, Path]] = []
    for path in scene_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        matched.append((int(match.group(1)), path))
    return [path for _, path in sorted(matched, key=lambda item: item[0])]


def assert_consecutive_indices(paths: Iterable[Path], pattern: re.Pattern[str], label: str) -> list[int]:
    indices: list[int] = []
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"{label} file does not match expected pattern: {path}")
        indices.append(int(match.group(1)))

    if not indices:
        raise ValueError(f"No {label} files found")

    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError(
            f"{label} indices are not consecutive in {paths[0].parent}: "
            f"got {indices[:5]}... expected {expected[:5]}..."
        )
    return indices


def load_scene_metadata(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        meta: dict[str, object] = {
            "camera_matrix": data["K"].astype(float).tolist() if "K" in data else None,
            "table": data["table"].tolist() if "table" in data else None,
            "objects": data["objects"].tolist() if "objects" in data else None,
        }
        if "azimuth" in data:
            meta["azimuth"] = float(data["azimuth"])
        if "elevation" in data:
            meta["elevation"] = float(data["elevation"])
        if "distance" in data:
            meta["distance"] = float(data["distance"])
    return meta


def verify_intrinsics(npz_files: list[Path]) -> list[list[float]] | None:
    k_ref: np.ndarray | None = None
    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=False) as data:
            if "K" not in data:
                continue
            k_cur = np.asarray(data["K"], dtype=np.float64)
        if k_ref is None:
            k_ref = k_cur
            continue
        if not np.allclose(k_ref, k_cur):
            raise ValueError(f"Inconsistent intrinsics found in scene: {npz_path.parent}")
    return None if k_ref is None else k_ref.astype(float).tolist()


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path} (pass --overwrite to replace)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_rgb_frames_as_three_channel(rgb_files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, src_path in enumerate(rgb_files, start=1):
        dst_path = dst_dir / f"{frame_idx:06d}.png"
        with Image.open(src_path) as image:
            rgb = image.convert("RGB")
            rgb.save(dst_path)


def copy_files_frame1based(files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, src_path in enumerate(files, start=1):
        dst_path = dst_dir / f"{frame_idx:06d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst_path)


def flatten_camera_matrix(camera_matrix: list[list[float]] | None) -> list[float]:
    if camera_matrix is None:
        return [700.0, 0.0, 320.0, 0.0, 700.0, 240.0, 0.0, 0.0, 1.0]
    flat = np.asarray(camera_matrix, dtype=np.float64).reshape(-1).tolist()
    if len(flat) != 9:
        raise ValueError(f"Expected 3x3 camera matrix, got: {camera_matrix}")
    return [float(v) for v in flat]


def write_text_executable(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_scene_run_scripts(
    scene_output_dir: Path,
    repo_root: Path,
    cam_values: list[float],
    preferred_mask_subdir: str | None,
) -> None:
    project_root_q = shlex.quote(str(repo_root))
    scene_dir_q = shlex.quote(str(scene_output_dir))
    cam_values_str = " ".join(f"{value:.10g}" for value in cam_values)

    auto_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"PROJECT_ROOT={project_root_q}",
        f"SCENE_DIR={scene_dir_q}",
        f'export CAM_VALUES="{cam_values_str}"',
        'DEFAULT_OUTPUT_DIR="$SCENE_DIR/infer_auto_mask"',
        'ARG1="${1:-}"',
        'ARG2="${2:-}"',
        'OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"',
        'CKPT_STEP=""',
        'if [[ -n "$ARG2" ]]; then',
        '  OUTPUT_DIR="$ARG1"',
        '  CKPT_STEP="$ARG2"',
        'elif [[ -n "$ARG1" ]]; then',
        '  if [[ "$ARG1" =~ ^[0-9]+$ || "$ARG1" =~ ^model_[0-9]{7}\\.pth$ ]]; then',
        '    CKPT_STEP="$ARG1"',
        '  else',
        '    OUTPUT_DIR="$ARG1"',
        '  fi',
        'fi',
        "",
        'bash "$PROJECT_ROOT/infer_vid_ar_frames.sh" \\',
        '  "$SCENE_DIR/frames" \\',
        '  "$OUTPUT_DIR" \\',
        '  auto_mask "" auto "" "$CKPT_STEP" .png',
    ]
    write_text_executable(scene_output_dir / "run_infer_auto_mask.sh", auto_lines)

    if preferred_mask_subdir is not None:
        external_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"PROJECT_ROOT={project_root_q}",
            f"SCENE_DIR={scene_dir_q}",
            f'export CAM_VALUES="{cam_values_str}"',
            'DEFAULT_OUTPUT_DIR="$SCENE_DIR/infer_external_mask"',
            'ARG1="${1:-}"',
            'ARG2="${2:-}"',
            'OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"',
            'CKPT_STEP=""',
            'if [[ -n "$ARG2" ]]; then',
            '  OUTPUT_DIR="$ARG1"',
            '  CKPT_STEP="$ARG2"',
            'elif [[ -n "$ARG1" ]]; then',
            '  if [[ "$ARG1" =~ ^[0-9]+$ || "$ARG1" =~ ^model_[0-9]{7}\\.pth$ ]]; then',
            '    CKPT_STEP="$ARG1"',
            '  else',
            '    OUTPUT_DIR="$ARG1"',
            '  fi',
            'fi',
            "",
            'bash "$PROJECT_ROOT/infer_vid_ar_frames.sh" \\',
            '  "$SCENE_DIR/frames" \\',
            '  "$OUTPUT_DIR" \\',
            f'  auto_mask "" auto "$SCENE_DIR/{preferred_mask_subdir}" "$CKPT_STEP" .png',
        ]
        write_text_executable(scene_output_dir / "run_infer_external_mask.sh", external_lines)
        default_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f'bash {shlex.quote(str(scene_output_dir / "run_infer_external_mask.sh"))} "$@"',
        ]
    else:
        default_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f'bash {shlex.quote(str(scene_output_dir / "run_infer_auto_mask.sh"))} "$@"',
        ]
    write_text_executable(scene_output_dir / "run_infer.sh", default_lines)


def prepare_scene(
    scene_dir: Path,
    repo_root: Path,
    processed_suffix: str,
    overwrite: bool,
) -> dict:
    rgb_files = sorted_matching_files(scene_dir, RGB_FRAME_RE)
    mask_files = sorted_matching_files(scene_dir, MASK_FRAME_RE)
    mask_visib_files = sorted_matching_files(scene_dir, MASK_VISIB_FRAME_RE)
    npz_files = sorted(scene_dir.glob("*.npz"))

    rgb_indices = assert_consecutive_indices(rgb_files, RGB_FRAME_RE, "RGB")
    if mask_files:
        mask_indices = assert_consecutive_indices(mask_files, MASK_FRAME_RE, "mask")
        if mask_indices != rgb_indices:
            raise ValueError(f"Mask indices do not match RGB indices in {scene_dir}")
    if mask_visib_files:
        mask_visib_indices = assert_consecutive_indices(mask_visib_files, MASK_VISIB_FRAME_RE, "mask_visib")
        if mask_visib_indices != rgb_indices:
            raise ValueError(f"Visible mask indices do not match RGB indices in {scene_dir}")
    if npz_files and len(npz_files) != len(rgb_files):
        raise ValueError(f"NPZ count does not match RGB count in {scene_dir}")

    scene_output_dir = scene_dir.parent / f"{scene_dir.name}{processed_suffix}"
    ensure_clean_dir(scene_output_dir, overwrite=overwrite)

    frames_dir = scene_output_dir / "frames"
    copy_rgb_frames_as_three_channel(rgb_files, frames_dir)
    if mask_files:
        copy_files_frame1based(mask_files, scene_output_dir / "masks_frame1based")
    if mask_visib_files:
        copy_files_frame1based(mask_visib_files, scene_output_dir / "masks_visib_frame1based")
    if npz_files:
        copy_files_frame1based(npz_files, scene_output_dir / "poses_npz_frame1based")

    camera_matrix = verify_intrinsics(npz_files) if npz_files else None
    scene_meta = {
        "scene_name": scene_dir.name,
        "source_dir": str(scene_dir),
        "processed_dir": str(scene_output_dir),
        "frames_dir": str(frames_dir),
        "frame_count": len(rgb_files),
        "source_frame_index_start": rgb_indices[0],
        "source_frame_index_end": rgb_indices[-1],
        "camera_matrix": camera_matrix,
        "camera_row_major": flatten_camera_matrix(camera_matrix),
    }
    if npz_files:
        scene_meta["source_metadata_example"] = load_scene_metadata(npz_files[0])
        scene_meta["npz_count"] = len(npz_files)
        scene_meta["poses_npz_dir"] = str(scene_output_dir / "poses_npz_frame1based")
    if mask_files:
        scene_meta["mask_dir"] = str(scene_output_dir / "masks_frame1based")
    if mask_visib_files:
        scene_meta["mask_visib_dir"] = str(scene_output_dir / "masks_visib_frame1based")
    preferred_mask_subdir = None
    if mask_visib_files:
        preferred_mask_subdir = "masks_visib_frame1based"
        scene_meta["preferred_external_mask_dir"] = str(scene_output_dir / preferred_mask_subdir)
    elif mask_files:
        preferred_mask_subdir = "masks_frame1based"
        scene_meta["preferred_external_mask_dir"] = str(scene_output_dir / preferred_mask_subdir)

    meta_path = scene_output_dir / "meta.json"
    meta_path.write_text(json.dumps(scene_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_scene_run_scripts(
        scene_output_dir=scene_output_dir,
        repo_root=repo_root,
        cam_values=scene_meta["camera_row_major"],
        preferred_mask_subdir=preferred_mask_subdir,
    )
    return scene_meta


def write_root_run_script(input_root: Path, scene_metas: list[dict], script_name: str, scene_script_name: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'CKPT_STEP="${1:-}"',
        "",
    ]
    for meta in scene_metas:
        scene_name = meta["scene_name"]
        processed_dir = Path(meta["processed_dir"])
        lines.extend(
            [
                f'echo "[run] {scene_name}"',
                f'bash {shlex.quote(str(processed_dir / scene_script_name))} "" "$CKPT_STEP"',
                "",
            ]
        )
    write_text_executable(input_root / script_name, lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    scene_dirs = find_scene_dirs(
        input_root=args.input_root,
        scene_names=args.scene,
        processed_suffix=args.processed_suffix,
    )

    scene_metas = []
    for scene_dir in scene_dirs:
        scene_meta = prepare_scene(
            scene_dir=scene_dir,
            repo_root=repo_root,
            processed_suffix=args.processed_suffix,
            overwrite=args.overwrite,
        )
        scene_metas.append(scene_meta)
        print(f"[prepared] {scene_dir.name} -> {scene_meta['processed_dir']}")

    manifest = {
        "input_root": str(args.input_root),
        "processed_suffix": args.processed_suffix,
        "scene_count": len(scene_metas),
        "scenes": scene_metas,
    }
    manifest_path = args.input_root / "processed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_root_run_script(
        input_root=args.input_root,
        scene_metas=scene_metas,
        script_name="run_infer_vid_ar_all_processed.sh",
        scene_script_name="run_infer.sh",
    )
    write_root_run_script(
        input_root=args.input_root,
        scene_metas=scene_metas,
        script_name="run_infer_vid_ar_all_processed_auto_mask.sh",
        scene_script_name="run_infer_auto_mask.sh",
    )
    if any("mask_dir" in item for item in scene_metas):
        write_root_run_script(
            input_root=args.input_root,
            scene_metas=[item for item in scene_metas if "mask_dir" in item],
            script_name="run_infer_vid_ar_all_processed_external_masks.sh",
            scene_script_name="run_infer_external_mask.sh",
        )
    print(f"[done] Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
