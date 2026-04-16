"""
Pose visualization utilities for training/inference (videos, single frames).
"""
import logging
import shutil
import os.path as osp
import subprocess

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _get_bbox_visualizer():
    """Lazy import to avoid requiring tools path at module load time."""
    import sys
    tools_path = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from visualize_3d_bbox import BBox3DVisualizer
    return BBox3DVisualizer


def _write_video_frames(frames, out_path, fps):
    """Prefer H.264 output for editor/browser compatibility; fall back to OpenCV codecs."""
    if not frames:
        return False

    h, w = frames[0].shape[:2]
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            for frame in frames:
                frame_u8 = np.ascontiguousarray(frame, dtype=np.uint8)
                proc.stdin.write(frame_u8.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="ignore")
            if proc.wait() == 0:
                return True
            logger.warning("ffmpeg video export failed for %s: %s", out_path, stderr.strip()[-1000:])
        except Exception as exc:
            logger.warning("ffmpeg video export raised for %s: %s", out_path, exc)
        finally:
            if proc is not None:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
                if proc.stderr is not None:
                    proc.stderr.close()

    for fourcc_name in ["avc1", "mp4v"]:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc_name), fps, (w, h))
        if not writer.isOpened():
            continue
        try:
            for frame in frames:
                writer.write(np.ascontiguousarray(frame, dtype=np.uint8))
        finally:
            writer.release()
        return True

    logger.warning("Failed to export video %s with both ffmpeg and OpenCV backends.", out_path)
    return False


def render_pose_vis_frame(batch, view_i, size, R, T, label=""):
    """Render a single training/inference frame with the 3D bbox overlay."""
    if view_i >= batch["input_images"].shape[1]:
        raise IndexError(f"view_i={view_i} out of range for input_images with shape {batch['input_images'].shape}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    img = batch["input_images"][0, view_i].detach().cpu().numpy()
    img = (img.transpose(1, 2, 0) * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if view_i >= batch["roi_cam"].shape[1]:
        return img

    K = batch["roi_cam"][0, view_i].detach().cpu().numpy().astype(np.float32)
    R = np.asarray(R, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).ravel()
    size = np.asarray(size, dtype=np.float64)
    BBox3DVisualizer = _get_bbox_visualizer()
    try:
        vis = BBox3DVisualizer(K)
        img = vis.draw_from_size(img.copy(), size, R, T, thickness=2)
    except Exception:
        pass
    if label:
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def write_pose_vis_videos(
    batch,
    out_dict,
    target_idx,
    data_ref,
    obj_cls,
    size,
    iteration,
    vis_output_dir,
    fps=5,
    output_prefix="pred_pose",
    pred_label="pred",
    drop_info=None,
):
    """Write two pose-visualization videos: target views only (pred), and all views (pred on target, GT on context)."""
    if "rot" not in out_dict or "trans" not in out_dict:
        return
    if target_idx is None:
        return
    if isinstance(target_idx, (list, tuple)):
        target_view_ids = [int(v) for v in target_idx]
    elif isinstance(target_idx, np.ndarray):
        target_view_ids = [int(v) for v in target_idx.reshape(-1).tolist()]
    elif torch.is_tensor(target_idx):
        target_view_ids = [int(v) for v in target_idx.reshape(-1).tolist()]
    else:
        target_view_ids = [int(target_idx)]
    N_views = batch["input_images"].shape[1]
    if N_views == 0:
        return
    pred_rot = out_dict["rot"][0].detach().cpu().numpy()  # (T, 3, 3) or (3, 3)
    pred_trans = out_dict["trans"][0].detach().cpu().numpy()  # (T, 3) or (3,)
    if pred_rot.ndim == 2:
        pred_rot = pred_rot[np.newaxis, ...]
        pred_trans = pred_trans[np.newaxis, ...]
    T_pred = pred_rot.shape[0]
    if T_pred != len(target_view_ids):
        logger.warning(
            "write_pose_vis_videos: pred pose count %s != target_view count %s, skipping (training should assert in GDRN.forward).",
            T_pred,
            len(target_view_ids),
        )
        return
    target_set = set(target_view_ids)
    size = np.asarray(size, dtype=np.float64)

    frames_all = []
    for view_i in range(N_views):
        if view_i in target_set:
            out_i = target_view_ids.index(view_i)
            R = pred_rot[out_i]
            T = pred_trans[out_i]
            frame_label = pred_label
            if drop_info is not None:
                kc = int(drop_info["keep_coarse"][out_i].item()) if out_i < len(drop_info["keep_coarse"]) else 1
                kr = int(drop_info["keep_rel"][out_i].item()) if out_i < len(drop_info["keep_rel"]) else 1
                km = int(drop_info["keep_motion"][out_i].item()) if out_i < len(drop_info["keep_motion"]) else 1
                frame_label = f"{pred_label} c{kc}r{kr}m{km}"
            frames_all.append(render_pose_vis_frame(batch, view_i, size, R, T, frame_label))
        else:
            R = batch["ego_rot"][0, view_i].detach().cpu().numpy()
            T = batch["trans"][0, view_i].detach().cpu().numpy()
            if R.ndim == 3:
                R = R[0]
            if T.ndim > 1:
                T = T.ravel()
            frames_all.append(render_pose_vis_frame(batch, view_i, size, R, T, "gt"))
    if not frames_all:
        return
    out_path = osp.join(vis_output_dir, f"{output_prefix}_all_views_{iteration:06d}.mp4")
    if _write_video_frames(frames_all, out_path, fps):
        logger.info("Wrote pose vis video: %s", out_path)
