#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash infer_vid_ar.sh /path/to/video.mp4 /path/to/output_dir [auto_mask|external_bbox(unimplemented)] [bbox_json] [auto|abs_head] [mask_dir] [ckpt_step] [frame_manifest]
#   e.g. bash infer_vid_ar.sh test_repo/IMG_5525.mp4 test_repo/IMG_5525_ auto_mask "" abs_head "" 50999
#   e.g. bash infer_vid_ar.sh test_repo/IMG_5560.mp4 test_repo/IMG_5560 auto_mask "" abs_head "/mnt/afs/TransparentObjectPose/test_repo/IMG_5560/pred_sam2_20260328_060359/masks_frame1based" 50999
#   e.g. GDRN_SAVE_COOR_XYZ_NPY=1
#   e.g. bash infer_vid_ar.sh test_repo/IMG_5881.mp4 test_repo/IMG_5881_extmask auto_mask "" abs_head "/mnt/afs/TransparentObjectPose/test_repo/IMG_5881/pred_sam2_20260410_090643/masks_frame1based" 68199
# 
# Examples:
#   # external per-frame mask mode
#   bash infer_vid_ar.sh \
#     /mnt/afs/TransparentObjectPose/test_repo/test_vid.mp4 \
#     /mnt/afs/TransparentObjectPose/test_repo/test_vid_ar \
#     auto_mask \
#     "" \
#     auto \
#     /mnt/afs/TransparentObjectPose/test_repo/test_vid/masks_frame1based
#
#   # auto mask mode (no bbox json needed)
#   bash infer_vid_ar.sh \
#     /mnt/afs/TransparentObjectPose/test_repo/test_vid.mp4 \
#     /mnt/afs/TransparentObjectPose/test_repo/test_vid_ar_auto \
#     auto_mask

if [[ $# -lt 2 ]]; then
  echo "Usage: bash infer_vid_ar.sh <video_path> <output_dir> [auto_mask|external_bbox(unimplemented)] [bbox_json] [auto|abs_head] [mask_dir] [ckpt_step] [frame_manifest]"
  exit 1
fi

VIDEO_PATH="$1"
OUTPUT_DIR="$2"
ROI_MODE="${3:-auto_mask}"     # auto_mask | external_bbox (not implemented in inference_vid_ar.py)
BBOX_JSON="${4:-}"             # unused unless external_bbox becomes supported
POSE_SOURCE="${5:-auto}"   # auto | abs_head
MASK_DIR="${6:-}"          # optional external per-frame mask directory
CKPT_STEP="${7:-}"         # optional ckpt step, e.g. 91799 -> model_0091799.pth
FRAME_MANIFEST="${8:-}"    # optional JSON manifest for explicit multi-view multi-time inference

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# -----------------------------
# Project/model settings
# -----------------------------
CONFIG="$PROJECT_ROOT/configs/gdrn/labsim/a6_cPnP_lm13_ctx_prior.py"
DEFAULT_WEIGHTS_NAME="${GDRN_WEIGHTS_NAME:-model_0027599.pth}"
LOCAL_CKPT_DIR="${GDRN_LOCAL_CKPT_DIR:-/mnt/afs/TransparentObjectPose/test_repo}"
ADS_CLI_BIN="${GDRN_ADS_CLI_BIN:-/mnt/afs/afs/ads-cli}"
# OSS_CKPT_DIR="${GDRN_CKPT_OSS_DIR:-s3://019CE739817A7E43855626E278830CD7:019CE739817A7E35801972419E02787C@jiycoss.aoss.cn-sh-01b.sensecoreapi-oss.cn/gdr_ckpt/multiview_rel_ar_vid_depth/}"
OSS_CKPT_DIR="${GDRN_CKPT_OSS_DIR:-s3://019CE739817A7E43855626E278830CD7:019CE739817A7E35801972419E02787C@jiycoss.aoss.cn-sh-01g.sensecoreapi-oss.cn/gdr_ckpt/multiview_abs_ar_vid_depth_TEST_ctx_info_interaction/}"
WEIGHTS="${GDRN_WEIGHTS_PATH:-${LOCAL_CKPT_DIR}/${DEFAULT_WEIGHTS_NAME}}"
OBJ_CLS=1
DATASET_NAME="labsim_test"
DEVICE="${DEVICE:-cuda}"
CAM=(700 0 320 0 700 240 0 0 1)

# AR settings (aligned with training config TRAIN_NUM_CONTEXT_VIEWS / TRAIN_NUM_TARGET_VIEWS):
# - first chunk: all target views
# - following chunks: AR_CONTEXT_SIZE context + AR_TARGET_SIZE target
AR_CONTEXT_SIZE=3
AR_TARGET_SIZE=3
TARGET_MODE="autoregressive"
MASK_THR=0.5
MIN_MASK_PIXELS=16
TEMPORAL_SMOOTH="${TEMPORAL_SMOOTH:-0}"
TEMPORAL_SMOOTH_ROT_ALPHA="${TEMPORAL_SMOOTH_ROT_ALPHA:-0.35}"
TEMPORAL_SMOOTH_TRANS_ALPHA="${TEMPORAL_SMOOTH_TRANS_ALPHA:-0.30}"
ROI_TEMPORAL_SMOOTH="${ROI_TEMPORAL_SMOOTH:-1}"
ROI_TEMPORAL_SMOOTH_TYPE="${ROI_TEMPORAL_SMOOTH_TYPE:-adaptive_ema}"
ROI_TEMPORAL_SMOOTH_ALPHA="${ROI_TEMPORAL_SMOOTH_ALPHA:-0.40}"
ROI_TEMPORAL_SMOOTH_MIN_ALPHA="${ROI_TEMPORAL_SMOOTH_MIN_ALPHA:-0.15}"
ROI_TEMPORAL_SMOOTH_MAX_ALPHA="${ROI_TEMPORAL_SMOOTH_MAX_ALPHA:-0.70}"
EXTERNAL_MASK_PAD_SCALE="${EXTERNAL_MASK_PAD_SCALE:-}"
MASK_POSTPROC="${MASK_POSTPROC:-overlap_cc}"
MASK_PREV_DILATE_KERNEL="${MASK_PREV_DILATE_KERNEL:-11}"
MASK_PREV_GATE="${MASK_PREV_GATE:-1}"
MASK_POST_OPEN_KERNEL="${MASK_POST_OPEN_KERNEL:-0}"
MASK_POST_DILATE_KERNEL="${MASK_POST_DILATE_KERNEL:-7}"
MASK_POST_CLOSE_KERNEL="${MASK_POST_CLOSE_KERNEL:-0}"
MASK_FALLBACK_TO_PREV="${MASK_FALLBACK_TO_PREV:-1}"
SYMM_MODE="${SYMM_MODE:-none}"
SYMM_AXIS="${SYMM_AXIS:-0 0 1}"

# Frame extraction settings
# If FPS is not explicitly provided:
# - default to source video fps when external masks are used (to keep mask/frame index aligned)
# - otherwise keep legacy default 30
FPS="${FPS:-}"
FRAME_EXT=".png"

mkdir -p "$OUTPUT_DIR"
FRAME_DIR="$OUTPUT_DIR/frames"
RESULT_DIR="$OUTPUT_DIR/pred_ar"
mkdir -p "$FRAME_DIR" "$RESULT_DIR"

normalize_ckpt_name() {
  local raw_step="$1"
  if [[ "$raw_step" =~ ^model_[0-9]{7}\.pth$ ]]; then
    echo "$raw_step"
    return 0
  fi
  if [[ "$raw_step" =~ ^[0-9]+$ ]]; then
    printf "model_%07d.pth\n" "$((10#$raw_step))"
    return 0
  fi
  echo "Error: invalid ckpt_step=${raw_step}. Expected digits like 91799 or filename like model_0091799.pth" >&2
  exit 1
}

resolve_weights_from_ckpt_step() {
  local raw_step="$1"
  local ckpt_name
  local local_path
  local remote_path

  ckpt_name="$(normalize_ckpt_name "$raw_step")"
  mkdir -p "$LOCAL_CKPT_DIR"
  local_path="${LOCAL_CKPT_DIR}/${ckpt_name}"

  if [[ -f "$local_path" ]]; then
    echo "Using existing local checkpoint: $local_path"
    WEIGHTS="$local_path"
    return 0
  fi

  if [[ ! -x "$ADS_CLI_BIN" ]]; then
    echo "Error: ads-cli not found or not executable: $ADS_CLI_BIN" >&2
    exit 1
  fi

  remote_path="${OSS_CKPT_DIR%/}/${ckpt_name}"
  echo "Downloading checkpoint from OSS:"
  echo "  - Remote: $remote_path"
  echo "  - Local:  $local_path"
  "$ADS_CLI_BIN" --threads=32 cp "$remote_path" "$local_path"
  WEIGHTS="$local_path"
}

if [[ -n "$CKPT_STEP" ]]; then
  resolve_weights_from_ckpt_step "$CKPT_STEP"
fi

if [[ -z "$FPS" ]]; then
  if [[ -n "$MASK_DIR" ]]; then
    SRC_FPS="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$VIDEO_PATH" 2>/dev/null || true)"
    if [[ -n "$SRC_FPS" && "$SRC_FPS" != "0/0" ]]; then
      FPS="$SRC_FPS"
      echo "Auto-selected extraction FPS from source video (mask-aligned mode): $FPS"
    else
      FPS="30"
      echo "Warning: failed to detect source FPS; falling back to FPS=$FPS"
    fi
  else
    FPS="30"
  fi
fi

if [[ "$ROI_MODE" == "external_bbox" ]]; then
  echo "Error: ROI_MODE=external_bbox is not implemented for multiview AR inference. Use auto_mask and provide per-frame masks via mask_dir." >&2
  exit 1
fi

render_video_if_exists() {
  local input_pattern="$1"
  local output_path="$2"
  local sample_frame="$3"
  if [[ ! -f "$sample_frame" ]]; then
    return 0
  fi
  ffmpeg -y -framerate "$FPS" -i "$input_pattern" -c:v libx264 -pix_fmt yuv420p "$output_path"
}

# if end with .MOV, convert to mp4 and crop
if [[ "$VIDEO_PATH" == *.MOV ]]; then
  VIDEO_PATH_MP4="${VIDEO_PATH%.MOV}.mp4"
  ffmpeg -i "$VIDEO_PATH" -vf "crop='min(iw,ih*4/3)':'min(iw*3/4,ih)',scale=640:480" -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac "$VIDEO_PATH_MP4"
  VIDEO_PATH="$VIDEO_PATH_MP4"
fi

if [[ -n "$FRAME_MANIFEST" ]]; then
  if [[ ! -f "$FRAME_MANIFEST" ]]; then
    echo "Error: frame_manifest does not exist: $FRAME_MANIFEST" >&2
    exit 1
  fi
  echo "[1/3] frame_manifest mode enabled, skipping video frame extraction"
else
  echo "[1/3] Extracting frames from: $VIDEO_PATH"
  echo "Frame extraction FPS: $FPS"
  ffmpeg -y -i "$VIDEO_PATH" -vf "fps=${FPS}" "$FRAME_DIR/%06d.png"

  if [[ -n "$MASK_DIR" ]]; then
    frame_count="$(find "$FRAME_DIR" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d ' ')"
    mask_count="$(
      find "$MASK_DIR" -maxdepth 1 -type f \
        \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) \
        | wc -l | tr -d ' '
    )"
    echo "External-mask sanity check: extracted_frames=${frame_count}, mask_files=${mask_count}"
    if [[ "$frame_count" != "$mask_count" ]]; then
      echo "Warning: extracted frame count and mask file count differ."
      echo "         This often indicates FPS mismatch between mask generation and inference frame extraction."
      echo "         You can force FPS explicitly, e.g.: FPS=60 bash infer_vid_ar.sh ..."
    fi
  fi
fi

echo "[2/3] Running AR multiview inference (roi_mode=${ROI_MODE})"
echo "Using weights: $WEIGHTS"
CMD=(
  python "$PROJECT_ROOT/inference_vid_ar.py"
  --config "$CONFIG"
  --weights "$WEIGHTS"
  --image_dir "$FRAME_DIR"
  --obj_cls "$OBJ_CLS"
  --dataset_name "$DATASET_NAME"
  --device "$DEVICE"
  --cam "${CAM[@]}"
  --image_ext "$FRAME_EXT"
  --output_dir "$RESULT_DIR"
  --roi_mode "$ROI_MODE"
  --target_mode "$TARGET_MODE"
  --ar_context_size "$AR_CONTEXT_SIZE"
  --ar_target_size "$AR_TARGET_SIZE"
  --mask_thr "$MASK_THR"
  --min_mask_pixels "$MIN_MASK_PIXELS"
  --temporal_smooth_rot_alpha "$TEMPORAL_SMOOTH_ROT_ALPHA"
  --temporal_smooth_trans_alpha "$TEMPORAL_SMOOTH_TRANS_ALPHA"
  --roi_temporal_smooth_type "$ROI_TEMPORAL_SMOOTH_TYPE"
  --roi_temporal_smooth_alpha "$ROI_TEMPORAL_SMOOTH_ALPHA"
  --roi_temporal_smooth_min_alpha "$ROI_TEMPORAL_SMOOTH_MIN_ALPHA"
  --roi_temporal_smooth_max_alpha "$ROI_TEMPORAL_SMOOTH_MAX_ALPHA"
  --mask_postproc "$MASK_POSTPROC"
  --mask_prev_dilate_kernel "$MASK_PREV_DILATE_KERNEL"
  --mask_post_open_kernel "$MASK_POST_OPEN_KERNEL"
  --mask_post_dilate_kernel "$MASK_POST_DILATE_KERNEL"
  --mask_post_close_kernel "$MASK_POST_CLOSE_KERNEL"
  --save_vis
  --symm_mode "$SYMM_MODE"
  --pose_source "$POSE_SOURCE"
)

if [[ -n "$FRAME_MANIFEST" ]]; then
  CMD+=(--frame_manifest "$FRAME_MANIFEST")
fi

if [[ "$TEMPORAL_SMOOTH" == "1" ]]; then
  CMD+=(--temporal_smooth)
fi

if [[ "$ROI_TEMPORAL_SMOOTH" == "1" ]]; then
  CMD+=(--roi_temporal_smooth)
fi

if [[ "${GDRN_SAVE_PNP_DECODE_DEBUG:-0}" == "1" ]]; then
  CMD+=(--save_pnp_decode_debug)
fi

if [[ "${GDRN_SAVE_PNP_INPUT_DEBUG:-0}" == "1" ]]; then
  CMD+=(--save_pnp_input_debug)
fi

if [[ "$MASK_FALLBACK_TO_PREV" == "1" ]]; then
  CMD+=(--mask_fallback_to_prev)
fi

if [[ "$MASK_PREV_GATE" == "1" ]]; then
  CMD+=(--mask_prev_gate)
fi

if [[ "$SYMM_MODE" == "continuous" ]]; then
  read -r -a SYMM_AXIS_ARR <<< "$SYMM_AXIS"
  CMD+=(--symm_axis "${SYMM_AXIS_ARR[@]}")
fi

if [[ "$ROI_MODE" == "external_bbox" ]]; then
  if [[ -z "$BBOX_JSON" ]]; then
    echo "Error: bbox_json is required when ROI_MODE=external_bbox"
    exit 1
  fi
  CMD+=(--bbox_json "$BBOX_JSON")
fi

if [[ -n "$MASK_DIR" ]]; then
  CMD+=(--mask_dir "$MASK_DIR")
fi

if [[ -n "$EXTERNAL_MASK_PAD_SCALE" ]]; then
  CMD+=(--external_mask_pad_scale "$EXTERNAL_MASK_PAD_SCALE")
fi

"${CMD[@]}"

if [[ -n "$FRAME_MANIFEST" ]]; then
  echo "[3/3] frame_manifest mode: skip sequential video rendering"
else
  echo "[3/3] Rendering result videos"
  render_video_if_exists "$RESULT_DIR/%06d_pred.png" "$RESULT_DIR/output.mp4" "$RESULT_DIR/000001_pred.png"
  render_video_if_exists "$RESULT_DIR/depth/%06d_depth.png" "$RESULT_DIR/depth/output_depth.mp4" "$RESULT_DIR/depth/000001_depth.png"
  render_video_if_exists "$RESULT_DIR/mask/%06d_mask.png" "$RESULT_DIR/mask/output_mask.mp4" "$RESULT_DIR/mask/000001_mask.png"
  render_video_if_exists "$RESULT_DIR/mask_raw/%06d_mask_raw.png" "$RESULT_DIR/mask_raw/output_mask_raw.mp4" "$RESULT_DIR/mask_raw/000001_mask_raw.png"
fi

echo "Done. Results:"
echo "  - Frames:  $FRAME_DIR"
echo "  - Predict: $RESULT_DIR"
echo "  - JSON:    $RESULT_DIR/results.json"
echo "  - Video:   $RESULT_DIR/output.mp4"
