#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash infer_vid_ar_frames.sh <image_dir> <output_dir> [auto_mask|external_bbox(unimplemented)] [bbox_json] [auto|abs_head] [mask_dir] [ckpt_step] [image_ext]
#
# Example:
#   bash infer_vid_ar_frames.sh /path/to/frames /path/to/output auto_mask "" auto /path/to/masks "" .png

if [[ $# -lt 2 ]]; then
  echo "Usage: bash infer_vid_ar_frames.sh <image_dir> <output_dir> [auto_mask|external_bbox(unimplemented)] [bbox_json] [auto|abs_head] [mask_dir] [ckpt_step] [image_ext]"
  exit 1
fi

IMAGE_DIR="$1"
OUTPUT_DIR="$2"
ROI_MODE="${3:-auto_mask}"     # auto_mask | external_bbox (not implemented in inference_vid_ar.py)
BBOX_JSON="${4:-}"             # unused unless external_bbox becomes supported
POSE_SOURCE="${5:-auto}"       # auto | abs_head
MASK_DIR="${6:-}"              # optional external per-frame mask directory
CKPT_STEP="${7:-}"             # optional ckpt step, e.g. 91799 -> model_0091799.pth
IMAGE_EXT="${8:-.png}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# -----------------------------
# Project/model settings
# -----------------------------
CONFIG="$PROJECT_ROOT/configs/gdrn/labsim/a6_cPnP_lm13_ctx_prior.py"
DEFAULT_WEIGHTS_NAME="${GDRN_WEIGHTS_NAME:-model_0081599.pth}"
LOCAL_CKPT_DIR="${GDRN_LOCAL_CKPT_DIR:-/mnt/afs/TransparentObjectPose/test_repo}"
ADS_CLI_BIN="${GDRN_ADS_CLI_BIN:-/mnt/afs/afs/ads-cli}"
# OSS_CKPT_DIR="${GDRN_CKPT_OSS_DIR:-s3://019CA7860AC073D39601CF4E30339D2C:019CA7860AC073BA9804E6B73B9DC91F@rencwoss.aoss.cn-sh-01b.sensecoreapi-oss.cn/gdr_ckpt/multiview_rel_ar_vid_depth/}"
OSS_CKPT_DIR="${GDRN_CKPT_OSS_DIR:-s3://019CA7860AC073D39601CF4E30339D2C:019CA7860AC073BA9804E6B73B9DC91F@rencwoss.aoss.cn-sh-01b.sensecoreapi-oss.cn/gdr_ckpt/multiview_abs_ar_vid_depth_TEST_ctx_info_interaction/}"
WEIGHTS="${GDRN_WEIGHTS_PATH:-${LOCAL_CKPT_DIR}/${DEFAULT_WEIGHTS_NAME}}"
OBJ_CLS=1
DATASET_NAME="labsim_test"
DEVICE="${DEVICE:-cuda}"

if [[ -n "${CAM_VALUES:-}" ]]; then
  read -r -a CAM <<< "${CAM_VALUES}"
else
  CAM=(700 0 320 0 700 240 0 0 1)
fi

# AR settings
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
RENDER_RESULT_VIDEOS="${RENDER_RESULT_VIDEOS:-1}"
FPS="${FPS:-30}"

RESULT_DIR="$OUTPUT_DIR/pred_ar"
mkdir -p "$RESULT_DIR"

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

render_video_if_exists() {
  local input_pattern="$1"
  local output_path="$2"
  local sample_frame="$3"
  if [[ ! -f "$sample_frame" ]]; then
    return 0
  fi
  ffmpeg -y -framerate "$FPS" -i "$input_pattern" -c:v libx264 -pix_fmt yuv420p "$output_path"
}

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Error: image_dir does not exist: $IMAGE_DIR" >&2
  exit 1
fi

if [[ -n "$CKPT_STEP" ]]; then
  resolve_weights_from_ckpt_step "$CKPT_STEP"
fi

if [[ "$ROI_MODE" == "external_bbox" ]]; then
  echo "Error: ROI_MODE=external_bbox is not implemented for multiview AR inference. Use auto_mask and provide per-frame masks via mask_dir." >&2
  exit 1
fi

echo "[1/2] Running AR multiview inference on frames: $IMAGE_DIR"
echo "Using weights: $WEIGHTS"
CMD=(
  python "$PROJECT_ROOT/inference_vid_ar.py"
  --config "$CONFIG"
  --weights "$WEIGHTS"
  --image_dir "$IMAGE_DIR"
  --obj_cls "$OBJ_CLS"
  --dataset_name "$DATASET_NAME"
  --device "$DEVICE"
  --cam "${CAM[@]}"
  --image_ext "$IMAGE_EXT"
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

if [[ "$TEMPORAL_SMOOTH" == "1" ]]; then
  CMD+=(--temporal_smooth)
fi

if [[ "$ROI_TEMPORAL_SMOOTH" == "1" ]]; then
  CMD+=(--roi_temporal_smooth)
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

if [[ "$RENDER_RESULT_VIDEOS" == "1" ]]; then
  echo "[2/2] Rendering result videos"
  render_video_if_exists "$RESULT_DIR/%06d_pred.png" "$RESULT_DIR/output.mp4" "$RESULT_DIR/000001_pred.png"
  render_video_if_exists "$RESULT_DIR/depth/%06d_depth.png" "$RESULT_DIR/depth/output_depth.mp4" "$RESULT_DIR/depth/000001_depth.png"
  render_video_if_exists "$RESULT_DIR/mask/%06d_mask.png" "$RESULT_DIR/mask/output_mask.mp4" "$RESULT_DIR/mask/000001_mask.png"
  render_video_if_exists "$RESULT_DIR/mask_raw/%06d_mask_raw.png" "$RESULT_DIR/mask_raw/output_mask_raw.mp4" "$RESULT_DIR/mask_raw/000001_mask_raw.png"
fi

echo "Done. Results:"
echo "  - Frames:  $IMAGE_DIR"
echo "  - Predict: $RESULT_DIR"
echo "  - JSON:    $RESULT_DIR/results.json"
if [[ "$RENDER_RESULT_VIDEOS" == "1" ]]; then
  echo "  - Video:   $RESULT_DIR/output.mp4"
fi
