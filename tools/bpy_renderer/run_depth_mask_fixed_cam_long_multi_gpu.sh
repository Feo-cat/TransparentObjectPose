#!/usr/bin/env bash
set -euo pipefail

BLENDER_BIN="${BLENDER_BIN:-/home/renchengwei/blender-4.4.3-linux-x64/blender}"
PY_SCRIPT="${PY_SCRIPT:-depth_mask_fixed_cam_long.py}"

OBJECT_PATH="${OBJECT_PATH:-glb_objs}"
PLANE_PATH="${PLANE_PATH:-glb_planes}"
OUTPUT_DIR="${OUTPUT_DIR:-./views_depth_mask_fixed_cam_long_multi}"
HDRS_DIR="${HDRS_DIR:-/home/renchengwei/bpy_render/hdri_bgs}"

GPU_IDS="${GPU_IDS:-0,1}"
NUM_IMAGES="${NUM_IMAGES:-240}"
RES_W="${RES_W:-640}"
RES_H="${RES_H:-480}"
CAMERA_DIST_MIN="${CAMERA_DIST_MIN:-16}"
CAMERA_DIST_MAX="${CAMERA_DIST_MAX:-32}"
ELEVATION_MIN="${ELEVATION_MIN:-10}"
ELEVATION_MAX="${ELEVATION_MAX:-45}"
DEVICE="${DEVICE:-CUDA}"
SEED="${SEED:--1}"
ENABLE_OTHER_OBJECTS="${ENABLE_OTHER_OBJECTS:-0}"

MOTION_SEGMENTS_MIN="${MOTION_SEGMENTS_MIN:-10}"
MOTION_SEGMENTS_MAX="${MOTION_SEGMENTS_MAX:-24}"
MOTION_SEGMENT_LEN_MIN="${MOTION_SEGMENT_LEN_MIN:-12}"
MOTION_SEGMENT_LEN_MAX="${MOTION_SEGMENT_LEN_MAX:-18}"
MOTION_COLLISION_SAMPLES="${MOTION_COLLISION_SAMPLES:-7}"
MOTION_POSE_ATTEMPTS="${MOTION_POSE_ATTEMPTS:-140}"
MOTION_XY_STEP_SCALE_MIN="${MOTION_XY_STEP_SCALE_MIN:-0.08}"
MOTION_XY_STEP_SCALE_MAX="${MOTION_XY_STEP_SCALE_MAX:-0.22}"
MOTION_SPIN_DEG_MAX="${MOTION_SPIN_DEG_MAX:-36}"
MOTION_TABLE_MIN_SEGMENT_DIST_SCALE="${MOTION_TABLE_MIN_SEGMENT_DIST_SCALE:-0.08}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PLAN_PATH="${PLAN_PATH:-${OUTPUT_DIR}/scene_plan_${RUN_ID}.json}"
PLAN_VALIDATION_MODE="${PLAN_VALIDATION_MODE:-sparse}"
PLAN_VALIDATION_MAX_SAMPLES="${PLAN_VALIDATION_MAX_SAMPLES:-24}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"

if [[ -z "${GPU_IDS//,/}" || "${GPU_COUNT}" -eq 0 ]]; then
    echo "No GPU ids provided. Set GPU_IDS, for example GPU_IDS=0,1,3"
    exit 1
fi

PLAN_GPU="${PLAN_GPU:-${GPU_ARRAY[0]}}"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --object_path "${OBJECT_PATH}"
    --plane_path "${PLANE_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --engine CYCLES
    --num_images "${NUM_IMAGES}"
    --camera_type fixed
    --camera_dist_min "${CAMERA_DIST_MIN}"
    --camera_dist_max "${CAMERA_DIST_MAX}"
    --elevation_min "${ELEVATION_MIN}"
    --elevation_max "${ELEVATION_MAX}"
    --enable_other_objects "${ENABLE_OTHER_OBJECTS}"
    --res_w "${RES_W}"
    --res_h "${RES_H}"
    --hdrs_dir "${HDRS_DIR}"
    --device "${DEVICE}"
    --plan_validation_mode "${PLAN_VALIDATION_MODE}"
    --plan_validation_max_samples "${PLAN_VALIDATION_MAX_SAMPLES}"
    --motion_segments_min "${MOTION_SEGMENTS_MIN}"
    --motion_segments_max "${MOTION_SEGMENTS_MAX}"
    --motion_segment_len_min "${MOTION_SEGMENT_LEN_MIN}"
    --motion_segment_len_max "${MOTION_SEGMENT_LEN_MAX}"
    --motion_collision_samples "${MOTION_COLLISION_SAMPLES}"
    --motion_pose_attempts "${MOTION_POSE_ATTEMPTS}"
    --motion_xy_step_scale_min "${MOTION_XY_STEP_SCALE_MIN}"
    --motion_xy_step_scale_max "${MOTION_XY_STEP_SCALE_MAX}"
    --motion_spin_deg_max "${MOTION_SPIN_DEG_MAX}"
    --motion_table_min_segment_dist_scale "${MOTION_TABLE_MIN_SEGMENT_DIST_SCALE}"
    --silent_mode
)

if [[ "${SEED}" != "-1" ]]; then
    COMMON_ARGS+=(--seed "${SEED}")
fi

echo "[1/2] Generating validated scene plan at ${PLAN_PATH}"
echo "     planning on GPU ${PLAN_GPU}"
CUDA_VISIBLE_DEVICES="${PLAN_GPU}" \
"${BLENDER_BIN}" -b -P "${PY_SCRIPT}" -- \
    "${COMMON_ARGS[@]}" \
    --mode plan \
    --plan_path "${PLAN_PATH}" \
    --worker_id planner

BASE_FRAMES=$(( NUM_IMAGES / GPU_COUNT ))
EXTRA_FRAMES=$(( NUM_IMAGES % GPU_COUNT ))

echo "[2/2] Launching ${GPU_COUNT} render workers across GPUs: ${GPU_IDS}"

PIDS=()
RANGES=()
START_FRAME=0

for IDX in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$IDX]}"
    FRAME_COUNT="${BASE_FRAMES}"
    if [[ "${IDX}" -lt "${EXTRA_FRAMES}" ]]; then
        FRAME_COUNT=$(( FRAME_COUNT + 1 ))
    fi

    if [[ "${FRAME_COUNT}" -le 0 ]]; then
        continue
    fi

    END_FRAME=$(( START_FRAME + FRAME_COUNT - 1 ))
    WORKER_ID="gpu${GPU_ID}"

    echo "  - GPU ${GPU_ID}: frames ${START_FRAME}-${END_FRAME}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${BLENDER_BIN}" -b -P "${PY_SCRIPT}" -- \
        "${COMMON_ARGS[@]}" \
        --mode render \
        --plan_path "${PLAN_PATH}" \
        --frame_start "${START_FRAME}" \
        --frame_end "${END_FRAME}" \
        --worker_id "${WORKER_ID}" &

    PIDS+=("$!")
    RANGES+=("${GPU_ID}:${START_FRAME}-${END_FRAME}")
    START_FRAME=$(( END_FRAME + 1 ))
done

FAIL=0
for IDX in "${!PIDS[@]}"; do
    PID="${PIDS[$IDX]}"
    if ! wait "${PID}"; then
        echo "Worker failed for ${RANGES[$IDX]}"
        FAIL=1
    fi
done

if [[ "${FAIL}" -ne 0 ]]; then
    exit 1
fi

echo "Completed rendering for plan ${PLAN_PATH}"
