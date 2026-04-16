# ---------------------------------------------------------------------------
# Local dataset cache: rsync from AFS to local disk on first run, then use
# the local copy for training (avoids AFS I/O bottleneck).
# Override by setting GDRN_DATASETS_ROOT before calling this script.
# ---------------------------------------------------------------------------
AFS_DATASETS="/mnt/afs/TransparentObjectPose/datasets"
LOCAL_DATASETS="/tmp"

if [ -z "$GDRN_DATASETS_ROOT" ]; then
    LOCAL_LABSIM="${LOCAL_DATASETS}/BOP_DATASETS/labsim"
    AFS_LABSIM="${AFS_DATASETS}/BOP_DATASETS/labsim"
    MARKER="${LOCAL_DATASETS}/.sync_done"

    if [ ! -f "$MARKER" ]; then
        _afs_size=$(du -sh "${AFS_LABSIM}" 2>/dev/null | cut -f1)
        echo "[train_script] Syncing dataset from AFS to local disk (one-time, ~${_afs_size})..."
        mkdir -p "${LOCAL_DATASETS}/BOP_DATASETS"
        rsync -a --partial --append-verify --info=progress2 --human-readable "${AFS_LABSIM}/" "${LOCAL_LABSIM}/" 2>&1 | \
            stdbuf -oL awk '
                BEGIN {
                    RS="\r"
                    ORS=""
                    log_step_bytes=100 * 1024 * 1024
                    next_log_bytes=log_step_bytes
                }
                function human_to_bytes(value, num, unit) {
                    num = value
                    gsub(/[[:alpha:]]+$/, "", num)
                    num += 0

                    unit = value
                    sub(/^[0-9.]+/, "", unit)
                    unit = toupper(unit)

                    if (unit == "" || unit == "B") return num
                    if (unit == "K" || unit == "KB") return num * 1024
                    if (unit == "M" || unit == "MB") return num * 1024 * 1024
                    if (unit == "G" || unit == "GB") return num * 1024 * 1024 * 1024
                    if (unit == "T" || unit == "TB") return num * 1024 * 1024 * 1024 * 1024
                    return -1
                }
                /[0-9]+%/ {
                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
                    n = split($0, fields, /[[:space:]]+/)
                    bytes_done = human_to_bytes(fields[1])
                    if (n >= 4 && (bytes_done >= next_log_bytes || fields[2] == "100%")) {
                        printf "[train_script] syncing: %s %s %s eta %s\n", fields[1], fields[2], fields[3], fields[4]
                        while (bytes_done >= next_log_bytes) {
                            next_log_bytes += log_step_bytes
                        }
                    } else if (n < 4) {
                        printf "[train_script] syncing: %s\n", $0
                    }
                    fflush()
                }
                END { print "" }
            '
        RSYNC_EXIT=${PIPESTATUS[0]}
        if [ $RSYNC_EXIT -eq 0 ]; then
            touch "$MARKER"
            echo "[train_script] Dataset sync done."
        else
            echo "[train_script] WARNING: rsync failed, falling back to AFS path."
            LOCAL_DATASETS="$AFS_DATASETS"
        fi
    else
        echo "[train_script] Local dataset cache found, skipping rsync."
    fi

    export GDRN_DATASETS_ROOT="${LOCAL_DATASETS}"
fi

echo "[train_script] GDRN_DATASETS_ROOT=${GDRN_DATASETS_ROOT}"
# ---------------------------------------------------------------------------

export GDRN_CKPT_OSS_DIR="s3://019CE739817A7E43855626E278830CD7:019CE739817A7E35801972419E02787C@jiycoss.aoss.cn-sh-01g.sensecoreapi-oss.cn/gdr_ckpt/multiview_abs_ar_vid_depth_TEST_ctx_info_interaction/"
# export GDRN_WEIGHTS_NAME="model_0050999.pth"
export GDRN_ADS_CLI_DIR="/mnt/afs/afs"
export GDRN_ADS_CLI_BIN="${GDRN_ADS_CLI_BIN:-${GDRN_ADS_CLI_DIR}/ads-cli}"
export GDRN_CKPT_SYNC_CMD="${GDRN_ADS_CLI_BIN} --threads=32 cp {local_path} {remote_dir}/"


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4   core/gdrn_modeling/main_gdrn.py   --config-file configs/gdrn/labsim/a6_cPnP_lm13.py   --num-gpus 4 --strategy ddp

# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1   core/gdrn_modeling/main_gdrn.py   --config-file configs/gdrn/labsim/a6_cPnP_lm13_ctx_prior.py   --num-gpus 1 --strategy ddp
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2   core/gdrn_modeling/main_gdrn.py   --config-file configs/gdrn/labsim/a6_cPnP_lm13_ctx_prior.py   --num-gpus 2 --strategy ddp
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4   core/gdrn_modeling/main_gdrn.py   --config-file configs/gdrn/labsim/a6_cPnP_lm13_ctx_prior.py   --num-gpus 4 --strategy ddp
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8   core/gdrn_modeling/main_gdrn.py   --config-file configs/gdrn/labsim/a6_cPnP_lm13_ctx_prior.py   --num-gpus 8 --strategy ddp
