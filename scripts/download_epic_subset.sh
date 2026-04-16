#!/usr/bin/env bash

set -euo pipefail

# Download a small EPIC-KITCHENS subset that matches VISOR video ids.
# Requires internet access and the official EPIC downloader repo.
# This wrapper adds per-file progress and speed output.
#
# Examples:
#   bash scripts/download_epic_subset.sh
#   bash scripts/download_epic_subset.sh --mode rgb-frames
#   bash scripts/download_epic_subset.sh --videos P01_01,P02_01,P03_03
#   EPIC_DL_CONNECTIONS=32 bash scripts/download_epic_subset.sh
#   OUTPUT_DIR=/mnt/afs/datasets/EPIC-KITCHENS bash scripts/download_epic_subset.sh

MODE="videos"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/datasets/EPIC-KITCHENS}"
DOWNLOADER_ROOT="${DOWNLOADER_ROOT:-/mnt/afs/tools/epic-kitchens-download-scripts}"
VIDEOS="P01_01,P02_01,P03_03,P04_02,P06_01"

usage() {
    cat <<EOF
Usage: $0 [--mode videos|rgb-frames] [--videos P01_01,P02_01,...] [--output-dir DIR] [--downloader-root DIR]

Defaults:
  --mode            ${MODE}
  --videos          ${VIDEOS}
  --output-dir      ${OUTPUT_DIR}
  --downloader-root ${DOWNLOADER_ROOT}

Notes:
  1. This script uses the official EPIC-KITCHENS downloader:
     https://github.com/epic-kitchens/epic-kitchens-download-scripts
  2. The downloader may ask you to accept terms or authenticate depending on the data source.
  3. VISOR dense hand masks need the original EPIC RGB/video source to reconstruct continuous hand clips.
  4. Optional env vars:
     EPIC_DL_CONNECTIONS=16
     EPIC_DL_MIN_SPLIT_SIZE=10M
     EPIC_DL_BACKEND=aria2c|urllib
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:?missing value for --mode}"
            shift 2
            ;;
        --videos)
            VIDEOS="${2:?missing value for --videos}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?missing value for --output-dir}"
            shift 2
            ;;
        --downloader-root)
            DOWNLOADER_ROOT="${2:?missing value for --downloader-root}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${MODE}" != "videos" && "${MODE}" != "rgb-frames" ]]; then
    echo "Unsupported mode: ${MODE}. Use 'videos' or 'rgb-frames'." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${DOWNLOADER_ROOT}" ]]; then
    echo "Cloning official EPIC downloader into ${DOWNLOADER_ROOT}"
    git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "${DOWNLOADER_ROOT}"
fi

if [[ ! -f "${DOWNLOADER_ROOT}/epic_downloader.py" ]]; then
    echo "Cannot find epic_downloader.py under ${DOWNLOADER_ROOT}" >&2
    exit 1
fi

echo "Download mode  : ${MODE}"
echo "Video ids      : ${VIDEOS}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Downloader dir : ${DOWNLOADER_ROOT}"

python3 /mnt/afs/TransparentObjectPose/scripts/run_epic_downloader_with_progress.py \
    --downloader-root "${DOWNLOADER_ROOT}" \
    -- \
    "--${MODE}" \
    --specific-videos "${VIDEOS}" \
    --output-path "${OUTPUT_DIR}"
