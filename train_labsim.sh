# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

# ./core/gdrn_modeling/train_gdrn.sh configs/gdrn/lm/a6_cPnP_lm13.py 4,5,6,7
# ./core/gdrn_modeling/train_gdrn.sh configs/gdrn/labsim/a6_cPnP_lm13_test.py 7
# ./core/gdrn_modeling/train_gdrn.sh configs/gdrn/labsim/a6_cPnP_lm13.py 7
# ./core/gdrn_modeling/train_gdrn.sh configs/gdrn/labsim/a6_cPnP_lm13.py 2,3,4,5,6,7
# ./core/gdrn_modeling/train_gdrn.sh configs/gdrn/labsim/a6_cPnP_lm13.py 0,1,2,3,4,5,6,7
# ./core/gdrn_modeling/train_gdrn.sh configs/gdrn/labsim/a6_cPnP_lm13.py 0,1

# source /root/anaconda3/etc/profile.d/conda.sh
# conda activate gdr

# nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total --format=csv -l 2 > /mnt/afs/gpu_memory_monitor.log

# if you want to sync checkpoint to OSS, you need to set the following environment variables
# export GDRN_CKPT_OSS_DIR="s3://019CA7860AC073D39601CF4E30339D2C:019CA7860AC073BA9804E6B73B9DC91F@rencwoss.aoss.cn-sh-01b.sensecoreapi-oss.cn/gdr_ckpt/"
# export GDRN_ADS_CLI_DIR="/mnt/afs/afs"
# export GDRN_ADS_CLI_BIN="${GDRN_ADS_CLI_BIN:-${GDRN_ADS_CLI_DIR}/ads-cli}"
# export GDRN_CKPT_SYNC_CMD="${GDRN_ADS_CLI_BIN} --threads=32 cp {local_path} {remote_dir}/"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi not found. Cannot detect GPU count." >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [ "$GPU_COUNT" -le 0 ]; then
  echo "Error: no GPUs detected." >&2
  exit 1
fi

GPU_IDS="$(seq -s, 0 $((GPU_COUNT - 1)))"
./core/gdrn_modeling/train_gdrn.sh configs/gdrn/labsim/a6_cPnP_lm13.py "$GPU_IDS"