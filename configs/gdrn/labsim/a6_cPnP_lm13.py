import os
from core.utils.oss_utils import (
    get_visible_gpu_count,
    print_gpu_memory_debug,
    resolve_ads_cli_bin,
    resolve_model_weights,
)

_base_ = ["../../_base_/gdrn_base.py"]


# IMS_PER_BATCH_PER_GPU = 7
# IMS_PER_BATCH_PER_GPU = 8
IMS_PER_BATCH_PER_GPU = 12
# IMS_PER_BATCH_PER_GPU = 32
# IMS_PER_BATCH_PER_GPU = 10
IMS_PER_BATCH_AUTO = IMS_PER_BATCH_PER_GPU * get_visible_gpu_count()
print(f"IMS_PER_BATCH_AUTO: {IMS_PER_BATCH_AUTO}")
print_gpu_memory_debug()


OSS_CKPT_REMOTE_DIR = os.environ.get("GDRN_CKPT_OSS_DIR", "").strip()
ADS_CLI_DIR = os.environ.get("GDRN_ADS_CLI_DIR", "").strip()
ADS_CLI_BIN = resolve_ads_cli_bin(
    ads_cli_bin=os.environ.get("GDRN_ADS_CLI_BIN", ""),
    ads_cli_dir=ADS_CLI_DIR,
)
MODEL_WEIGHTS = resolve_model_weights(
    oss_ckpt_remote_dir=OSS_CKPT_REMOTE_DIR,
    ads_cli_bin=ADS_CLI_BIN,
)

# OUTPUT_DIR = "/mnt/aoss/gdr_output/gdrn/labsim/a6_cPnP_lm13_ominiPose_multiView"
OUTPUT_DIR = "output/gdrn/labsim/a6_cPnP_lm13_ominiPose_multiView_depth_abs_AR_TEST_ctx_info_interaction"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    DZI_PATCH_GRID_ADSORPTION=False,
    # DZI_PATCH_GRID_ADSORPTION=True,
    MASK_NOISE_AUG_PROB=0.4,  # 50% of samples get mask noise augmentation
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),"
        # "Sometimes(0.5, Affine(scale=(1.0, 1.2))),"
        "Sometimes(0.5, GaussianBlur(np.random.rand())),"
        "Sometimes(0.5, Add((-20, 20), per_channel=0.3)),"
        "Sometimes(0.4, Invert(0.20, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),"
        "Sometimes(0.5, Multiply((0.7, 1.4))),"
        "Sometimes(0.5, ContrastNormalization((0.5, 2.0), per_channel=0.3))"
        "], random_order=False)"
    ),
    CHANGE_BG_PROB=0.0,
    WITH_DEPTH=True,
)

SOLVER = dict(
    # TOTAL_EPOCHS=800,
    TOTAL_EPOCHS=800*12,
    # TOTAL_EPOCHS=800*4,
    # IMS_PER_BATCH=6,
    # IMS_PER_BATCH=12,
    # IMS_PER_BATCH=24,
    # IMS_PER_BATCH=2,
    # IMS_PER_BATCH=36,
    # IMS_PER_BATCH=32,
    # IMS_PER_BATCH=16,
    # IMS_PER_BATCH=8,
    IMS_PER_BATCH=IMS_PER_BATCH_AUTO,
    # IMS_PER_BATCH=48,
    # IMS_PER_BATCH=64,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    CHECKPOINT_PERIOD=100,
    # CHECKPOINT_PERIOD=1,
    # AMP=dict(ENABLED=True), # Enable automatic mixed precision for training. WARNING: here may cause performance drop !!!
)

DATASETS = dict(
    TRAIN=("labsim_train",),
    TEST=("labsim_test",),
    DET_FILES_TEST=("datasets/BOP_DATASETS/labsim/test/test_bboxes/bbox_faster_all.json",),
    SYM_OBJS=["test_tube_rack", "tube"],
)

MODEL = dict(
    # LOAD_DETS_TEST=True,
    LOAD_DETS_TEST=False,
    WEIGHTS=MODEL_WEIGHTS,
    # WEIGHTS="/home/renchengwei/GDR-Net/gdrn_lm.pth",
    # WEIGHTS="/share/volumes/csi/renchengwei/output/gdrn/labsim/a6_cPnP_lm13_ominiPose_multiView/model_0057199.pth",
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            MASK_LOSS_TYPE="BCE",
            XYZ_LW=1.0,
            REGION_CLASS_AWARE=False,
            NUM_REGIONS=64,
            # REGION_LW should be balanced according mask proportion
            REGION_LW=0.4,
        ),
        PNP_NET=dict(
            R_ONLY=False,
            REGION_ATTENTION=True,
            MASK_ATTENTION="mul",
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
            PM_NORM_BY_EXTENT=True,
            PM_LOSS_SYM=True,
            # PM_R_ONLY=True,
            PM_R_ONLY=False,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
            # Keep an explicit absolute-pose constraint in addition to PM/centroid/z.
            ROT_LOSS_TYPE="angular",
            ROT_LW=1.0,
            TRANS_LOSS_TYPE="L1",
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=1.0,
            TRAIN_CONTEXT_THEN_TARGET_PROB=0.8,
            NUM_PM_POINTS=2500,
        ),
        DEPTH_HEAD_LOSS=dict(
            LAMBDA_BCE=1.0,
            LAMBDA_DICE=1.0,
            LAMBDA_DP_REG=1.0,
            LAMBDA_DP_GD=50.0,
            LAMBDA_DP_3D=5.0, # 3D 点云深度监督权重
            # LAMBDA_REPROJ=0.1, # 稠密光流位姿监督权重
            LAMBDA_REPROJ=0.0, # 稠密光流位姿监督权重
            REPROJ_MAX_POINTS=4096,
        ),
        DEPTH_HEAD_MASK_ACT="none",
        TRANS_HEAD=dict(FREEZE=True),
        FREEZE_IMAGE_ENCODER=False,
        RESNET_BACKBONE=False,
        # VGGT_BACKBONE=False,
        VGGT_BACKBONE=True,
        DINO_NUM_BLOCKS=12,
        DINO_TUNE_LAST_N_BLOCKS=4,
        ATTN_DECODER_DEPTH=2,
        ROI_DECODER_DEPTH=2,
        ROI_DECODER_DIM=256,
        ROI_HEAD_OUT_DIM=64,
        ROI_FEAT_INPUT_RES=64,
        # "flash" may trigger Triton/CUDA kernel instability on some environments.
        # Use "torch" for a stable fallback.
        ATTN_IMPL="flash",
        VIEW_INTERACTION=dict(
            ENABLED=True,
            TOKEN_DIM=256,
            NUM_HEADS=4,
            NUM_ATTN_LAYERS=2,
            RGB_EMBED_DIM=32,
            CTX_POSE_NOISE_ROT_DEG=5.0,
            CTX_POSE_NOISE_TRANS_RATIO=0.02,
            # ~20% of batches will train with zero xyz cue, matching inference first-chunk behavior
            XYZ_CUE_DROPOUT_PROB=0.2,
        ),
    ),
)

# TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="gt")  # gt | est
TEST = dict(EVAL_PERIOD=1000, VIS=False, TEST_BBOX_TYPE="gt", TARGET_IDX_MODE="all_except_first")  # gt | est
TRAIN = dict(
    VIS_IMG=True,
    VIS_POSE_VIDEO=True,
    VIS_POSE_VIDEO_INTERVAL=100,
    VIS_METRICS_INTERVAL=20,
    VIS_POSE_VIDEO_OUTPUT_DIR="/mnt/afs/TransparentObjectPose/debug",
    # BAD_CASE_STATS_ENABLED=True,
    BAD_CASE_STATS_ENABLED=False,
    BAD_CASE_STATS_PATH="/mnt/afs/TransparentObjectPose/debug/bad_cases_stats.txt",
    BAD_CASE_IMGS_DIR="/mnt/afs/TransparentObjectPose/debug/bad_cases_imgs",
    BAD_CASE_MIN_ITER=1000,
    BAD_CASE_COOLDOWN_ITERS=20,
    BAD_CASE_MAX_LOGS=500000,
    BAD_CASE_T_CM_THRESHOLD=6.0,
    BAD_CASE_FINAL_T_CM_THRESHOLD=10.0,
    BAD_CASE_MAX_MASK_COMPONENTS=2,
    BAD_CASE_MIN_COMPONENT_AREA=128,
)

CKPT_SYNC = dict(
    ENABLED=bool(OSS_CKPT_REMOTE_DIR),
    REMOTE_DIR=OSS_CKPT_REMOTE_DIR,
    CMD_TEMPLATE=os.environ.get(
        "GDRN_CKPT_SYNC_CMD",
        f"{ADS_CLI_BIN} --threads=32 cp {{local_path}} {{remote_dir}}/",
    ),
    DELETE_LOCAL_AFTER_SYNC=True,
    RETRY_TIMES=3,
    RETRY_INTERVAL_SEC=20,
)

# DEBUG = True
