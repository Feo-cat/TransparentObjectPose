_base_ = "./common_base.py"
# -----------------------------------------------------------------------------
# base model cfg for gdrn
# -----------------------------------------------------------------------------
MODEL = dict(
    DEVICE="cuda",
    WEIGHTS="",
    # PIXEL_MEAN = [103.530, 116.280, 123.675]  # bgr
    # PIXEL_STD = [57.375, 57.120, 58.395]
    # PIXEL_MEAN = [123.675, 116.280, 103.530]  # rgb
    # PIXEL_STD = [58.395, 57.120, 57.375]
    PIXEL_MEAN=[0, 0, 0],  # to [0,1]
    PIXEL_STD=[255.0, 255.0, 255.0],
    LOAD_DETS_TEST=False,
    CDPN=dict(
        NAME="GDRN",  # used module file name
        TASK="rot",
        USE_MTL=False,  # uncertainty multi-task weighting
        # DINO + attention-decoder depth controls (used when VGGT/DINO backbone is enabled).
        DINO_NUM_BLOCKS=12,
        # Fine-tune only the last N DINO blocks (0 means fully frozen).
        # Default keeps previous behavior: tune all kept blocks.
        DINO_TUNE_LAST_N_BLOCKS=12,
        ATTN_DECODER_DEPTH=6,
        ROI_DECODER_DEPTH=5,
        ROI_DECODER_DIM=512,
        ROI_HEAD_OUT_DIM=128,
        ROI_FEAT_INPUT_RES=256,
        ## view interaction (Proposal V1)
        VIEW_INTERACTION=dict(
            ENABLED=False,
            TOKEN_DIM=256,
            NUM_HEADS=4,
            NUM_ATTN_LAYERS=2,
            RGB_EMBED_DIM=32,
            CTX_POSE_NOISE_ROT_DEG=5.0,   # std of rotation noise (degrees) added to context GT pose during training
            CTX_POSE_NOISE_TRANS_RATIO=0.02,  # std of translation noise as fraction of ||t||
            # Probability of zeroing xyz cue during training to simulate first-chunk inference
            # (where no cached poses exist).  Set e.g. to 0.2 to cover ~20% of all-target batches.
            XYZ_CUE_DROPOUT_PROB=0.0,
            GEOMETRIC_PRIOR=dict(
                ENABLED=False,
                RETRIEVAL_RES=8,
                PRIOR_RES=64,
                WITH_8X8_RETRIEVAL=True,
                WITH_64X64_PRIOR=True,
                INJECT_TO_PNP=True,
                APPEND_PRIOR_CONF=True,
                APPEND_PRIOR_RESIDUAL=True,
                USE_VIEW_DIR_OBJ=True,
                USE_CAM_CENTER_OBJ=True,
                USE_GLOBAL_IMAGE_COORD=False,  # Phase 2 only; v1 stays on existing roi-local coords
                SOFT_ASSIGN_SIGMA=0.05,
                BANK_TOPK=256,
                PRIOR_DROPOUT_PROB=0.15,
                HYP_SOURCE="noisy_gt",  # v1: use structured noisy GT context pose
            ),
            GEOMETRIC_PRIOR_LOSS=dict(
                XYZ_LW=1.0,
                CONF_LW=0.2,
                CONF_ERR_THR=0.05,
            ),
            RELATIVE_GEOMETRY_LOSS=dict(
                ENABLED=False,
                LW=0.25,
                TRANS_LW=0.05,
                NUM_POINTS=1024,
            ),
        ),
        ## backbone
        BACKBONE=dict(
            PRETRAINED="torchvision://resnet34",
            ARCH="resnet",
            NUM_LAYERS=34,
            INPUT_CHANNEL=3,
            INPUT_RES=256,
            OUTPUT_RES=64,
            FREEZE=False,
        ),
        ## rot head
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_CONCAT=False,
            XYZ_BIN=64,  # for classification xyz, the last one is bg
            NUM_LAYERS=3,
            NUM_FILTERS=256,
            CONV_KERNEL_SIZE=3,
            NORM="BN",
            NUM_GN_GROUPS=32,
            OUT_CONV_KERNEL_SIZE=1,
            NUM_CLASSES=13,
            ROT_CLASS_AWARE=False,
            XYZ_LOSS_TYPE="L1",  # L1 | CE_coor
            XYZ_LOSS_MASK_GT="visib",  # trunc | visib | obj
            XYZ_LW=1.0,
            MASK_CLASS_AWARE=False,
            MASK_LOSS_TYPE="L1",  # L1 | BCE | CE
            MASK_LOSS_GT="trunc",  # trunc | visib | gt
            MASK_LW=1.0,
            MASK_THR_TEST=0.5,
            # for region classification, 0 is bg, [1, num_regions]
            # num_regions <= 1: no region classification
            NUM_REGIONS=8,
            REGION_CLASS_AWARE=False,
            REGION_LOSS_TYPE="CE",  # CE
            REGION_LOSS_MASK_GT="visib",  # trunc | visib | obj
            REGION_LW=1.0,
        ),
        ## for direct regression
        PNP_NET=dict(
            FREEZE=False,
            R_ONLY=False,
            LR_MULT=1.0,
            # ConvPnPNet | SimplePointPnPNet | PointPnPNet | ResPointPnPNet
            PNP_HEAD_CFG=dict(type="ConvPnPNet", norm="GN", num_gn_groups=32, drop_prob=0.0),  # 0.25
            # PNP_HEAD_CFG=dict(
            #     type="ConvPnPNet",
            #     norm="GN",
            #     num_gn_groups=32,
            #     spatial_pooltype="max", # max | mean | soft | topk
            #     spatial_topk=1,
            #     region_softpool=False,
            #     region_topk=8,  # NOTE: default the same as NUM_REGIONS
            # ),
            WITH_2D_COORD=False,  # using 2D XY coords
            REGION_ATTENTION=False,  # region attention
            MASK_ATTENTION="none",  # none | concat | mul
            TRANS_WITH_BOX_INFO="none",  # none | ltrb | wh  # TODO
            ## for losses
            # {allo/ego}_{quat/rot6d/log_quat/lie_vec}
            ROT_TYPE="ego_rot6d",
            TRANS_TYPE="centroid_z",  # trans | centroid_z (SITE) | centroid_z_abs
            Z_TYPE="REL",  # REL | ABS | LOG | NEG_LOG  (only valid for centroid_z)
            # Continuous-symmetry GT canonicalization mode: face_camera | closest_pose
            CONT_SYM_GT_MODE="face_camera",
            # Degenerate fallback threshold: |u_perp| < thresh → skip camera-facing canon
            CONT_SYM_CANON_DEGENERATE_THRESH=0.1,
            # point matching loss
            NUM_PM_POINTS=3000,
            PM_LOSS_TYPE="L1",  # L1 | Smooth_L1
            PM_SMOOTH_L1_BETA=1.0,
            PM_LOSS_SYM=False,  # use symmetric PM loss
            PM_NORM_BY_EXTENT=False,  # 10. / extent.max(1, keepdim=True)[0]
            # if False, the trans loss is in point matching loss
            PM_R_ONLY=True,  # only do R loss in PM
            PM_DISENTANGLE_T=False,  # disentangle R/T
            PM_DISENTANGLE_Z=False,  # disentangle R/xy/z
            PM_T_USE_POINTS=False,
            PM_LW=1.0,
            ROT_LOSS_TYPE="angular",  # angular | L2
            ROT_LW=0.0,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=0.0,
            Z_LOSS_TYPE="L1",
            Z_LW=0.0,
            TRANS_LOSS_TYPE="L1",
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=0.0,
            TRAIN_NUM_CONTEXT_VIEWS=3,           # number of context views per AR chunk
            TRAIN_NUM_TARGET_VIEWS=3,            # number of target views per AR chunk
            TRAIN_CONTEXT_THEN_TARGET_PROB=0.5,  # probability of training with context->target chunks; all-target uses 1-p
            # bind term loss: R^T@t
            BIND_LOSS_TYPE="L1",
            BIND_LW=0.0,
        ),
        ## trans head
        TRANS_HEAD=dict(
            ENABLED=False,
            FREEZE=True,
            LR_MULT=1.0,
            NUM_LAYERS=3,
            NUM_FILTERS=256,
            NORM="BN",
            NUM_GN_GROUPS=32,
            CONV_KERNEL_SIZE=3,
            OUT_CHANNEL=3,
            TRANS_TYPE="centroid_z",  # trans | centroid_z
            Z_TYPE="REL",  # REL | ABS | LOG | NEG_LOG
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=0.0,
            Z_LOSS_TYPE="L1",
            Z_LW=0.0,
            TRANS_LOSS_TYPE="L1",
            TRANS_LW=0.0,
        ),
    ),
    # some d2 keys but not used
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)

TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    TEST_BBOX_TYPE="gt",  # gt | est
    USE_PNP=False,  # use pnp or direct prediction
    # ransac_pnp | net_iter_pnp (learned pnp init + iter pnp) | net_ransac_pnp (net init + ransac pnp)
    # net_ransac_pnp_rot (net_init + ransanc pnp --> net t + pnp R)
    PNP_TYPE="ransac_pnp",
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
)
