# WEIGHTS_PATH="output/gdrn/labsim/a6_cPnP_lm13/model_0031359.pth"
# WEIGHTS_PATH="output/gdrn/labsim/a6_cPnP_lm13/model_0025479_symm.pth"
# WEIGHTS_PATH="output/gdrn/labsim/a6_cPnP_lm13/model_0085259.pth"
CKPT_STEP=0078199
WEIGHTS_PATH="output/gdrn/labsim/a6_cPnP_lm13_ominiPose_singleView/model_${CKPT_STEP}.pth"
CONFIG="configs/gdrn/labsim/a6_cPnP_lm13_test.py"

# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/test/000001/rgb/000002.png \
#     --bbox 252 173 383 253 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1

# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/003.png \
#     --bbox 230 123 299 176 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_name 003_pred.png


# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4949_640_480.png \
#     --bbox 271 209 412 320 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_name 4949_640_480_pred.png

# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4950_640_480.png \
#     --bbox 207 236 362 357 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4950_640_480_pred.png

# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4951_640_480.png \
#     --bbox 292 220 427 321 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4951_640_480_pred.png


# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4952_640_480.png \
#     --bbox 293 212 448 331 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4952_640_480_pred.png


# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4960_640_480.png \
#     --bbox 260 182 389 280 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4960_640_480_pred.png


# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4961_640_480.png \
#     --bbox 240 225 351 308 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4961_640_480_pred.png


# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4962_640_480.png \
#     --bbox 252 203 357 283 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4962_640_480_pred.png


# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4963_640_480.png \
#     --bbox 209 244 346 344 \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4963_640_480_pred.png



# python inference_single_image.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image /home/renchengwei/GDR-Net/test_repo/4976_640_480.png \
#     --bbox 316 190 375 352 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 4976_640_480_pred.png


# python inference_single_image.py \
#     --config $CONFIG \
#     --weights $WEIGHTS_PATH \
#     --image /share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000140/000010.png \
#     --bbox 283 60 318 181 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 000140_000010_pred_${CKPT_STEP}.png



# python inference_single_image.py \
#     --config $CONFIG \
#     --weights $WEIGHTS_PATH \
#     --image /share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000150/000010.png \
#     --bbox 266 0 311 152 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 000150_000010_pred_${CKPT_STEP}.png


# python inference_single_image.py \
#     --config $CONFIG \
#     --weights $WEIGHTS_PATH \
#     --image /share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000784/000010.png \
#     --bbox 287 115 333 195 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 000784_000010_pred_${CKPT_STEP}.png


# python inference_single_image.py \
#     --config $CONFIG \
#     --weights $WEIGHTS_PATH \
#     --image /share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000784/000002.png \
#     --bbox 374 74 501 178 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 000784_000002_pred_${CKPT_STEP}.png




# python inference_single_image.py \
#     --config $CONFIG \
#     --weights $WEIGHTS_PATH \
#     --image /share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000091/000007.png \
#     --bbox 0 51 47 224 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 000091_000007_pred_${CKPT_STEP}.png


# python inference_single_image.py \
#     --config $CONFIG \
#     --weights $WEIGHTS_PATH \
#     --image /share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000091/000001.png \
#     --bbox 395 91 428 196 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 000091_000001_pred_${CKPT_STEP}.png


# python inference_single_image.py \
#     --config $CONFIG \
#     --weights $WEIGHTS_PATH \
#     --image /share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000288/000003.png \
#     --bbox 245 120 286 239 \
#     --obj_cls 1 \
#     --dataset_name labsim_test \
#     --cam 700 0 320 0 700 240 0 0 1 \
#     --output_dir /home/renchengwei/GDR-Net/test_repo \
#     --output_name 000288_000003_pred_${CKPT_STEP}.png




python inference_single_image.py \
    --config $CONFIG \
    --weights $WEIGHTS_PATH \
    --image /home/renchengwei/GDR-Net/test_repo/4979_640_480.png \
    --bbox 306 135 434 269 \
    --obj_cls 1 \
    --dataset_name labsim_test \
    --cam 700 0 320 0 700 240 0 0 1 \
    --output_dir /home/renchengwei/GDR-Net/test_repo \
    --output_name 4979_640_480_pred_${CKPT_STEP}.png


python inference_single_image.py \
    --config $CONFIG \
    --weights $WEIGHTS_PATH \
    --image /home/renchengwei/GDR-Net/test_repo/4980_640_480.png \
    --bbox 280 149 379 287 \
    --obj_cls 1 \
    --dataset_name labsim_test \
    --cam 700 0 320 0 700 240 0 0 1 \
    --output_dir /home/renchengwei/GDR-Net/test_repo \
    --output_name 4980_640_480_pred_${CKPT_STEP}.png


python inference_single_image.py \
    --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
    --weights $WEIGHTS_PATH \
    --image /home/renchengwei/GDR-Net/test_repo/5042_640_480.png \
    --bbox 242 187 298 324 \
    --obj_cls 1 \
    --dataset_name labsim_test \
    --cam 700 0 320 0 700 240 0 0 1 \
    --output_dir /home/renchengwei/GDR-Net/test_repo \
    --output_name 5042_640_480_pred_${CKPT_STEP}.png


python inference_single_image.py \
    --config $CONFIG \
    --weights $WEIGHTS_PATH \
    --image /home/renchengwei/GDR-Net/test_repo/5043_640_480.png \
    --bbox 260 162 318 324 \
    --obj_cls 1 \
    --dataset_name labsim_test \
    --cam 700 0 320 0 700 240 0 0 1 \
    --output_dir /home/renchengwei/GDR-Net/test_repo \
    --output_name 5043_640_480_pred_${CKPT_STEP}.png


python inference_single_image.py \
    --config $CONFIG \
    --weights $WEIGHTS_PATH \
    --image /home/renchengwei/GDR-Net/test_repo/5044_640_480.png \
    --bbox 198 198 342 266 \
    --obj_cls 1 \
    --dataset_name labsim_test \
    --cam 700 0 320 0 700 240 0 0 1 \
    --output_dir /home/renchengwei/GDR-Net/test_repo \
    --output_name 5044_640_480_pred_${CKPT_STEP}.png