# ffmpeg -framerate 25 -i %06d.png -c:v libx264 -pix_fmt yuv420p output.mp4
CKPT_STEP=0078199
WEIGHTS_PATH="output/gdrn/labsim/a6_cPnP_lm13_ominiPose_singleView/model_${CKPT_STEP}.pth"
BBOX_JSON="/home/renchengwei/sam2/results_test_tube_2/bbox.json"

# python inference_batch.py \
#     --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#     --weights output/gdrn/labsim/a6_cPnP_lm13/model_final.pth \
#     --image_dir /home/renchengwei/sam2/notebooks/videos/test_tube_rack \
#     --bbox_json /home/renchengwei/sam2/results/bbox.json \
#     --obj_cls 0 \
#     --dataset_name labsim_test \
#     --save_vis \
#     --cam 700 0 320 0 700 240 0 0 1 


python inference_batch.py \
    --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
    --weights $WEIGHTS_PATH \
    --image_dir /home/renchengwei/sam2/notebooks/videos/test_tube_2 \
    --bbox_json $BBOX_JSON \
    --obj_cls 1 \
    --dataset_name labsim_test \
    --save_vis \
    --cam 700 0 320 0 700 240 0 0 1 \
    --symm_mode continuous \
    --symm_axis 0 0 1 \
    --symm_offset 0 0 0 \
    