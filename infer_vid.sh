# python inference_vid.py \
#   --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#   --weights /mnt/afs/TransparentObjectPose/test_repo/model_0007999.pth \
#   --image_dir /mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002/rgb/000786 \
#   --bbox_json /mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002/000786_bbox.json \
#   --obj_cls 1 \
#   --dataset_name labsim_test \
#   --cam 700 0 320 0 700 240 0 0 1 \
#   --output_dir /mnt/afs/TransparentObjectPose/test_repo \
#   --save_vis \

# python inference_vid.py \
#   --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#   --weights /mnt/afs/TransparentObjectPose/test_repo/model_0009999.pth \
#   --image_dir /mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002/rgb/000723 \
#   --bbox_json /mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002/000723_bbox.json \
#   --obj_cls 1 \
#   --dataset_name labsim_test \
#   --cam 700 0 320 0 700 240 0 0 1 \
#   --output_dir /mnt/afs/TransparentObjectPose/test_repo/000723 \
#   --save_vis \

# python inference_vid.py \
#   --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#   --weights /mnt/afs/TransparentObjectPose/test_repo/model_0009999.pth \
#   --image_dir /mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002/rgb/000321 \
#   --bbox_json /mnt/afs/TransparentObjectPose/datasets/BOP_DATASETS/labsim/train/000002/000321_bbox.json \
#   --obj_cls 1 \
#   --dataset_name labsim_test \
#   --cam 700 0 320 0 700 240 0 0 1 \
#   --output_dir /mnt/afs/TransparentObjectPose/test_repo/000321 \
#   --save_vis \


# python inference_vid.py \
#   --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
#   --weights /mnt/afs/TransparentObjectPose/test_repo/model_0025299.pth \
#   --image_dir /mnt/afs/TransparentObjectPose/test_repo/test_vid \
#   --bbox_json /mnt/afs/TransparentObjectPose/test_repo/test_vid/bbox.json \
#   --obj_cls 1 \
#   --dataset_name labsim_test \
#   --cam 700 0 320 0 700 240 0 0 1 \
#   --roi_mode external_bbox \
#   --output_dir /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred \
#   --save_vis \


# auto ROI from model-predicted mask (bbox_json not required)
python inference_vid.py \
  --config configs/gdrn/labsim/a6_cPnP_lm13_test.py \
  --weights /mnt/afs/TransparentObjectPose/test_repo/model_0025299.pth \
  --image_dir /mnt/afs/TransparentObjectPose/test_repo/test_vid \
  --obj_cls 1 \
  --dataset_name labsim_test \
  --cam 700 0 320 0 700 240 0 0 1 \
  --roi_mode auto_mask \
  --mask_thr 0.5 \
  --min_mask_pixels 16 \
  --output_dir /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred_auto_mask \
  --save_vis \