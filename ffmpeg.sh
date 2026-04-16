ffmpeg -framerate 6 -i %06d.png -c:v libx264 -pix_fmt yuv420p output.mp4

ffmpeg -framerate 6 -i /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred/depth/%06d_depth.png -c:v libx264 -pix_fmt yuv420p /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred/depth/output_depth.mp4

ffmpeg -framerate 6 -i /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred/mask/%06d_mask.png -c:v libx264 -pix_fmt yuv420p /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred/mask/output_mask.mp4

ffmpeg -framerate 6 -i /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred/%06d_pred.png -c:v libx264 -pix_fmt yuv420p /mnt/afs/TransparentObjectPose/test_repo/test_vid/pred/output_pred.mp4

# -frames:v 69 
ffmpeg -framerate 12 -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/depth/%06d_depth.png -c:v libx264 -pix_fmt yuv420p -frames:v 69 /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/depth/output_depth1.mp4

ffmpeg -framerate 12 -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/mask/%06d_mask.png -c:v libx264 -pix_fmt yuv420p -frames:v 69 /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/mask/output_mask1.mp4

ffmpeg -framerate 12 -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/%06d_pred.png -c:v libx264 -pix_fmt yuv420p -frames:v 69 /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/output_pred1.mp4

# 
ffmpeg -framerate 12 -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/depth/%06d_depth.png -c:v libx264 -pix_fmt yuv420p /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/depth/output_depth2.mp4

ffmpeg -framerate 12 -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/mask/%06d_mask.png -c:v libx264 -pix_fmt yuv420p /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/mask/output_mask2.mp4

ffmpeg -framerate 12 -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/%06d_pred.png -c:v libx264 -pix_fmt yuv420p /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/pred_ar/output_pred2.mp4


# MOV to mp4, and crop
ffmpeg -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525.MOV   -vf "crop='min(iw,ih*4/3)':'min(iw*3/4,ih)',scale=640:480"   -c:v libx264 -crf 18 -pix_fmt yuv420p   -c:a aac /mnt/afs/TransparentObjectPose/test_repo/IMG_5525.mp4

# mp4 to imgs
ffmpeg -i /mnt/afs/TransparentObjectPose/test_repo/IMG_5525.mp4 /mnt/afs/TransparentObjectPose/test_repo/IMG_5525/%06d.png