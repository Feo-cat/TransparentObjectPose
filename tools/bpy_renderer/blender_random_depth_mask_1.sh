# /home/renchengwei//blender-4.4.3-linux-x64/blender -b -P blender_script_random_depth_mask.py -- \
#     --object_path glb_objs \
#     --plane_path glb_planes \
#     --output_dir ./views_tube \
#     --engine CYCLES \
#     --num_images 12 \
#     --camera_type random \
#     --camera_dist_min 16 \
#     --camera_dist_max 32 \
#     --elevation_min 10 \
#     --elevation_max 45 \
#     --res_w 640 \
#     --res_h 480 \
#     --hdrs_dir /home/renchengwei/bpy_render/hdri_bgs \
#     --silent_mode
#     # --auto_offset True
#     # --camera_dist 1.2

/home/renchengwei/blender-4.4.3-linux-x64/blender -b -P blender_script_random_depth_mask.py -- \
    --object_path glb_objs \
    --plane_path glb_planes \
    --output_dir ./views_tube_1 \
    --engine CYCLES \
    --num_images 12 \
    --camera_type cluster \
    --camera_dist_min 16 \
    --camera_dist_max 32 \
    --elevation_min 10 \
    --elevation_max 45 \
    --enable_other_objects "${ENABLE_OTHER_OBJECTS:-0}" \
    --res_w 640 \
    --res_h 480 \
    --hdrs_dir /home/renchengwei/bpy_render/hdri_bgs \
    --silent_mode \
    --cluster_dist_range 4 \
    --cluster_az_range 45 \
    --cluster_el_range 30 \
    # --auto_offset True
    # --camera_dist 1.2



# /home/renchengwei/blender-4.4.3-linux-x64/blender -b -P depth_mask_fixed_cam_legacy.py -- \
#     --object_path glb_objs \
#     --plane_path glb_planes \
#     --output_dir ./views_tube_fixed_cam_2 \
#     --engine CYCLES \
#     --num_images 12 \
#     --camera_dist_min 16 \
#     --camera_dist_max 32 \
#     --elevation_min 10 \
#     --elevation_max 45 \
#     --res_w 640 \
#     --res_h 480 \
#     --hdrs_dir /home/renchengwei/bpy_render/hdri_bgs \
#     --silent_mode \








# move from views_tube_tmp to views_tube

# pip: /home/renchengwei/blender-4.4.3-linux-x64/4.4/python/bin/python3.11 -m pip install
