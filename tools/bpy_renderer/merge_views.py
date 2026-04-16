import os
import shutil

# src_dir = "./views_tube_1"
# src_dir = "./views_tube_fixed_cam_2"
# src_dir = "./views_tube_fixed_cam_1"
# src_dir = "./views_tube_fixed_cam"
src_dir = "./views_tube_fixed_cam_vid"
# src_dir = "./views_tube_1"
# src_dir = "./views_tube"
# dst_dir = "./views_tube_1"
dst_dir = "/share/volumes/csi/renchengwei/blender_renderings/views_tube_vid"
# dst_dir = "./views_tube_fixed_cam_1"
# dst_dir = "./views_tube_fixed_cam"
# dst_dir = "./views_tube_1"
# dst_dir = "/share/volumes/csi/renchengwei/blender_renderings/views_tube"
# dst_dir = "./views_tube_fixed_cam"

# analyze dst scene cnt
dst_scene_cnt = {}
for views_sub_dir in os.listdir(dst_dir):
    views_sub_dir_path = os.path.join(dst_dir, views_sub_dir)
    scene_name_prefix = views_sub_dir.split("__")[0]
    scene_name_middle = views_sub_dir.split("__")[1].split(".blend")[0]
    scene_name = f"{scene_name_prefix}__{scene_name_middle}"
    dst_scene_cnt[scene_name] = dst_scene_cnt.get(scene_name, 0) + 1

print(dst_scene_cnt)

for views_sub_dir in os.listdir(src_dir):
    views_sub_dir_path = os.path.join(src_dir, views_sub_dir)
    scene_name_prefix = views_sub_dir.split("__")[0]
    scene_name_middle = views_sub_dir.split("__")[1].split(".blend")[0]
    # copy to dst
    dst_scene_name = f"{scene_name_prefix}__{scene_name_middle}"
    cnt = dst_scene_cnt.get(dst_scene_name, 0)
    if cnt == 0:
        dst_scene_cp_name = dst_scene_name + f".blend"
    else:
        dst_scene_cp_name = dst_scene_name + f".blend_{cnt - 1}"
    dst_scene_cnt[dst_scene_name] = dst_scene_cnt.get(dst_scene_name, 0) + 1
    dst_scene_path = os.path.join(dst_dir, dst_scene_cp_name)
    # copy dir
    shutil.copytree(views_sub_dir_path, dst_scene_path)
    print(f"Copying {views_sub_dir_path} to {dst_scene_path}")
    