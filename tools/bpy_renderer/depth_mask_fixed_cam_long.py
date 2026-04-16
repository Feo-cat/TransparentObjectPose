import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

from mathutils import Vector, Matrix
import numpy as np

import bpy
from contextlib import contextmanager

# ==========================================
# Configs & Utils
# ==========================================
@contextmanager
def suppress_output():
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stdout = os.dup(1)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(save_stdout, 1)
            os.dup2(save_stderr, 2)
            os.close(null_fd)
        except Exception:
            pass
        
def randomize_glass_ior(objects, mat_target_name="Glass.002", min_ior=1.0, max_ior=1.5):
    modified_count = 0
    new_ior = random.uniform(min_ior, max_ior)
    for obj in objects:
        if obj.type != 'MESH': continue
        for slot in obj.material_slots:
            if mat_target_name != slot.material.name: continue
            mat = slot.material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_GLASS':
                        node.inputs['IOR'].default_value = new_ior
                        modified_count += 1
                    elif node.type == 'BSDF_PRINCIPLED':
                        trans_input = node.inputs.get('Transmission Weight') or node.inputs.get('Transmission')
                        if trans_input and trans_input.default_value > 0:
                            node.inputs['IOR'].default_value = new_ior
                            modified_count += 1
    if modified_count > 0:
        print(f"  [Material] Randomized Glass IOR to {new_ior:.3f} for {modified_count} slots.")

def randomize_water_height(objects, target_name="tube.002", min_scale=0.0, max_scale=0.084, height_value=None):
    found = False
    new_scale = height_value if height_value is not None else random.uniform(min_scale, max_scale)
    
    # 记录当前活动物体
    original_active = bpy.context.view_layer.objects.active
    
    for obj in objects:
        if target_name in obj.name:
            # 1. 选中水体
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # ========================================================
            # 步骤 A: 修复液面倾斜 (Shear)
            # ========================================================
            try:
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            except Exception as e:
                print(f"  [Water] Apply transform failed (likely multi-user data): {e}")

            # ========================================================
            # 步骤 B: 调整高度 (Z轴)
            # ========================================================
            obj.scale[2] = new_scale
            
            print(f"  [Water] Rectified rotation & Set height (Z) to {new_scale:.5f}")
            found = True
            
            obj.select_set(False)

    # 恢复现场
    if original_active:
        bpy.context.view_layer.objects.active = original_active

    if not found:
        print(f"  [Water] Warning: Could not find water object with name '{target_name}'")
        
        
def check_and_apply_solidify(obj):
    if obj.type != 'MESH': return

    print(f"\n========== DIAGNOSIS: {obj.name} ==========")
    
    # 1. 打印所有 Modifier
    print(f"  [Modifiers List]:")
    if len(obj.modifiers) == 0:
        print("    (None)")
    else:
        for mod in obj.modifiers:
            # 打印名字和类型
            print(f"    - Name: '{mod.name}' | Type: {mod.type}")
            # 如果是实体化，额外打印参数
            if mod.type == 'SOLIDIFY':
                print(f"      -> Thickness: {mod.thickness:.4f}, Offset: {mod.offset}")

    # 2. 打印形态键 (Shape Keys)
    if obj.data.shape_keys:
        print(f"  [CRITICAL WARNING] Shape Keys detected! (Blocks modifier applying)")
        for key_block in obj.data.shape_keys.key_blocks:
             print(f"    - Key: {key_block.name} (Value: {key_block.value})")
    else:
        print(f"  [Shape Keys]: None (Safe)")

    # 3. 检查数据用户数
    print(f"  [Data Users]: {obj.data.users} (If > 1, apply might fail)")
    print("===========================================\n")

    # ==========================================
    # 原有的修复逻辑 (尝试 Convert)
    # ==========================================
    solidify_mods = [m for m in obj.modifiers if m.type == 'SOLIDIFY']
    
    if solidify_mods:
        try:
            bpy.context.view_layer.objects.active = obj
            
            # 1. 解决 Multi-user
            if obj.data.users > 1:
                obj.data = obj.data.copy()
                
            # 2. 移除形态键 (如果有)
            if obj.data.shape_keys:
                print(f"    [Auto-Fix] Clearing Shape Keys...")
                obj.active_shape_key_index = 0
                bpy.ops.object.shape_key_remove(all=True)

            # 3. 强行转网格
            bpy.ops.object.convert(target='MESH')
            print(f"    [Action] Successfully converted '{obj.name}' to Mesh.")
            
        except Exception as e:
            print(f"    [Error] Failed to convert: {repr(e)}")
        
        

# 【修改点 1】 配置表支持 None
TABLE_CONFIG = {
    "plane_checker.glb": [-8., -8., 8., 8.],
    "plane_furn_cabinet.glb": [-7.96, -3.46, 9.1, 3.46],
    "plane_furn.glb": [7.32],
    "plane_gray.glb": [-4.0, -4.0, 4.0, 4.0],
    "plane_wood.glb": [-2.96, -3.98, 2.96, 3.98],
    "plane_table.glb": [-4.6, -7.6, 4.6, 7.6],
    "plane_desk.glb": [-4.52, -8.0, 4.52, 8.0],
    "plane_round_table.glb": [6.28],
    "plane_office_table.glb": [-3.78, -6.6, 3.78, 6.6],
}
TABLE_CONFIG[None] = 5.
TABLE_LIST = list(TABLE_CONFIG.keys())

# Fixed Object Config
fixed_obj_names_list = ["tube_water.blend", "tube.blend"]
# fixed_obj_names_list = ["tube_water.blend"]
whole_obj_visib_ratio_threshold = 0.004
occ_ratio_threshold = 0.6



fixed_obj_rand_proc_func = {
    "tube.blend": {},
    "tube_water.blend": {randomize_water_height: ["tube.002", 0.0, 0.065]},
}

OBJECT_LIST = [] 

def get_all_children(obj):
    children = []
    for c in obj.children:
        children.append(c)
        children.extend(get_all_children(c))
    return children

def get_world_bbox(obj):
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    meshes = []
    if obj.type == 'MESH': meshes.append(obj)
    for c in get_all_children(obj):
        if c.type == 'MESH': meshes.append(c)
    if obj.type == 'EMPTY' and obj.instance_type == 'COLLECTION':
        for c in obj.instance_collection.objects:
            if c.type == 'MESH': meshes.append(c)
            for cc in get_all_children(c):
                if cc.type == 'MESH': meshes.append(cc)
    if not meshes: return Vector((-0.1, -0.1, -0.1)), Vector((0.1, 0.1, 0.1))
    for m in meshes:
        for v in m.bound_box:
            p = m.matrix_world @ Vector(v)
            bbox_min = Vector((min(bbox_min[0], p[0]), min(bbox_min[1], p[1]), min(bbox_min[2], p[2])))
            bbox_max = Vector((max(bbox_max[0], p[0]), max(bbox_max[1], p[1]), max(bbox_max[2], p[2])))
    return bbox_min, bbox_max

def bbox_overlap_2d(bbox_a, bbox_b):
    (ax_min, ay_min), (ax_max, ay_max) = bbox_a
    (bx_min, by_min), (bx_max, by_max) = bbox_b
    return not (ax_max <= bx_min or ax_min >= bx_max or ay_max <= by_min or ay_min >= by_max)

def bbox_overlap_3d(bbox_a, bbox_b):
    (a_min, a_max) = bbox_a
    (b_min, b_max) = bbox_b
    return not (
        a_max.x <= b_min.x or a_min.x >= b_max.x or
        a_max.y <= b_min.y or a_min.y >= b_max.y or
        a_max.z <= b_min.z or a_min.z >= b_max.z
    )

def get_scene_objects_set(): 
    return set(bpy.context.scene.objects)

def get_new_objects(old_set): 
    return [o for o in bpy.context.scene.objects if o not in old_set]

def get_meshes_from_objects(objs):
    meshes = []
    for o in objs:
        if o.type == 'MESH': meshes.append(o)
        if o.type == 'EMPTY' and o.instance_type == 'COLLECTION':
            for c in o.instance_collection.objects:
                if c.type == 'MESH': meshes.append(c)
    return meshes

def get_world_bbox_from_meshes(meshes):
    if not meshes: raise RuntimeError("No mesh found")
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    for m in meshes:
        for v in m.bound_box:
            p = m.matrix_world @ Vector(v)
            bbox_min.x = min(bbox_min.x, p.x); bbox_min.y = min(bbox_min.y, p.y); bbox_min.z = min(bbox_min.z, p.z)
            bbox_max.x = max(bbox_max.x, p.x); bbox_max.y = max(bbox_max.y, p.y); bbox_max.z = max(bbox_max.z, p.z)
    return bbox_min, bbox_max

def find_root_object(objs):
    for o in objs:
        if o.parent is None: return o
    return objs[0]

def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}: bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials: bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures: bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images: bpy.data.images.remove(image, do_unlink=True)

def load_object(object_path: str) -> None:
    print(f"\n[DEBUG] Opening File: {object_path}")
    if not os.path.exists(object_path):
        print(f"[ERROR] File not found: {object_path}")
        return

    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".blend"):
        with bpy.data.libraries.load(object_path, link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        linked_count = 0
        for obj in data_to.objects:
            if obj is not None:
                if obj.name not in bpy.context.scene.objects:
                    try:
                        bpy.context.collection.objects.link(obj)
                        linked_count += 1
                    except RuntimeError:
                        pass
        print(f"[DEBUG] Linked {linked_count} objects from blend file.")
    else: 
        raise ValueError(f"Unsupported file type: {object_path}")

def create_emission_material(name, color=(1, 1, 1, 1)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs['Color'].default_value = color
    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(emission.outputs[0], out.inputs[0])
    return mat

def create_opaque_material(name, color=(0.5, 0.5, 0.5, 1)):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
    bsdf = nodes.new(type="ShaderNodeBsdfDiffuse")
    bsdf.inputs['Color'].default_value = color
    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(bsdf.outputs[0], out.inputs[0])
    return mat

def create_holdout_material(name="Mat_Holdout"):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
    holdout = nodes.new(type="ShaderNodeHoldout")
    out = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(holdout.outputs[0], out.inputs[0])
    return mat

def setup_compositor_nodes(scene):
    scene.render.use_compositing = True
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()
    
    scene.view_layers[0].use_pass_z = True

    rl_node = nodes.new(type="CompositorNodeRLayers")
    composite_node = nodes.new(type="CompositorNodeComposite") 
    
    mask_out_node = nodes.new(type="CompositorNodeOutputFile")
    mask_out_node.name = "Mask_Output"
    mask_out_node.format.file_format = 'PNG'
    mask_out_node.format.color_mode = 'BW'
    mask_out_node.file_slots[0].path = "mask"

    links.new(rl_node.outputs['Alpha'], mask_out_node.inputs[0])
    
    depth_out.mute = True
    mask_out.mute = True
    return depth_out, mask_out

# ==========================================
# Main Script
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
parser.add_argument("--plane_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--engine", type=str, default="CYCLES")
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--camera_dist_min", type=float, default=1.2)
parser.add_argument("--camera_dist_max", type=float, default=2.2)
parser.add_argument("--elevation", type=float, default=30)
parser.add_argument("--elevation_min", type=float, default=-10)
parser.add_argument("--elevation_max", type=float, default=40)
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--res_w", type=int, default=256)
parser.add_argument("--res_h", type=int, default=256)
parser.add_argument("--hdrs_dir", type=str, default="/home/renchengwei/bpy_render/hdri_bgs")
parser.add_argument("--auto_offset", type=bool, default=False)
parser.add_argument("--normalize_scene", type=bool, default=False)
parser.add_argument("--device", type=str, default='CUDA')
parser.add_argument("--enable_other_objects", type=int, default=0, choices=[0, 1])

parser.add_argument("--camera_type", type=str, default="fixed", choices=["random", "fixed"])
parser.add_argument("--mode", type=str, default="full", choices=["full", "plan", "render"])
parser.add_argument("--plan_path", type=str, default="")
parser.add_argument("--frame_start", type=int, default=0)
parser.add_argument("--frame_end", type=int, default=-1)
parser.add_argument("--worker_id", type=str, default="main")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--plan_validation_mode", type=str, default="sparse", choices=["full", "sparse", "off"])
parser.add_argument("--plan_validation_max_samples", type=int, default=24)
parser.add_argument("--rgb_video_fps", type=int, default=24)
parser.add_argument("--rgb_video_crf", type=int, default=18)
parser.add_argument("--rgb_video_name", type=str, default="rgb_24fps.mp4")
parser.add_argument("--skip_rgb_video", action="store_true")
parser.add_argument("--motion_segments_min", type=int, default=10)
parser.add_argument("--motion_segments_max", type=int, default=24)
parser.add_argument("--motion_segment_len_min", type=int, default=12)
parser.add_argument("--motion_segment_len_max", type=int, default=18)
parser.add_argument("--motion_pose_attempts", type=int, default=120)
parser.add_argument("--motion_collision_samples", type=int, default=7)
parser.add_argument("--motion_xy_step_scale_min", type=float, default=0.08)
parser.add_argument("--motion_xy_step_scale_max", type=float, default=0.24)
parser.add_argument("--motion_free_step_scale_min", type=float, default=0.06)
parser.add_argument("--motion_free_step_scale_max", type=float, default=0.22)
parser.add_argument("--motion_z_drift_scale", type=float, default=0.08)
parser.add_argument("--motion_spin_deg_max", type=float, default=45.0)
parser.add_argument("--motion_table_margin", type=float, default=0.12)
parser.add_argument("--motion_table_min_segment_dist_scale", type=float, default=0.08)
parser.add_argument("--motion_free_min_segment_dist_scale", type=float, default=0.06)

parser.add_argument("--silent_mode", action="store_true")

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

if args.seed >= 0:
    random.seed(args.seed)
    np.random.seed(args.seed)

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 0, 0); cam.data.lens = 35; cam.data.sensor_width = 32
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"; cam_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.res_w; render.resolution_y = args.res_h

scene.cycles.device = "GPU"
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPTIX"
scene.cycles.samples = 512

# ========================================================
# 【新增】消除水体/玻璃噪点的关键设置 (Anti-Firefly Settings)
# ========================================================
scene.cycles.sample_clamp_indirect = 20.
scene.cycles.max_bounces = 32
scene.cycles.transmission_bounces = 32
scene.cycles.transparent_max_bounces = 32
scene.cycles.diffuse_bounces = 8
scene.cycles.glossy_bounces = 16

cprefs = bpy.context.preferences.addons['cycles'].preferences
cprefs.compute_device_type = args.device 
cprefs.get_devices()
print(f"[GPU Setup] Activating {args.device} devices:")
for device in cprefs.devices:
    if device.type == args.device:
        device.use = True
        print(f"  - Activated: {device.name}")

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1)

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def smoothstep(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def shortest_angle_delta(start_angle, end_angle):
    return (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi

def interpolate_pose(start_pose, end_pose, t):
    eased_t = smoothstep(t)
    start_loc, start_rot = start_pose
    end_loc, end_rot = end_pose
    # Keep translation speed perceptually stronger across many short segments.
    cur_loc = start_loc.lerp(end_loc, t)
    cur_rot = Vector((
        start_rot[0] + shortest_angle_delta(start_rot[0], end_rot[0]) * eased_t,
        start_rot[1] + shortest_angle_delta(start_rot[1], end_rot[1]) * eased_t,
        start_rot[2] + shortest_angle_delta(start_rot[2], end_rot[2]) * eased_t,
    ))
    return cur_loc, cur_rot

def get_motion_area_scale(table_bbox_cfg):
    if isinstance(table_bbox_cfg, (float, int)):
        return float(table_bbox_cfg)
    if len(table_bbox_cfg) == 4:
        xmin, ymin, xmax, ymax = table_bbox_cfg
        return max(xmax - xmin, ymax - ymin) * 0.5
    return float(table_bbox_cfg[0])

def build_motion_frame_nodes(total_frames, min_segments, max_segments, min_len, max_len):
    total_steps = max(1, total_frames - 1)
    max_segments = min(max_segments, total_steps)
    min_segments = min(min_segments, max_segments)

    feasible_min_segments = max(1, math.ceil(total_steps / max_len))
    feasible_max_segments = max(1, total_steps // min_len)
    min_segments = max(min_segments, feasible_min_segments)
    max_segments = min(max_segments, feasible_max_segments)

    if min_segments > max_segments:
        min_segments = feasible_min_segments
        max_segments = feasible_max_segments

    for _ in range(200):
        segment_count = random.randint(min_segments, max_segments)
        if segment_count * min_len > total_steps or segment_count * max_len < total_steps:
            continue

        segment_lengths = []
        remaining = total_steps
        for idx in range(segment_count):
            remaining_segments = segment_count - idx
            cur_min = max(min_len, remaining - (remaining_segments - 1) * max_len)
            cur_max = min(max_len, remaining - (remaining_segments - 1) * min_len)
            seg_len = random.randint(cur_min, cur_max)
            segment_lengths.append(seg_len)
            remaining -= seg_len

        frame_nodes = [0]
        cur_frame = 0
        for seg_len in segment_lengths:
            cur_frame += seg_len
            frame_nodes.append(cur_frame)
        return frame_nodes

    frame_nodes = [0]
    for step in range(1, total_steps + 1):
        frame_nodes.append(step)
    return frame_nodes

def pose_for_frame_from_plan(plan, frame_idx):
    keyframes = plan["keyframes"]
    if frame_idx <= keyframes[0]["frame"]:
        return keyframes[0]["loc"].copy(), keyframes[0]["rot"].copy()
    if frame_idx >= keyframes[-1]["frame"]:
        return keyframes[-1]["loc"].copy(), keyframes[-1]["rot"].copy()

    for idx in range(len(keyframes) - 1):
        start_kf = keyframes[idx]
        end_kf = keyframes[idx + 1]
        if start_kf["frame"] <= frame_idx <= end_kf["frame"]:
            local_t = (frame_idx - start_kf["frame"]) / max(1, end_kf["frame"] - start_kf["frame"])
            return interpolate_pose(
                (start_kf["loc"], start_kf["rot"]),
                (end_kf["loc"], end_kf["rot"]),
                local_t,
            )

    return keyframes[-1]["loc"].copy(), keyframes[-1]["rot"].copy()

def apply_pose_to_object(obj_dat, pose):
    root = obj_dat["root"]
    loc, rot = pose
    root.location = loc
    root.rotation_euler = rot

def get_pose_bbox(obj_dat, pose, has_table):
    apply_pose_to_object(obj_dat, pose)
    bpy.context.view_layer.update()
    meshes = obj_dat["meshes"]
    if not meshes:
        return None
    bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
    if has_table:
        return ((bbox_min.x, bbox_min.y), (bbox_max.x, bbox_max.y))
    return (bbox_min.copy(), bbox_max.copy())

def bbox_inside_table(bbox2d, table_bbox_cfg, margin):
    if bbox2d is None:
        return True
    (x_min, y_min), (x_max, y_max) = bbox2d
    if isinstance(table_bbox_cfg, (float, int)) or len(table_bbox_cfg) == 1:
        radius = float(table_bbox_cfg if isinstance(table_bbox_cfg, (float, int)) else table_bbox_cfg[0]) - margin
        if radius <= 0:
            return False
        corners = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
        ]
        return all(x * x + y * y <= radius * radius for x, y in corners)

    xmin, ymin, xmax, ymax = table_bbox_cfg
    return (
        x_min >= xmin + margin and
        y_min >= ymin + margin and
        x_max <= xmax - margin and
        y_max <= ymax - margin
    )

def generate_initial_pose(obj_dat, table_name, table_bbox_cfg):
    is_liquid = obj_dat["is_liquid"]
    if table_name is None:
        radius = table_bbox_cfg if isinstance(table_bbox_cfg, (float, int)) else 0.6
        if is_liquid:
            rot = Vector((0.0, 0.0, random.uniform(0, 2 * math.pi)))
        else:
            rot = Vector((
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
            ))
        loc = Vector((
            random.uniform(-radius, radius),
            random.uniform(-radius, radius),
            random.uniform(-radius, radius),
        ))
    else:
        if len(table_bbox_cfg) == 4:
            xmin, ymin, xmax, ymax = table_bbox_cfg
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
        else:
            radius = table_bbox_cfg[0]
            r = random.uniform(0, radius)
            theta = random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
        loc = Vector((x, y, 0.0))
        if is_liquid:
            rot = Vector((0.0, 0.0, random.uniform(0, 2 * math.pi)))
        else:
            base_x = 0.0 if random.random() < 0.5 else math.pi / 2
            rot = Vector((base_x, 0.0, random.uniform(0, 2 * math.pi)))
    return loc, rot

def try_finalize_pose(obj_dat, raw_pose, table_name, table_bbox_cfg, table_margin):
    loc, rot = raw_pose
    root = obj_dat["root"]
    root.location = loc.copy()
    root.rotation_euler = rot.copy()
    bpy.context.view_layer.update()

    if table_name is not None and obj_dat["meshes"]:
        bbox_min, _ = get_world_bbox_from_meshes(obj_dat["meshes"])
        root.location.z -= bbox_min.z
        bpy.context.view_layer.update()

    finalized_pose = (root.location.copy(), root.rotation_euler.copy())
    bbox = get_pose_bbox(obj_dat, finalized_pose, table_name is not None)
    if table_name is not None and not bbox_inside_table(bbox, table_bbox_cfg, table_margin):
        return None, None
    return finalized_pose, bbox

def sample_next_pose(obj_dat, prev_pose, table_name, table_bbox_cfg, attempt_idx, max_attempts):
    prev_loc, prev_rot = prev_pose
    decay = 1.0 - 0.75 * (attempt_idx / max(1, max_attempts - 1))
    area_scale = get_motion_area_scale(table_bbox_cfg)
    spin_limit = math.radians(args.motion_spin_deg_max) * decay

    if table_name is None:
        step_min = args.motion_free_step_scale_min * area_scale * 0.35
        step_max = args.motion_free_step_scale_max * area_scale * decay
        step_len = random.uniform(step_min, max(step_min, step_max))
        direction = Vector(np.random.normal(size=3).tolist())
        if direction.length == 0:
            direction = Vector((1.0, 0.0, 0.0))
        direction.normalize()
        loc = prev_loc + direction * step_len

        bound = area_scale
        loc.x = clamp(loc.x, -bound, bound)
        loc.y = clamp(loc.y, -bound, bound)
        z_bound = max(bound * args.motion_z_drift_scale, 0.25)
        loc.z = clamp(loc.z, -z_bound, z_bound)

        if obj_dat["is_liquid"]:
            rot = Vector((prev_rot[0], prev_rot[1], prev_rot[2] + random.uniform(-spin_limit, spin_limit)))
        else:
            rot = Vector((
                prev_rot[0] + random.uniform(-spin_limit * 0.3, spin_limit * 0.3),
                prev_rot[1] + random.uniform(-spin_limit * 0.3, spin_limit * 0.3),
                prev_rot[2] + random.uniform(-spin_limit, spin_limit),
            ))
    else:
        step_min = args.motion_xy_step_scale_min * area_scale * 0.4
        step_max = args.motion_xy_step_scale_max * area_scale * decay
        step_len = random.uniform(step_min, max(step_min, step_max))
        theta = random.uniform(0, 2 * math.pi)
        loc = prev_loc.copy()
        loc.x += step_len * math.cos(theta)
        loc.y += step_len * math.sin(theta)
        loc.z = prev_loc.z

        if obj_dat["is_liquid"]:
            rot = Vector((prev_rot[0], prev_rot[1], prev_rot[2] + random.uniform(-spin_limit * 0.6, spin_limit * 0.6)))
        else:
            rot = Vector((prev_rot[0], prev_rot[1], prev_rot[2] + random.uniform(-spin_limit, spin_limit)))

    return loc, rot

def get_min_segment_motion_distance(obj_dat, table_name, table_bbox_cfg):
    area_scale = get_motion_area_scale(table_bbox_cfg)
    if table_name is None:
        base_scale = args.motion_free_min_segment_dist_scale
    else:
        base_scale = args.motion_table_min_segment_dist_scale

    if obj_dat["is_liquid"]:
        base_scale *= 0.7

    return area_scale * base_scale

def build_segment_sample_frames(start_frame, end_frame, sample_count):
    if end_frame <= start_frame:
        return [start_frame]
    samples = {start_frame, end_frame}
    for idx in range(1, max(1, sample_count) - 1):
        alpha = idx / (sample_count - 1)
        frame_idx = int(round(start_frame + (end_frame - start_frame) * alpha))
        samples.add(frame_idx)
    return sorted(samples)

def is_pose_bbox_collision_free(candidate_bbox, other_bboxes, has_table):
    if candidate_bbox is None:
        return True
    overlap_fn = bbox_overlap_2d if has_table else bbox_overlap_3d
    return not any(overlap_fn(candidate_bbox, other_bbox) for other_bbox in other_bboxes if other_bbox is not None)

def segment_candidate_is_valid(obj_dat, planned_motion_plans, start_pose, end_pose, start_frame, end_frame, table_name, table_bbox_cfg):
    has_table = table_name is not None
    sample_frames = build_segment_sample_frames(start_frame, end_frame, args.motion_collision_samples)
    for frame_idx in sample_frames:
        local_t = (frame_idx - start_frame) / max(1, end_frame - start_frame)
        cur_pose = interpolate_pose(start_pose, end_pose, local_t)

        other_bboxes = []
        for other_plan in planned_motion_plans:
            other_pose = pose_for_frame_from_plan(other_plan, frame_idx)
            other_bbox = get_pose_bbox(other_plan["obj_dat"], other_pose, has_table)
            other_bboxes.append(other_bbox)

        candidate_bbox = get_pose_bbox(obj_dat, cur_pose, has_table)
        if has_table and not bbox_inside_table(candidate_bbox, table_bbox_cfg, args.motion_table_margin):
            return False
        if not is_pose_bbox_collision_free(candidate_bbox, other_bboxes, has_table):
            return False
    return True

def plan_motion_for_object(obj_dat, planned_motion_plans, frame_nodes, table_name, table_bbox_cfg):
    has_table = table_name is not None
    start_pose = None

    for attempt_idx in range(args.motion_pose_attempts):
        raw_pose = generate_initial_pose(obj_dat, table_name, table_bbox_cfg)
        finalized_pose, candidate_bbox = try_finalize_pose(
            obj_dat,
            raw_pose,
            table_name,
            table_bbox_cfg,
            args.motion_table_margin,
        )
        if finalized_pose is None:
            continue

        other_bboxes = []
        for other_plan in planned_motion_plans:
            other_pose = pose_for_frame_from_plan(other_plan, frame_nodes[0])
            other_bbox = get_pose_bbox(other_plan["obj_dat"], other_pose, has_table)
            other_bboxes.append(other_bbox)

        if is_pose_bbox_collision_free(candidate_bbox, other_bboxes, has_table):
            start_pose = finalized_pose
            break

    if start_pose is None:
        return None

    keyframes = [{
        "frame": frame_nodes[0],
        "loc": start_pose[0].copy(),
        "rot": start_pose[1].copy(),
    }]

    for segment_idx in range(1, len(frame_nodes)):
        prev_pose = (keyframes[-1]["loc"].copy(), keyframes[-1]["rot"].copy())
        start_frame = frame_nodes[segment_idx - 1]
        end_frame = frame_nodes[segment_idx]
        next_pose = None
        min_motion_dist = get_min_segment_motion_distance(obj_dat, table_name, table_bbox_cfg)

        for attempt_idx in range(args.motion_pose_attempts):
            raw_pose = sample_next_pose(
                obj_dat,
                prev_pose,
                table_name,
                table_bbox_cfg,
                attempt_idx,
                args.motion_pose_attempts,
            )
            finalized_pose, _ = try_finalize_pose(
                obj_dat,
                raw_pose,
                table_name,
                table_bbox_cfg,
                args.motion_table_margin,
            )
            if finalized_pose is None:
                continue

            motion_dist = (finalized_pose[0] - prev_pose[0]).length
            relax = 1.0 - 0.55 * (attempt_idx / max(1, args.motion_pose_attempts - 1))
            effective_min_motion_dist = min_motion_dist * relax
            if motion_dist < max(1e-3, effective_min_motion_dist):
                continue

            if segment_candidate_is_valid(
                obj_dat,
                planned_motion_plans,
                prev_pose,
                finalized_pose,
                start_frame,
                end_frame,
                table_name,
                table_bbox_cfg,
            ):
                next_pose = finalized_pose
                break

        if next_pose is None:
            return None

        keyframes.append({
            "frame": end_frame,
            "loc": next_pose[0].copy(),
            "rot": next_pose[1].copy(),
        })

    return {
        "obj_dat": obj_dat,
        "keyframes": keyframes,
    }

def set_camera_location(cam_pt):
    x, y, z = cam_pt
    bpy.data.objects["Camera"].location = x, y, z
    return bpy.data.objects["Camera"]

def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix()); t = np.asarray(location)
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64)
    R = R.T; t = -R @ t
    return np.concatenate([cam_rec @ R, (cam_rec @ t)[:,None]], 1)

def get_calibration_matrix_K_from_blender(camera):
    f_mm = camera.data.lens
    scale = scene.render.resolution_percentage / 100
    w_mm = camera.data.sensor_width; h_mm = camera.data.sensor_height
    res_x = scene.render.resolution_x; res_y = scene.render.resolution_y
    pixel_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camera.data.sensor_fit == 'VERTICAL':
        s_u = res_x * scale / w_mm / pixel_aspect; s_v = res_y * scale / h_mm
    else:
        s_u = res_x * scale / w_mm; s_v = res_y * scale * pixel_aspect / h_mm
    return np.asarray(((f_mm*s_u, 0, res_x*scale/2), (0, f_mm*s_u, res_y*scale/2), (0, 0, 1)), np.float64)

def calculate_occlusion_ratio(mask_path, visib_mask_path):
    try:
        img_mask = bpy.data.images.load(str(mask_path))
        img_visib = bpy.data.images.load(str(visib_mask_path))
        pixels_mask = np.array(img_mask.pixels[:])[0::4]
        pixels_visib = np.array(img_visib.pixels[:])[0::4]
        bpy.data.images.remove(img_mask)
        bpy.data.images.remove(img_visib)
        count_mask = np.count_nonzero(pixels_mask > 0.5)
        count_visib = np.count_nonzero(pixels_visib > 0.5)
        if count_mask == 0: return 1.0, None, None
        occlusion_ratio = 1.0 - (count_visib / count_mask)
        return occlusion_ratio, np.array(pixels_mask > 0.5), np.array(pixels_visib > 0.5)
    except Exception as e:
        print(f"[Warning] Failed to calculate occlusion: {e}")
        return 1.0, None, None 

def rename_output(folder, prefix, index, ext=".png"):
    target_pattern = f"{prefix}*{ext}" 
    dst = folder / f"{prefix}{ext}"
    for _ in range(20): 
        candidates = list(folder.glob(target_pattern))
        if candidates:
            src = candidates[0] 
            if src.name == dst.name: return
            if dst.exists(): os.remove(dst)
            os.rename(src, dst)
            return
        time.sleep(0.1)
    print(f"[Error] File missing! Expected pattern: {folder}/{target_pattern}")
    print(f"        Files found in dir: {[f.name for f in folder.glob('*'+ext)]}")

def clean_before_render(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    base = str(file_path).replace('.png', '').replace('.exr', '')
    ext = '.png' if str(file_path).endswith('.png') else '.exr'
    artifact = f"{base}0001{ext}"
    if os.path.exists(artifact):
        os.remove(artifact)

def vector_to_list(vec):
    return [float(v) for v in vec]

def sanitize_worker_tag(worker_id):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(worker_id))

def list_available_object_assets(object_dir):
    supported_exts = {".glb", ".fbx", ".blend"}
    object_dir = Path(object_dir)
    if not object_dir.exists():
        return []
    assets = []
    for entry in sorted(object_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in supported_exts:
            continue
        if entry.name in fixed_obj_names_list:
            continue
        assets.append(entry.name)
    return assets

def generate_fixed_object_randomization(fixed_obj_name):
    rand_cfg = {}
    if fixed_obj_name == "tube_water.blend":
        rand_cfg["water_height"] = random.uniform(0.0, 0.065)
        rand_cfg["water_target_name"] = "tube.002"
    return rand_cfg

def apply_fixed_object_randomization(objects, fixed_obj_name, rand_cfg):
    if fixed_obj_name == "tube_water.blend" and "water_height" in rand_cfg:
        randomize_water_height(
            objects,
            target_name=rand_cfg.get("water_target_name", "tube.002"),
            min_scale=0.0,
            max_scale=0.065,
            height_value=rand_cfg["water_height"],
        )

def choose_scene_assets():
    table_name = random.choice(TABLE_LIST)
    fixed_obj_name = random.choice(fixed_obj_names_list)
    available_objects = list_available_object_assets(args.object_path) if args.enable_other_objects else []
    num_objs = min(random.randint(0, 2), len(available_objects))
    extra_objects = random.sample(available_objects, num_objs)
    hdr_candidates = sorted([p.name for p in Path(args.hdrs_dir).iterdir() if p.is_file()])
    if not hdr_candidates:
        raise RuntimeError(f"No HDR files found in {args.hdrs_dir}")

    obj_names = [fixed_obj_name] + extra_objects
    scene_id_prefix = str(table_name) if table_name else "no_table"
    scene_id = f"{scene_id_prefix}__{'_'.join(obj_names)}"
    return {
        "table_name": table_name,
        "fixed_obj_name": fixed_obj_name,
        "obj_names": obj_names,
        "hdr_name": random.choice(hdr_candidates),
        "bg_strength": random.uniform(0.8, 2.5),
        "fixed_obj_randomization": generate_fixed_object_randomization(fixed_obj_name),
        "num_images": args.num_images,
        "scene_id": scene_id,
    }

def reserve_output_scene_dir(out_dir_path, scene_id):
    out_dir_scene = out_dir_path / scene_id
    if not out_dir_scene.exists():
        out_dir_scene.mkdir(parents=True, exist_ok=True)
        return out_dir_scene
    for idx in range(1000):
        candidate = Path(str(out_dir_scene) + f"_{idx}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(f"Failed to reserve output directory for scene {scene_id}")

def serialize_motion_plans(motion_plans):
    serialized = []
    for plan in motion_plans:
        serialized.append({
            "keyframes": [
                {
                    "frame": int(kf["frame"]),
                    "loc": vector_to_list(kf["loc"]),
                    "rot": vector_to_list(kf["rot"]),
                }
                for kf in plan["keyframes"]
            ]
        })
    return serialized

def deserialize_motion_plans(serialized_plans, loaded_scene_objects_data):
    motion_plans = []
    for obj_dat, plan_dict in zip(loaded_scene_objects_data, serialized_plans):
        motion_plans.append({
            "obj_dat": obj_dat,
            "keyframes": [
                {
                    "frame": int(kf["frame"]),
                    "loc": Vector(kf["loc"]),
                    "rot": Vector(kf["rot"]),
                }
                for kf in plan_dict["keyframes"]
            ],
        })
    return motion_plans

def write_plan_file(plan_path, plan_dict):
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f, indent=2, ensure_ascii=False)

def load_plan_file(plan_path):
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_render_bundle():
    reset_scene()

    mat_opaque = create_opaque_material("Mat_Opaque", color=(0.5, 0.5, 0.5, 1))
    mat_white = create_emission_material("Mat_White", color=(1, 1, 1, 1))
    mat_holdout = create_holdout_material("Mat_Holdout")

    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    rl_node = nodes.new(type="CompositorNodeRLayers")
    composite_node = nodes.new(type="CompositorNodeComposite")

    mask_out_node = nodes.new(type="CompositorNodeOutputFile")
    mask_out_node.name = "Mask_Output"
    mask_out_node.format.file_format = 'PNG'
    mask_out_node.format.color_mode = 'BW'
    mask_out_node.file_slots[0].path = "mask"
    links.new(rl_node.outputs['Alpha'], mask_out_node.inputs[0])
    mask_out_node.mute = True

    return {
        "mat_opaque": mat_opaque,
        "mat_white": mat_white,
        "mat_holdout": mat_holdout,
        "tree": tree,
        "nodes": nodes,
        "links": links,
        "rl_node": rl_node,
        "composite_node": composite_node,
        "mask_out_node": mask_out_node,
    }

def prepare_scene_bundle(scene_cfg, out_dir_scene):
    render_bundle = create_render_bundle()

    table_name = scene_cfg["table_name"]
    fixed_obj_name = scene_cfg["fixed_obj_name"]
    obj_names = scene_cfg["obj_names"]

    table_glb = os.path.join(args.plane_path, table_name) if table_name is not None else None
    table_objs = []
    if table_glb is not None:
        objs_before_table = get_scene_objects_set()
        load_object(table_glb)
        bpy.context.view_layer.update()
        table_objs = get_new_objects(objs_before_table)

    table_bbox_cfg = TABLE_CONFIG.get(table_name, [-5, -5, 5, 5])
    obj_glbs = [os.path.join(args.object_path, n) for n in obj_names]

    fixed_obj_parts = []
    other_obj_parts = []
    fixed_obj_root_ref = None
    loaded_scene_objects_data = []

    print(f"[Compose] Table: {table_name} | Objects: {obj_names}")

    for obj_glb in obj_glbs:
        old_objs = get_scene_objects_set()
        try:
            load_object(obj_glb)
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            continue

        bpy.context.view_layer.update()
        current_new_objs = get_new_objects(old_objs)
        if not current_new_objs:
            if fixed_obj_name in obj_glb:
                raise RuntimeError(f"Failed to load fixed object: {obj_glb}")
            continue

        root = find_root_object(current_new_objs)
        meshes = get_meshes_from_objects(current_new_objs)

        for o in current_new_objs:
            if o.animation_data:
                o.animation_data_clear()
            if o.rotation_mode != 'XYZ':
                o.rotation_mode = 'XYZ'

        bpy.context.view_layer.objects.active = root
        root.select_set(True)
        try:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        except Exception as e:
            print(f"[Warning] Failed to apply rotation to {root.name}: {e}")
        root.select_set(False)

        is_target = fixed_obj_name in obj_glb
        if is_target:
            print(f"  [Info] Identified target: {obj_glb}")
            print("  --- Inspecting Modifiers ---")
            for o in current_new_objs:
                check_and_apply_solidify(o)
            print("  ----------------------------")
            apply_fixed_object_randomization(current_new_objs, fixed_obj_name, scene_cfg.get("fixed_obj_randomization", {}))
            fixed_obj_parts.extend(current_new_objs)
            fixed_obj_root_ref = root
        else:
            other_obj_parts.extend(current_new_objs)

        loaded_scene_objects_data.append({
            "root": root,
            "meshes": meshes,
            "is_liquid": is_target and ("water" in fixed_obj_name),
            "all_objs": current_new_objs,
        })

    if fixed_obj_root_ref is None:
        raise RuntimeError("Fixed object root not found after scene load")

    world = bpy.context.scene.world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_nodes.clear()
    env_tex = world_nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(os.path.join(args.hdrs_dir, scene_cfg["hdr_name"]))
    bg = world_nodes.new(type="ShaderNodeBackground")
    bg.inputs["Strength"].default_value = scene_cfg["bg_strength"]
    out = world_nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    print(f"Background strength: {scene_cfg['bg_strength']}")

    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    render_bundle["mask_out_node"].base_path = str(out_dir_scene)

    global_obj_mat_backup = {}
    all_meshes_in_scene = [o for o in (fixed_obj_parts + table_objs + other_obj_parts) if o.type == 'MESH' and o.data]
    for o in all_meshes_in_scene:
        global_obj_mat_backup[o] = [slot.material for slot in o.material_slots]
    print(f"[System] Created non-destructive material backup for {len(global_obj_mat_backup)} objects.")

    scene_bundle = {
        **render_bundle,
        "table_name": table_name,
        "table_bbox_cfg": table_bbox_cfg,
        "obj_names": obj_names,
        "fixed_obj_name": fixed_obj_name,
        "table_objs": table_objs,
        "fixed_obj_parts": fixed_obj_parts,
        "other_obj_parts": other_obj_parts,
        "fixed_obj_root_ref": fixed_obj_root_ref,
        "loaded_scene_objects_data": loaded_scene_objects_data,
        "global_obj_mat_backup": global_obj_mat_backup,
        "out_dir_scene": Path(out_dir_scene),
    }
    return scene_bundle

def set_camera_from_pose(camera_pose):
    cam_pt = az_el_to_points(np.array([camera_pose["az"]]), np.array([camera_pose["el"]])) * camera_pose["dist"]
    cam_pt = cam_pt[0]
    scene.frame_set(1)
    camera = set_camera_location(cam_pt)
    bpy.context.view_layer.update()
    return camera

def clear_validation_outputs(validation_dir, frame_idx):
    for suffix in ("mask", "mask_visib"):
        target = validation_dir / f"{frame_idx:03d}_{suffix}.png"
        if target.exists():
            target.unlink()

def downsample_sorted_frames(frame_indices, max_samples):
    if len(frame_indices) <= max_samples:
        return frame_indices
    sampled = []
    last_idx = len(frame_indices) - 1
    for sample_idx in range(max_samples):
        src_idx = int(round(sample_idx * last_idx / max(1, max_samples - 1)))
        sampled.append(frame_indices[src_idx])
    return sorted(set(sampled))

def build_validation_frame_indices(motion_plans, total_frames):
    if args.plan_validation_mode == "off":
        return []
    if args.plan_validation_mode == "full":
        return list(range(total_frames))

    frame_indices = {0, max(0, total_frames - 1)}
    for plan in motion_plans:
        keyframes = plan["keyframes"]
        for idx, kf in enumerate(keyframes):
            frame_indices.add(int(kf["frame"]))
            if idx + 1 < len(keyframes):
                next_frame = int(keyframes[idx + 1]["frame"])
                mid_frame = int(round((int(kf["frame"]) + next_frame) * 0.5))
                frame_indices.add(mid_frame)

    frame_indices = sorted(frame_indices)
    return downsample_sorted_frames(frame_indices, max(2, args.plan_validation_max_samples))

def render_mask_pair(scene_bundle, frame_idx, worker_tag, output_dir):
    links = scene_bundle["links"]
    rl_node = scene_bundle["rl_node"]
    composite_node = scene_bundle["composite_node"]
    mask_out_node = scene_bundle["mask_out_node"]
    mat_white = scene_bundle["mat_white"]
    mat_holdout = scene_bundle["mat_holdout"]
    table_objs = scene_bundle["table_objs"]
    fixed_obj_parts = scene_bundle["fixed_obj_parts"]
    other_obj_parts = scene_bundle["other_obj_parts"]
    global_obj_mat_backup = scene_bundle["global_obj_mat_backup"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_out_node.base_path = str(output_dir)

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"

    for o in table_objs:
        o.hide_render = True
    for o in other_obj_parts:
        o.hide_render = True
    for o in fixed_obj_parts:
        o.hide_render = False
    scene.view_layers[0].material_override = mat_white
    bpy.context.view_layer.update()

    links.clear()
    links.new(rl_node.outputs['Alpha'], composite_node.inputs['Image'])
    links.new(rl_node.outputs['Alpha'], mask_out_node.inputs[0])
    mask_out_node.mute = False

    temp_prefix = f"w{worker_tag}_{frame_idx:03d}_mask_tmp"
    temp_junk = output_dir / f"{temp_prefix}.png"
    scene.render.filepath = str(temp_junk)

    final_mask_path = output_dir / f"{frame_idx:03d}_mask.png"
    clear_validation_outputs(output_dir, frame_idx)
    clean_before_render(final_mask_path)
    mask_out_node.file_slots[0].path = f"{frame_idx:03d}_mask"

    if args.silent_mode:
        with suppress_output():
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(write_still=True)
    rename_output(output_dir, f"{frame_idx:03d}_mask", frame_idx, ".png")
    if temp_junk.exists():
        temp_junk.unlink()

    scene.view_layers[0].material_override = None
    bpy.context.view_layer.update()

    target_set = set(fixed_obj_parts)
    pass4_objs = [o for o in (fixed_obj_parts + table_objs + other_obj_parts) if o.type == 'MESH' and o.data]
    objs_with_added_slots = []

    for o in pass4_objs:
        o.hide_render = False
        target_mat = mat_white if o in target_set else mat_holdout
        if len(o.material_slots) == 0:
            o.data.materials.append(target_mat)
            objs_with_added_slots.append(o)
        else:
            for slot in o.material_slots:
                slot.material = target_mat

    scene.render.film_transparent = True
    mask_out_node.mute = False
    temp_visib_prefix = f"w{worker_tag}_{frame_idx:03d}_mask_visib_tmp"
    temp_junk_visib = output_dir / f"{temp_visib_prefix}.png"
    scene.render.filepath = str(temp_junk_visib)
    final_mask_visib_path = output_dir / f"{frame_idx:03d}_mask_visib.png"
    clean_before_render(final_mask_visib_path)
    mask_out_node.file_slots[0].path = f"{frame_idx:03d}_mask_visib"

    if args.silent_mode:
        with suppress_output():
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(write_still=True)
    rename_output(output_dir, f"{frame_idx:03d}_mask_visib", frame_idx, ".png")
    if temp_junk_visib.exists():
        temp_junk_visib.unlink()

    for o in objs_with_added_slots:
        o.data.materials.clear()
    for o, original_mats in global_obj_mat_backup.items():
        if o in pass4_objs and len(o.material_slots) == len(original_mats):
            for idx, mat in enumerate(original_mats):
                o.material_slots[idx].material = mat

    scene.view_layers[0].material_override = None
    bpy.context.view_layer.update()
    scene.render.film_transparent = False
    mask_out_node.base_path = str(scene_bundle["out_dir_scene"])

    return final_mask_path, final_mask_visib_path

def validate_motion_plan(scene_bundle, motion_plans, camera_pose, worker_tag="planner"):
    validation_frames = build_validation_frame_indices(motion_plans, args.num_images)
    if not validation_frames:
        print("  [Plan] Validation skipped by configuration.")
        return True

    print(
        f"  [Plan] Validation mode={args.plan_validation_mode}, "
        f"frames={len(validation_frames)}/{args.num_images}"
    )
    camera = set_camera_from_pose(camera_pose)
    validation_dir = scene_bundle["out_dir_scene"] / "_plan_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in tqdm(validation_frames, desc="Plan validation"):
        for plan in motion_plans:
            cur_pose = pose_for_frame_from_plan(plan, frame_idx)
            apply_pose_to_object(plan["obj_dat"], cur_pose)
        bpy.context.view_layer.update()

        mask_path, visib_mask_path = render_mask_pair(scene_bundle, frame_idx, worker_tag, validation_dir)
        occ_ratio, count_mask, count_visib = calculate_occlusion_ratio(mask_path, visib_mask_path)
        whole_obj_visib_ratio = 0.0
        if count_visib is not None:
            whole_obj_visib_ratio = count_visib.sum() / count_visib.shape[0]

        if whole_obj_visib_ratio < whole_obj_visib_ratio_threshold:
            print(f"  [Reject] View {frame_idx} occluded. Replanning... (VisRatio: {whole_obj_visib_ratio:.4f})")
            return False

    for path in validation_dir.glob("*"):
        try:
            path.unlink()
        except Exception:
            pass
    try:
        validation_dir.rmdir()
    except Exception:
        pass
    print("  [Success] Motion plan passed mask-based validation.")
    return True

def build_scene_plan(out_dir_path):
    base_scene_cfg = choose_scene_assets()
    out_dir_scene = reserve_output_scene_dir(out_dir_path, base_scene_cfg["scene_id"])
    scene_bundle = prepare_scene_bundle(base_scene_cfg, out_dir_scene)

    sequence_attempt_count = 0
    while True:
        sequence_attempt_count += 1
        print(f"\n--- Sequence Attempt #{sequence_attempt_count} ---")

        camera_pose = {
            "dist": random.uniform(args.camera_dist_min, args.camera_dist_max),
            "az": random.uniform(0, 2 * np.pi),
            "el": np.deg2rad(random.uniform(args.elevation_min, args.elevation_max)),
        }

        frame_nodes = build_motion_frame_nodes(
            args.num_images,
            args.motion_segments_min,
            args.motion_segments_max,
            args.motion_segment_len_min,
            args.motion_segment_len_max,
        )
        print(f"  [Motion] Frame nodes: {frame_nodes}")

        planned_motion_plans = []
        valid_plan = True
        for obj_dat in scene_bundle["loaded_scene_objects_data"]:
            motion_plan = plan_motion_for_object(
                obj_dat,
                planned_motion_plans,
                frame_nodes,
                scene_bundle["table_name"],
                scene_bundle["table_bbox_cfg"],
            )
            if motion_plan is None:
                valid_plan = False
                break
            planned_motion_plans.append(motion_plan)
            print(
                f"  [Motion] Planned {obj_dat['root'].name} with "
                f"{len(motion_plan['keyframes']) - 1} segments."
            )

        if not valid_plan:
            print("  [Setup Failed] Could not generate collision-free long trajectory. Retrying sequence...")
            continue

        if validate_motion_plan(scene_bundle, planned_motion_plans, camera_pose, worker_tag="planner"):
            plan_dict = {
                "version": 1,
                "num_images": args.num_images,
                "scene_id": out_dir_scene.name,
                "output_scene_dir": str(out_dir_scene),
                "table_name": base_scene_cfg["table_name"],
                "fixed_obj_name": base_scene_cfg["fixed_obj_name"],
                "obj_names": base_scene_cfg["obj_names"],
                "hdr_name": base_scene_cfg["hdr_name"],
                "bg_strength": base_scene_cfg["bg_strength"],
                "fixed_obj_randomization": base_scene_cfg["fixed_obj_randomization"],
                "camera_pose": camera_pose,
                "motion_plans": serialize_motion_plans(planned_motion_plans),
            }
            return plan_dict

def render_full_frame(scene_bundle, camera, frame_idx, worker_tag, plan_dict):
    links = scene_bundle["links"]
    rl_node = scene_bundle["rl_node"]
    composite_node = scene_bundle["composite_node"]
    mask_out_node = scene_bundle["mask_out_node"]
    out_dir_scene = scene_bundle["out_dir_scene"]
    mat_opaque = scene_bundle["mat_opaque"]
    mat_white = scene_bundle["mat_white"]
    mat_holdout = scene_bundle["mat_holdout"]
    table_objs = scene_bundle["table_objs"]
    fixed_obj_parts = scene_bundle["fixed_obj_parts"]
    other_obj_parts = scene_bundle["other_obj_parts"]
    global_obj_mat_backup = scene_bundle["global_obj_mat_backup"]
    fixed_obj_root_ref = scene_bundle["fixed_obj_root_ref"]

    world2obj_matrix = np.array(fixed_obj_root_ref.matrix_world.inverted())
    RT = get_3x4_RT_matrix_from_blender(camera)
    RT_4x4 = np.concatenate([RT, np.zeros((1, 4))], 0)
    RT_4x4[-1, -1] = 1
    obj2cam = RT_4x4 @ np.linalg.inv(world2obj_matrix)

    scene.view_layers[0].material_override = None
    bpy.context.view_layer.update()
    for o in table_objs:
        o.hide_render = False
    for o in fixed_obj_parts:
        o.hide_render = False
    for o in other_obj_parts:
        o.hide_render = False

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False

    links.clear()
    links.new(rl_node.outputs['Image'], composite_node.inputs['Image'])
    links.new(rl_node.outputs['Alpha'], mask_out_node.inputs[0])
    mask_out_node.mute = True

    f_rgb_path = out_dir_scene / f"{frame_idx:03d}.png"
    scene.render.filepath = str(f_rgb_path)
    clean_before_render(f_rgb_path)

    if args.silent_mode:
        with suppress_output():
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(write_still=True)

    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.view_layers[0].use_pass_z = True
    scene.render.use_compositing = True

    scene.view_layers[0].material_override = mat_opaque
    bpy.context.view_layer.update()
    scene.render.film_transparent = True

    links.clear()
    links.new(rl_node.outputs['Depth'], composite_node.inputs['Image'])
    mask_out_node.mute = True

    depth_temp_prefix = f"w{worker_tag}_{frame_idx:03d}_depth_tmp"
    temp_depth_exr = out_dir_scene / f"{depth_temp_prefix}.exr"
    scene.render.filepath = str(temp_depth_exr)
    clean_before_render(temp_depth_exr)

    if args.silent_mode:
        with suppress_output():
            bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(write_still=True)
    rename_output(out_dir_scene, depth_temp_prefix, frame_idx, ".exr")
    final_depth_path = out_dir_scene / f"{frame_idx:03d}_depth.exr"
    depth_intermediate = out_dir_scene / f"{depth_temp_prefix}.exr"
    if depth_intermediate.exists():
        if final_depth_path.exists():
            final_depth_path.unlink()
        os.rename(depth_intermediate, final_depth_path)

    scene.view_layers[0].material_override = None
    bpy.context.view_layer.update()

    mask_path, visib_mask_path = render_mask_pair(scene_bundle, frame_idx, worker_tag, out_dir_scene)

    K = get_calibration_matrix_K_from_blender(camera)
    np.savez(
        out_dir_scene / f"{frame_idx:03d}.npz",
        table=str(scene_bundle["table_name"]),
        objects=scene_bundle["obj_names"],
        K=K,
        RT=RT,
        obj2cam_poses=obj2cam,
        azimuth=plan_dict["camera_pose"]["az"],
        elevation=plan_dict["camera_pose"]["el"],
        distance=plan_dict["camera_pose"]["dist"],
    )

    return {
        "rgb": f_rgb_path,
        "depth": final_depth_path,
        "mask": mask_path,
        "mask_visib": visib_mask_path,
    }

def resolve_frame_range(total_frames):
    frame_start = max(0, args.frame_start)
    frame_end = total_frames - 1 if args.frame_end < 0 else min(args.frame_end, total_frames - 1)
    if frame_end < frame_start:
        raise ValueError(f"Invalid frame range: start={frame_start}, end={frame_end}")
    return frame_start, frame_end

def all_rgb_frames_exist(out_dir_scene, total_frames):
    for frame_idx in range(total_frames):
        if not (out_dir_scene / f"{frame_idx:03d}.png").exists():
            return False
    return True

def maybe_encode_rgb_video(plan_dict, out_dir_scene):
    if args.skip_rgb_video:
        return

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("[Video] Skip RGB video encoding because ffmpeg is not available.")
        return

    total_frames = int(plan_dict["num_images"])
    if not all_rgb_frames_exist(out_dir_scene, total_frames):
        print("[Video] RGB frames are not complete yet. Skip video encoding for now.")
        return

    lock_path = out_dir_scene / ".rgb_video_encode.lock"
    video_path = out_dir_scene / args.rgb_video_name

    lock_fd = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, f"{os.getpid()}\n".encode("utf-8"))
        os.close(lock_fd)
        lock_fd = None
    except FileExistsError:
        print("[Video] Another worker is already encoding the RGB video.")
        return
    except Exception as e:
        print(f"[Video] Failed to acquire encode lock: {e}")
        return

    try:
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate", str(args.rgb_video_fps),
            "-i", str(out_dir_scene / "%03d.png"),
            "-c:v", "libx264",
            "-crf", str(args.rgb_video_crf),
            "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        print(f"[Video] Encoding RGB video to {video_path}")
        subprocess.run(ffmpeg_cmd, check=True)
        print("[Video] RGB video encoding completed.")
    except subprocess.CalledProcessError as e:
        print(f"[Video] ffmpeg failed with exit code {e.returncode}")
    except Exception as e:
        print(f"[Video] Failed to encode RGB video: {e}")
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                pass
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass

def render_from_plan(plan_dict):
    out_dir_scene = Path(plan_dict["output_scene_dir"]).resolve()
    out_dir_scene.mkdir(parents=True, exist_ok=True)
    scene_bundle = prepare_scene_bundle(plan_dict, out_dir_scene)
    motion_plans = deserialize_motion_plans(plan_dict["motion_plans"], scene_bundle["loaded_scene_objects_data"])
    camera = set_camera_from_pose(plan_dict["camera_pose"])

    frame_start, frame_end = resolve_frame_range(plan_dict["num_images"])
    worker_tag = sanitize_worker_tag(args.worker_id)
    print(f"[Render] Worker {worker_tag} rendering frames {frame_start}..{frame_end} into {out_dir_scene}")

    for frame_idx in tqdm(range(frame_start, frame_end + 1), desc=f"Worker {worker_tag}"):
        for plan in motion_plans:
            cur_pose = pose_for_frame_from_plan(plan, frame_idx)
            apply_pose_to_object(plan["obj_dat"], cur_pose)
        bpy.context.view_layer.update()
        render_full_frame(scene_bundle, camera, frame_idx, worker_tag, plan_dict)

    maybe_encode_rgb_video(plan_dict, out_dir_scene)

def save_images():
    out_dir_path = Path(args.output_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if args.mode == "render":
        if not args.plan_path:
            raise ValueError("--plan_path is required in render mode")
        plan_dict = load_plan_file(Path(args.plan_path).resolve())
        render_from_plan(plan_dict)
        return

    plan_dict = build_scene_plan(out_dir_path)
    default_plan_path = Path(plan_dict["output_scene_dir"]) / "scene_plan.json"
    plan_path = Path(args.plan_path).resolve() if args.plan_path else default_plan_path
    write_plan_file(plan_path, plan_dict)
    print(f"[Plan] Wrote scene plan to {plan_path}")

    if args.mode == "plan":
        return

    render_from_plan(plan_dict)

if __name__ == "__main__":
    save_images()
