import argparse
import json
import math
import os
import random
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

def randomize_water_height(objects, target_name="tube.002", min_scale=0.0, max_scale=0.084):
    found = False
    new_scale = random.uniform(min_scale, max_scale)
    
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
fixed_obj_names = random.choice(fixed_obj_names_list)
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

# 【修改点】 增加 "continuous" 模式
parser.add_argument("--camera_type", type=str, default="random", choices=["random", "fixed", "continuous"])

parser.add_argument("--silent_mode", action="store_true")

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

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

def save_images():
    out_dir_path = Path(args.output_dir).resolve() 
    os.makedirs(out_dir_path, exist_ok=True)
    
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

    table_name = random.choice(TABLE_LIST)
    
    if table_name is not None:
        table_glb = os.path.join(args.plane_path, table_name)
    else:
        table_glb = None
        args.elevation_min = -15

    available_objects = OBJECT_LIST if args.enable_other_objects else []
    num_objs = random.randint(0, 2)
    num_objs = min(num_objs, len(available_objects))
    obj_names = random.sample(available_objects, num_objs)
    obj_names = [fixed_obj_names] + obj_names
    obj_glbs = [os.path.join(args.object_path, n) for n in obj_names]

    print(f"[Compose] Table: {table_name} | Objects: {obj_names}")

    objs_before_table = get_scene_objects_set()
    table_objs = []
    
    if table_glb is not None:
        load_object(table_glb)
        bpy.context.view_layer.update()
        table_objs = get_new_objects(objs_before_table)
    
    table_bbox_cfg = TABLE_CONFIG.get(table_name, [-5, -5, 5, 5])
    
    placed_bboxes = []
    fixed_obj_parts = []
    other_obj_parts = []
    fixed_obj_root_ref = None

    # 【修改点】新建列表用于存储加载的物体信息
    loaded_scene_objects_data = []

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
            if fixed_obj_names in obj_glb: return
            continue

        bpy.context.view_layer.update()
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
        
        is_target = fixed_obj_names in obj_glb
        
        if is_target:
            print(f"  [Info] Identified target: {obj_glb}")
            
            print("  --- Inspecting Modifiers ---")
            for o in current_new_objs:
                check_and_apply_solidify(o)
            print("  ----------------------------")
            
            for func, params in fixed_obj_rand_proc_func[fixed_obj_names].items():
                func(current_new_objs, *params)
            fixed_obj_parts.extend(current_new_objs)
            fixed_obj_root_ref = root
        else:
            other_obj_parts.extend(current_new_objs)

        is_liquid_obj = is_target and ("water" in fixed_obj_names)
        loaded_scene_objects_data.append({
            "root": root,
            "meshes": meshes,
            "is_liquid": is_liquid_obj,
            "all_objs": current_new_objs
        })
    
    if fixed_obj_root_ref is None: return

    # Environment Setup
    world = bpy.context.scene.world; world.use_nodes = True
    nodes = world.node_tree.nodes; nodes.clear()
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(os.path.join(args.hdrs_dir, random.choice(os.listdir(args.hdrs_dir))))
    bg_strength = random.uniform(0.8, 2.5)
    print(f"Background strength: {bg_strength}")
    bg = nodes.new(type="ShaderNodeBackground"); bg.inputs["Strength"].default_value = bg_strength
    out = nodes.new(type="ShaderNodeOutputWorld")
    world.node_tree.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])

    empty = bpy.data.objects.new("Empty", None); bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    scene_id_prefix = str(table_name) if table_name else "no_table"
    scene_id = f"{scene_id_prefix}__{'_'.join(obj_names)}"
    out_dir_scene = out_dir_path / scene_id
    if not os.path.exists(out_dir_scene):
        out_dir_scene.mkdir(parents=True, exist_ok=True)
    else:
        for i in range(1000):
            new_out_dir_scene = Path(str(out_dir_scene) + f"_{i}")
            if not os.path.exists(new_out_dir_scene):
                new_out_dir_scene.mkdir(parents=True, exist_ok=True)
                out_dir_scene = new_out_dir_scene
                break
    print(f"Output directory: {out_dir_scene}")
    mask_out_node.base_path = str(out_dir_scene)

    global_obj_mat_backup = {}
    all_meshes_in_scene = [o for o in (fixed_obj_parts + table_objs + other_obj_parts) if o.type == 'MESH' and o.data]
    for o in all_meshes_in_scene:
        original_mats = [slot.material for slot in o.material_slots]
        global_obj_mat_backup[o] = original_mats
    
    print(f"[System] Created non-destructive material backup for {len(global_obj_mat_backup)} objects.")

    # ----------------------------------------------------
    # Camera Mode Setup
    # ----------------------------------------------------
    if args.camera_type == "random":
        pass 
    elif args.camera_type == "fixed":
        # Fixed Pose
        fix_dist = args.camera_dist
        fix_az = 0.0 
        fix_el = np.deg2rad(args.elevation)
    elif args.camera_type == "continuous":
        # Continuous Trajectory Setup (Start & End)
        # Random Start Pose
        traj_dist_start = random.uniform(args.camera_dist_min, args.camera_dist_max)
        traj_az_start = random.uniform(0, 2*np.pi)
        traj_el_start = np.deg2rad(random.uniform(args.elevation_min, args.elevation_max))
        
        # Random End Pose (with constrained azimuth rotation)
        traj_dist_end = random.uniform(args.camera_dist_min, args.camera_dist_max)
        # Limit azimuth rotation to +/- 120 degrees
        # traj_az_end = traj_az_start + random.uniform(-math.pi/1.5, math.pi/1.5)
        # Limit elevation rotation to +/- 180 degrees
        traj_az_end = traj_az_start + random.uniform(-math.pi/1., math.pi/1.)
        traj_el_end = np.deg2rad(random.uniform(args.elevation_min, args.elevation_max))
        
        print(f"[Continuous Trajectory] Generating smooth path...")

    # ========================================================
    # 【核心逻辑】全局重试循环 (Global Retry Loop)
    # ========================================================
    # 如果整个渲染序列中有一张图被遮挡，就重新随机摆放物体，从头开始渲染。
    
    sequence_attempt_count = 0
    
    while True:
        sequence_attempt_count += 1
        print(f"--- Sequence Attempt #{sequence_attempt_count} ---")
        
        # 1. 每一轮尝试都要重置结果容器
        valid_cam_poses = []
        valid_obj2cam_poses = []
        valid_azimuths = []
        valid_elevations = []
        valid_distances = []
        
        # 2. 随机摆放物体 (Place Objects)
        placed_bboxes = [] 
        for obj_dat in loaded_scene_objects_data:
            root = obj_dat['root']
            meshes = obj_dat['meshes']
            is_liquid_obj = obj_dat['is_liquid']
            
            placed = False
            for _ in range(100):
                if table_name is None:
                    # --- 无桌子 ---
                    if is_liquid_obj:
                        root.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
                    else:
                        root.rotation_euler = (
                            random.uniform(0, 2 * math.pi),
                            random.uniform(0, 2 * math.pi),
                            random.uniform(0, 2 * math.pi)
                        )

                    radius = table_bbox_cfg if isinstance(table_bbox_cfg, (float, int)) else 0.6
                    rx = random.uniform(-radius, radius)
                    ry = random.uniform(-radius, radius)
                    rz = random.uniform(-radius, radius)
                    root.location = (rx, ry, rz)
                    
                    bpy.context.view_layer.update()
                    
                    if meshes:
                        bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
                        bbox3d = (bbox_min, bbox_max)
                        if not any(bbox_overlap_3d(bbox3d, b) for b in placed_bboxes):
                            placed_bboxes.append(bbox3d)
                            placed = True
                            break
                    else:
                        placed = True
                        break
                        
                else:
                    # --- 有桌子 ---
                    if len(table_bbox_cfg) == 4:
                        xmin, ymin, xmax, ymax = table_bbox_cfg
                        x = random.uniform(xmin, xmax); y = random.uniform(ymin, ymax)
                    else:
                        R = table_bbox_cfg[0]
                        r = random.uniform(0, R); theta = random.uniform(0, 2 * math.pi)
                        x = r * math.cos(theta); y = r * math.sin(theta)
                    
                    root.location = (x, y, 0.0)
                    
                    if is_liquid_obj:
                        root.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
                    else:
                        if random.random() < 0.5:
                            root.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
                        else:
                            root.rotation_euler = (math.pi / 2, 0, random.uniform(0, 2 * math.pi))
                    
                    bpy.context.view_layer.update()
                    
                    if meshes:
                        bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
                        root.location.z = -bbox_min.z
                        bpy.context.view_layer.update()
                        bbox_min, bbox_max = get_world_bbox_from_meshes(meshes)
                        bbox2d = ((bbox_min.x, bbox_min.y), (bbox_max.x, bbox_max.y))
                        if not any(bbox_overlap_2d(bbox2d, b) for b in placed_bboxes):
                            placed_bboxes.append(bbox2d)
                            placed = True
                            break
                    else:
                        placed = True
                        break
        
        # 3. 计算位姿矩阵 (物体定好后，矩阵就定了)
        world2obj_matrix = np.array(fixed_obj_root_ref.matrix_world.inverted())

        # ================= RENDER LOOP (Inner) =================
        # 这个循环负责渲染所有帧。如果中途失败，break 掉，回到外层 while 继续重试。
        sequence_failed = False
        
        for i in tqdm(range(args.num_images), desc=f"Attempt {sequence_attempt_count}"):
            
            # 2. 计算当前帧的相机位置
            if args.camera_type == "random":
                dist = random.uniform(args.camera_dist_min, args.camera_dist_max)
                az = random.uniform(0, 2*np.pi)
                el = np.deg2rad(random.uniform(args.elevation_min, args.elevation_max))
            elif args.camera_type == "fixed":
                dist = fix_dist
                az = fix_az
                el = fix_el
            elif args.camera_type == "continuous":
                # 根据进度 t 进行线性插值 (LERP)
                t = i / max(1, args.num_images - 1)
                dist = traj_dist_start * (1 - t) + traj_dist_end * t
                az = traj_az_start * (1 - t) + traj_az_end * t
                el = traj_el_start * (1 - t) + traj_el_end * t

            cam_pt = az_el_to_points(np.array([az]), np.array([el])) * dist
            cam_pt = cam_pt[0]

            scene.frame_set(1)
            camera = set_camera_location(cam_pt)
            bpy.context.view_layer.update()

            # 3. 重新计算相对位姿矩阵
            RT = get_3x4_RT_matrix_from_blender(camera)
            RT_4x4 = np.concatenate([RT, np.zeros((1, 4))], 0); RT_4x4[-1, -1] = 1
            obj2cam = RT_4x4 @ np.linalg.inv(world2obj_matrix)

            # PASS 1: RGB
            scene.view_layers[0].material_override = None 
            bpy.context.view_layer.update()
            for o in table_objs: o.hide_render = False
            for o in fixed_obj_parts: o.hide_render = False
            for o in other_obj_parts: o.hide_render = False
            
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGBA"
            scene.render.film_transparent = False 
            
            links.clear()
            links.new(rl_node.outputs['Image'], composite_node.inputs['Image'])
            links.new(rl_node.outputs['Alpha'], mask_out_node.inputs[0])
            mask_out_node.mute = True 
            
            f_rgb_path = str(out_dir_scene / f"{i:03d}.png")
            scene.render.filepath = f_rgb_path
            
            clean_before_render(f_rgb_path)
            
            if args.silent_mode:
                with suppress_output(): bpy.ops.render.render(write_still=True)
            else:
                bpy.ops.render.render(write_still=True)

            # PASS 2: Depth
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
            
            f_depth = f"{i:03d}_depth"
            temp_depth_exr = out_dir_scene / "temp_depth_main.exr"
            scene.render.filepath = str(temp_depth_exr)
            clean_before_render(temp_depth_exr)
            
            if args.silent_mode:
                with suppress_output(): bpy.ops.render.render(write_still=True)
            else:
                bpy.ops.render.render(write_still=True)
            rename_output(out_dir_scene, "temp_depth_main", i, ".exr")
            final_depth_path = out_dir_scene / f"{f_depth}.exr"
            if (out_dir_scene / "temp_depth_main.exr").exists():
                    os.rename(out_dir_scene / "temp_depth_main.exr", final_depth_path)

            scene.view_layers[0].material_override = None
            bpy.context.view_layer.update()

            # PASS 3: Original Mask
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "BW"
            
            for o in table_objs: o.hide_render = True
            for o in other_obj_parts: o.hide_render = True
            for o in fixed_obj_parts: o.hide_render = False
            scene.view_layers[0].material_override = mat_white
            bpy.context.view_layer.update()
            
            links.clear()
            links.new(rl_node.outputs['Alpha'], mask_out_node.inputs[0])
            mask_out_node.mute = False
            
            f_mask = f"{i:03d}_mask"
            mask_out_node.file_slots[0].path = f_mask
            
            temp_junk = out_dir_scene / "junk_pass3.png"
            scene.render.filepath = str(temp_junk)
            
            final_mask_path = out_dir_scene / f"{f_mask}.png"
            clean_before_render(final_mask_path)
            
            if args.silent_mode:
                with suppress_output(): bpy.ops.render.render(write_still=True)
            else:
                bpy.ops.render.render(write_still=True)
            rename_output(out_dir_scene, f_mask, i, ".png")
            if temp_junk.exists(): os.remove(temp_junk)

            scene.view_layers[0].material_override = None
            bpy.context.view_layer.update()

            # PASS 4: Visible Mask
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
            f_mask_visib = f"{i:03d}_mask_visib"
            mask_out_node.file_slots[0].path = f_mask_visib
            
            temp_junk_visib = out_dir_scene / "junk_pass4.png"
            scene.render.filepath = str(temp_junk_visib)
            
            final_mask_visib_path = out_dir_scene / f"{f_mask_visib}.png"
            clean_before_render(final_mask_visib_path)
            
            if args.silent_mode:
                with suppress_output(): bpy.ops.render.render(write_still=True)
            else:
                bpy.ops.render.render(write_still=True)
            rename_output(out_dir_scene, f_mask_visib, i, ".png")
            if temp_junk_visib.exists(): os.remove(temp_junk_visib)
            
            for o in objs_with_added_slots:
                o.data.materials.clear()
            for o, original_mats in global_obj_mat_backup.items():
                if o in pass4_objs and len(o.material_slots) == len(original_mats):
                    for idx, mat in enumerate(original_mats):
                        o.material_slots[idx].material = mat

            scene.view_layers[0].material_override = None
            bpy.context.view_layer.update()
            
            # Occlusion Check
            mask_path = out_dir_scene / f"{f_mask}.png"
            visib_mask_path = out_dir_scene / f"{f_mask_visib}.png"
            
            occ_ratio, count_mask, count_visib = calculate_occlusion_ratio(mask_path, visib_mask_path)
            
            whole_obj_visib_ratio = 0
            if count_visib is not None:
                whole_obj_visib_ratio = count_visib.sum() / count_visib.shape[0]
            
            if whole_obj_visib_ratio < whole_obj_visib_ratio_threshold:
                print(f"  [Reject] View {i} occluded. Restarting entire sequence... (VisRatio: {whole_obj_visib_ratio:.4f})")
                
                # 【关键逻辑】 只要有一帧被遮挡，整个序列作废，标记 failure，跳出 for 循环
                sequence_failed = True
                
                # 可选：清理当前这一帧的垃圾文件
                files_to_clean = [Path(f_rgb_path), final_depth_path, mask_path, visib_mask_path]
                for f in files_to_clean:
                    if f.exists():
                        try: os.remove(f)
                        except: pass
                
                break # Break out of 'for' loop, back to 'while' loop
            else:
                valid_cam_poses.append(RT)
                valid_obj2cam_poses.append(obj2cam)
                valid_azimuths.append(az)
                valid_elevations.append(el)
                valid_distances.append(dist)
                # Continue to next frame
        
        # 检查 sequence 是否成功
        if not sequence_failed:
            print("  [Success] Full sequence rendered without occlusion.")
            break # 跳出 while True，结束渲染
        else:
            # 失败了，外层 while 会继续，重新随机摆放物体
            pass

    K = get_calibration_matrix_K_from_blender(camera)
    for i in range(args.num_images):
        # 确保数据完整才保存 (理论上 break while 时肯定是完整的)
        if i < len(valid_cam_poses):
            np.savez(
                out_dir_scene / f"{i:03d}.npz", 
                table=str(table_name), 
                objects=obj_names, 
                K=K, 
                RT=valid_cam_poses[i], 
                obj2cam_poses=valid_obj2cam_poses[i], 
                azimuth=valid_azimuths[i], 
                elevation=valid_elevations[i], 
                distance=valid_distances[i]
            )

if __name__ == "__main__":
    save_images()
