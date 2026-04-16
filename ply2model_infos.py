import json
import numpy as np
import trimesh
import open3d as o3d


def compute_diameter(vertices, max_points=5000):
    """
    BOP-style diameter: max Euclidean distance between any two points
    """
    if len(vertices) > max_points:
        idx = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[idx]

    max_dist = 0.0
    for i in range(len(vertices)):
        dists = np.linalg.norm(vertices[i] - vertices, axis=1)
        max_dist = max(max_dist, dists.max())
    return float(max_dist)


# def ply_to_info(ply_path):
#     mesh = trimesh.load(ply_path, process=False)

#     # 处理 Scene 情况
#     if isinstance(mesh, trimesh.Scene):
#         mesh = trimesh.util.concatenate(mesh.dump())

#     vertices = np.asarray(mesh.vertices) # already in mm

#     min_xyz = vertices.min(axis=0)
#     max_xyz = vertices.max(axis=0)
#     size_xyz = max_xyz - min_xyz

#     info = {
#         "diameter": round(compute_diameter(vertices), 6),
#         "min_x": round(float(min_xyz[0]), 6),
#         "min_y": round(float(min_xyz[1]), 6),
#         "min_z": round(float(min_xyz[2]), 6),
#         "size_x": round(float(size_xyz[0]), 6),
#         "size_y": round(float(size_xyz[1]), 6),
#         "size_z": round(float(size_xyz[2]), 6),
#     }

#     return info


def ply_to_info(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    size = maxs - mins
    diameter = np.linalg.norm(size)

    # return {
    #     "min": mins.tolist(),
    #     "max": maxs.tolist(),
    #     "size": size.tolist(),
    #     "diameter": float(diameter),
    # }
    
    info = {
        "diameter": round(compute_diameter(pts), 6),
        "min_x": round(float(mins[0]), 6),
        "min_y": round(float(mins[1]), 6),
        "min_z": round(float(mins[2]), 6),
        "size_x": round(float(size[0]), 6),
        "size_y": round(float(size[1]), 6),
        "size_z": round(float(size[2]), 6),
    }
    return info


# =========================
# 直接在这里填路径
# =========================
# ply_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000001.ply"
ply_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000002.ply"
# ply_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000002_original_scale.ply"
# ply_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/tube_thick_vA_Blender.ply"
# ply_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000001_original_scale.ply"
# ply_path = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000001_rescaled.ply"
obj_id = "1"  # 你想用的 object id

result = {
    obj_id: ply_to_info(ply_path)
}

# 打印
print(json.dumps(result, indent=2))