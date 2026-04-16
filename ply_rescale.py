from plyfile import PlyData, PlyElement

# ply_file = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000001.ply"
# ply_file = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000001_original_scale.ply"
ply_file = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000002_original_scale.ply"
# save_file = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000001_rescaled.ply"
save_file = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/models/obj_000002.ply"

global_scale = 20.

plydata = PlyData.read(ply_file)

faces = plydata['face'].data
bad = [f for f in faces if len(f[0]) != 3]

print("Total faces:", len(faces))
print("Non-triangle faces:", len(bad))


plydata["vertex"]["x"] = plydata["vertex"]["x"] / global_scale
plydata["vertex"]["y"] = plydata["vertex"]["y"] / global_scale
plydata["vertex"]["z"] = plydata["vertex"]["z"] / global_scale
plydata.write(save_file)