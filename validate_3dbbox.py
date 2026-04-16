import cv2
import numpy as np
import mmcv
import sys
sys.path.append("/mnt/afs/TransparentObjectPose/tools")
from visualize_3d_bbox import BBox3DVisualizer


def demo_example():
    """
    演示示例：展示如何使用BBox3DVisualizer
    """
    # 相机内参（示例）
    # K = np.array([
    #     [572.4114, 0, 325.2611],
    #     [0, 573.57043, 242.04899],
    #     [0, 0, 1]
    # ])
    K = np.array([[700.0, 0, 320.0],
                  [0, 700.0, 240.0],
                  [0, 0, 1]])
    
    # 创建可视化器
    visualizer = BBox3DVisualizer(K)
    
    # 示例1: 使用边界框尺寸绘制
    # image = np.zeros((480, 640, 3), dtype=np.uint8)  # 创建空白图像
    # image = cv2.imread(f"/home/renchengwei/bpy_render/views_tube_fixed_cam/no_table__tube.blend_2/007.png")
    # image = cv2.imread(f"/home/renchengwei/bpy_render/views_tube_vid/plane_round_table.glb__tube.blend_0/010.png")
    # image = cv2.imread("/home/renchengwei/bpy_render/views_tube_1/no_table__tube.blend_2/001.png")
    image = cv2.imread(f"/home/renchengwei/bpy_render/views_tube_vid/plane_furn.glb__tube.blend/005.png")
    # image = cv2.imread(f"/share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/rgb/000014/000005.png")
    
    # 6D姿态
    # R = np.eye(3)  # 单位旋转矩阵
    # T = np.array([0, 0, 0.5])  # 平移向量（米）
    # R_t_json = mmcv.load("/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/scene_gt.json")
    # R_t_npz = np.load("/home/renchengwei/bpy_render/views_tube_fixed_cam/no_table__tube.blend_2/007.npz")
    # R_t_npz = np.load("/home/renchengwei/bpy_render/views_tube_vid/plane_round_table.glb__tube.blend_0/010.npz")
    # R_t_npz = np.load("/home/renchengwei/bpy_render/views_tube_1/no_table__tube.blend_2/001.npz")
    R_t_npz = np.load("/home/renchengwei/bpy_render/views_tube_vid/plane_furn.glb__tube.blend/005.npz")
    # R_t_npz = np.load("/share/volumes/csi/renchengwei/BOP_DATASETS/labsim/train/000002/npz/000014/000005.npz")
    R_t = R_t_npz["obj2cam_poses"]
    R = R_t[:3, :3]
    scale_x = np.linalg.norm(R[0, :])
    scale_y = np.linalg.norm(R[1, :])
    scale_z = np.linalg.norm(R[2, :])
    scale = np.array([scale_x, scale_y, scale_z])
    R = R / scale
    T = R_t[:3, 3] / 20.
    # T = R_t[:3, 3]
    print(R, T)
    
    # add some noise to the T and R
    # T = T + np.random.normal(0, 0.03, 3)
    T[2] = T[2] #+ np.random.normal(0, 0.03)
    # R = add_rotation_noise(R, sigma_deg=5.0)
    
    # 物体尺寸（米）
    # size = [0.1, 0.1, 0.1]
    # size = [159.70 / 1000.0, 90.00 / 1000.0, 106.87 / 1000.0]
    # size = [66.46975708007812 / 1000.,
    #   66.50539016723633 / 1000.,
    #   239.7760009765625 / 1000.]
    size = [-33.172504 / 1000.0, -33.315807 / 1000.0, -128.711121 / 1000.0, 66.469757 / 1000.0, 66.50539 / 1000.0, 239.776001 / 1000.0]
    
    # 绘制3D框
    image_with_bbox = visualizer.draw_from_size(image, size, R, T, thickness=2)
    
    print("演示完成！")
    print("使用方法:")
    print("1. 从模型文件: visualizer.draw_from_model(image, model_path, R, T)")
    print("2. 从尺寸: visualizer.draw_from_size(image, size, R, T)")
    print("3. 从角点: visualizer.draw_from_corners(image, corners_3d, R, T)")
    
    return image_with_bbox

if __name__ == "__main__":
    image_with_bbox = demo_example()
    cv2.imwrite("output_example4_bop.png", image_with_bbox)