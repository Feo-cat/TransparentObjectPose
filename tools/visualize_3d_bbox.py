#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化工具：在图片上绘制3D边界框
给定6D物体姿态（旋转矩阵R和平移向量T），在输入图片上绘制物体的3D边界框

使用示例：
    # 方法1: 使用3D模型文件
    visualizer = BBox3DVisualizer(camera_K)
    image_with_bbox = visualizer.draw_from_model(image, model_path, R, T)
    
    # 方法2: 直接提供3D边界框尺寸
    visualizer = BBox3DVisualizer(camera_K)
    image_with_bbox = visualizer.draw_from_size(image, size=[0.1, 0.1, 0.1], R, T)
    
    # 方法3: 直接提供8个3D角点
    visualizer = BBox3DVisualizer(camera_K)
    corners_3d = np.array([...])  # (8, 3)
    image_with_bbox = visualizer.draw_from_corners(image, corners_3d, R, T)
"""

import cv2
import numpy as np
import mmcv
from typing import Union, Tuple, Optional
import os.path as osp


def get_3D_corners(pts):
    """
    从3D点云计算3D边界框的8个角点
    
    Args:
        pts: (N, 3) numpy array of 3D points
    
    Returns:
        corners: (8, 3) numpy array of 8 corner points
        
    8个角点的顺序:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    min_z = np.min(pts[:, 2])
    max_z = np.max(pts[:, 2])
    
    corners = np.array([
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
    ])
    
    return corners


def get_3D_corners_from_size(size):
    """
    从边界框尺寸创建3D边界框的8个角点（中心在原点）
    
    Args:
        size: list or array of [min_x, min_y, min_z, size_x, size_y, size_z]
    
    Returns:
        corners: (8, 3) numpy array of 8 corner points
    """
    min_x, min_y, min_z, size_x, size_y, size_z = size
    max_x = min_x + size_x
    max_y = min_y + size_y
    max_z = min_z + size_z
    # half_w, half_h, half_d = w / 2, h / 2, d / 2
    
    # corners = np.array([
    #     [half_w, half_h, half_d],
    #     [-half_w, half_h, half_d],
    #     [-half_w, -half_h, half_d],
    #     [half_w, -half_h, half_d],
    #     [half_w, half_h, -half_d],
    #     [-half_w, half_h, -half_d],
    #     [-half_w, -half_h, -half_d],
    #     [half_w, -half_h, -half_d],
    # ])
    
    corners = np.array([
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
    ])
    
    return corners


def points_to_2D(points, R, T, K):
    """
    将3D点投影到2D图像平面
    
    Args:
        points: (N, 3) numpy array of 3D points
        R: (3, 3) rotation matrix
        T: (3,) translation vector
        K: (3, 3) camera intrinsic matrix
    
    Returns:
        points_2D: (N, 2) projected 2D points
        z: (N,) depth values
    """
    # 将点从物体坐标系转换到相机坐标系
    points_in_camera = np.matmul(R, points.T) + T.reshape((3, 1))  # (3, N)
    
    # 投影到图像平面
    points_in_image = np.matmul(K, points_in_camera)  # (3, N)
    
    N = points_in_camera.shape[1]
    points_2D = np.zeros((2, N))
    points_2D[0, :] = points_in_image[0, :] / (points_in_image[2, :] + 1e-15)
    points_2D[1, :] = points_in_image[1, :] / (points_in_image[2, :] + 1e-15)
    z = points_in_camera[2, :]
    
    return points_2D.T, z


def colormap(rgb=False, maximum=255):
    """
    生成颜色映射表
    
    Args:
        rgb: 如果为True，返回RGB顺序；否则返回BGR顺序
        maximum: 颜色的最大值（255或1.0）
    
    Returns:
        colors: list of colors
    """
    colors = [
        [31, 119, 180],
        [255, 127, 14],
        [46, 160, 44],
        [214, 40, 39],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [126, 126, 126],
        [188, 189, 34],
        [26, 190, 207],
    ]
    
    if not rgb:
        colors = [[c[2], c[1], c[0]] for c in colors]
    
    if maximum != 255:
        colors = [[c[0] / 255.0 * maximum, c[1] / 255.0 * maximum, c[2] / 255.0 * maximum] for c in colors]
    
    return colors


def draw_projected_box3d(image, qs, color=(255, 0, 255), middle_color=None, bottom_color=None, thickness=2, 
                         depth=None, draw_orientation=True):
    """
    在图像上绘制投影的3D边界框（支持深度遮挡判断）
    
    Args:
        image: numpy array, 输入图像
        qs: (8, 2) numpy array, 投影后的8个角点的2D坐标
        color: 顶面的颜色 (B, G, R)
        middle_color: 垂直边的颜色，如果为None则使用colormap
        bottom_color: 底面的颜色，如果为None则使用colormap
        thickness: 线条粗细
        depth: (8,) numpy array, 8个角点的深度值（Z坐标），用于判断遮挡关系
        draw_orientation: 是否绘制方向指示（前面的边加粗）
    
    Returns:
        image: 绘制了3D框的图像
        
    8个角点的顺序:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    qs = qs.astype(np.int32)
    color = mmcv.color_val(color)  # 顶面颜色
    colors = colormap(rgb=False, maximum=255)
    
    # 定义12条边：每条边由两个顶点索引定义
    edges = [
        # 顶面 (0-1-2-3)
        (0, 1), (1, 2), (2, 3), (3, 0),
        # 底面 (4-5-6-7)
        (4, 5), (5, 6), (6, 7), (7, 4),
        # 垂直边
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # 如果提供了深度信息，则根据深度判断哪些边在前面
    if depth is not None:
        # 计算每条边的平均深度
        edge_depths = []
        for i, j in edges:
            avg_depth = (depth[i] + depth[j]) / 2.0
            edge_depths.append(avg_depth)
        
        # 找出中位深度
        median_depth = np.median(edge_depths)
        
        # 先绘制后面的边（虚线，半透明）
        for idx, (i, j) in enumerate(edges):
            if edge_depths[idx] > median_depth:  # 后面的边
                # 确定颜色
                if idx < 4:  # 顶面
                    edge_color = color
                elif idx < 8:  # 底面
                    k = idx - 4
                    if bottom_color is None:
                        edge_color = tuple(int(_c) for _c in colors[k % len(colors)])
                    else:
                        edge_color = tuple(int(_c) for _c in mmcv.color_val(bottom_color))
                else:  # 垂直边
                    k = idx - 8
                    if middle_color is None:
                        edge_color = tuple(int(_c) for _c in colors[k % len(colors)])
                    else:
                        edge_color = tuple(int(_c) for _c in mmcv.color_val(middle_color))
                
                # 绘制虚线（后面的边）
                draw_dashed_line(image, qs[i], qs[j], edge_color, thickness)
        
        # 再绘制前面的边（实线）
        for idx, (i, j) in enumerate(edges):
            if edge_depths[idx] <= median_depth:  # 前面的边
                # 确定颜色
                if idx < 4:  # 顶面
                    edge_color = color
                elif idx < 8:  # 底面
                    k = idx - 4
                    if bottom_color is None:
                        edge_color = tuple(int(_c) for _c in colors[k % len(colors)])
                    else:
                        edge_color = tuple(int(_c) for _c in mmcv.color_val(bottom_color))
                else:  # 垂直边
                    k = idx - 8
                    if middle_color is None:
                        edge_color = tuple(int(_c) for _c in colors[k % len(colors)])
                    else:
                        edge_color = tuple(int(_c) for _c in mmcv.color_val(middle_color))
                
                # 绘制实线
                cv2.line(image, tuple(qs[i]), tuple(qs[j]), edge_color, thickness, cv2.LINE_AA)
    else:
        # 没有深度信息，使用原来的绘制方法
        for k in range(0, 4):
            # 绘制底面 (4-5-6-7)
            i, j = k + 4, (k + 1) % 4 + 4
            if bottom_color is None:
                _bottom_color = tuple(int(_c) for _c in colors[k % len(colors)])
            else:
                _bottom_color = tuple(int(_c) for _c in mmcv.color_val(bottom_color))
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _bottom_color, thickness, cv2.LINE_AA)
            
            # 绘制垂直边 (连接顶面和底面)
            i, j = k, k + 4
            if middle_color is None:
                _middle_color = tuple(int(_c) for _c in colors[k % len(colors)])
            else:
                _middle_color = tuple(int(_c) for _c in mmcv.color_val(middle_color))
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _middle_color, thickness, cv2.LINE_AA)
            
            # 绘制顶面 (0-1-2-3)
            i, j = k, (k + 1) % 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    
    return image


def draw_dashed_line(image, pt1, pt2, color, thickness=1, gap=10):
    """
    绘制虚线
    
    Args:
        image: 图像
        pt1: 起点 (x, y)
        pt2: 终点 (x, y)
        color: 颜色
        thickness: 线条粗细
        gap: 虚线间隔
    """
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))
    
    for i in range(0, len(pts) - 1, 2):
        cv2.line(image, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


class BBox3DVisualizer:
    """
    3D边界框可视化类
    """
    
    def __init__(self, camera_K):
        """
        初始化可视化器
        
        Args:
            camera_K: (3, 3) numpy array, 相机内参矩阵
        """
        self.K = camera_K
    
    def draw_from_corners(self, image, corners_3d, R, T, 
                         color=(255, 0, 255), middle_color=None, bottom_color=None, 
                         thickness=2, copy=True, use_depth=True):
        """
        从3D角点绘制3D边界框
        
        Args:
            image: numpy array, 输入图像
            corners_3d: (8, 3) numpy array, 物体坐标系下的8个角点
            R: (3, 3) numpy array, 旋转矩阵
            T: (3,) numpy array, 平移向量
            color: 顶面颜色
            middle_color: 垂直边颜色
            bottom_color: 底面颜色
            thickness: 线条粗细
            copy: 是否复制图像（True则不修改原图）
            use_depth: 是否使用深度信息判断遮挡关系（推荐True）
        
        Returns:
            image_with_bbox: 绘制了3D框的图像
        """
        if copy:
            image = image.copy()
        
        # 将3D角点投影到2D，并获取深度信息
        corners_2d, depth = points_to_2D(corners_3d, R, T, self.K)
        
        # 绘制3D框（传入深度信息）
        image = draw_projected_box3d(image, corners_2d, color=color, 
                                    middle_color=middle_color, 
                                    bottom_color=bottom_color, 
                                    thickness=thickness,
                                    depth=depth if use_depth else None)
        
        return image
    
    def draw_from_size(self, image, size, R, T,
                      color=(255, 0, 255), middle_color=None, bottom_color=None,
                      thickness=2, copy=True, use_depth=True):
        """
        从边界框尺寸绘制3D边界框
        
        Args:
            image: numpy array, 输入图像
            size: list or array, 边界框尺寸 [min_x, min_y, min_z, size_x, size_y, size_z]
            R: (3, 3) numpy array, 旋转矩阵
            T: (3,) numpy array, 平移向量
            color: 顶面颜色
            middle_color: 垂直边颜色
            bottom_color: 底面颜色
            thickness: 线条粗细
            copy: 是否复制图像
            use_depth: 是否使用深度信息判断遮挡关系
        
        Returns:
            image_with_bbox: 绘制了3D框的图像
        """
        corners_3d = get_3D_corners_from_size(size)
        return self.draw_from_corners(image, corners_3d, R, T, color, 
                                     middle_color, bottom_color, thickness, copy, use_depth)
    
    def draw_from_model(self, image, model_path, R, T,
                       color=(255, 0, 255), middle_color=None, bottom_color=None,
                       thickness=2, copy=True, vertex_scale=1.0, use_depth=True):
        """
        从3D模型文件绘制3D边界框
        
        Args:
            image: numpy array, 输入图像
            model_path: str, 3D模型文件路径 (.ply格式)
            R: (3, 3) numpy array, 旋转矩阵
            T: (3,) numpy array, 平移向量
            color: 顶面颜色
            middle_color: 垂直边颜色
            bottom_color: 底面颜色
            thickness: 线条粗细
            copy: 是否复制图像
            vertex_scale: 顶点缩放因子（某些数据集需要缩放，如BOP数据集通常为0.001）
            use_depth: 是否使用深度信息判断遮挡关系
        
        Returns:
            image_with_bbox: 绘制了3D框的图像
        """
        # 加载3D模型
        from lib.pysixd import inout
        model = inout.load_ply(model_path, vertex_scale=vertex_scale)
        pts_3d = model["pts"]
        
        # 计算3D边界框角点
        corners_3d = get_3D_corners(pts_3d)
        
        return self.draw_from_corners(image, corners_3d, R, T, color,
                                     middle_color, bottom_color, thickness, copy, use_depth)
    
    def draw_multiple(self, image, corners_list, R_list, T_list,
                     colors=None, thickness=2, copy=True, use_depth=True):
        """
        绘制多个3D边界框
        
        Args:
            image: numpy array, 输入图像
            corners_list: list of (8, 3) arrays, 多个物体的角点
            R_list: list of (3, 3) arrays, 旋转矩阵列表
            T_list: list of (3,) arrays, 平移向量列表
            colors: list of colors, 每个框的颜色，如果为None则使用不同颜色
            thickness: 线条粗细
            copy: 是否复制图像
            use_depth: 是否使用深度信息判断遮挡关系
        
        Returns:
            image_with_bboxes: 绘制了多个3D框的图像
        """
        if copy:
            image = image.copy()
        
        if colors is None:
            color_list = colormap(rgb=False, maximum=255)
            colors = [color_list[i % len(color_list)] for i in range(len(corners_list))]
        
        for i, (corners_3d, R, T) in enumerate(zip(corners_list, R_list, T_list)):
            color = colors[i] if i < len(colors) else (255, 0, 255)
            image = self.draw_from_corners(image, corners_3d, R, T, 
                                          color=color, thickness=thickness, 
                                          copy=False, use_depth=use_depth)
        
        return image

# rotation noise utils
def skew(w):
    """ w: (3,) """
    wx, wy, wz = w
    return np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])

def so3_exp(w):
    """
    Rodrigues' formula
    w: (3,)
    return: (3,3) rotation matrix
    """
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)

    k = w / theta
    K = skew(k)

    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )

def add_rotation_noise(R, sigma_deg=1.0):
    """
    R: (3,3) rotation matrix
    sigma_deg: std of rotation noise (degrees)
    """
    sigma = np.deg2rad(sigma_deg)
    delta_w = np.random.randn(3) * sigma
    R_noise = so3_exp(delta_w)
    return R_noise @ R




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
    # idx = 977
    # idx = 1012
    idx = 1376
    # image = cv2.imread(f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/rgb/{idx:06d}.png")
    image = cv2.imread(f"/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/rgb/{idx:06d}.png")
    
    # 6D姿态
    # R = np.eye(3)  # 单位旋转矩阵
    # T = np.array([0, 0, 0.5])  # 平移向量（米）
    # R_t_json = mmcv.load("/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/scene_gt.json")
    R_t_json = mmcv.load("/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000002/scene_gt.json")
    R_t_json = R_t_json[str(idx)][0]
    R = np.array(R_t_json["cam_R_m2c"]).reshape(3, 3)
    T = np.array(R_t_json["cam_t_m2c"])# / 1000.0
    
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


def visualize_example(K=None, image=None, RT=None, size=None, output_path=None):
    """
    演示示例：展示如何使用BBox3DVisualizer
    """
    # 相机内参（示例）
    # K = np.array([
    #     [572.4114, 0, 325.2611],
    #     [0, 573.57043, 242.04899],
    #     [0, 0, 1]
    # ])
    if K is None:
        K = np.array([[700.0, 0, 320.0],
                      [0, 700.0, 240.0],
                      [0, 0, 1]])
    
    # 创建可视化器
    visualizer = BBox3DVisualizer(K)
    
    
    # 6D姿态
    R, T = RT[:3, :3], RT[:3, 3]
    
    # 物体尺寸（米）
    # size = [0.1, 0.1, 0.1]
    if size is None:
        size = [159.70 / 1000.0, 90.00 / 1000.0, 106.87 / 1000.0]
    
    # 绘制3D框
    image_with_bbox = visualizer.draw_from_size(image, size, R, T, thickness=2)
    
    # print("演示完成！")
    # print("使用方法:")
    # print("1. 从模型文件: visualizer.draw_from_model(image, model_path, R, T)")
    # print("2. 从尺寸: visualizer.draw_from_size(image, size, R, T)")
    # print("3. 从角点: visualizer.draw_from_corners(image, corners_3d, R, T)")
    
    if output_path is not None:
        cv2.imwrite(output_path, image_with_bbox)
    else:
        cv2.imshow("image_with_bbox", image_with_bbox)
        
    return


def get_cylinder_points(size, symmetry_axis='z', num_circle_points=32):
    """
    生成圆柱形边界框的3D点（两个圆）
    
    Args:
        size: [min_x, min_y, min_z, size_x, size_y, size_z] 或 [width, height, depth]
        symmetry_axis: 'x', 'y', 或 'z' - 旋转对称轴
        num_circle_points: 每个圆上的点数
    
    Returns:
        top_circle: (N, 3) 顶部圆的点
        bottom_circle: (N, 3) 底部圆的点
    """
    # 支持两种格式：6个值 [min_x, min_y, min_z, size_x, size_y, size_z] 或 3个值 [w, h, d]
    if len(size) == 6:
        w, h, d = size[3], size[4], size[5]  # size_x, size_y, size_z
    else:
        w, h, d = size
    
    if symmetry_axis == 'x':
        # X轴对称：Y-Z平面是圆形
        radius = max(h, d) / 2.0
        half_height = w / 2.0
        angles = np.linspace(0, 2 * np.pi, num_circle_points, endpoint=False)
        circle_y = radius * np.cos(angles)
        circle_z = radius * np.sin(angles)
        top_circle = np.stack([
            np.full(num_circle_points, half_height),
            circle_y,
            circle_z
        ], axis=1)
        bottom_circle = np.stack([
            np.full(num_circle_points, -half_height),
            circle_y,
            circle_z
        ], axis=1)
        
    elif symmetry_axis == 'y':
        # Y轴对称：X-Z平面是圆形
        radius = max(w, d) / 2.0
        half_height = h / 2.0
        angles = np.linspace(0, 2 * np.pi, num_circle_points, endpoint=False)
        circle_x = radius * np.cos(angles)
        circle_z = radius * np.sin(angles)
        top_circle = np.stack([
            circle_x,
            np.full(num_circle_points, half_height),
            circle_z
        ], axis=1)
        bottom_circle = np.stack([
            circle_x,
            np.full(num_circle_points, -half_height),
            circle_z
        ], axis=1)
        
    else:  # 'z' - 默认
        # Z轴对称：X-Y平面是圆形
        radius = max(w, h) / 2.0
        half_height = d / 2.0
        angles = np.linspace(0, 2 * np.pi, num_circle_points, endpoint=False)
        circle_x = radius * np.cos(angles)
        circle_y = radius * np.sin(angles)
        top_circle = np.stack([
            circle_x,
            circle_y,
            np.full(num_circle_points, half_height)
        ], axis=1)
        bottom_circle = np.stack([
            circle_x,
            circle_y,
            np.full(num_circle_points, -half_height)
        ], axis=1)
    
    return top_circle, bottom_circle


def draw_cylinder_bbox(image, K, R, T, size, symmetry_axis='z', 
                       color=(255, 0, 255), thickness=2, num_circle_points=48):
    """
    绘制圆柱形3D边界框（适用于旋转对称物体）
    全部使用实线，只绘制左右两条垂直连接线
    
    Args:
        image: numpy array, 输入图像
        K: (3, 3) 相机内参矩阵
        R: (3, 3) 旋转矩阵
        T: (3,) 平移向量
        size: [width, height, depth] 物体尺寸
        symmetry_axis: 'x', 'y', 或 'z' - 旋转对称轴
        color: 边框颜色 (B, G, R)
        thickness: 线条粗细
        num_circle_points: 每个圆上的点数（越多越圆滑）
    
    Returns:
        image: 绘制了圆柱形边界框的图像
    """
    image = image.copy()
    
    # 获取圆柱的3D点
    top_circle, bottom_circle = get_cylinder_points(size, symmetry_axis, num_circle_points)
    
    # 将3D点投影到2D
    top_2d, _ = points_to_2D(top_circle, R, T, K)
    bottom_2d, _ = points_to_2D(bottom_circle, R, T, K)
    
    top_2d = top_2d.astype(np.int32)
    bottom_2d = bottom_2d.astype(np.int32)
    
    color = mmcv.color_val(color)
    
    # 绘制顶部圆（实线）
    for i in range(num_circle_points):
        j = (i + 1) % num_circle_points
        cv2.line(image, tuple(top_2d[i]), tuple(top_2d[j]), color, thickness, cv2.LINE_AA)
    
    # 绘制底部圆（实线）
    for i in range(num_circle_points):
        j = (i + 1) % num_circle_points
        cv2.line(image, tuple(bottom_2d[i]), tuple(bottom_2d[j]), color, thickness, cv2.LINE_AA)
    
    # 找到左右两边最边上的点（x坐标最小和最大的点）
    # 顶部圆
    top_left_idx = np.argmin(top_2d[:, 0])
    top_right_idx = np.argmax(top_2d[:, 0])
    # 底部圆
    bottom_left_idx = np.argmin(bottom_2d[:, 0])
    bottom_right_idx = np.argmax(bottom_2d[:, 0])
    
    # 绘制左边的垂直线（实线）
    cv2.line(image, tuple(top_2d[top_left_idx]), tuple(bottom_2d[bottom_left_idx]), 
             color, thickness, cv2.LINE_AA)
    
    # 绘制右边的垂直线（实线）
    cv2.line(image, tuple(top_2d[top_right_idx]), tuple(bottom_2d[bottom_right_idx]), 
             color, thickness, cv2.LINE_AA)
    
    return image


def visualize_symmetric_object(K=None, image=None, RT=None, size=None, 
                               symmetry_axis='z', output_path=None,
                               color=(255, 0, 255), thickness=2):
    """
    可视化旋转对称物体的3D边界框（圆柱形）
    
    Args:
        K: (3, 3) 相机内参矩阵，如果为None则使用默认值
        image: numpy array, 输入图像
        RT: (3, 4) 位姿矩阵 [R|T]
        size: [width, height, depth] 物体尺寸
        symmetry_axis: 'x', 'y', 或 'z' - 旋转对称轴
            - 'x': 物体绕X轴旋转对称（圆柱沿X方向）
            - 'y': 物体绕Y轴旋转对称（圆柱沿Y方向）
            - 'z': 物体绕Z轴旋转对称（圆柱沿Z方向）
        output_path: 输出图像路径，如果为None则显示图像
        color: 边框颜色 (B, G, R)
        thickness: 线条粗细
    
    Returns:
        image_with_bbox: 绘制了圆柱形边界框的图像
    
    使用示例:
        # 对于沿Z轴旋转对称的物体（如瓶子、杯子）
        visualize_symmetric_object(K, image, RT, size, symmetry_axis='z')
        
        # 对于沿Y轴旋转对称的物体（如轮子）
        visualize_symmetric_object(K, image, RT, size, symmetry_axis='y')
    """
    if K is None:
        K = np.array([[700.0, 0, 320.0],
                      [0, 700.0, 240.0],
                      [0, 0, 1]])
    
    # 6D姿态
    R, T = RT[:3, :3], RT[:3, 3]
    
    # 物体尺寸
    if size is None:
        size = [159.70 / 1000.0, 90.00 / 1000.0, 106.87 / 1000.0]
    
    # 绘制圆柱形边界框
    image_with_bbox = draw_cylinder_bbox(
        image, K, R, T, size, 
        symmetry_axis=symmetry_axis,
        color=color,
        thickness=thickness,
        num_circle_points=48
    )
    
    if output_path is not None:
        cv2.imwrite(output_path, image_with_bbox)
    else:
        cv2.imshow("symmetric_object_bbox", image_with_bbox)
    
    return image_with_bbox


if __name__ == "__main__":
    image = demo_example()
    cv2.imwrite("image_with_bbox.png", image)