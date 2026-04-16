import cv2
import os

# 配置输入和输出文件夹
# input_folder = "/home/renchengwei/ObjectPose_ImpCorrLearning/views/plane_checker__test_tube_rack_alarm_clock/"       # 图片所在的文件夹（默认为当前目录）
input_folder = "/home/renchengwei/GDR-Net/datasets/BOP_DATASETS/labsim/train/000001/rgb"
output_folder = "bbox_output" # 结果保存的文件夹

# 之前识别出的数据列表 [文件名, x, y, w, h]
# 格式: (Filename, X-Left, Y-Top, Width, Height)
# idx, rel XYWH

# data_list = [
#     ("000000.png", 0.331287, 0.447372, 0.132942, 0.210510),
#     ("000001.png", 0.288574, 0.381474, 0.248265, 0.324003),
#     ("000002.png", 0.383342, 0.382389, 0.205019, 0.205019),
#     ("000003.png", 0.463962, 0.347151, 0.170849, 0.193120),
#     ("000004.png", 0.550454, 0.352185, 0.186866, 0.205019),
#     ("000005.png", 0.617192, 0.382389, 0.169781, 0.221493),
#     ("000006.png", 0.657768, 0.423118, 0.126001, 0.238883),
#     ("000007.png", 0.688468, 0.486729, 0.197010, 0.331325),
#     ("000008.png", 0.664709, 0.547136, 0.289375, 0.378918),
#     ("000009.png", 0.545916, 0.595187, 0.222637, 0.290138),
#     ("000010.png", 0.411639, 0.615323, 0.316070, 0.440241),
#     ("000011.png", 0.357448, 0.480779, 0.192739, 0.262680),
# ]

# {
#     "11": [{"bbox_obj": [97, 197, 151, 129], "bbox_visib": [97, 197, 151, 129], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "10": [{"bbox_obj": [223, 254, 164, 137], "bbox_visib": [223, 254, 164, 137], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "2": [{"bbox_obj": [131, 167, 108, 64], "bbox_visib": [131, 167, 108, 64], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "0": [{"bbox_obj": [0, 171, 147, 110], "bbox_visib": [0, 171, 147, 110], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "4": [{"bbox_obj": [271, 153, 115, 64], "bbox_visib": [271, 153, 115, 64], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "7": [{"bbox_obj": [467, 169, 110, 100], "bbox_visib": [467, 169, 110, 100], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "6": [{"bbox_obj": [428, 159, 81, 72], "bbox_visib": [428, 159, 81, 72], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "5": [{"bbox_obj": [358, 160, 105, 63], "bbox_visib": [358, 160, 105, 63], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "9": [{"bbox_obj": [362, 235, 171, 119], "bbox_visib": [362, 235, 171, 119], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "8": [{"bbox_obj": [425, 201, 148, 104], "bbox_visib": [425, 201, 148, 104], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "3": [{"bbox_obj": [210, 152, 86, 56], "bbox_visib": [210, 152, 86, 56], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}],
#     "1": [{"bbox_obj": [57, 162, 134, 102], "bbox_visib": [57, 162, 134, 102], "px_count_all": 99999, "px_count_valid": 99999, "px_count_visib": 99999, "visib_fract": 1.0}]
#   }

data_list = [
    ("000011.png", 97, 197, 151, 129),
    ("000010.png", 223, 254, 164, 137),
    ("000002.png", 131, 167, 108, 64),
    ("000000.png", 0, 171, 147, 110),
    ("000004.png", 271, 153, 115, 64),
    ("000007.png", 467, 169, 110, 100),
    ("000006.png", 428, 159, 81, 72),
    ("000005.png", 358, 160, 105, 63),
    ("000009.png", 362, 235, 171, 119),
    ("000008.png", 425, 201, 148, 104),
    ("000003.png", 210, 152, 86, 56),
    ("000001.png", 57, 162, 134, 102),
]

save_txt_list = []

def draw_boxes(abs=False):
    # 如果输出目录不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"开始处理 {len(data_list)} 张图片...")

    for filename, cx, cy, w, h in data_list:
        # 构建完整文件路径
        img_path = os.path.join(input_folder, filename)
        
        # 1. 读取图片
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"[错误] 找不到图片或无法读取: {filename}")
            continue

        # 2. 计算右下角坐标 (OpenCV 需要 左上角 和 右下角 两个点)
        if abs:
            top_left = (int(cx), int(cy))
            bottom_right = (int(cx + w), int(cy + h))
        else:
            top_left = (int(cx * img.shape[1] - w * img.shape[1] / 2), int(cy * img.shape[0] - h * img.shape[0] / 2))
            bottom_right = (int(cx * img.shape[1] + w * img.shape[1] / 2), int(cy * img.shape[0] + h * img.shape[0] / 2))

        # 3. 画框 (图像, 左上点, 右下点, 颜色(BGR), 线宽)
        # 这里使用红色 (0, 0, 255), 线宽为 2 像素
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

        # (可选) 添加文字标签
        if abs:
            text = f"Rack: {int(w)}x{int(h)}"
        else:
            text = f"Rack: {int(w * img.shape[1])}x{int(h * img.shape[0])}"
        print("Image shape: ", img.shape, "w, h: ", w, h)
        cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 4. 保存结果
        save_path = os.path.join(output_folder, f"boxed_{filename}")
        cv2.imwrite(save_path, img)
        print(f"[成功] 已保存: {save_path}")
        
        save_txt_list.append(f"{top_left[0]} {top_left[1]} {bottom_right[0]} {bottom_right[1]}")
        
    with open(os.path.join(output_folder, "bbox_output.txt"), "w") as f:
        for line in save_txt_list:
            f.write(line + "\n")

    print("\n所有图片处理完成！")

if __name__ == "__main__":
    draw_boxes(abs=True)