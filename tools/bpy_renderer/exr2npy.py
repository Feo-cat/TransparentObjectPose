import os
import pickle
import numpy as np
import OpenEXR
import Imath
# from PIL import Image
from tqdm import tqdm

def read_exr(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    
    channels = list(exr_file.header()['channels'].keys())
    print("Channels in EXR:", channels)
    
    # 获取图像尺寸
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # 设置通道类型
    pt = Imath.PixelType(Imath.PixelType.FLOAT)  # 32-bit float
    
    # 读取 RGB 通道
    r = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape(height, width)
    g = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape(height, width)
    b = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape(height, width)
    if "A" in channels:
        a = np.frombuffer(exr_file.channel('A', pt), dtype=np.float32).reshape(height, width)
    else:
        a = np.ones((height, width))
    
    # 合并为 HWC 数组
    img = np.stack([r, g, b], axis=2)  # float32, HWC
    alpha = a
    
    return img, alpha


# file_path = "/data/user/rencw/ICL-I2PReg/views/plane_furn__test_tube_rack_mass_chalice"
# file_path = "/data/user/rencw/ICL-I2PReg/views/plane_furn__test_tube_rack"
# file_path = "/data/user/rencw/ICL-I2PReg/views/plane_checker__test_tube_rack"
# file_path = "/data/user/rencw/ICL-I2PReg/views/plane_checker__test_tube_rack_0"
# file_path = "/data/user/rencw/ICL-I2PReg/views/plane_furn__test_tube_rack_0"
# file_path_list = os.listdir("/data/user/rencw/ICL-I2PReg/views")
# process_path = "./views_tube"
# process_path = "./views_tube_1"
# process_path = "./views_tube_fixed_cam"
process_path = "/share/volumes/csi/renchengwei/blender_renderings/views_tube_vid"
file_path_list = os.listdir(process_path)

for file_path in tqdm(file_path_list):
    file_path = os.path.join(process_path, file_path)
    for file in os.listdir(file_path):
        if file.endswith(".exr"):
            exr_path = os.path.join(file_path, file)
            img, _ = read_exr(exr_path)
            print(img.shape, img.max(), img.min(), exr_path)
            # handle inf
            img[img == 1.e10] = 0
            img = img[:, :, 0]
            # save as npy
            np.save(os.path.join(file_path, file.replace(".exr", ".npy")), img)
        
        
        
        