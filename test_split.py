import os
import shutil

train_path = "/share/volumes/csi/renchengwei/BOP_DATASETS/labsim_vid/train/000002"
test_path = "/share/volumes/csi/renchengwei/BOP_DATASETS/labsim_vid/test/000002"

os.makedirs(test_path, exist_ok=True)
os.makedirs(os.path.join(test_path, "rgb"), exist_ok=True)
os.makedirs(os.path.join(test_path, "depth"), exist_ok=True)
os.makedirs(os.path.join(test_path, "mask"), exist_ok=True)
os.makedirs(os.path.join(test_path, "mask_visib"), exist_ok=True)
os.makedirs(os.path.join(test_path, "npz"), exist_ok=True)

# test_split_num = 12
test_split_num = 56
train_num = len(os.listdir(os.path.join(train_path, "rgb")))
print(f"Train number: {train_num}")


# rgb
for i in range(test_split_num):
    # copy reverse rgb file
    print(f"Copying {train_num - i - 1:06d}")
    rgb_file = os.path.join(train_path, "rgb", f"{train_num - i - 1:06d}")
    shutil.move(rgb_file, os.path.join(test_path, "rgb", f"{i:06d}"))

    # depth
    depth_file = os.path.join(train_path, "depth", f"{train_num - i - 1:06d}")
    shutil.move(depth_file, os.path.join(test_path, "depth", f"{i:06d}"))

    # mask
    mask_file = os.path.join(train_path, "mask", f"{train_num - i - 1:06d}")
    shutil.move(mask_file, os.path.join(test_path, "mask", f"{i:06d}"))

    # mask_visib
    mask_visib_file = os.path.join(train_path, "mask_visib", f"{train_num - i - 1:06d}")
    shutil.move(mask_visib_file, os.path.join(test_path, "mask_visib", f"{i:06d}"))

    # npz
    npz_file = os.path.join(train_path, "npz", f"{train_num - i - 1:06d}")
    shutil.move(npz_file, os.path.join(test_path, "npz", f"{i:06d}"))
