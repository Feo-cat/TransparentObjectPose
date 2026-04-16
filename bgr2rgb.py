"""将 BGR 格式的图片批量转换为 RGB 格式。

用法:
    # 转换单张图片（原地覆盖）
    python bgr2rgb.py path/to/image.png

    # 转换单张图片并保存到指定路径
    python bgr2rgb.py path/to/image.png -o path/to/output.png

    # 批量转换整个目录下的图片（原地覆盖）
    python bgr2rgb.py path/to/dir/

    # 批量转换整个目录，输出到另一个目录
    python bgr2rgb.py path/to/dir/ -o path/to/output_dir/

    # 指定图片格式（默认为 png）
    python bgr2rgb.py path/to/dir/ --ext png jpg jpeg bmp
"""

import argparse
import glob
import os

import cv2


def bgr2rgb(input_path, output_path):
    """读取 BGR 图片并保存为 RGB 格式。"""
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[跳过] 无法读取: {input_path}")
        return False

    if len(img.shape) == 2:
        # 灰度图，无需转换
        print(f"[跳过] 灰度图无需转换: {input_path}")
        return False

    # BGR -> RGB（交换 R 和 B 通道）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, img_rgb)
    print(f"[完成] {input_path} -> {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="将 BGR 图片转换为 RGB")
    parser.add_argument("input", help="输入图片路径或目录")
    parser.add_argument("-o", "--output", default=None, help="输出路径或目录（默认原地覆盖）")
    parser.add_argument(
        "--ext",
        nargs="+",
        default=["png"],
        help="要处理的图片扩展名（默认: png）",
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        # 单张图片
        output_path = args.output if args.output else args.input
        bgr2rgb(args.input, output_path)
    elif os.path.isdir(args.input):
        # 批量处理目录
        if args.output and not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)

        files = []
        for ext in args.ext:
            files.extend(glob.glob(os.path.join(args.input, f"*.{ext}")))
        files.sort()

        if not files:
            print(f"在 {args.input} 中未找到匹配的图片文件")
            return

        count = 0
        for fpath in files:
            if args.output:
                out_path = os.path.join(args.output, os.path.basename(fpath))
            else:
                out_path = fpath
            if bgr2rgb(fpath, out_path):
                count += 1

        print(f"\n共转换 {count}/{len(files)} 张图片")
    else:
        print(f"路径不存在: {args.input}")


if __name__ == "__main__":
    main()
