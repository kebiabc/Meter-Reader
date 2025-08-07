import cv2
import os
import numpy as np

# 原始图片和标注路径
image_path = "test_data/minus.jpg"
label_path = "test_data/1.txt"

# 创建增强数据保存目录
os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)

# 读取图像和图像尺寸
img = cv2.imread(image_path)
h, w = img.shape[:2]

# 读取负号坐标（假设只有一行）
with open(label_path, 'r') as f:
    line = f.readline()
cls_id, x_c, y_c, bw, bh = map(float, line.strip().split())

# 反算真实像素坐标
x = int((x_c - bw / 2) * w)
y = int((y_c - bh / 2) * h)
box_w = int(bw * w)
box_h = int(bh * h)

# 裁剪负号图像
neg_img = img[y:y+box_h, x:x+box_w]

# 增强参数
angles = [-5, 0, 5]
scales = [0.9, 1.0, 1.1]
brightness = [0.8, 1.0, 1.2]

count = 0
for ang in angles:
    for sc in scales:
        for br in brightness:
            # 图像增强
            center = (neg_img.shape[1]//2, neg_img.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, ang, sc)
            transformed = cv2.warpAffine(neg_img, M, (neg_img.shape[1], neg_img.shape[0]))
            transformed = cv2.convertScaleAbs(transformed, alpha=br, beta=0)

            # 保存图像
            out_img_path = f"dataset/images/train/neg_{count}.jpg"
            out_txt_path = f"dataset/labels/train/neg_{count}.txt"
            cv2.imwrite(out_img_path, transformed)

            # YOLO格式：中心点 (0.5, 0.5)，宽高=1（填满整张图）
            with open(out_txt_path, "w") as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

            count += 1

print(f"已生成 {count} 个增强样本")