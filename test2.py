import os
import cv2
from ultralytics import YOLO

# 设置输入输出路径
image_dir = "dataset"
output_label_dir = "labels"
os.makedirs(output_label_dir, exist_ok=True)

# 加载模型
model = YOLO("runs/detect/train/weights/best.pt")

# 遍历所有图片
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取失败：{image_path}")
        continue

    # 推理
    results = model(image)

    h, w = image.shape[:2]
    label_lines = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = xyxy
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 保存为txt标签
    label_path = os.path.join(output_label_dir, os.path.splitext(filename)[0] + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

    print(f"处理完成：{filename}，检测到 {len(label_lines)} 个目标")