from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 加载图片
image_path = "test_data/wr3.jpg"
# image_path = "test_data/minus.jpg"
image = cv2.imread(image_path)

# 推理（自动完成预处理）
results = model(image)

# 可视化结果
results[0].plot()  # 可视化框图

# 创建窗口
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

cv2.imshow("Detection", results[0].plot())

cv2.waitKey(0)

# 获取检测框
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"Class: {cls_id}, Conf: {conf:.2f}, BBox: {xyxy}")

