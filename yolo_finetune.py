from ultralytics import YOLO

def train_model():
    """
    训练 YOLOv8 模型
    """
    # 设置参数
    data_yaml_path = "C:\\py_proj\\Meter-Digit-Recognition\\finetune\\finetune.yaml"  # 数据集配置文件路径
    model_path = "runs/detect/train/weights/best.pt"                  # 预训练模型权重文件路径
    epochs = 20                               # 训练的总轮数
    imgsz = 1024                                # 输入图像的大小
    batch = 16                                 # 每个批次的大小

    # 加载模型
    model = YOLO(model_path)

    # 训练模型
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch
    )

    print("训练完成！")

if __name__ == "__main__":
    # 调用训练函数
    train_model()