# import os
# import tarfile
# import urllib.request
# import os
# from pathlib import Path

# # 自动下载数据集
# url = 'https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_det.tar.gz'
# tar_name = 'meter_det.tar.gz'

# if not os.path.exists('meter_det'):
#     print("Downloading dataset...")
#     urllib.request.urlretrieve(url, tar_name)

#     print("Extracting dataset...")
#     with tarfile.open(tar_name, 'r:gz') as tar:
#         tar.extractall()

#     print("Done.")
# else:
#     print("Dataset already exists.")

# # ================== 训练部分 ===================
from torchvision.datasets import CocoDetection
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from tqdm import tqdm

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# train_dataset = CocoDetection(
#     root='meter_det/train',
#     annFile='meter_det/coco_annotations/instances_train.json',
#     transform=transform
# )

# val_dataset = CocoDetection(
#     root='meter_det/test',
#     annFile='meter_det/coco_annotations/instances_val.json',
#     transform=transform
# )

# def collate_fn(batch):
#     return tuple(zip(*batch))

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 使用预训练 Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
# model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for images, targets in tqdm(train_loader):
#         images = list(img.to(device) for img in images)
#         targets = [{
#             'boxes': torch.tensor(
#                 [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in [obj['bbox'] for obj in anns]],
#                 dtype=torch.float32
#             ).to(device),
#             'labels': torch.tensor([obj['category_id'] for obj in anns], dtype=torch.int64).to(device)
#         } for anns in targets]

#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         epoch_loss += losses.item()
    
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# torch.save(model.state_dict(), 'meter_detector.pth')

# ================== 推理部分 ===================
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model.eval()
model.load_state_dict(torch.load('meter_detector.pth', map_location=device))
model.to(device)

# 读取测试图像
img = Image.open('meter_det/test/20190822_86.jpg').convert("RGB")
img_tensor = transforms.ToTensor()(img).to(device)

with torch.no_grad():
    prediction = model([img_tensor])[0]

# 绘制结果
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
    if score > 0.5:
        # print(box)
        x1, y1, x2, y2 = map(int,box.tolist())
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_cv, f'{score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
