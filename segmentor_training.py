import albumentations as A
import cv2
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset,DataLoader, random_split
import torch.nn as nn
import torchvision.transforms as T
import segmentation_models_pytorch as smp


import numpy as np
import matplotlib.pyplot as plt

import urllib.request
import tarfile
import os

url = 'https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_seg.tar.gz'
tar_name = 'meter_seg.tar.gz'

if not os.path.exists('meter_seg'):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, tar_name)

    print("Extracting dataset...")
    with tarfile.open(tar_name, 'r:gz') as tar:
        tar.extractall()

    print("Done.")
else:
    print("Dataset already exists.")


augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.Blur(blur_limit=3),
    A.ColorJitter(),
    A.RGBShift()
])

augmented_image = Path('augmented_data/augmented_images')
augmented_label = Path('augmented_data/augmented_labels')
augmented_image.mkdir(parents=True, exist_ok=True)
augmented_label.mkdir(parents=True, exist_ok=True)

img_dir = './meter_seg/images/train'
ann_dir = './meter_seg/annotations/train'

for fname in sorted(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, fname)
    ann_path = os.path.join(ann_dir, fname.replace('.jpg', '.png'))

    for t in range(3):
        img = cv2.imread(img_path)
        ann = cv2.imread(ann_path)
        aug = augmentor(image=img, mask=ann)
        cv2.imwrite(f'{augmented_image}/{fname[:-4]}_{t}.jpg', aug['image'])
        cv2.imwrite(f'{augmented_label}/{fname[:-4]}_{t}.png', aug['mask'])


class SegDataset(Dataset):
    def __init__(self, image_dir, label_dir, size=(256, 256)):
        self.image_dir = sorted(list(Path(image_dir).glob('*.jpg')))
        self.label_dir = sorted(list(Path(label_dir).glob('*.png')))
        self.size = size
        self.image_transform = T.ToTensor()
        
    def __len__(self):
        return len(self.image_dir)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_dir[idx]))
        mask = cv2.imread(str(self.label_dir[idx]), 0)
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        image = self.image_transform(image)
        mask = torch.from_numpy(mask).long()

        return image, mask


dataset = SegDataset('augmented_data/augmented_images', 'augmented_data/augmented_labels')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(encoder_name="efficientnet-b1", classes=3, activation=None).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

torch.save(model.state_dict(), 'meter_seg.pth')

model.eval()
val_imgs, val_masks = next(iter(val_loader))
with torch.no_grad():
    preds = model(val_imgs.to(device))
    preds = torch.argmax(preds, dim=1).cpu().numpy()

for i in range(3):
    plt.subplot(3,3,i*3+1); plt.imshow(val_imgs[i].permute(1,2,0)); plt.axis('off')
    plt.subplot(3,3,i*3+2); plt.imshow(val_masks[i], cmap='gray'); plt.axis('off')
    plt.subplot(3,3,i*3+3); plt.imshow(preds[i], cmap='gray'); plt.axis('off')
plt.suptitle('Image - Ground Truth - Prediction')
plt.tight_layout()
plt.show()
