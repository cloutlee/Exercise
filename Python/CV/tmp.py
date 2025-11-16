pip install torch torchvision opencv-python


dataset/
    images/
        img001.jpg
        img002.jpg
    masks/
        img001.png   # 白色(255)=線, 黑色(0)=背景
        img002.png
U-Net 輸出是單一通道 segmentation mask（0=背景, 1=線）。




U-Net 模型程式

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool(d1)

        d2 = self.down2(p1)
        p2 = self.pool(d2)

        d3 = self.down3(p2)
        p3 = self.pool(d3)

        d4 = self.down4(p3)
        p4 = self.pool(d4)

        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
        merge4 = torch.cat([up4, d4], dim=1)
        c4 = self.conv4(merge4)

        up3 = self.up3(c4)
        merge3 = torch.cat([up3, d3], dim=1)
        c3 = self.conv3(merge3)

        up2 = self.up2(c3)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)

        up1 = self.up1(c2)
        merge1 = torch.cat([up1, d1], dim=1)
        c1 = self.conv1(merge1)

        return torch.sigmoid(self.out(c1))











Dataset 載入方式

from torch.utils.data import Dataset
import cv2
import os

class LineDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filenames = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        mask = cv2.imread(os.path.join(self.mask_dir, name.replace('.jpg','.png')), 0)
        mask = mask / 255.0
        mask = mask.reshape(1, mask.shape[0], mask.shape[1])

        img = img.transpose(2,0,1)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)












訓練程式

from torch.utils.data import DataLoader
import torch.optim as optim

dataset = LineDataset("dataset/images", "dataset/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(30):
    for imgs, masks in loader:
        imgs, masks = imgs.cuda(), masks.cuda()

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")












推論（偵測是否存在垂直不規則線）

import numpy as np

def detect_line(model, path):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    inp = torch.tensor(rgb.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).cuda()

    pred = model(inp)[0,0].detach().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)

    h = mask.shape[0]
    # 檢查是否從頂端連到底部
    top = np.any(mask[0:10, :] == 1)
    bottom = np.any(mask[h-10:h, :] == 1)

    return bool(top and bottom), mask





