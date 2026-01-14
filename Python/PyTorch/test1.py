import torch.nn as nn
import torch
from torchvision import transforms


# print(torch.zeros(5))
# print(torch.ones(5))
# print(torch.full((5,), 3.14))
# print(torch.arange(0, 10, 2))
# print(torch.linspace(0, 1, steps=5))
# print(torch.eye(3))
# print(torch.rand(3, 3))
# print(torch.randn(3, 3))
# print(torch.randint(0, 10, (3, 3)))
# print(torch.empty(3, 3))
# print(torch.tensor([1.0, 2.0, 3.0]))




# 將多個圖像預處理步驟（Transforms）串聯起來,定義一個組合操作
# my_transform = transforms.Compose([
#     transforms.Resize(256),              # 1. 將圖片縮放至 256
#     transforms.CenterCrop(224),          # 2. 從中心裁剪出 224x224
#     transforms.ToTensor(),               # 3. 轉為 Tensor 並歸一化到 [0, 1]
#     transforms.Normalize(                # 4. 標準化（減均值、除以標準差）
#         mean=[0.485, 0.456, 0.406], 
#         std=[0.229, 0.224, 0.225]
#     )
# ])
# 使用時只需要調用一次
# transformed_img = my_transform(img)




# x = torch.randn(2, 3, 4, 4)   # batch=2, RGB圖像 4×4

# #攤平成 (2, 3*4*4) = (2, 48)  → 接全連接層最常見寫法
# x_flat = x.flatten(start_dim=1)           # 目前最推薦
# print(x_flat.shape)
# # 或
# x_flat = x.view(x.size(0), -1)            # 更快，但要確保 contiguous
# print(x_flat.shape)
# # 或
# x_flat = x.reshape(x.shape[0], -1)        # 安全，但可能稍微慢一點
# print(x_flat.shape)



# x = torch.randn(4, 4)
# print(x.size())
# print(x.ndim)
# print(x.shape)
# print(x)

# y = x.view(16)
# # y = torch.flatten(x)
# print(y.size())
# print(y.ndim)
# print(y.shape)
# print(y)




# a = torch.randn(5)
# print(a)
# re = nn.ReLU()
# output = re(a)
# print(output)



# bce_loss = nn.BCELoss()
# predictions = torch.tensor([0.8, 0.2, 0.9, 0.1])
# targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
# loss = bce_loss(predictions, targets)
# print(f"BCELoss: {loss.item()}")





# # 假設輸入是：批次=1張, 通道=3(RGB), 28×28 的圖片
# x = torch.randn(1, 3, 28, 28)
# # 定義一個普通的 2D 卷積層
# conv = nn.Conv2d(
#     in_channels=3,      # 輸入通道數
#     out_channels=16,    # 輸出通道數（也就是濾波器/特徵圖數量）
#     kernel_size=3,      # 卷積核大小 3×3
#     stride=1,           # 步幅
#     padding=1           # 周圍補零寬度（保持尺寸常用）
# )

# out = conv(x)
# print(out.shape)





# def conv_size_calc():
#     x = torch.randn(1, 3, 224, 224)
    
#     conv_layers = [
#         nn.Conv2d(3, 64, 7, stride=2, padding=3),     # VGG/AlexNet 常見開頭,大幅降維
#         nn.Conv2d(3, 64, 3, stride=1, padding=1),     # 保持尺寸最常見
#         nn.Conv2d(3, 64, 3, stride=2, padding=1),     # 尺寸減半
#         nn.Conv2d(3, 64, 3, stride=1, padding=0),     # 尺寸變小
#         nn.Conv2d(3, 64, 1, stride=1),                # 1x1 卷積（通道變換）
#     ]
    
#     for i, conv in enumerate(conv_layers, 1):
#         out = conv(x)
#         print(f"conv{i}: {out.shape}")

# conv_size_calc()


