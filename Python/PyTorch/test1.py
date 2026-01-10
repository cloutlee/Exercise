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



