from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img


# -----------matplotlib讀圖、顯示圖片-----------

# image = plt.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
# # image = img.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
# print(image.shape)  #(高度, 寬度, 通道數) RGB
# print(type(image))  #<class 'numpy.ndarray'>
# plt.axis('off')
# plt.imshow(image)
# plt.show()




# -----------opencv讀圖、顯示圖片-----------

# image_array = cv2.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png', cv2.IMREAD_COLOR)
# print(image_array.shape)    #(高度, 寬度, 通道數BGR)
# print(type(image_array))    #<class 'numpy.ndarray'>
# # cv2.imwrite('output.jpg', image_array)
# cv2.namedWindow('Image11', cv2.WINDOW_NORMAL)
# cv2.imshow('Image11', image_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# -----------PIL讀圖、顯示圖片-----------

# imgp = Image.open('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
# # imgp = Image.fromarray(data)
# # imgp.save('1.png')
# print(type(imgp))   #<class 'PIL.PngImagePlugin.PngImageFile'>
# print(imgp.size)    #(寬度, 高度)
# print(imgp.format)  #PNG
# print(imgp.mode)    #P
# imgp.show()




# ima = np.random.random((5, 5))
# plt.imshow(ima, cmap='gray')
# plt.show()


# img = img.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\1.png')

# # 假設這些是numpy索引點的座標 (x, y)
# indices = np.array([[10, 20], [50, 60], [80, 40]])

# # 顯示圖片
# fig, ax = plt.subplots()
# ax.imshow(img)

# # 在圖片上畫線連接索引點
# x = indices[:, 0]
# y = indices[:, 1]
# ax.plot(x, y, 'r-', linewidth=2)  # 紅色線
# ax.axis('off')
# plt.show()




# image = Image.open('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\1.png')

# # 將圖片轉換為RGB模式（如果是灰度圖的話）
# image = image.convert('RGB')

# # 創建NumPy數組
# width, height = image.size
# x = np.linspace(0, width-1, width)  # 用圖片寬度的索引
# y = np.sin(x / width * 2 * np.pi) * (height / 4) + height / 2  # 假設用sin函數來生成Y座標

# # 將y數據轉換為整數，因為像素位置需要整數
# y = y.astype(int)

# # 創建繪圖對象
# draw = ImageDraw.Draw(image)

# # 在圖片上繪製從(x, y)的點形成的線條
# for i in range(1, len(x)):
#     start = (int(x[i-1]), y[i-1])
#     end = (int(x[i]), y[i])
#     draw.line([start, end], fill=(255, 0, 0), width=2)  # 用紅色繪製線條

# # 顯示圖片
# image.show()



# img_rgb = np.ones((400, 600, 3), dtype=np.uint8) * 255  # 白底

# # ------------------------------
# # 2. 建立 NumPy 座標點 (x, y)
# # ------------------------------
# # 例如：5 個點的座標 (x, y)
# points = np.array([
#     [100, 200],
#     [150, 150],
#     [250, 180],
#     [300, 250],
#     [380, 220]
# ], dtype=np.int32)

# # ------------------------------
# # 3. 在圖片上畫折線
# # ------------------------------
# # 複製一份避免修改原始圖
# canvas = img_rgb.copy()

# # 畫線：從點 i 到點 i+1
# for i in range(len(points) - 1):
#     pt1 = tuple(points[i])      # (x, y)
#     pt2 = tuple(points[i+1])
#     cv2.line(canvas, pt1, pt2, color=(255, 0, 0), thickness=3)  # 藍色線

# # 畫點（可選）
# for pt in points:
#     cv2.circle(canvas, tuple(pt), radius=6, color=(0, 255, 0), thickness=-1)  # 綠色實心點

# # ------------------------------
# # 4. 顯示結果 (Matplotlib)
# # ------------------------------
# plt.figure(figsize=(10, 6))
# plt.imshow(canvas)
# plt.axis('off')
# plt.show()





# # 假設圖片為一個 500x800 像素的彩色 (3 通道 BGR) NumPy 陣列
# # 這裡我們創建一個全黑的圖片作為示例
# height, width = 500, 800
# # dtype=np.uint8 是圖片像素常用的數據類型
# image_array = np.zeros((height, width, 3), dtype=np.uint8) 

# # 將圖片的背景設為灰色，以便線條更明顯
# image_array[:, :] = [150, 150, 150] # BGR 格式的灰色

# # --- 2. 定義繪製線條所需的 NumPy 索引/座標 ---

# # 繪製直線需要兩個點的座標 (x1, y1) 和 (x2, y2)
# # ⚠️ 注意：在 OpenCV 中，座標系統是 (X 寬度, Y 高度)，原點 (0, 0) 在左上角。

# # 點 A 的像素索引
# x1_index = 100 
# y1_index = 50 
# point1 = (x1_index, y1_index)

# # 點 B 的像素索引
# x2_index = 700 
# y2_index = 450 
# point2 = (x2_index, y2_index)

# # 線條顏色 (使用 BGR 格式：藍色 Blue, 綠色 Green, 紅色 Red)
# # 例如：純紅色
# line_color_bgr = (0, 0, 255) 

# # 線條寬度 (像素)
# line_thickness = 5 

# # --- 3. 使用 OpenCV 函數在 NumPy 陣列上畫線 ---

# # 參數順序： 圖片陣列, 起點, 終點, 顏色, 寬度
# cv2.line(
#     image_array, 
#     point1, 
#     point2, 
#     line_color_bgr, 
#     line_thickness
# )

# # 也可以畫圓形 (例如在起點畫一個綠色的圓)
# # cv2.circle(image_array, point1, radius=10, color=(0, 255, 0), thickness=-1) # thickness=-1 表示填充

# # --- 4. 顯示結果 (可選，但有助於驗證) ---

# # OpenCV 使用 BGR 顏色空間，Matplotlib 使用 RGB。
# # 為了正確顯示，需要將 BGR 轉換為 RGB：
# image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(8, 5))
# plt.imshow(image_rgb)
# plt.title(f'在圖片上繪製線段：從 {point1} 到 {point2}')
# plt.xlabel('X 座標 (寬度)')
# plt.ylabel('Y 座標 (高度)')
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] #顯示中文
# plt.show()





img_array = np.zeros((10, 10, 3), dtype=np.uint8)

# 在中間列（第5列，index為4）設置綠色線，綠色在RGB是(0,255,0)
img_array[:, 5, 1] = 255  # 在所有行，列索引4，綠色通道設定255

# 轉換成圖片物件
img = Image.fromarray(img_array)

# # 儲存圖片
# img.save("green_line.png")
img.show()