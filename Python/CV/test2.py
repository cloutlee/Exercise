from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img


# 色彩直方圖

# img = cv2.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     # plt.xlim([0, 256])

# plt.show()



# 方向梯度圖

# gray = cv2.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png', cv2.IMREAD_GRAYSCALE)
# gray_norm = np.float32(gray) / 255.0
# gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=1)
# gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=1)
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
# h, w = gray.shape
# arrow_img = np.zeros((h, w, 3), dtype=np.uint8)
# cell_size = 10

# for y in range(0, h, cell_size):
#     for x in range(0, w, cell_size):
#         cell_mag = mag[y:y+cell_size, x:x+cell_size]
#         cell_angle = angle[y:y+cell_size, x:x+cell_size]
#         avg_mag = np.mean(cell_mag) * 50
#         avg_angle = np.mean(cell_angle)
#         center_x = x + cell_size // 2
#         center_y = y + cell_size // 2
#         end_x = int(center_x + avg_mag * np.cos(np.deg2rad(avg_angle)))
#         end_y = int(center_y + avg_mag * np.sin(np.deg2rad(avg_angle)))
#         cv2.arrowedLine(arrow_img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)

# cv2.imshow('Gradient Arrows', arrow_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# 二值化

# img = cv2.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
# grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, binaryimg = cv2.threshold(grayimg, 128, 255, cv2.THRESH_BINARY)
# plt.show(binaryimg)



# 邊緣檢測
# img = cv2.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
# grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(grayimg, 100, 200)
# plt.show()


