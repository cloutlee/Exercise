from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img


# -----------matplotlib讀圖、顯示圖片-----------

image = plt.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
# image = img.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png')
print(image.shape)  #(高度, 寬度, 通道數) RGB
print(type(image))  #<class 'numpy.ndarray'>
plt.axis('off')
plt.imshow(image)
plt.show()




# -----------opencv讀圖、顯示圖片-----------

# image_array = cv2.imread('C:\\Users\\clout\\Desktop\\hub\\Exercise\\Python\\11.png', cv2.IMREAD_COLOR)
# print(image_array.shape)    #(高度, 寬度, 通道數) BGR
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


