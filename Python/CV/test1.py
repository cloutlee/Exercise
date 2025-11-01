from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

data = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

# 從numpy匯出成圖片

# img = Image.fromarray(data)
# # img.save('1.png')
# img.show()

# cv2.imshow('Random Image', data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.imshow(data)
plt.show()
