import pytesseract
import cv2
import numpy as np

img = cv2.imread('12.png', cv2.IMREAD_COLOR)
#縮放
resize_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
#灰階
gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
#二值化反轉
sim_img = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY_INV)[1]
#模糊
mblur = cv2.medianBlur(sim_img, 3)
kernel = np.ones((2, 2), np.uint8)
open_img = cv2.morphologyEx(mblur, cv2.MORPH_OPEN, kernel)
text = pytesseract.image_to_string(open_img, lang="chi_tra+eng")
print(text)
cv2.imshow('imgage', img)
cv2.waitKey(0)