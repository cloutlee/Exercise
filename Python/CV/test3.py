import cv2
import numpy as np

def detect_vertical_irregular_line(image_path, min_height_ratio=0.7, max_width_ratio=0.1):
    # 1. 讀取圖片並轉灰階
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 2. 邊緣偵測
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 3. 形態學操作：閉運算連接斷裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))  # 垂直結構元
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 4. 找輪廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 篩選條件 1：高度要夠高（至少覆蓋圖片 70%）
        if h < height * min_height_ratio:
            continue

        # 篩選條件 2：寬度要窄（像線）
        if w > width * max_width_ratio:
            continue

        # 篩選條件 3：從上到下（y 接近 0，y+h 接近 height）
        if y > height * 0.2 or (y + h) < height * 0.8:
            continue

        # 篩選條件 4：不規則性（可以用輪廓面積 vs 外接矩形面積比）
        area = cv2.contourArea(cnt)
        rect_area = w * h
        if area / rect_area < 0.3:  # 太稀疏不算線
            continue

        # 額外：檢查輪廓是否「連續由上到下」
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        col_sums = np.sum(mask > 0, axis=0)  # 每列有幾個白點
        vertical_coverage = np.sum(col_sums > 0) / width

        if vertical_coverage > 0.7:  # 線要「貫穿」大部分高度
            # 畫出偵測結果
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Vertical Line", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print("找到一條由上往下的不規則線！")
            cv2.imshow("Result", img)
            cv2.waitKey(0)
            return True, img

    print("沒有找到符合條件的線。")
    return False, img

# 使用範例
found, result_img = detect_vertical_irregular_line("your_image.jpg")




# 掃描線算法
# Bresenham
# Line Segment Detection
# https://github.com/Vincentqyw/LineSegmentsDetection
# https://github.com/jbn/ZigZag?tab=readme-ov-file
# https://pypi.org/project/zigzag/
# Crack Detection github
# https://github.com/konskyrt/Concrete-Crack-Detection-Segmentation
# Road Extraction github
# https://github.com/tsdinh442/road-extraction

