import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读入灰度图像，默认读入彩色图像
try:
    img1 = cv2.imread("img/train/airplane/f0.png", 0)
except:
    pass
img2 = cv2.imread("img/train/airplane/35.png", 0)
img3 = cv2.imread("img/train/airplane/349.png", 0)
imgs = np.hstack([img1, img2, img3])
cv2.imshow("one image", img1)  # 展示图像，imshow(窗口名, image_name)
cv2.imshow("more image", imgs)  # 展示多个图像
cv2.waitKey(0)  # 等待响应
