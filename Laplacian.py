import cv2
import numpy as np

img1 = cv2.imread("1.jpg", 0)
(high, wide) = img1.shape
# Laplace算子
laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
# 图像周边填充0
padding = np.zeros((high + 2, wide + 2), np.uint8)
padding[1:-1, 1:-1] = img1
cv2.imshow("image", padding)
# 创建结果图像
result_image = np.zeros((high, wide), np.uint8)
# 卷积运算
for i in range(0, high):  # 5*5的矩阵从左到右运算3次，从上到下运算3次
    for j in range(0, wide):
        window = padding[i:i + 3, j:j + 3]
        result_image[i, j] = np.abs(np.sum(laplace * window))  # 矩阵内积
cv2.imshow("result", result_image)
cv2.waitKey(0)
