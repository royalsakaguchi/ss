#方法一代码
import cv2
import numpy as np
#读取照片1.jpg，得到矩阵img
from matplotlib import pyplot as plt

img=cv2.imread("1.jpg")
#将矩阵img复制给img1
img1=np.copy(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img1[i,j]=img[i,img.shape[1]-1-j]
plt.imshow(img1)
plt.show()