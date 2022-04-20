import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 错切
shearM = np.array([
    [1, 0.3, 0],
    [0.2, 1,   0]

], dtype=np.float32)

img = cv.imread('1.jpg')
img_shear = cv.warpAffine(img, shearM, dsize=(300, 300))

plt.imshow(img_shear)
plt.show()
