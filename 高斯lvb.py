# import numpy as np
# import cv2 as cv
# import random
# import matplotlib.pyplot as plt
#
# def gasuss_noise(image, mean=0, var=0.001):
#     """
#         添加高斯噪声
#         mean : 均值
#         var : 方差
#     """
#     image = np.array(image/255, dtype=float)
#     noise = np.random.normal(mean, var ** 0.5, image.shape)
#     out = image + noise
#     if out.min() < 0:
#         low_clip = -1.
#     else:
#         low_clip = 0.
#     out = np.clip(out, low_clip, 1.0)
#     out = np.uint8(out*255)
#     return out
# src = cv.imread("1.jpg")
# img = src.copy()
#
# # 调用噪声函数生成高斯噪声图片‘
# img_gauss = gasuss_noise(img, mean=0, var=0.01)  # 均值为0，方差为0.01
#
# # 高斯滤波
# img_gaussianBlur = cv.GaussianBlur(img_gauss, (3, 3), 1)
#
# # 显示图像
# cv.imshow("gauss", img_gauss)
# cv.imshow("gaussianBlur", img_gaussianBlur)
# cv.waitKey(0)

import cv2
import cv2 as cv
import numpy as np


def main():

    # 1.创建原图片
    # img_src = np.zeros((500, 500), dtype=np.uint8)
    # img_src[:, 250:] = 255
    img = cv.imread(r"1.jpg")

    # 2.执行双边滤波与高斯滤波
    img_dst = cv2.bilateralFilter(img, 10, 100, 100)
    # opencv中的bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)函数实现图像双边滤波，参数src表示的是图像的输入图像；
    # d是过滤时周围每个像素图像领域的直径；sigmaColor是颜色空间过滤器的sigma值(对应上式[公式]，参数越大，会有越远的像素被混合到一起；
    # sigmaSpace是坐标空间滤波器的sigma值(对应上式[公式]，参数越大，那些颜色足够相近的的颜色的影响越大。
    img_gauss = cv2.GaussianBlur(img, (3, 3), 0, 0)
    # 从结果图中可以看出卷积核越大，高斯噪声过滤的效果越好，但是最终的结果图也就越模糊，清晰度越差。
    # 3.显示结果
    cv2.imshow("img_src", img)
    cv2.imshow("img_dst", img_dst)
    cv2.imshow("img_gauss", img_gauss)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
