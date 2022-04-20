'''中值滤波 不使用opencv'''
import cv2 as cv
import numpy as np



def median_filter(image):
    img = image.copy()
    for i in range(1, image.shape[0] - 1):  # 行
        for j in range(1, image.shape[1] - 1):  # 列
            img[i, j] = np.median(image[i - 1:i + 2, j - 1:j + 2])  # 计算附近3x3矩阵像素点的中位数
    return img


'''均值滤波 不使用opencv'''


def mean_filter(image):
    img = image.copy()
    for i in range(1, image.shape[0] - 1):  # 行
        for j in range(1, image.shape[1] - 1):  # 列
            img[i, j] = np.mean(image[i - 1:i + 2, j - 1:j + 2])  # 计算附近3x3矩阵像素点的平均数
    return img


if __name__ == "__main__":
    image = cv.imread(r"1.jpg")
    cv.imshow("original", image)
    median=median_filter(image)
    cv.imshow("median", median)
    mean=mean_filter(image)

    cv.imshow("mean", mean)
    cv.waitKey(0)
    cv.destroyAllWindows()