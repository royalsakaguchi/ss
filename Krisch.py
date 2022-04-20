from skimage import data, color
import matplotlib.pyplot as plt
import numpy as np
import cv2


def conv_cal(img, kernal):
    edge_img = cv2.filter2D(img, -1, kernal)
    return edge_img


def def_krisch(img):
    # 定义算子
    krisch1 = np.array([[5, 5, 5],
                        [-3, 0, -3],
                        [-3, -3, -3]])
    krisch2 = np.array([[-3, -3, -3],
                        [-3, 0, -3],
                        [5, 5, 5]])
    krisch3 = np.array([[5, -3, -3],
                        [5, 0, -3],
                        [5, -3, -3]])
    krisch4 = np.array([[-3, -3, 5],
                        [-3, 0, 5],
                        [-3, -3, 5]])
    krisch5 = np.array([[-3, -3, -3],
                        [-3, 0, 5],
                        [-3, 5, 5]])
    krisch6 = np.array([[-3, -3, -3],
                        [5, 0, -3],
                        [5, 5, -3]])
    krisch7 = np.array([[-3, 5, 5],
                        [-3, 0, 5],
                        [-3, -3, -3]])
    krisch8 = np.array([[5, 5, -3],
                        [5, 0, -3],
                        [-3, -3, -3]])
    gray_img = cv2.imread(img, 0)
    w, h = gray_img.shape
    img = np.zeros([w + 2, h + 2])
    img[2:w + 2, 2:h + 2] = gray_img[0:w, 0:h]
    edge1 = conv_cal(img, krisch1)
    edge2 = conv_cal(img, krisch2)
    edge3 = conv_cal(img, krisch3)
    edge4 = conv_cal(img, krisch4)
    edge5 = conv_cal(img, krisch5)
    edge6 = conv_cal(img, krisch6)
    edge7 = conv_cal(img, krisch7)
    edge8 = conv_cal(img, krisch8)
    edge_img = np.zeros([w, h], np.uint8)
    for i in range(w):
        for j in range(h):
            edge_img[i][j] = max(list(
                [edge1[i][j], edge2[i][j], edge3[i][j], edge4[i][j], edge5[i][j], edge6[i][j], edge7[i][j],
                 edge8[i][j]]))
    return [edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge_img]


if __name__ == '__main__':
    img = '1.jpg'
    edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge_img = def_krisch(img)
    cv2.imshow('1', edge1)
    cv2.imshow('2', edge2)
    cv2.imshow('3', edge3)
    cv2.imshow('4', edge4)
    cv2.imshow('5', edge5)
    cv2.imshow('6', edge6)
    cv2.imshow('7', edge7)
    cv2.imshow('8', edge8)
    cv2.imshow('9', edge_img)
    cv2.waitKey(0)
