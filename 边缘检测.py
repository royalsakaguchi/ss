#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
# import math
from scipy import signal


# In[2]:


img=cv2.imread("1.jpg",0)
print(img)


# In[7]:


# 垂直提取
Vertical_extraction = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
(high, wide) = img.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img

# 创建输出图像
img_Vertical_extraction=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Vertical_extraction[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Vertical_extraction))
plt.imshow(img_Vertical_extraction)
plt.show()


# In[4]:


# 水平提取
Horizontal_extraction= np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
(high, wide) = img.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img

# 创建输出图像
img_Horizontal_extraction=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Horizontal_extraction[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Horizontal_extraction))
plt.imshow(img_Horizontal_extraction)
plt.show()


# In[6]:


# Roberts算子
Roberts_x= np.array([[1, 0],[0, -1]])
Roberts_y= np.array([[0, -1],[1, 0]])
(high, wide) = img.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img

# 创建输出图像
img_Roberts_x=np.zeros((high,wide),np.uint8)
img_Roberts_y=np.zeros((high,wide),np.uint8)
img_Roberts=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Roberts_x[i,j]=np.abs(np.sum(padding[i:i+2,j:j+2]*Roberts_x))
        img_Roberts_y[i,j]=np.abs(np.sum(padding[i:i+2,j:j+2]*Roberts_y))
# 处理Roberts合体
img_Roberts=img_Roberts_x+img_Roberts_y

cv2.imshow("img_Robots_x",img_Roberts_x)
cv2.imshow("img_Robots_y",img_Roberts_y)
cv2.imshow("img_Robots",img_Roberts)
cv2.waitKey(0)


# In[9]:


# Prewitt算子
Prewitt_x= np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
Prewitt_y= np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
(high, wide) = img.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img

# 创建输出图像
img_Prewitt_x=np.zeros((high,wide),np.uint8)
img_Prewitt_y=np.zeros((high,wide),np.uint8)
img_Prewitt=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Prewitt_x[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Prewitt_x))
        img_Prewitt_y[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Prewitt_y))
# 处理Prewitt合体
img_Prewitt=img_Prewitt_x+img_Prewitt_y

cv2.imshow("img_Prewitt_x",img_Prewitt_x)
cv2.imshow("img_Prewitt_y",img_Prewitt_y)
cv2.imshow("img_Prewitt",img_Prewitt)
cv2.waitKey(0)


# In[10]:


# Sobel算子
# Sobel算子是高斯平滑和微分操作的结合体，因此它的抗噪声能力比较好。
Sobel_x= np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
Sobel_y= np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
(high, wide) = img.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img

# 创建输出图像
img_Sobel_x=np.zeros((high,wide),np.uint8)
img_Sobel_y=np.zeros((high,wide),np.uint8)
img_Sobel=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Sobel_x[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Sobel_x))
        img_Sobel_y[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Sobel_y))
# 处理Sobel合体
# 理论上应该使用img_Sobel=np.sqrt(img_Sobel_x*img_Sobel_x+img_Sobel_y*img_Sobel_y)
# 为简便运算使用img_Sobel=np.abs(img_Sobel_x)+np.abs(img_Sobel_y)，
# 此处上面已处理就不进行再次运算
img_Sobel=img_Sobel_x+img_Sobel_y

cv2.imshow("img_Sobel_x",img_Sobel_x)
cv2.imshow("img_Sobel_y",img_Sobel_y)
cv2.imshow("img_Sobel",img_Sobel)
cv2.waitKey(0)


# In[8]:


# Scharr算子
# Scharr算子比Sobel更好用，（实际看效果时感觉较差，原因未知）
Scharr_x= np.array([[-3, 0, 3],[-10, 0, 10],[-3, 0, 3]])
Scharr_y= np.array([[3, 10, 3],[0, 0, 0],[-3, -10, -3]])
(high, wide) = img.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img

# 创建输出图像
img_Scharr_x=np.zeros((high,wide),np.uint8)
img_Scharr_y=np.zeros((high,wide),np.uint8)
img_Scharr=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Scharr_x[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Scharr_x))
        img_Scharr_y[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Scharr_y))
# 处理Scharr合体
# 理论上应该使用img_Scharr=np.sqrt(img_Scharr_x*img_Scharr_x+img_Scharr_y*img_Scharr_y)
# 为简便运算使用img_Scharr=np.abs(img_Scharr_x)+np.abs(img_Scharr_y)，
# 此处上面已处理就不进行再次运算
img_Scharr=img_Scharr_x+img_Scharr_y

cv2.imshow("img_Scharr_x",img_Scharr_x)
cv2.imshow("img_Scharr_y",img_Scharr_y)
cv2.imshow("img_Scharr",img_Scharr)
cv2.waitKey(0)


# In[9]:


# Laplace算子
laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
#图像周边填充0
padding = np.zeros((high+2, wide+2), np.uint8)
padding[1:-1, 1:-1] = img
#创建结果图像
img_Laplace = np.zeros((high, wide), np.uint8)
# 卷积运算
for i in range(0, high):  # 5*5的矩阵从左到右运算3次，从上到下运算3次
    for j in range(0, wide):
        window = padding[i:i+3, j:j+3]
        img_Laplace[i, j] = np.abs(np.sum(laplace*window)) #矩阵内积
cv2.imshow("Laplace", img_Laplace)
cv2.waitKey(0)


# In[23]:


# Kirsch算子

#定义Kirsch 卷积模板
m1 = np.array([[5, 5, 5],[-3, 0, -3],[-3, -3, -3]])

m2 = np.array([[-3, 5, 5],[-3, 0, 5],[-3, -3, -3]])

m3 = np.array([[-3, -3, 5],[-3, 0, 5],[-3, -3, 5]])

m4 = np.array([[-3, -3, -3],[-3, 0, 5],[-3, 5, 5]])

m5 = np.array([[-3, -3, -3],[-3, 0, -3],[5, 5, 5]])

m6 = np.array([[-3, -3, -3],[5, 0, -3],[5, 5, -3]])

m7 = np.array([[5, -3, -3],[5, 0, -3],[5, -3, -3]])

m8 = np.array([[5, 5, -3],[5, 0, -3],[-3, -3, -3]])
#周围填充一圈
#卷积时，必须在原图周围填充一个像素
padding = np.zeros((high+2, wide+2), np.uint8)
padding[1:-1, 1:-1] = img

# 计算结果
temp=list(range(8))
img_Kirsch=np.zeros((high, wide), np.uint8)

for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
        temp[0] = np.abs( ( np.dot( np.array([1,1,1]) , ( m1*img[i-1:i+2,j-1:j+2]) ) ).dot(np.array([[1],[1],[1]])) )
			#利用矩阵的二次型表达，可以计算出矩阵的各个元素之和
        temp[1] = np.abs(
            (np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

        temp[2] = np.abs( ( np.dot( np.array([1,1,1]) , ( m1*img[i-1:i+2,j-1:j+2]) ) ).dot(np.array([[1],[1],[1]])) )

        temp[3] = np.abs(
            (np.dot(np.array([1, 1, 1]), (m3 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

        temp[4] = np.abs(
            (np.dot(np.array([1, 1, 1]), (m4 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

        temp[5] = np.abs(
            (np.dot(np.array([1, 1, 1]), (m5 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

        temp[6] = np.abs(
            (np.dot(np.array([1, 1, 1]), (m6 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

        temp[7] = np.abs(
            (np.dot(np.array([1, 1, 1]), (m7 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

        img_Kirsch[i,j] = np.max(temp)

        if img_Kirsch[i, j] > 125:  #此处的阈值一般写255，根据实际情况选择0~255之间的值
            img_Kirsch[i, j] = 255
        else:
            img_Kirsch[i, j] = 0

cv2.imshow("Kirsch",img_Kirsch)
cv2.waitKey(0)


# In[27]:


# Marr-Hildreth算子/LOG算子

# 二维高斯卷积核拆分为水平核垂直一维卷积核，分别进行卷积
def gaussConv(image, size, sigma):
    H, W = size
    # 先水平一维高斯核卷积
    xr, xc = np.mgrid[0:1, 0:W]
    xc = xc.astype(np.float32)
    xc -= (W-1.0)/2.0
    xk = np.exp(-np.power(xc, 2.0)/(2*sigma*sigma))
    image_xk = signal.convolve2d(image, xk, 'same', 'symm')

    # 垂直一维高斯核卷积
    yr, yc = np.mgrid[0:H, 0:1]
    yr = yr.astype(np.float32)
    yr -= (H-1.0)/2.0
    yk = np.exp(-np.power(yr, 2.0)/(2*sigma*sigma))
    image_yk = signal.convolve2d(image_xk, yk, 'same','symm')
    image_conv = image_yk/(2*np.pi*np.power(sigma, 2.0))

    return image_conv

def DoG(image, size, sigma, k=1.1):
    Is = gaussConv(image, size, sigma)
    Isk = gaussConv(image, size, sigma*k)
    doG = Isk - Is
    doG /= (np.power(sigma, 2.0)*(k-1))
    return doG

def zero_cross_default(doG):
    zero_cross = np.zeros(doG.shape, np.uint8);
    rows, cols = doG.shape
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if doG[r][c-1]*doG[r][c+1] < 0:
                zero_cross[r][c]=255
                continue
            if doG[r-1][c] * doG[r+1][c] <0:
                zero_cross[r][c] = 255
                continue
            if doG[r-1][c-1] * doG[r+1][c+1] <0:
                zero_cross[r][c] = 255
                continue
            if doG[r-1][c+1] * doG[r+1][c-1] <0:
                zero_cross[r][c] = 255
                continue
    return zero_cross

def Marr_Hildreth(image, size, sigma, k=1.1):
    doG = DoG(image, size, sigma, k)
    zero_cross = zero_cross_default(doG)
    return zero_cross

k = 1.1
marri_edge = Marr_Hildreth(img, (11, 11), 1, k)
marri_edge2 = Marr_Hildreth(img, (11, 11), 2, k)
marri_edge3 = Marr_Hildreth(img, (7, 7), 1, k)

cv2.imshow("img", img)
cv2.imshow("marri_edge", marri_edge)
cv2.imshow("marri_edge2", marri_edge2)
cv2.imshow("marri_edge3", marri_edge3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


# Canny边缘检测
# 1.消除噪声，高斯滤波（可选3*3与5*5）
# 高斯卷积
# Gauss3X3=np.array([[1, 2, 1],
#                   [2, 8, 2],
#                   [1, 2, 1]])
# Gauss5X5=np.array([[2, 4, 5, 4, 2],
#                   [4, 9, 12, 9, 4],
#                   [5, 12, 15, 12, 5],
#                   [4, 9, 12, 9, 4]
#                   [2, 4, 5, 4, 2]])

# 卷积使用过于复杂，此处使用高斯卷积核代替，高斯卷积核等于高斯卷积除以各元素之和
Gauss3X3=np.array([[0.057, 0.125,0.057],
                   [0.125, 0.272, 0.125],
                   [0.057, 0.125, 0.057]])
Gauss5X5=np.array([[0.003, 0.013, 0.022, 0.013, 0.003],
                   [0.013, 0.059, 0.097, 0.059, 0.013],
                   [0.022, 0.097, 0.159, 0.097, 0.022],
                   [0.013, 0.059, 0.097, 0.059, 0.013],
                   [0.003, 0.013, 0.022, 0.013, 0.003]])

(high, wide) = img.shape
# 3X3
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img

# 创建输出图像
img_Gauss3X3=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Gauss3X3[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Gauss3X3))
        
# 5X5
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+4,wide+4),np.uint8)
padding[2:-2,2:-2]=img

# 创建输出图像
img_Gauss5X5=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Gauss5X5[i,j]=np.abs(np.sum(padding[i:i+5,j:j+5]*Gauss5X5))

cv2.imshow("img_Gauss3X3",img_Gauss3X3)
cv2.imshow("img_Gauss5X5",img_Gauss5X5)
cv2.waitKey(0)


# In[15]:


# 2.计算梯度幅值和方向。 此处，按照Sobel滤波器的步骤。

# 3X3
Sobel_x= np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
Sobel_y= np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img_Gauss3X3

# 创建输出图像
img_Sobel_Gauss3X3_x=np.zeros((high,wide),np.uint8)
img_Sobel_Gauss3X3_y=np.zeros((high,wide),np.uint8)
img_Sobel_Gauss3X3=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Sobel_Gauss3X3_x[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Sobel_x))
        img_Sobel_Gauss3X3_y[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Sobel_y))
        
# Sobel梯度幅值
img_Sobel_Gauss3X3=img_Sobel_Gauss3X3_x+img_Sobel_Gauss3X3_y
# Sobel方向
theta_Sobel_Gauss3X3=np.arctan(img_Sobel_Gauss3X3_y/img_Sobel_Gauss3X3_x)/3.14

# 5X5
Sobel_x= np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
Sobel_y= np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=img_Gauss5X5

# 创建输出图像
img_Sobel_Gauss5X5_x=np.zeros((high,wide),np.uint8)
img_Sobel_Gauss5X5_y=np.zeros((high,wide),np.uint8)
img_Sobel_Gauss5X5=np.zeros((high,wide),np.uint8)

# 进行数据处理
for i in range(0,high):
    for j in range(0,wide):
        img_Sobel_Gauss5X5_x[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Sobel_x))
        img_Sobel_Gauss5X5_y[i,j]=np.abs(np.sum(padding[i:i+3,j:j+3]*Sobel_y))
        
# Sobel梯度幅值
img_Sobel_Gauss5X5=img_Sobel_Gauss5X5_x+img_Sobel_Gauss5X5_y
# Sobel方向
theta_Sobel_Gauss5X5=np.arctan(img_Sobel_Gauss5X5_y/img_Sobel_Gauss5X5_x)/3.14

cv2.imshow("img_Sobel_Gauss3X3",img_Sobel_Gauss3X3)
cv2.imshow("img_Sobel_Gauss5X5",img_Sobel_Gauss5X5)
cv2.imshow("different between img_Sobel_Gauss3X3 and img_Sobel_Gauss5X5",np.abs(img_Sobel_Gauss3X3-img_Sobel_Gauss5X5))
cv2.waitKey(0)


# In[32]:


# 3.非极大值抑制NMS。 这一步排除非边缘像素， 仅仅保留了一些细线条(候选边缘)。
# 3X3
(high, wide) = img_Sobel_Gauss3X3.shape
NMS_Gauss3X3=np.copy(img_Sobel_Gauss3X3)
img_Sobel_Gauss3X3_NMS_Gauss3X3=np.copy(img_Sobel_Gauss3X3)
NMS_Gauss3X3[0, :] = NMS_Gauss3X3[high - 1, :] = NMS_Gauss3X3[:, 0] = NMS_Gauss3X3[:, wide - 1] = 0
for i in range(1, high - 1):
    for j in range(1, wide - 1):
        if img_Sobel_Gauss3X3[i,j]==0:
            NMS_Gauss3X3[i, j] = 0
        else:
            gradX = img_Sobel_Gauss3X3_x[i, j]
            gradY = img_Sobel_Gauss3X3_y[i, j]
            gradTemp = img_Sobel_Gauss3X3[i, j]
            
            # 如果Y方向幅度值较大
            if np.abs(gradY) > np.abs(gradX):
                weight = np.abs(gradX) / np.abs(gradY)
                grad2 = img_Sobel_Gauss3X3[i - 1, j]
                grad4 = img_Sobel_Gauss3X3[i + 1, j]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = img_Sobel_Gauss3X3[i - 1, j - 1]
                    grad3 = img_Sobel_Gauss3X3[i + 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = img_Sobel_Gauss3X3[i - 1, j + 1]
                    grad3 = img_Sobel_Gauss3X3[i + 1, j - 1]

            # 如果X方向幅度值较大
            else:
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = img_Sobel_Gauss3X3[i, j - 1]
                grad4 = img_Sobel_Gauss3X3[i, j + 1]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = img_Sobel_Gauss3X3[i + 1, j - 1]
                    grad3 = img_Sobel_Gauss3X3[i - 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = img_Sobel_Gauss3X3[i - 1, j - 1]
                    grad3 = img_Sobel_Gauss3X3[i + 1, j + 1]
            
            gradTemp1 = weight * grad1 + (1 - weight) * grad2
            gradTemp2 = weight * grad3 + (1 - weight) * grad4
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS_Gauss3X3[i, j] = gradTemp
            else:
                NMS_Gauss3X3[i, j] = 0
                

# 5X5
(high, wide) = img_Sobel_Gauss5X5.shape
NMS_Gauss5X5=np.copy(img_Sobel_Gauss5X5)
img_Sobel_Gauss5X5_NMS_Gauss5X5=np.copy(img_Sobel_Gauss5X5)
NMS_Gauss5X5[0, :] = NMS_Gauss5X5[high - 1, :] = NMS_Gauss5X5[:, 0] = NMS_Gauss5X5[:, wide - 1] = 0
for i in range(1, high - 1):
    for j in range(1, wide - 1):
        if img_Sobel_Gauss5X5[i,j]==0:
            NMS_Gauss5X5[i, j] = 0
        else:
            gradX = img_Sobel_Gauss5X5_x[i, j]
            gradY = img_Sobel_Gauss5X5_y[i, j]
            gradTemp = img_Sobel_Gauss5X5[i, j]
            
            # 如果Y方向幅度值较大
            if np.abs(gradY) > np.abs(gradX):
                weight = np.abs(gradX) / np.abs(gradY)
                grad2 = img_Sobel_Gauss5X5[i - 1, j]
                grad4 = img_Sobel_Gauss5X5[i + 1, j]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = img_Sobel_Gauss5X5[i - 1, j - 1]
                    grad3 = img_Sobel_Gauss5X5[i + 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = img_Sobel_Gauss5X5[i - 1, j + 1]
                    grad3 = img_Sobel_Gauss5X5[i + 1, j - 1]

            # 如果X方向幅度值较大
            else:
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = img_Sobel_Gauss5X5[i, j - 1]
                grad4 = img_Sobel_Gauss5X5[i, j + 1]
                # 如果x,y方向梯度符号相同
                if gradX * gradY > 0:
                    grad1 = img_Sobel_Gauss5X5[i + 1, j - 1]
                    grad3 = img_Sobel_Gauss5X5[i - 1, j + 1]
                # 如果x,y方向梯度符号相反
                else:
                    grad1 = img_Sobel_Gauss5X5[i - 1, j - 1]
                    grad3 = img_Sobel_Gauss5X5[i + 1, j + 1]
            
            gradTemp1 = weight * grad1 + (1 - weight) * grad2
            gradTemp2 = weight * grad3 + (1 - weight) * grad4
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS_Gauss5X5[i, j] = gradTemp
            else:
                NMS_Gauss5X5[i, j] = 0

plt.subplot(1, 2, 1),plt.imshow(NMS_Gauss3X3, cmap = "gray")
plt.title("NMS_Gauss3X3") 
plt.xticks([]),plt.yticks([]) 

plt.subplot(1, 2, 2),plt.imshow(NMS_Gauss5X5, cmap = "gray")
plt.title("NMS_Gauss5X5") 
plt.xticks([]),plt.yticks([]) 

plt.show()


# In[34]:


#  4.滞后阈值。最后一步，Canny 使用了滞后阈值，滞后阈值需要两个阈值(高阈值和低阈值):
#  双阈值算法检测，链接边缘
# 　　Ⅰ.如果某一像素位置的幅值超过 高 阈值, 该像素被保留为边缘像素。
# 　　Ⅱ.如果某一像素位置的幅值小于 低 阈值, 该像素被排除。
# 　　Ⅲ.如果某一像素位置的幅值在两个阈值之间,该像素仅仅在连接到一个高于 高阈值的像素时被保留。

#  tips：对于Canny函数的使用，推荐的高低阈值比在2:1到3:1之间。
# 3X3
(high, wide) = NMS_Gauss3X3.shape
DT_Gauss3X3 = np.zeros([high, wide])
# 定义高低阈值
TL = 0.31 * np.max(NMS_Gauss3X3)
TH = 0.4 * np.max(NMS_Gauss3X3)

for i in range(1, high - 1):
    for j in range(1, wide - 1):
        if (NMS_Gauss3X3[i, j] < TL):
            DT_Gauss3X3[i, j] = 0
        elif (NMS_Gauss3X3[i, j] > TH):
            DT_Gauss3X3[i, j] = 255
        elif ((NMS_Gauss3X3[i - 1, j - 1:j + 1] < TH).any() or (NMS_Gauss3X3[i + 1, j - 1:j + 1]).any()
              or (NMS_Gauss3X3[i, [j - 1, j + 1]] < TH).any()):
            DT_Gauss3X3[i, j] = 255
            
# 5X5
(high, wide) = NMS_Gauss5X5.shape
DT_Gauss5X5 = np.zeros([high, wide])
# 定义高低阈值
TL = 0.31 * np.max(NMS_Gauss5X5)
TH = 0.4 * np.max(NMS_Gauss5X5)

for i in range(1, high - 1):
    for j in range(1, wide - 1):
        if (NMS_Gauss5X5[i, j] < TL):
            DT_Gauss5X5[i, j] = 0
        elif (NMS_Gauss5X5[i, j] > TH):
            DT_Gauss5X5[i, j] = 255
        elif ((NMS_Gauss5X5[i - 1, j - 1:j + 1] < TH).any() or (NMS_Gauss5X5[i + 1, j - 1:j + 1]).any()
              or (NMS_Gauss5X5[i, [j - 1, j + 1]] < TH).any()):
            DT_Gauss5X5[i, j] = 255
cv2.imshow( "DT_Gauss5X5",DT_Gauss5X5)       
cv2.imshow( "DT_Gauss3X3",DT_Gauss3X3)
cv2.waitKey(0)


# In[ ]:




