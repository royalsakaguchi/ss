#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


f=np.array([[1,5,15,8],[1,7,14,9],[3,7,10,11],[1,0,4,6]])


# In[3]:


# Roberts算子
Roberts_x= np.array([[1, 0],[0, -1]])
Roberts_y= np.array([[0, -1],[1, 0]])
(high, wide) = f.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=f

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

img_Roberts


# In[9]:


# Sobel算子
# Sobel算子是高斯平滑和微分操作的结合体，因此它的抗噪声能力比较好。
Sobel_x= np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
Sobel_y= np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
(high, wide) = f.shape
# 创建待处理图像，扩大边缘检测范围，使得整个图像都可以被检测到
padding=np.zeros((high+2,wide+2),np.uint8)
padding[1:-1,1:-1]=f

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
print(img_Sobel)


# In[7]:


# Laplace算子
laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
#图像周边填充0
padding = np.zeros((high+2, wide+2), np.uint8)
padding[1:-1, 1:-1] = f
#创建结果图像
Laplace = np.zeros((high, wide), np.uint8)
# 卷积运算
for i in range(0, high):  # 5*5的矩阵从左到右运算3次，从上到下运算3次
    for j in range(0, wide):
        window = padding[i:i+3, j:j+3]
        Laplace[i, j] = np.abs(np.sum(laplace*window)) #矩阵内积
Laplace


# In[ ]:




