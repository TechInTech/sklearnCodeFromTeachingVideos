#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/11 15:41
# @Author  : Despicable Me
# @Site    : 
# @File    : K_means_Application.py
# @Software: PyCharm
# @Explain :

# 利用图像的灰度,颜色,纹理,形状等特征,把图像分成若干互不重叠的区域
# 并使这些特征在同一区域内呈现相似性,在不同区域之间存在明显的差异性
# 然后就可以将分割的图像中具有独特性质的区域提取出来用于不同的研究
# 图像分割的应用: 轮毂裂纹图像的分割; 肝脏CT图像的分割
# 图像分割常用方法:
# 1. 阈值分割: 对图像灰度值进行度量,设置不同的阈值,达到分割的目的
# 2. 边缘分割: 对图像边缘进行检测,即检测图像中灰度值发生跳变的地方,则为一片区域的边缘
# 3. 直方图法: 对图像的颜色建立直方图,直方图的波峰波谷能够表示一块区域的颜色值范围,来达到分割的目的
# 4. 特定理论: 基于聚类分析,小波变换等理论完成图像分割
# 本例描述
# 目标: 利用K-means聚类算法对图像像素点进行聚类实现简单的图像分割
# 输出: 同一聚类中的点使用相同的颜色标记,不同聚类颜色不同
# 实验步骤:
# 1. 建立工程并导入sklearn包
# 2. 加载图片进行预处理
# 3. 加载K-means聚类算法
# 4. 对像素点进行聚类并输出

import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def loadData(filePath):
    f = open(filePath, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            # 获取图片的pixel值(R,G,B)
            # x, y, z 分别代表获取到的R, G, B值
            x,y,z = img.getpixel((i,j))
            # 将每个像素点RGB颜色处理到0-1范围内并放进data
            data.append([x/256.0, y/256.0, z/256.0])  # 像素点进行灰度处理,并RGB像素
                                                      # 以列表的形式存储在data列表中
            # x, y, z的每个值都在0~255范围内
            # x/256.0 >= 0/256, 0/256 = 0, x/256.0 >=0
            # x/256.0 <= 255/256
            # 255/256 = (256-1)/256 = 256/256 - 1/256 = 1- 1/256
            # x/256.0 <= 1- 1/256
    f.close()
    return np.mat(data), m, n
    # numpy.mat()方法将数据变为矩阵, 返回类型为numpy.matrixlib.defmatrix.matrix

imgData, row, col = loadData('bull.jpg')
# imgData, row, col = loadData('person.png')

km = KMeans(n_clusters=3)
# aa = []

label = km.fit_predict(imgData)
label = label.reshape([row, col])
pic_new = image.new("L", (row, col))

for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256 / (label[i][j] + 1)))
        # putpixel(xy, value) ,xy 指定像素位置，value 为像素点需要修改的值
        # Modifies the pixel at the given position. The color is given as
        # a single numerical value for single-band images, and a tuple for
        # multi-band images.

        # aa.append(int(256/ (label[i][j]+1)))

pic_new.save("result_bull.jpg", "JPEG")
# pic_new.save("result_person.jpg", "JPEG")