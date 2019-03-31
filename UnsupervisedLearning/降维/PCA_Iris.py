#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 23:13
# @Author  : Despicable Me
# @Site    : 
# @File    : PCA_Iris.py
# @Software: PyCharm
# @Explain :

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  #加载PCA算法包
from sklearn.datasets import load_iris #加载鸢尾花数据集导入函数

data = load_iris()
# 以字典形式加载鸢尾花数据集
y = data.target   #使用y表示数据集中的标签
x = data.data     #使用x表示数据集中属性数据

pca = PCA(n_components= 2)   #加载PCA算法，设置将为后的主成分数目为2

reduced_X = pca.fit_transform(x)   #对原始数据进行降维，保存在reduced_x中

red_x, red_y = [], []        #第一类数据点
blue_x, blue_y = [], []      #第二类数据点
green_x, green_y = [], []    #第三类数据点


#降维并不改变数据所属的类别，只是将原数据的维度从高维降为低维
#降维后的数据，按照所属维度一次存入各类中
#三类样本的标签分别用0,1,2表示

for i in range(len(reduced_X)):
    if y[i] == 0:     #
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker ='x')  #c = color
plt.scatter(blue_x,blue_y,c='b', marker = 'D')
plt.scatter(green_x, green_y, c='g', marker = '.')

plt.show()