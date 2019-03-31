#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/11 22:09
# @Author  : Despicable Me
# @Site    : 
# @File    : house.py
# @Software: PyCharm
# @Explain :

import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

datasets_X = []
datasets_Y = []
f = open('prices.txt','r')
lines = f.readlines()

for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
length = len(datasets_X)

datasets_X = np.array(datasets_X).reshape(length, 1)
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)

X = np.arange(minX, maxX).reshape([-1, 1])
# 我们事先不知道np.arange(minX, maxX)的维度，但我们想要得到的数组维度为一列，则通过.reshape(-1,1)
# 我们可以得到np.arange(minX, maxX) 的shape中 n_row * n_col / 1 行的 一列数组
# 一个参数为-1时，那么reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值

linear = linear_model.LinearRegression()
linear.fit(datasets_X, datasets_Y)

# 查看回归方程系数
print('Coefficients:', linear.coef_)

# 查看回归方程的截距
print('intercept:', linear.intercept_)

# 绘制图像
plt.scatter(datasets_X, datasets_Y, color = 'r')
plt.plot(X, linear.predict(X), color = 'b')
plt.grid(True)
plt.axis([500, 4500, 100, 900])
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()