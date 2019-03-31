#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 23:12
# @Author  : Despicable Me
# @Site    : 
# @File    : ridge_regression_traffic.py
# @Software: PyCharm
# @Explain :

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

data = np.genfromtxt('data.csv', delimiter= ',', usecols= (1, 2, 3, 4, 5))

# delimiter=','表示按逗号','分隔
# usecols=(1, 2, 3, 4, 5)表示采用数据中的哪几列
# 本例中总共采集了5列数据，数据的第2列(下标为1)到第6列(下标为5)
# 这五列分别代表了数据的5个特征：
# HR：一天中的第几个小时(0-23)
# WEEK_DAY：一周中的第几天(0-6)
# DAY_OF_YEAR：一年中的第几天(1-365)
# WEEK_OF_YEAR：一年中的第几周(1-53)
# TRAFFIC_COUNT：交通流量

plt.figure(1)

plt.plot(data[:,4])   # applying the plt to show the traffic information

X= data[:, :4]        #

y = data[:, 4]

poly = PolynomialFeatures(6)    # 用来创建最高次数6次方的多项式特征，多次试验后决定采用6次

X = poly.fit_transform(X)       # X为创建的多项式特征

# 4. 划分训练集和测试集
# 将所有数据划分为训练集和测试集
# test_size表示测试集的比例
# random_state是随机数种子

train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size= 0.3, random_state= 0)

# 创建回归器，并进行训练
clf = Ridge(alpha= 1.0, fit_intercept= True)

clf.fit(train_set_X, train_set_y)   # 调用fit函数使用训练集训练回归器

clf.score(test_set_X, test_set_y)

# 利用测试集计算回归曲线的拟合优度
# clf.score返回值为0.7375
# 拟合优度，用于评价拟合好坏，最大为1，无最小值
# 当对所有输入都输出同一个值时，拟合优度为0

start = 200

end = 300

y_pre = clf.predict(X)   # 调用predict函数的拟合值

time = np.arange(start, end)

plt.figure(2)

plt.plot(time, y[start:end], 'b', label = 'real')

plt.plot(time, y_pre[start:end], 'r', label = 'predict')

plt.legend(loc = 'upper left')

plt.show()            #