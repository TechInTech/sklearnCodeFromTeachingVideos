#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/11 19:28
# @Author  : Despicable Me
# @Site    : 
# @File    : Stock_Index_Predict.py
# @Software: PyCharm
# @Explain :

import pandas as pd
import numpy as np
from sklearn import svm                                 # 从sklearn 包导入svm 算法包
from sklearn.model_selection import train_test_split    # 从sklearn 中导入(交叉验证包)(cross_validation 在0.20 版之后被舍弃)
                                                        # 统一采用 model_selection
# 数据的加载 && 数据预处理
data = pd.read_csv('000777.csv', encoding = 'gbk', parse_dates=[0], index_col= 0)
data.sort_index(0, ascending= True, inplace= True)
# axis : index, columns to direct sorting
# level : int or level name or list of ints or list of level names
#     if not None, sort on values in specified index level(s)
# ascending : boolean, default True
#     Sort ascending vs. descending
# inplace : bool, default False
#     if True, perform operation in-place
dayfeature = 150
featurenum = 5 * dayfeature

x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))
y = np.zeros((data.shape[0] - dayfeature))

for i in range(0, data.shape[0] - dayfeature):
    x[i, 0:featurenum] = np.array(data[i:i+dayfeature][[u'收盘价', u'最高价',
         u'最低价', u'开盘价',u'成交量']]).reshape((1,featurenum))
    # 将数据中的'收盘价' '最高价' '最低价' '开盘价' '成交量' 存入x数组中
    x[i, featurenum] = data.loc[i + dayfeature][u'开盘价']
    # .loc or .iloc
    # df: dataframe , 列名为 A B C D
    # A    B    C     D
    #
    # 0    ss   小红  8
    #
    # 1    aa   小明  d
    #
    # 4    f          f
    #
    # 6    ak   小紫  7
    # 选取标签为A和C的列，并且选完类型还是dataframe
    # df = df.loc[:, ['A', 'C']]
    # df = df.iloc[:, [0, 2]]
    # 如果你想要选取某一行的数据，可以使用df.loc[[i]]或者df.iloc[[i]]
    # 最后一列记录当日的开盘价

for i in range(0, data.shape[0] - dayfeature):
    if data.loc[i + dayfeature][u'收盘价'] >= data.loc[i + dayfeature][u'开盘价']:
        y[i] = 1
    else:
        y[i] = 0
    # 如果当天收盘价高于开盘价，y[i] = 1代表涨，0代表跌

# 创建svm实例，并进行交叉验证

clf = svm.SVC(kernel='sigmoid')
# 调用svm函数， 并设置kernel参数，默认为rbf， 其他可选项为： 'linear' 'poly' 'sigmoid'
result = []
for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # x和y的验证集合测试集，切分比率为 8:2
    clf.fit(x_train, y_train)
    # 训练数据进行训练
    result.append(np.mean(y_test == clf.predict(x_test)))
    # 将预测数据与测试集的验证数据比对

print("svm classifier accuacy:")

print(result)