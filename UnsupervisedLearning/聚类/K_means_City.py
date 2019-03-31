#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 19:45
# @Author  : Despicable Me
# @Site    : 
# @File    : K_meansDemo.py
# @Software: PyCharm
# @Explain :


import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    f = open(filePath, 'r+')
    lines = f.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(',')  #listobj.strip() delete the spce before or behind of string
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName

if __name__ == '__main__':

    #  一个python的文件有两种使用的方法
    # 第一是直接作为脚本执行
    # 第二是import到其他的python脚本中被调用（模块重用）执行
    # 因此if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程
    # 在if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行
    # 而import到其他脚本中是不会被执行的

    data, cityName =loadData('city.txt')
    km = KMeans(n_clusters=3)
    label = km.fit_predict(data)  # sklearn.cluster.fit_predict 中.fit_predict(X)相当于先Calling fit(X),然后在predict(X)
    # label 为聚类后各数据所属的标签
    # km.fit_predict 计算簇中心以及为簇分配序号（每组样本所属的簇号）

    expenses = np.sum(km.cluster_centers_, axis=1) #km.cluster_centers_为数据的聚类中心，按矩阵的横轴计算每一行的总和
    # 聚类中心的数值加和，也就是平均消费水平


    #print(expenses)
    CityCluster = [],[],[]  #创建一个元组，注意元组的标志为逗号，且元组中每个元素都是列表
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print('Expeses: %.2f' % expenses[i])
        print(CityCluster[i])