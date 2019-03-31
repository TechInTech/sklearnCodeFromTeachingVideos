#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/16 21:44
# @Author  : Despicable Me
# @Site    : 
# @File    : KNN.py
# @Software: PyCharm
# @Explain :

import numpy as np
from os import listdir
from sklearn import neighbors

def img2vector(filename):
    retMat = np.zeros([1024], int)
    fr = open(filename)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i*32 + j] =lines[i][j]
    return retMat

def readDataset(path):
    filelist = listdir(path)
    numFiles = len(filelist)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles, 10])
    for i in range(numFiles):
        filePath = filelist[i]
        digit = int(filePath.split('_')[0])
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet, hwLabels

train_dataSet, train_hwLabels = readDataset('trainingDigits')

knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)

knn.fit(train_dataSet, train_hwLabels)

dataSet, hwLabes = readDataset('testDigits')

res = knn.predict(dataSet)

error_num = np.sum(res != hwLabes)

num = len(dataSet)

print('Total num:', num, 'Wrongnum:', error_num, 'WrongRate:', error_num/float(num))