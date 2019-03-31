#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/11 9:44
# @Author  : Despicable Me
# @Site    : 
# @File    : NMF_FetchFace.py
# @Software: PyCharm
# @Explain :


import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState   #加载RandomState用于创建随机种子

#设置基本参数及加载数据

n_row, n_col = 2, 3  #设置图像展示时的排列情况 2 * 3 矩阵
n_components = n_row * n_col   #设置提取的特征的数目
image_shape = (64, 64)   #设置人脸数据图片的大小

dataset = fetch_olivetti_faces(shuffle= True, random_state= RandomState(0))
faces = dataset.data    #加载数据，并打乱数据

def plot_gallery(title, images, n_col = n_col, n_row = n_row):
    plt.figure(figsize= (2. * n_col, 2.26 * n_row))   #创建图片，并指定图片的大小（英寸）
    plt.suptitle(title, size = 16)    #设置标题，及字号大小

    for i, comp in enumerate(images):
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        # enumerate(sequence, [start=0])
        # sequence -- 一个序列、迭代器或其他支持迭代对象
        # start -- 下标起始位置
        #
        # example:
        # seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
        #
        # >>> list(enumerate(seasons, start=1))       # 下标从 1 开始
        # [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

        plt.subplot(n_row, n_col, i + 1)    #选择画制的子图
        vmax = max(comp.max(), -comp.min())

        plt.imshow(comp.reshape(image_shape), cmap = plt.cm.gray,          #对数值归一化，并以灰度图形式显示
                   interpolation= 'nearest', vmin = -vmax, vmax = vmax)

        # matplotlib.pyplot.imshow(
        #     X, cmap=None, norm=None, aspect=None,
        #     interpolation=None, alpha=None, vmin=None, vmax=None,
        #     origin=None, extent=None, shape=None, filternorm=1,
        #     filterrad=4.0, imlim=None, resample=None,
        #     url=None, hold=None, data=None, **kwargs)

        plt.xticks(())    #去除子图的坐标轴标签
        plt.yticks(())    #
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)   #对子图位置及间隔进行调整

    # subplots_adjust(left=None, bottom=None, right=None, top=None,
    #            wspace=None, hspace=None)
    # left  = 0.125  # 子图(subplot)距画板(figure)左边的距离
    # right = 0.9    # 右边
    # bottom = 0.1   # 底部
    # top = 0.9      # 顶部
    # wspace = 0.2   # 子图水平间距
    # hspace = 0.2   # 子图垂直间距

#创建特征提取的对象NMF, 使用PCA作为对比

estimators = [('Eigenfaces - PCA using randomized SVD',             #将PCA和NMF实例存放在同一个列表中
               decomposition.PCA(n_components= 6, whiten= True)),    #PCA实例
              ('Non-negtive compenents - NMF',
               decomposition.NMF(n_components= 6, init= 'nndsvda',  #NMF实例
                                 tol= 5e-3))]

#降维后数据点的可视化
for name, estimator in estimators:                    #分别调用PCA和NMF方法
    #estimators 为由元组构成的列表，每次从estimators迭代出的对象也为元组，分别赋予元组 name, estimators
    estimator.fit(faces)                              #调动PCA或NMF提取特征
    components_ = estimator.components_               #获取提取的特征

    plot_gallery(name, components_[:n_components])    #按照固定的格式进行排列

plt.show()

