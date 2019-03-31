#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 11:49
# @Author  : Despicable Me
# @Site    : 
# @File    : Test_tensorflow.py
# @Software: PyCharm
# @Explain :

import tensorflow as tf
# import cv2
# import pygame

mat1 = tf.constant([[3., 3.]])     # 创建一个1*2的矩阵
mat2 = tf.constant([[2.], [2.]])   # 创建一个2*1的矩阵

# 构建阶段
product = tf.matmul(mat1, mat2)    # 创建op执行两个矩阵的乘法

sess = tf.Session()                # 创建会话sess
# 执行阶段
res = sess.run(product)            # 会话(sess)将图（graph中的op分发到CPU或GPU上进行计算，
                                   # 然后将产生的tensorflow返回，tensor就是numpy.ndarray对象
print(res)

sess.close()