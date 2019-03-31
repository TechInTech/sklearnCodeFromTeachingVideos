#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 14:19
# @Author  : Despicable Me
# @Site    : 
# @File    : Test_tensorflow2.py
# @Software: PyCharm
# @Explain :

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    # 防止警告信息出现
import tensorflow as tf

sess = tf.InteractiveSession()            # 创建交互式会话

a = tf.Variable([1.0, 2.0])

b = tf.constant([3.0, 4.0])

sess.run(tf.global_variables_initializer())    # 变量初始化

res = tf.add(a, b)                             # 创建加法操作

print(res.eval())