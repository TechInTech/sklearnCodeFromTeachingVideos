#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 15:32
# @Author  : Despicable Me
# @Site    : 
# @File    : Test_tensorflow3.py
# @Software: PyCharm
# @Explain :

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    # 防止警告信息出现
import tensorflow as tf

sess = tf.InteractiveSession()

input1 = tf.placeholder(tf.float32)       # 创建占位符

input2 = tf.placeholder(tf.float32)

res= tf.multiply(input1, input2)

print(res.eval(feed_dict ={input1:[7.], input2:[2.]}))   # 参数赋值，求值，并输出