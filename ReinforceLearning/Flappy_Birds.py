#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 11:13
# @Author  : Despicable Me
# @Site    : 
# @File    : Flappy_Birds.py
# @Software: PyCharm
# @Explain :

from __future__ import print_function

import tensorflow as tf
import cv2                            # 导入opencv库
import sys                            # 导入sys模块
sys.path.append("F:\Projects_Python\Dong\SklearnLibLearn\ReinforceLearning\game")              # 添加game目录到系统环境变量
import wrapped_flappy_bird as game    # 加载游戏
import random
import numpy as np
from collections import deque         # 导入双端队列
import pygame


# 设置超参数
GAME = 'bird'                         # 设置游戏名称
ACTIONS = 2                           # 设置游戏动作数目(点击不点击屏幕)
GAMMA = 0.99                          # 设置增强学习更新公式中对的累计折扣因子
OBSERVE = 10000.                      # 观察期 一万次迭代(随机指定动作获得D）
EXPLORE = 2000000.                    # 探索期
FINAL_EPSILON = 0.0001                # 设置ε的最终最小值
INITIAL_EPSILON = 0.1                 # 设置ε贪心策略中的ε初始值
REPLAY_MEMORY = 50000                 # 设置Replay Memory的容量
BATCH = 32                            # 设置每次网络参数更新时用的样本数目
FRAME_PRE_ACTION = 1                  # 设置几帧图像进行一次动作

#### 创建深度神经网络  ####

#1 首先定义一个函数，该函数用于生成形状为shape的张量(高维数组)
#  张量的初始值服从正态分布，且方差为0.01
def weight_variable(shape):                                   # 生成权值变量函数
    initial = tf.truncated_normal(shape, stddev= 0.01)
    # shape表示生成张量的维度，mean是均值，stddev是标准差。
    # 这个函数产生正太分布，均值和标准差自己设定
    # 这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，
    # 那就重新生成和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与
    # 均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的
    return tf.Variable(initial)

#2 定义另一个函数，用于生成偏置项，初始置位0.01
def bias_varaiable(shape):                                    # 生成偏置项
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

#3 定义卷积操作，实现卷积核W在数据x上卷积操作
#  strides为卷积核的移动步长，padding为卷积的一种模式，
#  参数same表示滑动范围超过边界
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides= [1, stride, stride, 1], padding = 'SAME')

#4 定义池化操作，此程序中通过调用max_pool执行最大池化操作，大小为2*2，stride步长为2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME')

#4 定义深度神经网络的参数和偏置
def createNetwork():
    W_conv1 = weight_variable([8, 8, 4,32])
    b_conv1 = bias_varaiable([32])                            # 第一个卷积层

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_varaiable([64])                            # 第二个卷积层

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_varaiable([64])                            # 第三个卷积层

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_varaiable([512])                             # 第一个全连接层

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_varaiable([ACTIONS])                         # 第二个全连接层

    # 输入层
    s = tf.placeholder('float', [None, 80, 80, 4])

    # 隐含层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)          ## ##
    h_pool1 = max_pool_2x2(h_conv1)                                   ##
                                                                      ##
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)       ##
                                                                      ## 对各层进行连接
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)       ##
                                                                      ##
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])                    ##
                                                                      ##
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)        ## ##

    # 输出层
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

#5 定义神经网络训练函数1
def trainNetwork(s, readout, h_fc1, sess):
    # 定义损失函数
    a = tf.placeholder('float', [None, ACTIONS])
    y = tf.placeholder('float', [None])

    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)

    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 开启游戏模拟器，会打开一个模拟器的窗口，实时显示游戏的信息
    game_state = game.GameState()

    # 创建双端队列用于存放replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # 获取游戏的初始状态，设置动作为不执行跳跃， 并将初始状态修改为*0*80*4大小
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 用于加载或保存网络参数
    saver= tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state('saved_networks')
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print('Successfully loaded:', checkpoint.model_checkpoint_path)
    else:
        print('Could not find old network weights')

    # 开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    while 'flappy bird' != 'angry bird':
        # 使用epsilon贪心策略选择一个动作
        readout_t = readout.eval(feed_dict={s:[s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PRE_ACTION == 0:
            # 执行一个随机动作
            if random.random() <= epsilon:
                print('--------Random Action--------')
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            # 由神经网络计算的Q(s,a)值选择对性的动作
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1             # 不执行跳跃动作

        # 随着游戏的进行，不断降低epsilon,减小随机动作
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITINAL_EPSILON - FINAL_EPSILON)/EXPLORE

        # 执行选择的动作，并获得下一状态及回报
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)),cv2.COLOR_BGR2GRAY)

        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis = 2)

        # 将状态转移过程存储在D中，用于更新参数时采样
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 过了观察期，才会进行网络参数的更新
        if t > OBSERVE:
            # 从D中随机获取采样、采取的动作、获取的回报、下一状态分组存放
            minibatch = random.sample(D, BATCH)

            # 分别将当前状态、采取的动作、获得的回报、下一状态分组存放
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # 计算Q(s,a)的新值
            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s:s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 如果游戏结束，则只有反馈值
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # 使用梯度下降更新网络参数
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # 状态发生改变， 用于下次循环
        s_t = s_t1
        t += 1

        # 每进行10000次迭代，保留一下网络参数
        if t % 10000 ==0:
            saver.save(sess, 'saved_networks/' + GAME +'-dpn', global_step = t)

        # 打印游戏信息
        state = ''
        if t <= OBSERVE:
            state = 'observe'
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = 'explore'
        else:
            state = 'train'

        print('TIMESTEP', t, '/ STATE', state, '/ EPSILON', epsilon, \
              '/ ACTION', action_index, '/ REWARD', r_t, \
              '/ Q_MAX %e' % np.max(readout_t))

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

if '__name__' == '__main__':
    playGame()