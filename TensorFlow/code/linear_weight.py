# -*- coding: utf-8 -*-
#
# @author hztancong
#


import tensorflow as tf
import numpy as np

# 生成训练数据
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3


# 构建模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, end='')
        print(sess.run(W), sess.run(b))
