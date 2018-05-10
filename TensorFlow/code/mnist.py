# -*- coding: utf-8 -*-
#
# @author hztancong
#

import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets('MNIST_data', one_hot=True)
mnist.count()
print("read data end")
x_data = tf.placeholder("float", shape=[None, 784])
y_data = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))



y = tf.nn.softmax(tf.matmul(x_data, W) + b)

# 交叉熵损失函数
loss = -tf.reduce_sum(y_data * tf.log(y))


# sess = tf.Session()
sess = tf.InteractiveSession()
# 全局变量初始化
sess.run(tf.global_variables_initializer())
# tf.global_variables_initializer().run()


train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

correct_prection = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prection, "float"))

for i in range(1000):
    batch = mnist.train.next_batch(50)
    # sess.run(train)
    if i % 100 == 0:
        train_acc = sess.run(accuracy, feed_dict={x_data: batch[0]
                                             , y_data: batch[1]})
        print(i)
        print(train_acc)
    sess.run(train, feed_dict={x_data:batch[0], y_data: batch[1]})

print(sess.run(accuracy, feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels}))