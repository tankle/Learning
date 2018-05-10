# -*- coding: utf-8 -*-
#
# @author hztancong
#

import tensorflow as tf

tf.Session()

tf.global_variables_initializer()

tf.placeholder()
tf.random_normal()
tf.train.GradientDescentOptimizer()

tf.nn.dropout(h_fc1, keep_prob)
initial = tf.truncated_normal(shape, stddev=0.1)
tf.reshape(x, [-1, 28, 28,1])

tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

