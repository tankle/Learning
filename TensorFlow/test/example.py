# -*- coding: utf-8 -*-
#
# @author hztancong
#

import tensorflow as tf

def tensorboard():
    v1 = tf.constant([2.0, 2.0], tf.float32, name='v1')
    v2 = tf.Variable(tf.random_normal([2]), name='v2')

    v3 = tf.add_n([v1, v2], name='v3')

    writer = tf.summary.FileWriter("log/", tf.get_default_graph())
    writer.close()

def device():
    v1 = tf.constant([2.0, 2.0], tf.float32, name='v1')
    v2 = tf.Variable(tf.random_normal([2]), name='v2')

    v3 = tf.add_n([v1, v2], name='v3')

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    sess.run(v3)
    sess.close()


device()