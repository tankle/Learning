# -*- coding: utf-8 -*-
#
# mnist 训练过程
#
# @author hztancong
#
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# batch 大小
from code.mnist import mnist_inference

BATCH_SIZE = 100
# 学习率
learing_rate_base = 0.8
# 学习率的衰减
learing_rate_decay = 0.99

# 移动平均的衰减指数
moving_rate_decay = 0.99

# 正则参数
lambda_rate = 0.0001
# 训练步骤
train_steps = 30000

# 输入维度
INPUT_NODE = 28*28
# 输出维度
OUTPUT_NODE = 10
# 隐藏层层数
LAYER1_NODE = 500


SAVE_PATH = "model/"
MODEL_NAME = "mnist"

def train(mnist):

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], 'x_input')
    y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], 'y_input')

    regularize = tf.contrib.layers.l2_regularizer(lambda_rate)

    y_pred = mnist_inference.inference(x, regularize)

    global_steps = tf.Variable(0, trainable=False)

    # 创建指数移动平均参数
    ema = tf.train.ExponentialMovingAverage(moving_rate_decay, global_steps)
    ema_op = ema.apply(tf.trainable_variables())

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 动态学习率
    learing_rate = tf.train.exponential_decay(learing_rate_base,
                                              global_steps,
                                              mnist.train.num_examples / BATCH_SIZE,
                                              learing_rate_decay)
    # 优化方法
    train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss, global_step=global_steps)

    # 以下两种控制流程的效果一样
    #train_op = tf.group(train_step, ema_op)
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op("train")

    # 保存器
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(train_steps):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, now_steps = sess.run([train_op, loss, global_steps], feed_dict={x: xs, y:ys})

            # 每运行1000次，打印一次输出并保存模型
            if i % 1000 == 0:
                print("After %d steps, loss value is %g. " % (now_steps, loss_value))
                save_path = os.path.join(SAVE_PATH, MODEL_NAME)
                saver.save(sess, global_step=global_steps, save_path=save_path)


def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_DATA", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    # tf.app.run()
    main()