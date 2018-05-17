# -*- coding: utf-8 -*-
#
# mnist 预测过程
# @author hztancong
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from code.mnist import mnist_train, mnist_inference

eval_per_sec = 20

def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_train.INPUT_NODE], name='x_input')
    y = tf.placeholder(tf.float32, [None, mnist_train.OUTPUT_NODE], name='y_input')

    # 不需要正则化值
    y_pred = mnist_inference.inference(x, None)

    # 定义评价指标
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ema = tf.train.ExponentialMovingAverage(mnist_train.moving_rate_decay)
    ema_to_restore = ema.variables_to_restore()

    saver = tf.train.Saver(ema_to_restore)


    while True:
        with tf.Session() as sess:
            ckp = tf.train.get_checkpoint_state(mnist_train.SAVE_PATH)

            if ckp and ckp.model_checkpoint_path:
                saver.restore(sess, ckp.model_checkpoint_path)
                global_step = ckp.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_value = sess.run(accuracy, feed_dict={x: mnist.validation.images,
                                                               y: mnist.validation.labels})
                print("After %s training steps, validation accuracy is %g. " %(global_step, accuracy_value))

            else:
                print("no checkpoint")
                return

        break


if __name__ == "__main__":
    mnist = input_data.read_data_sets("../MNIST_DATA", one_hot=True)
    evaluate(mnist)