# -*- coding: utf-8 -*-
#
# 定义网络结构，方便训练和测试复用
# @author hztancong
#


import tensorflow as tf

from code.mnist import mnist_train


def get_weights_variable(shape, regularize=None):
    """
    获取权重变量
    :param shape:
    :param regularize:
    :return:
    """
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularize is not None:
        tf.add_to_collection("losses", regularize(weights))

    return weights


def inference(input_tensor, regularize):
    """
    定义网络结构
    :param input_tensor:
    :param regularize:
    :return:
    """
    with tf.variable_scope("layer1"):
        w1 = get_weights_variable([mnist_train.INPUT_NODE, mnist_train.LAYER1_NODE], regularize)
        b1 = tf.get_variable("bias", [mnist_train.LAYER1_NODE], initializer=tf.constant_initializer(0.1))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)

    with tf.variable_scope("layer2"):
        w2 = get_weights_variable([mnist_train.LAYER1_NODE, mnist_train.OUTPUT_NODE], regularize)
        b2 = tf.get_variable("bias", [mnist_train.OUTPUT_NODE], initializer=tf.constant_initializer(0.1))

        layer2 = tf.matmul(layer1, w2) + b2

    return layer2

