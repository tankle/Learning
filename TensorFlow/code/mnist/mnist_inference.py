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


#####################################################################
# 以下是实现了LeNet-5 的卷积模型
def inference_cnn(input_tensor, train, regularize):
    """

    :param input_tensor:
    :param train:       是否是训练
    :param regularize:
    :return:
    """

    with tf.variable_scope("cnn_layer1"):
        weight_1 = tf.get_variable("weight", [mnist_train.conv1_size, mnist_train.conv1_size,
                                             mnist_train.channel_num, mnist_train.conv1_deep],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias_1 = tf.get_variable('bias', [mnist_train.conv1_deep], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, weight_1, strides=[1, 1, 1, 1], padding='SAME')

        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias_1))

    with tf.name_scope("pool_layer2"):
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("cnn_layer3"):
        weight_2 = tf.get_variable('weight', [mnist_train.conv2_size, mnist_train.conv2_size,
                                              mnist_train.conv1_deep, mnist_train.conv2_deep],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias_2 = tf.get_variable('bias', [mnist_train.conv2_deep], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, weight_2, [1, 1, 1, 1], 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias_2))

    with tf.name_scope('pool_layer4'):
        pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        pool_shape = pool2.get_shape().as_list()

        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

        # 结构重构
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope("fc_layer5"):
        weight_3 = tf.get_variable('weight', [nodes, mnist_train.fc_size],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias_3 = tf.get_variable('bias', [mnist_train.fc_size], initializer=tf.constant_initializer(0.0))

        if regularize is not None:
            tf.add_to_collection('losses', regularize(weight_3))

        fc_1 = tf.nn.relu(tf.matmul(reshaped, weight_3) + bias_3)

        if train:
            tf.nn.dropout(fc_1, 0.5)

    with tf.variable_scope("fc_layer6"):
        weight_4 = tf.get_variable('weight', [mnist_train.fc_size, mnist_train.OUTPUT_NODE],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularize is not None:
            tf.add_to_collection('losses', regularize(weight_4))

        bias_4 = tf.get_variable('bias', [mnist_train.OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))

        fc_2 = tf.matmul(fc_1, weight_4) + bias_4

    return fc_2