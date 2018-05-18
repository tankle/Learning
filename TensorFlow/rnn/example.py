# -*- coding: utf-8 -*-
#
# @author hztancong
#

import numpy as np


def rnn_scratch():
    x = [1, 2]
    state = [0.0, 0.0]

    # 两个不同的权重
    # 前一个节点输入的权重
    w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    # 当前节点输入的权重
    w_cess_input = np.asarray([0.5, 0.6])
    b_cell = np.asarray([0.1, -0.1])

    # 输出权重
    w_output = np.asarray([[1.0], [2.0]])
    b_output = 0.1

    for i in range(len(x)):
        input_value = np.dot(state, w_cell_state) + x[i] * w_cess_input + b_cell

        state = np.tanh(input_value)

        output = np.dot(state, w_output) + b_output

        print("input value:", input_value)
        print("state value:", state)
        print("output value:", output)


def lstm_scratch():
    pass

if __name__ == "__main__":
    rnn_scratch()
