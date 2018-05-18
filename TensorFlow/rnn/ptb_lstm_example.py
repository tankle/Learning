# -*- coding: utf-8 -*-
#
# @author hztancong
#
from rnn import reader
import tensorflow as tf

ptb_path = "data/simple-examples/data/"
train, valid, test, vocabulary = reader.ptb_raw_data(ptb_path)
#print(train)
print(len(train))
print(train[:100])

print(vocabulary)
# 隐藏层大小
hidden_size = 200

tf.get_variable_scope().reuse_variables()

cell()