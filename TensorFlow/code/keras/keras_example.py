# -*- coding: utf-8 -*-
#
# @author hztancong
#

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from numpy.random import seed
import keras

seed(131)

data_dim = 4    # 特征数量
timestep = 3    # 日期数量
num_classes = 2 # 标签分类
num_samples = 5 # 训练样本数据量
num_predict = 3 # 带预测数据


x_train = np.random.random((num_samples, timestep, data_dim))
y_train = np.random.randint(num_classes, size=(num_samples, num_classes))

print(x_train)
print(y_train)

x_test = np.random.random((num_predict, timestep, data_dim))
y_test = np.random.randint(num_classes, size=(num_predict, num_classes))


model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(timestep, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

tb_cb = keras.callbacks.TensorBoard(log_dir="log/", write_images=1, histogram_freq=1)
cbks = [tb_cb]

hist = model.fit(x_train, y_train, batch_size=100, epochs=50,  callbacks=cbks,validation_data=(x_test, y_test))
y = model.predict_proba(x_test)

print("preding")
print(y)