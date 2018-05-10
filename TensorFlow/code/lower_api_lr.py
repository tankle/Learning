# -*- coding: utf-8 -*-
#
# @author hztancong
# 


import tensorflow as tf

x = tf.constant([[1.0, 5], [2.0, 2], [3.0, 6], [4.0, 2]], dtype=tf.float32)
y_true = tf.constant([[0], [-1.0], [-2.0], [-3.0]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.001)

train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(0, 300):
    _, loss_value = sess.run((train, loss))
    if i % 30 == 0:
        print(loss_value)

print("============")

print(sess.run(y_pred))
print(train)
print(linear_model.trainable_weights[0])
for tv in tf.trainable_variables():
    vv = tf.get_default_graph().get_tensor_by_name(tv.name)
    print(tv.name)
    print(sess.run(vv))