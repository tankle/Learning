# -*- coding: utf-8 -*-
#
# @author hztancong
#
import re
import pandas as pd
import tensorflow as tf


def test_re():
    line = "Toy Story (1995)"
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    rlt = pattern.match(line)
    groups = rlt.groups()
    for g in groups:
        print(g)
    print(rlt.group(1))


def test_pd():
    d = {'col1': [7, 4], 'col2': [1, 2]}
    one = pd.DataFrame(data=d)
    b = {'col2': [1, 2], 'col3': [3, 4]}
    two = pd.DataFrame(data=b)
    c = pd.merge(one, two)
    print(d)
    print(b)
    print(c)
    print(c.values)
    print(c.values.take(0, 1))


def test_tf():
    t1 = [[[1, 2, 3], [4, 5, 6]], [[11, 21, 31], [41, 51, 61]]]
    t2 = [[[7, 8, 9], [10, 11, 12]],[[71, 81, 91], [101, 111, 121]]]
    # 两个矩阵相连
    tt = tf.concat([t1, t2], 2)

    t1 = tf.constant([1, 2, 3])
    t2 = tf.constant([4, 5, 6])
    # concated = tf.concat(1, [t1,t2])这样会报错
    t1 = tf.expand_dims(t1, 1)
    t2 = tf.expand_dims(t2, 1)
    concated = tf.concat([t1, t2], 1)  # 这样就是正确的

    sess = tf.Session()
    aa = sess.run(tt)
    print(aa)
    bb = sess.run(concated)
    print(bb)

    cc = tf.reshape(concated, [-1, 6])
    print(sess.run(cc))

    ss = tf.reduce_sum(concated, axis=1, keep_dims=True)
    print(sess.run(ss))

    movie_categories_embed_matrix = tf.Variable(tf.random_uniform([5, 3], -1, 1), name="movie_categories_embed_matrix")
    sess.run(tf.global_variables_initializer())
    movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, [0, 1], name="movie_categories_embed_layer")
    movie_title_embed_layer_expand = tf.expand_dims(movie_categories_embed_layer, -1)
    print(sess.run(movie_categories_embed_matrix))
    print(sess.run(movie_categories_embed_layer))
    print(sess.run(movie_title_embed_layer_expand))

    ss = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
    print(sess.run(ss))

if __name__ == "__main__":
    # test_re()
    # test_pd()

    test_tf()