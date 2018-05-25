# -*- coding: utf-8 -*-
#
# @author hztancong
#

import tensorflow as tf
import numpy as np
import time
import threading


def fifo_queue():
    print("begin")
    queue = tf.FIFOQueue(2, "int32")
    init = queue.enqueue_many(([0, 10],))
    x = queue.dequeue()
    y = x + 1

    q_inc = queue.enqueue([y])

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(5):
            val, _ = sess.run([x, q_inc])
            print(val)

    print("end")


def coordinate():
    """
    使用coordinate 来管理多线程
    :return:
    """

    def main_loop(coord, work_id):
        while not coord.should_stop():
            if np.random.rand() < 0.1:
                print("stop from ", work_id)
                coord.request_stop()
            else:
                print("work in ", work_id)

            time.sleep(1)

        print("stop for ", work_id)

    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=main_loop, args=(coord, i)) for i in range(5)]

    for t in threads: t.start()

    coord.join(threads)


if __name__ == "__main__":
    # fifo_queue()
    coordinate()