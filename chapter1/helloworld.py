# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    b = tf.Variable(tf.zeros([100]))
    W = tf.Variable(tf.random_uniform([784,100],-1,1))
    x = tf.placeholder(name="x",dtype=float)
    relu = tf.nn.relu(tf.matmul(W,x)+b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs = np.arange(100).reshape(100,1)
        result = sess.run(relu,feed_dict={x:inputs})
        print(result)

