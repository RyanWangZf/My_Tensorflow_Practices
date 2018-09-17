# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    x = tf.placeholder(tf.float32,[None,784],name="x")
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    # predicted label
    y = tf.nn.softmax(tf.matmul(x,W)+b)
    # ground truth label
    y_ = tf.placeholder("float",[None,10],name="y_true")

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
    # define training operation
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
        
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_op,feed_dict={x:batch_xs,y_:batch_ys})
        print("iteration {}".format(i))
    
    # test model on test sets
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

        
