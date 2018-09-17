# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == "__main__":

    mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
    sess = tf.InteractiveSession()

    # define the NN's structure Variables
    in_units = 784
    h1_units = 300
    w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    w2 = tf.Variable(tf.zeros([h1_units,10]))
    b2 = tf.Variable(tf.zeros([10]))
    
    x = tf.placeholder(tf.float32,[None,in_units])
    # makes the dropout ratio a placeholder cause it is below 1 but equals 1 while training
    dropout_rate = tf.placeholder(tf.float32)
    
    # NN's structure
    hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    hidden1 = tf.nn.dropout(hidden1,dropout_rate)
    y = tf.nn.softmax(tf.matmul(hidden1,w2)+b2)
    
    y_ = tf.placeholder(tf.float32,[None,10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

    train_op = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
    
    # training
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        print("iteration:",i)
        batch_xs,batch_ys = mnist.train.next_batch(100)
        train_op.run({x:batch_xs,y_:batch_ys,dropout_rate:0.75})

    # prediction
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(acc.eval({x:mnist.test.images,y_:mnist.test.labels,dropout_rate:1.}))

    

