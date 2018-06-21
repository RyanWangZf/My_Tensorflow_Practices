# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_X = np.linspace(-1,1,100)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
    
    # input data
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    
    # trainable parameters
    w = tf.Variable(0.0,name="weight")
    b = tf.Variable(0.0,name="biases")
    
    # define the loss function
    loss = tf.square(Y - X*w - b)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    # build session graph for running
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 1
        for i in range(10):
            for (x,y) in zip(train_X,train_Y): # batch_size = 1
                _,w_value,b_value = sess.run([train_op,w,b],feed_dict={X:x,Y:y}) # assign values on placeholders by feed_dict
            print("Epoch: {}, w: {}, b: {}".format(epoch,w_value,b_value))
            epoch+=1
        
   
    plt.plot(train_X,train_Y,"+",label="raw data")
    plt.plot(train_X,train_X.dot(w_value)+b_value,label="fitted line")
    plt.legend()
    plt.show()
    
    
    
    