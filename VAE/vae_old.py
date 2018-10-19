# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


import pdb

def dense(x,nUnit,scopeName,parameters,activation="relu"):
    n_in = x.get_shape()[-1].value

    with tf.name_scope(scopeName) as scope:
        kernel = tf.get_variable(scope+"w",shape=[n_in,nUnit],dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        bias   = tf.Variable(tf.constant(.1,shape=[nUnit],dtype=tf.float32),name="b")
        z = tf.matmul(x,kernel) + bias
        if activation == "relu":
            z = tf.nn.relu(z,name=scope)

        parameters += [kernel,bias]

    return z
    
def main():
    
    # load data
    mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
    x_all,y_all = mnist.train.images,mnist.train.labels
    print("MNIST train images:",mnist.train.images.shape)
    print("MNIST test  images:",mnist.test.images.shape)
    
    # set params
    nClass = y_all.shape[1] # 10
    nFeature = x_all.shape[1] # 784
    batchSize = 100
    nEpoch = 2000
    latentDim = 2 # dim of the z code
    epsilonStd = 1.

    trainableParam = []

    # construct calculation map
    x = tf.placeholder(tf.float32,[None,nFeature])
    y = tf.placeholder(tf.float32,[None,nClass])
    dropoutRate = tf.placeholder(tf.float32)

    # encoder
    enc_h1 = dense(x,nUnit=256,scopeName="enc_h1",parameters=trainableParam)
    enc_drop1 = tf.nn.dropout(enc_h1,dropoutRate,name="enc_drop1")  # numBatch,256
    
    # get mean and variance of p(z|x)
    z_mean = dense(enc_drop1,latentDim,"z_mean",parameters=trainableParam,activation=None)
    z_log_var = dense(enc_drop1,latentDim,"z_log_var",parameters=trainableParam,activation=None)

    def sampling(z_mean,z_log_var):
        epsilon = tf.random_normal(shape=(tf.shape(z_mean)[0],latentDim),mean=.0,stddev=epsilonStd)
        return z_mean+tf.exp(z_log_var/2)*epsilon
    
    # resampling layer for adding noise
    z = sampling(z_mean,z_log_var) # numBatch,latentDim 
    
    # decoder
    
    dec_h1 = dense(z,nUnit=256,scopeName="dec_h1",parameters=trainableParam)

    dec_mean = dense(dec_h1,nUnit=nFeature,scopeName="dec_mean",
        parameters=trainableParam,activation=None) # numBatch,nFeature
    dec_output = tf.nn.sigmoid(dec_mean)

    # loss function

    reconstructionLoss = nFeature * tf.keras.metrics.binary_crossentropy(x,dec_output)

    klDivergenceLoss = -0.5 * tf.reduce_sum((1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),[1])
    
    vaeLoss = tf.reduce_mean(reconstructionLoss + klDivergenceLoss)
    
    train_op = tf.train.AdamOptimizer(0.001).minimize(vaeLoss)
    # run
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(nEpoch):
        x_data,y_data = mnist.train.next_batch(batchSize)    
        _,epochLoss = sess.run([train_op,vaeLoss],feed_dict={x:x_data,dropoutRate:0.5})

        print("epoch {}/{}, vae_loss: {}".format(i+1,nEpoch,epochLoss))
    
    '''
    # test block 1
    x_test,y_test = mnist.test.next_batch(1)
    dec = sess.run(dec_output,feed_dict={x:x_test,dropoutRate:1.})
    pdb.set_trace()
    
    
    # test block 2
    x_data,y_data = mnist.train.next_batch(2)
    res = sess.run(reconstructionLoss,feed_dict={x:x_data,dropoutRate:1.})
    print(res)
    pdb.set_trace()
    '''

    x_test,y_test = mnist.test.next_batch(batchSize)
    
    x_test_encoded = sess.run(z_mean,feed_dict={x:x_test,dropoutRate:1.})
    plt.figure(figsize=(6,6))
    plt.scatter(x_test_encoded[:,0],x_test_encoded[:,1],c=np.argmax(y_test,1))
    plt.colorbar()
    plt.show()
    
    # observe how latent vector influence the generated images
    n = 15 # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size*n,digit_size*n))
    
    grid_x = norm.ppf(np.linspace(0.05,0.95,n))
    grid_y = norm.ppf(np.linspace(0.05,0.95,n))
    
    for i,yi in enumerate(grid_x):
        for j,xi in enumerate(grid_y):
            z_sample = np.array([[xi,yi]])
            x_decoded = sess.run(dec_output,feed_dict={z:z_sample,dropoutRate:1.})
            digit = x_decoded[0].reshape(digit_size,digit_size)
            figure[i * digit_size : (i+1)*digit_size,
                   j * digit_size : (j+1)*digit_size] = digit
    
    plt.figure(figsize=(10,10))
    plt.imshow(figure,cmap="Greys_r")
    plt.show()
    

if __name__ == "__main__":
    main()

    
    

