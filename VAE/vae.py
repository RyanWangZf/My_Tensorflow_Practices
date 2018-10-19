# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


import pdb

class vae():
    def __init__(self):
        # set params
        self.nClass = 10
        self.nFeature = 784
        self.batchSize = 100
        self.nEpoch = 2000
        self.latentDim = 2 # dim of the z code
        self.epsilonStd = 1.
        self.trainableParam = []
        
    def dense(self,x,nUnit,scopeName,parameters,activation="relu"):
        '''
        input:
            x: input tensor
            nUnit: num of units in this layer
            scopeName: scope name of this layer
            parameters: list of trainable variables
            activation: activation function
        '''
        nInput = x.get_shape()[-1].value
        with tf.name_scope(scopeName) as scope:
            kernel = tf.get_variable(scope+"w",shape=[nInput,nUnit],dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.Variable(tf.constant(.1,shape=[nUnit],dtype=tf.float32),name="b")
            z = tf.matmul(x,kernel) + bias
            if activation == "relu":
                z = tf.nn.relu(z,name=scope)
            parameters += [kernel,bias]
        return z
    
    def sampling(self,z_mean,z_log_var):
        epsilon = tf.random_normal(shape=(tf.shape(z_mean)[0],self.latentDim),mean=.0,
            stddev=self.epsilonStd)
        return z_mean+tf.exp(z_log_var/2)*epsilon
    
    def loadData(self):
        # load data
        mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
        x_all,y_all = mnist.train.images,mnist.train.labels
        print("MNIST train images:",mnist.train.images.shape)
        print("MNIST test  images:",mnist.test.images.shape)
        self.mnist = mnist

    def inference(self,x,dropoutRate):
        dense = self.dense
        trainableParam = self.trainableParam
        latentDim = self.latentDim
        nFeature = self.nFeature

        # encoder
        enc_h1 = dense(x,nUnit=256,scopeName="enc_h1",parameters=trainableParam)
        enc_drop1 = tf.nn.dropout(enc_h1,dropoutRate,name="enc_drop1")  # numBatch,256
        enc_h2 = dense(enc_drop1,nUnit=512,scopeName="enc_h2",parameters=trainableParam)
        enc_drop2 = tf.nn.dropout(enc_h2,dropoutRate,name="enc_drop2")  # numBatch,256
        enc_h3 = dense(enc_drop2,nUnit=1024,scopeName="enc_h3",parameters=trainableParam)
        enc_drop3 = tf.nn.dropout(enc_h3,dropoutRate,name="enc_drop3")  # numBatch,256
        
        # get mean and variance of p(z|x)
        z_mean = dense(enc_drop3,latentDim,"z_mean",parameters=trainableParam,activation=None)
        z_log_var = dense(enc_drop3,latentDim,"z_log_var",parameters=trainableParam,activation=None)

        # resampling layer for adding noise
        z = self.sampling(z_mean,z_log_var) # numBatch,latentDim 
        
        # decoder
        dec_h1 = dense(z,nUnit=256,scopeName="dec_h1",parameters=trainableParam)
        dec_h2 = dense(dec_h1,nUnit=512,scopeName="dec_h2",parameters=trainableParam)
        dec_h3 = dense(dec_h2,nUnit=1024,scopeName="dec_h3",parameters=trainableParam)
            
        dec_mean = dense(dec_h3,nUnit=nFeature,scopeName="dec_mean",
            parameters=trainableParam,activation=None) # numBatch,nFeature
        dec_output = tf.nn.sigmoid(dec_mean)

        # loss function
        reconstructionLoss = nFeature * tf.keras.metrics.binary_crossentropy(x,dec_output)
        klDivergenceLoss = -0.5 * tf.reduce_sum((1 + z_log_var - tf.square(z_mean) - 
                tf.exp(z_log_var)),[1])
    
        vaeLoss = tf.reduce_mean(reconstructionLoss + klDivergenceLoss)
        train_op = tf.train.AdamOptimizer(0.001).minimize(vaeLoss)
        
        # link op
        self.train_op = train_op
        self.vaeLoss  = vaeLoss
        self.dec_output = dec_output
        self.z_mean   = z_mean
        self.z = z

    def fit(self):
        # placeholder
        self.x = tf.placeholder(tf.float32,[None,self.nFeature])
        self.y = tf.placeholder(tf.float32,[None,self.nClass])
        self.dropoutRate = tf.placeholder(tf.float32)
        
        self.inference(self.x,self.dropoutRate)

        # run
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(self.nEpoch):
            x_data,y_data = self.mnist.train.next_batch(self.batchSize)    
            _,epochLoss = self.sess.run([self.train_op,self.vaeLoss],
                feed_dict={self.x:x_data,self.dropoutRate:0.5})
            print("epoch {}/{}, vae_loss: {}".format(i+1,self.nEpoch,epochLoss))
    
    def generate(self):
        # observe how latent vector influence the generated images
        n = 15 # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size*n,digit_size*n))
    
        grid_x = norm.ppf(np.linspace(0.05,0.95,n))
        grid_y = norm.ppf(np.linspace(0.05,0.95,n))
    
        for i,yi in enumerate(grid_x):
            for j,xi in enumerate(grid_y):
                z_sample = np.array([[xi,yi]])
                x_decoded = self.sess.run(self.dec_output,feed_dict={self.z:z_sample,self.dropoutRate:1.})
                digit = x_decoded[0].reshape(digit_size,digit_size)
                figure[i * digit_size : (i+1)*digit_size,
                   j * digit_size : (j+1)*digit_size] = digit
    
        plt.figure(figsize=(10,10))
        plt.imshow(figure,cmap="Greys_r")
        plt.show()
    
    def test(self):
        self.loadData()
        self.fit()
        x_test,y_test = self.mnist.test.next_batch(self.batchSize)
        x_test_encoded = self.sess.run(self.z_mean,feed_dict={self.x:x_test,self.dropoutRate:1.})
        plt.figure(figsize=(6,6))
        plt.scatter(x_test_encoded[:,0],x_test_encoded[:,1],c=np.argmax(y_test,1))
        plt.colorbar()
        plt.show()
        self.generate()


def main():
    vae_model = vae()
    vae_model.test()

    
if __name__ == "__main__":
    main()
