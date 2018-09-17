# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pdb

def variable_with_weight_loss(shape,stddev,w1=.0):
    # initialize weight variables with l2 loss when w1 > 0.0
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 > .0:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return var

def loss_func(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_en = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits,labels = labels,name="cross_entropy_per_example")
    cross_en_mean = tf.reduce_mean(cross_en,name="cross_entropy")
    tf.add_to_collection("losses",cross_en_mean)
    return tf.add_n(tf.get_collection("losses"),name="total_loss")

if __name__ == "__main__":
    max_steps = 3000
    batch_size = 32

    # load data sets
    mnist = input_data.read_data_sets("../MNIST_data/",one_hot=False)

    # Define the placeholders
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.int32)
    
    # nn structure
    # conv1
    weight1 = variable_with_weight_loss(shape=[5,5,1,64],stddev=5e-2,w1=.0)
    bias1 = tf.Variable(tf.constant(.0,shape=[64]))
    x_image = tf.reshape(x,[-1,28,28,1])
    conv1 = tf.nn.relu(tf.nn.conv2d(x_image,weight1,[1,1,1,1],padding="SAME"))
    conv1 = conv1 + bias1
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
    norm1 = tf.nn.lrn(pool1,4,bias=1.,alpha=0.001/9,beta=.75) # (batch_size,14,14,64)
    # conv2
    weight2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=.0)
    bias2 = tf.Variable(tf.constant(.0,shape=[64]))
    conv2 = tf.nn.relu(tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding="SAME"))
    conv2 = conv2 + bias2    
    pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
    norm2 = tf.nn.lrn(pool2,4,bias=1.,alpha=0.001/9.,beta=0.75) # (batch_size,7,7,64)
    # fcn1, dim_ : 7*7*64
    dim_ = pool2.get_shape()[1].value * pool2.get_shape()[2].value * pool2.get_shape()[3].value  
    pool2_flat = tf.reshape(pool2,[-1,dim_])
    weight3 = variable_with_weight_loss(shape=[dim_,384],stddev=0.04,w1=0.004)
    bias3 = tf.Variable(tf.constant(.0,shape=[384]))
    local3 = tf.nn.relu(tf.matmul(pool2_flat,weight3)+bias3) # (batch_size,384)
    # fcn2
    weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
    bias4 = tf.Variable(tf.constant(.1,shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3,weight4)+bias4) # (batch_size,192)
    # output
    weight5 = variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
    bias5 = tf.Variable(tf.constant(.0,shape=[10]))
    logits = tf.matmul(local4,weight5) + bias5 # (batch_size,10)
    # loss function & train op
    loss_ = loss_func(logits,y_) #(1)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss_)
    top_k_op = tf.nn.in_top_k(logits,y_,1)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # training
    x_val,y_val = mnist.validation.images[:1000],mnist.validation.labels[:1000]
    for step in range(max_steps):
        image_batch,label_batch = mnist.train.next_batch(batch_size)
        _,loss_value = sess.run([train_op,loss_],feed_dict={x:image_batch,y_:label_batch})
        if step % 100 == 0:
            val_res = sess.run(top_k_op,
                feed_dict={x:x_val,y_:y_val})
            val_acc = 100*np.sum(val_res)/y_val.shape[0]
            print("iteration {}, train_loss {}, val_acc {} %".format(step,loss_value,val_acc))
    
    # test
    x_test = mnist.test.images[:1000]
    y_test = mnist.test.labels[:1000]
    test_res = sess.run(top_k_op,feed_dict={x:x_test,y_:y_test})
    true_count = np.sum(test_res)
    precision = 100 * true_count / y_test.shape[0]
    print("test precision: {} %".format(precision))
    pdb.set_trace()


