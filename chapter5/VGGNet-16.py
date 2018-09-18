# -*- coding: utf-8 -*-
import tensorflow as tf
import pdb
import time
import math
from datetime import datetime

# ---
# Function Tools
# ---

def time_tensorflow_run(session,target,feed_dict,info_string):
    # target: operation; info_string: test name.
    # Assess the time consumed every iteration.
    num_steps_burn_in = 10
    total_duration = .0
    total_duration_squared = .0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target,feed_dict=feed_dict)
        duration = time.time() - start_time
        if i >= num_steps_burn_in: # warming up
            if not i%10:
                print("%s : step %d, duration = %.3f"%(datetime.now(),
                    i-num_steps_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn*mn
    sd = math.sqrt(vr)
    print("%s: %s across %d steps,%.3f +/- %.3f sec / batch"%(datetime.now(),
        info_string,num_batches,mn,sd))

# ---
# Standard Layer Functions
# ---

def conv_op(input_op,kh,kw,n_out,dh,dw,scope_name,parameters):
    # This is the standard convolutional layer in the VGGNet-16 with Relu activation;
    # kh,kw,n_out are kernel's height, width & number(output channels);
    # dh,dw are the kernels's strides.
    # parameters is the tf.Variable object which contains the whole trainable params in the NNs.
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(scope_name) as scope:
        # initialize the weights with xavier_initializer
        kernel = tf.get_variable(scope+"w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding="SAME")
        biases = tf.Variable(tf.constant(.0,shape=[n_out],dtype=tf.float32),
            trainable=True,name="b")
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        parameters += [kernel,biases]
    return activation

def fc_op(input_op,n_out,scope_name,parameters):
    # This is the standard fully-connected layer with Relu activation.
    # n_out: the num of hidden units.
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(scope_name) as scope:
        kernel = tf.get_variable(scope+"w",shape=[n_in,n_out],dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(.1,shape=[n_out],dtype=tf.float32),name="b")
        z = tf.matmul(input_op,kernel) + biases
        activation = tf.nn.relu(z,name=scope)
        parameters += [kernel,biases]
    return activation

def maxpool_op(input_op,kh,kw,dh,dw,scope_name):
    # This is the standard max pooling layer;
    # The kh,kw,dh,dw are the kernel's heigth,width & strides.
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],
        padding="SAME",name=scope_name)

# ---
# NN's Inference
# ---

def inference(input_op,dropout_rate):
    # This is the inference of VGGNet-16;
    # input_op: the input operation or tensor;
    # dropout_rate: the placeholder controls the dropout ratio.
   
    param = []

    # The 1st block.
    conv1_1 = conv_op(input_op,kh=3,kw=3,n_out=64,dh=1,dw=1,scope_name="conv1_1",
        parameters=param)
    conv1_2 = conv_op(conv1_1,kh=3,kw=3,n_out=64,dh=1,dw=1,scope_name="conv1_2",
        parameters=param)
    pool1 = maxpool_op(conv1_2,kh=2,kw=2,dw=2,dh=2,scope_name="pool1")
     
    # The 2nd block.
    conv2_1 = conv_op(pool1,kh=3,kw=3,n_out=128,dh=1,dw=1,scope_name="conv2_1",
        parameters=param)
    conv2_2 = conv_op(conv2_1,kh=3,kw=3,n_out=128,dh=1,dw=1,scope_name="conv2_2",
        parameters=param)
    pool2 = maxpool_op(conv2_2,kh=2,kw=2,dw=2,dh=2,scope_name="pool2")

    # The 3rd block.
    conv3_1 = conv_op(pool2,kh=3,kw=3,n_out=256,dh=1,dw=1,scope_name="conv3_1",
        parameters=param)
    conv3_2 = conv_op(conv3_1,kh=3,kw=3,n_out=256,dh=1,dw=1,scope_name="conv3_2",
        parameters=param)
    conv3_3 = conv_op(conv3_2,kh=3,kw=3,n_out=256,dh=1,dw=1,scope_name="conv3_3",
        parameters=param)
    pool3 = maxpool_op(conv3_3,kh=2,kw=2,dw=2,dh=2,scope_name="pool3")
    
    # The 4th block.
    conv4_1 = conv_op(pool3,kh=3,kw=3,n_out=512,dh=1,dw=1,scope_name="conv4_1",
        parameters=param)
    conv4_2 = conv_op(conv4_1,kh=3,kw=3,n_out=512,dh=1,dw=1,scope_name="conv4_2",
        parameters=param)
    conv4_3 = conv_op(conv4_2,kh=3,kw=3,n_out=512,dh=1,dw=1,scope_name="conv4_3",
        parameters=param)
    pool4 = maxpool_op(conv4_3,kh=2,kw=2,dw=2,dh=2,scope_name="pool4")

    # The 5th block.
    conv5_1 = conv_op(pool4,kh=3,kw=3,n_out=512,dh=1,dw=1,scope_name="conv5_1",
        parameters=param)
    conv5_2 = conv_op(conv5_1,kh=3,kw=3,n_out=512,dh=1,dw=1,scope_name="conv5_2",
        parameters=param)
    conv5_3 = conv_op(conv5_2,kh=3,kw=3,n_out=512,dh=1,dw=1,scope_name="conv5_3",
        parameters=param)
    pool5 = maxpool_op(conv5_3,kh=2,kw=2,dw=2,dh=2,scope_name="pool5")
    
    # The Fully-connected Block
    shp = pool5.get_shape().as_list()
    from functools import reduce
    from operator import mul
    input_dim = reduce(mul,shp)
    resh1 = tf.reshape(pool5,[-1,input_dim],name="resh1")
    
    fc6 = fc_op(resh1,n_out=4096,scope_name="fc6",parameters=param)
    fc6_drop = tf.nn.dropout(fc6,dropout_rate,name="fc6_drop")
    fc7 = fc_op(fc6_drop,n_out=4096,scope_name="fc7",parameters=param)
    fc7_drop = tf.nn.dropout(fc7,dropout_rate,name="fc7_drop")
    fc8 = fc_op(fc7_drop,n_out=1000,scope_name="fc8",parameters=param)
    
    softmax_ = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax_,1)

    return predictions,softmax_,fc8,param

# ---
# Main Functions
# ---

if __name__ == "__main__":
    
    with tf.Graph().as_default():

        # generate images.
        num_batches=100
        image_size = 224 # 224x224x3
        batch_size = 1
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
            dtype=tf.float32,stddev=1e-1))

        # placeholders
        dropout_rate = tf.placeholder(tf.float32)

        # Inference & Run Session
        pred,softmax_,fc8,param = inference(images,dropout_rate)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        
        # Foward
        time_tensorflow_run(sess,pred,{dropout_rate:1.},"Forward")

        # Forward & Backward
        obj = tf.nn.l2_loss(fc8)
        grad = tf.gradients(obj,param)
        time_tensorflow_run(sess,grad,{dropout_rate:.5},"Foward_Backward")


        '''
        res = sess.run(res_op,feed_dict={dropout_rate:1.})
        print("The result shape:",res.shape)

        pred,softmax_,fc8,param = sess.run(res_op,feed_dict={dropout_rate:1.})
        print("The predictions shape:",pred.shape)
        print("The softmax shape:",softmax_.shape)
        print("The fc8 shape:",fc8.shape)
        print("The prediction:",pred)
        '''
        

    
    





