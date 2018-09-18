# -*- coding: utf-8 -*-

from datetime import datetime
import math
import time
import tensorflow as tf
import pdb

# ---
# Tool Functions
# ---

def print_activations(t):
    # Print the Shape of Layer's Tensor.
    print(t.op.name," ",t.get_shape().as_list());

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

def run_benchmark():
    pass

# ---
# Standard Layer Functions
# ---

def conv_layer_pool(inputs,kernel_shape,kernel_strides,pool_shape,pool_strides,
    scope_name,parameters):
    # This is the standard convolutional layer in the AlexNet with max pooling layer.
    # inputs: input tensor; kernel_shape: conv kernel size; kernel_strides: conv kernel strides
    # pool_shape: pool kernel shape; pool_strides: pool kernel strides;
    # scope_name: this layer scope name; parameters: collected list of trainable parameters.
    # output the output tensor after conv, lrn & max pool.
    with tf.name_scope(scope_name) as scope:
        kernel = tf.Variable(tf.truncated_normal(kernel_shape,
            dtype=tf.float32,stddev=.1),name="weights")
        conv = tf.nn.conv2d(inputs,kernel,kernel_strides,padding="SAME")
        biases = tf.Variable(tf.constant(.0,shape=[kernel_shape[-1]],dtype=tf.float32),
            trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        conv = tf.nn.relu(bias,name=scope)
        print_activations(conv)
        parameters += [kernel,biases]
        lrn = tf.nn.lrn(conv,4,bias=1.,alpha=0.001/9,beta=0.75,name=scope_name+"_lrn")
        pool = tf.nn.max_pool(lrn,ksize=pool_shape,strides=pool_strides,
            padding="VALID",name=scope_name+"_pool")
        print_activations(pool)
    return pool,parameters

def conv_layer(inputs,kernel_shape,kernel_strides,scope_name,parameters):
    # This is the standard convolutional layer in the AlexNet without both 
    # max pooling layer and the Local Responsen Norm
    # Other inputs params share the same meaning in the function conv_layer_pool.
    with tf.name_scope(scope_name) as scope:
        kernel = tf.Variable(tf.truncated_normal(kernel_shape,
            dtype=tf.float32,stddev=.1),name="weights")
        conv = tf.nn.conv2d(inputs,kernel,kernel_strides,padding="SAME")
        biases = tf.Variable(tf.constant(.0,shape=[kernel_shape[-1]],dtype=tf.float32),
            trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        conv = tf.nn.relu(bias,name=scope)
        print_activations(conv)
        parameters += [kernel,biases]
    return conv,parameters

def fcn_layer(inputs,n_hidden,scope_name,parameters):
    # This is the standard fully-connected layer.
    # n_hidden: the num of hidden units; scope_name: the scope name of this layer.
    from functools import reduce
    from operator import mul
    num_input_dim = reduce(mul,inputs.get_shape().as_list()[1:])
    inputs = tf.reshape(inputs,shape=[-1,num_input_dim])
    with tf.name_scope(scope_name) as scope:
        weights = tf.Variable(tf.truncated_normal([num_input_dim,n_hidden],
            dtype=tf.float32,stddev=.1),name="weights")
        biases = tf.Variable(tf.constant(.0,shape=[n_hidden],dtype=tf.float32),
            trainable=True,name="biases")
        fcn = tf.nn.bias_add(tf.matmul(inputs,weights),biases)
        print_activations(fcn)
        parameters += [weights,biases]
    return fcn,parameters

# ---
# Networks Inference
# ---

def inference(images,dropout_rate):
    # input: images, output: pool5 & praameters
    # it contains the NN's structure information
    parameters = []
    # conv layer1
    layer1,parameters = conv_layer_pool(images,kernel_shape=[11,11,3,64],kernel_strides=[1,4,4,1],
        pool_shape=[1,3,3,1],pool_strides=[1,2,2,1],scope_name="conv1",parameters=parameters)
    # conv layer2
    layer2,parameters = conv_layer_pool(layer1,kernel_shape=[5,5,64,192],kernel_strides=[1,1,1,1],
        pool_shape=[1,3,3,1],pool_strides=[1,2,2,1],scope_name="conv2",parameters=parameters)
    # conv layer3
    layer3,parameters = conv_layer(layer2,kernel_shape=[3,3,192,384],kernel_strides=[1,1,1,1],
        scope_name="conv3",parameters=parameters)
    # conv layer4
    layer4,parameters = conv_layer(layer3,kernel_shape=[3,3,384,256],kernel_strides=[1,1,1,1],
        scope_name="conv4",parameters=parameters)
    # conv layer5
    layer5,parameters = conv_layer(layer4,kernel_shape=[3,3,256,256],kernel_strides=[1,1,1,1],
        scope_name="conv5",parameters=parameters)
    # max_pooling layer6
    pool_last = tf.nn.max_pool(layer5,ksize=[1,3,3,1],strides=[1,2,2,1],
        padding="VALID",name="pool_last")
    print_activations(pool_last)
    # fully-connected layer 1
    fcn1,parameters = fcn_layer(pool_last,n_hidden=4096,scope_name="fcn1",parameters=parameters)
    fcn1_drop = tf.nn.dropout(fcn1,dropout_rate)
    # fully-connected layer 2
    fcn2,parameters = fcn_layer(fcn1_drop,n_hidden=4096,scope_name="fcn2",parameters=parameters)
    fcn2_drop = tf.nn.dropout(fcn2,dropout_rate)
    # fully-connected layer 3
    fcn3,parameters = fcn_layer(fcn2_drop,n_hidden=1000,scope_name="fcn3",parameters=parameters)
    fcn3_drop = tf.nn.dropout(fcn3,dropout_rate)
    
    return fcn3_drop,parameters

# ---
# Main Function
# ---

if __name__ == "__main__":
    # Define global variables
    batch_size = 32;
    num_batches = 100;
    
    # load data sets
    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
        dtype=tf.float32,stddev=1e-1))
    dropout_rate = tf.placeholder(tf.float32)
    
    output_layer,param = inference(images,dropout_rate)
     
    # run session forward & backward
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    time_tensorflow_run(sess,target=output_layer,
        feed_dict={dropout_rate:0.5},info_string="Forward") # Forward
    
    obj = tf.nn.l2_loss(output_layer)
    grad = tf.gradients(obj,param)
    time_tensorflow_run(sess,target=grad,
        feed_dict={dropout_rate:0.5},info_string="Forward-backward") # Backward
    
    
    # res = sess.run(output_layer,feed_dict={dropout_rate:0.5})
    # print("Output tensor's shape: ",res.shape)
    pdb.set_trace()





