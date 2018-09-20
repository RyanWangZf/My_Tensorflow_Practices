# -*- coding: utf-8 -*-

import tensorflow as tf
import time
from datetime import datetime
import math

slim = tf.contrib.slim

# generate truncated normal weights.
trunc_normal = lambda stddev: tf.truncated_normal_initializer(.0,stddev)

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

def inception_v3_arg_scope(weight_decay=0.00004,stddev=.1,
    batch_norm_var_collection="moving_vars"):
    # To generate default params for defining a Convolutional layer simply.
    # weight_decay: The decay ratio of L2 Loss;
    
    # Params Dict for BatchNormalization layer
    batch_norm_params = {
        "decay":0.9997,
        "epsilon":0.001,
        "updates_collections":tf.GraphKeys.UPDATE_OPS,
        "variables_collections":{
            "beta":None,
            "gamma":None,
            "moving_mean":[batch_norm_var_collection], # mean of every batch
            "moving_variance":[batch_norm_var_collection], # variance of every batch
            }
        }

    # slim.arg_scope to assign values on functions' params;
    # The 1st arg_scope will assign the param "weights_regularizer" for function slim.conv2d
    # and fully_connected;
    # Actually add a l2_regularizer on the weights.
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay)):
        # This arg_scope assign params of slim.conv2d;
        # Including BN layer and Relu activation layer.
        with slim.arg_scope([slim.conv2d],
            weights_initializer=trunc_normal(stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params) as sc:

            return sc

def inception_v3_base(inputs,scope=None):
    # inputs: inputs images' tensor;
    # scope: params environment.
    # This Function defines 3 module groups:
    # Inception1, Inception2, Inception3.
    # Inception1: 3 modules;
    # Inception2: 5 modules;
    # Inception3: 3 modules;
    # Each module has different number of branches.

    end_points = {} # save several key points in the calculation map.

    # Normal Convolutional Layer (Not Inception Module, Base)
    with tf.variable_scope(scope,"InceptionV3",[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
            stride=1,padding="VALID"):
            # input_tensorf,output_channels,kernel_size,kernel_strides
            # input_shape: (batch_size,299,299,3);
            # output_shape:(batch_size,35,35,192).
            net = slim.conv2d(inputs,32,[3,3],stride=2,scope="Conv2d_1a_3x3")
            net = slim.conv2d(net,32,[3,3],scope="Conv2d_2a_3x3")
            net = slim.conv2d(net,64,[3,3],padding="SAME",scope="Conv2d_2b_3x3")
            net = slim.max_pool2d(net,[3,3],stride=2,scope="MaxPool_3a_3x3")
            net = slim.conv2d(net,80,[1,1],scope="Conv2d_3b_1x1")
            net = slim.conv2d(net,192,[3,3],scope="Conv2d_4a_3x3")
            net = slim.max_pool2d(net,[3,3],stride=2,scope="MaxPool_5a_3x3")
    
    # Inception Modules
    with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
        stride=1,padding="SAME"):
        # ---
        # 1st Inception Module Group.
        # ---
        with tf.variable_scope("Mixed_5b"):
            # Module named as Mixed_5b;
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")

            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,48,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,64,[5,5],scope="Conv2d_0b_5x5")

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0c_3x3")

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,32,[1,1],scope="Conv2d_0b_1x1")
            # Add 4 branches output together on axis 3 (channels);
            # Output shape: None,35,35,256
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

        with tf.variable_scope("Mixed_5c"):
            # Module named as Mixed_5c;
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                # None,35,35,64

            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,48,[1,1],scope="Conv2d_0b_1x1")
                branch_1 = slim.conv2d(branch_1,64,[5,5],scope="Conv2d_0c_5x5")
                # None,35,35,64

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0c_3x3")
                # NOne,35,35,96

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,64,[1,1],scope="Conv2d_0b_1x1")
                # None,35,35,64

            # Output shape: None,35,35,288
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

        with tf.variable_scope("Mixed_5d"):
            # Module named as Mixed_5d;
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
            
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,48,[1,1],scope="Conv2d_0b_1x1")
                branch_1 = slim.conv2d(branch_1,64,[5,5],scope="Conv2d_0c_5x5")
            
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0c_3x3")

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,64,[1,1],scope="Conv2d_0b_1x1")
            # Output shape: None,35,35,288
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        
        # ---
        # 2nd Inception Module Group.
        # ---
        with tf.variable_scope("Mixed_6a"):
            # Module named as Mixed_6a;
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,384,[3,3],stride=2,padding="VALID",
                    scope="Conv2d_1a_3x3")
                # None,17,17,384

            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,96,[3,3],scope="Conv2d_0b_3x3")
                branch_1 = slim.conv2d(branch_1,96,[3,3],stride=2,
                    padding="VALID",scope="Conv2d_1a_1x1")
                # None,17,17,96

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.max_pool2d(net,[3,3],stride=2,
                    padding="VALID",scope="MaxPool_1a_3x3")
                # None,17,17,288

            net = tf.concat([branch_0,branch_1,branch_2],3)
            # None,17,17,768

        with tf.variable_scope("Mixed_6b"):
            # Module named as Mixed_6b;
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                # None,17,17,192
            
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,128,[1,1],scope="Conv2d_0a_1x1")
                # Factorization Into Small Convolutions e.g. 1x7 + 7x1 = 7x7.
                branch_1 = slim.conv2d(branch_1,128,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
                # None,17,17,192

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,128,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,128,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,128,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,128,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
                # None,17,17,192

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
                # None,17,17,192

            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # None,17,17,768

        with tf.variable_scope("Mixed_6c"):
            # Module named as Mixed_6c.
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                # None,17,17,192

            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                # Factorization Into Small Convolutions e.g. 1x7 + 7x1 = 7x7.
                branch_1 = slim.conv2d(branch_1,160,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
                # None,17,17,192

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,160,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
                # None,17,17,192

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
                # None,17,17,192

            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # None,17,17,768

        with tf.variable_scope("Mixed_6d"):
            # Module named as Mixed_6d, totally same as Mixed_6c.
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                # None,17,17,192

            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                # Factorization Into Small Convolutions e.g. 1x7 + 7x1 = 7x7.
                branch_1 = slim.conv2d(branch_1,160,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
                # None,17,17,192

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,160,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
                # None,17,17,192

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
                # None,17,17,192

            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # None,17,17,768
        
        with tf.variable_scope("Mixed_6e"):
            # Module named as Mixed_6e, totally same as Mixed_6c,Mixed_6d.
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                # None,17,17,192

            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                # Factorization Into Small Convolutions e.g. 1x7 + 7x1 = 7x7.
                branch_1 = slim.conv2d(branch_1,160,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
                # None,17,17,192

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,160,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
                # None,17,17,192

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
                # None,17,17,192

            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # None,17,17,768
        
        # Save Mixed_6e output in the end_points as the Auxiliary Classifier.
        end_points["Mixed_6e"] = net
        
        # ---
        # 3rd Inception Module Group.
        # ---
        with tf.variable_scope("Mixed_7a"):
            # Module named as Mixed_7a.
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                branch_0 = slim.conv2d(branch_0,320,[3,3],
                    stride=2,padding="VALID",scope="Conv2d_1a_3x3")
                # None,8,8,320

            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,192,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_1x7")
                branch_1 = slim.conv2d(branch_1,192,[3,3],
                    stride=2,padding="VALID",scope="Conv2d_1a_3x3")
                # None,8,8,192

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.max_pool2d(net,[3,3],stride=2,
                    padding="VALID",scope="MaxPool_1a_3x3")
                # None,8,8,768

            net = tf.concat([branch_0,branch_1,branch_2],3)
            # None,8,8,1280

        with tf.variable_scope("Mixed_7b"):
            # Module named as Mixed_7b.
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,320,[1,1],scope="Conv2d_0a_1x1")
                # None,8,8,320
            
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,384,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = tf.concat([
                    slim.conv2d(branch_1,384,[1,3],scope="Conv2d_0b_1x3"),
                    slim.conv2d(branch_1,384,[3,1],scope="Conv2d_0b_3x1")],3)
                # None,8,8,768

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,448,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,384,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = tf.concat([
                    slim.conv2d(branch_2,384,[1,3],scope="Conv2d_0c_1x3"),
                    slim.conv2d(branch_2,384,[3,1],scope="Conv2d_0d_3x1")],3)
                # None,8,8,768

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
                # None,8,8,192

            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # None,8,8,2048

        with tf.variable_scope("Mixed_7c"):
            # Module named as Mixed_7c, same as the Mixed_7b.
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,320,[1,1],scope="Conv2d_0a_1x1")
                # None,8,8,320
            
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,384,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = tf.concat([
                    slim.conv2d(branch_1,384,[1,3],scope="Conv2d_0b_1x3"),
                    slim.conv2d(branch_1,384,[3,1],scope="Conv2d_0b_3x1")],3)
                # None,8,8,768

            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,448,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,384,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = tf.concat([
                    slim.conv2d(branch_2,384,[1,3],scope="Conv2d_0c_1x3"),
                    slim.conv2d(branch_2,384,[3,1],scope="Conv2d_0d_3x1")],3)
                # None,8,8,768

            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
                # None,8,8,192

            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
            # None,8,8,2048
        
        return net,end_points

def inception_v3(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.8,
    prediction_fn=slim.softmax,spatial_squeeze=True,reuse=None,scope="InceptionV3"):
    # num_classes: Number of prediction classes;
    # is_training: True for training, False for predicting, About the BN & Dropout layers;
    # dropout_keep_prob: dropout ratio while training;
    # spatial_squeeze: whether squeeze output, e.g [5,3,1] --> [5,3];
    # reuse: whether reuse the networks and Variable;

    with tf.variable_scope(scope,"InceptionV3",[inputs,num_classes],reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            net,end_points = inception_v3_base(inputs,scope=scope)


            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                stride=1,padding="SAME"):
                
                # Mixed_6e: None,17,17,768
                aux_logits = end_points["Mixed_6e"]
                with tf.variable_scope("AuxLogits"):
                    # auxiliary logits
                    aux_logits = slim.avg_pool2d(aux_logits,[5,5],stride=3,padding="VALID",
                        scope="AvgPool_1a_5x5")
                    # None,5,5,768
                    aux_logits = slim.conv2d(aux_logits,128,[1,1],scope="Conv2d_1b_1x1")

                    aux_logits = slim.conv2d(
                        aux_logits,768,[5,5],
                        weights_initializer=trunc_normal(.01),
                        padding="VALID",scope="Conv2d_2a_5x5")
                    # None,1,1,768

                    aux_logits = slim.conv2d(
                        aux_logits,num_classes,[1,1],activation_fn=None,
                        normalizer_fn=None,weights_initializer=trunc_normal(.001),
                        scope="Conv2d_2b_1x1")
                    # None,1,1,1000

                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits,[1,2],name="SpatialSqueeze")
                        # None,1000

                    end_points["AuxLogits"]=aux_logits

            with tf.variable_scope("Logits"):
                # Last logits.
                net = slim.avg_pool2d(net,[8,8],padding="VALID",scope="AvgPool_1a_8x8")
                net = slim.dropout(net,keep_prob=dropout_keep_prob,scope="Dropout_1b")
                # None,1,1,2018
                end_points["PreLogits"] = net
                logits = slim.conv2d(net,num_classes,[1,1],activation_fn=None,
                    normalizer_fn=None,scope="Conv2d_1c_1x1")
                # None,1,1,1000

                if spatial_squeeze:
                    logits = tf.squeeze(logits,[1,2],name="SpatialSqueeze")
                    # None,1000

            end_points["Logits"]=logits
            # prediction_fn : slim.softmax as default.
            end_points["Predictions"]=prediction_fn(logits,scope="Predictions")

        return logits,end_points


if __name__ == "__main__":
    batch_size = 2
    height,width = 299,299
    inputs = tf.random_uniform((batch_size,height,width,3))
    with slim.arg_scope(inception_v3_arg_scope()):
        logits,end_points = inception_v3(inputs,is_training=False)
    
    print_activations(logits)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_batches = 100

    time_tensorflow_run(sess,logits,feed_dict={},info_string="Forward")



    


