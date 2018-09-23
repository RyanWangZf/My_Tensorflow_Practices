# -*- coding: utf-8 -*-
import time
from datetime import datetime
import collections
import tensorflow as tf
import math


slim = tf.contrib.slim

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
                print("%s : step %d, duration = %.3f sec"%(datetime.now(),
                    i-num_steps_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn*mn
    sd = math.sqrt(vr)
    print("%s: %s across %d steps,%.3f +/- %.3f sec / batch"%(datetime.now(),
        info_string,num_batches,mn,sd))

def resnet_arg_scope(is_training=True,weight_decay=.0001,batch_norm_decay=.997,
    batch_norm_epsilon=1e-5,batch_norm_scale=True):
    
    batch_norm_params = {
        "is_training":is_training,
        "decay":batch_norm_decay,
        "epsilon":batch_norm_epsilon,
        "scale":batch_norm_scale,
        "updates_collections":tf.GraphKeys.UPDATE_OPS,
        }
    
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(), #scaled truncated normal
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding="SAME") as arg_sc:
                return arg_sc

# Block("block1",bottleneck,[(256,64,1)]*2+[(256,64,2)])
# block1: scope;
# bottleneck: residual learning unit in ResNetV2;
# args: ResNetV2 units defined in this block
# (256,64,3): The 3rd layer outputs depth 256, 
# the 1st and 2nd layer outputs depth_bottleneck 64,
# the 2nd layer has stride 3.
class Block(collections.namedtuple("Block",["scope","unit_fn","args"])):
    "A named tuple describing a ResNet block."

@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections=None):
    # Only function with @slim.add_arg_scope can be with arg_scope
    # net: inputs tensor;
    # blocks: defined Block class;
    # outputs_collections: end_points collections;
    for block in blocks:
        # each block
        with tf.variable_scope(block.scope,"block",[net]) as sc:
            for i,unit in enumerate(block.args):
                # each residual unit, named as bock1/unit_1.
                with tf.variable_scope("unit_%d"%(i+1),values=[net]):
                    unit_depth,unit_depth_bottleneck,unit_stride = unit
                    net = block.unit_fn(net,
                        depth=unit_depth,depth_bottleneck=unit_depth_bottleneck,
                        stride=unit_stride)
        # collect activations at the block's end before performing subsampling                
        net = slim.utils.collect_named_outputs(outputs_collections,sc.name,net)
    return net

def subsample(inputs,factor,scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
    if stride == 1:
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=1,
            padding="SAME",scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total//2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,
            padding="VALID",scope=scope)
    
@slim.add_arg_scope
def bottleneck(inputs,depth,depth_bottleneck,stride,outputs_collections=None,scope=None):
    # depth: the 3rd layer outputs feature maps
    # depth_bottleneck: the 2nd layer feature maps
            
    with tf.variable_scope(scope,"bottleneck_v2",[inputs]) as sc:
        # get the last dimension of the tensor inputs  
        depth_in = slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
        # activation layer ahead of mapping  
        preact = slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope="preact")
        
        # make sure that shortcut share same size with the residual
        if depth == depth_in: # use maxpool2d 1x1 with stride to reshape the map
            shortcut = subsample(inputs,stride,"shortcut")
        else: # use 1x1 conv2d layer to transform shortcut to depth_bottleneck
            shortcut = slim.conv2d(preact,depth,[1,1],stride=stride,scope="shortcut")
        
        residual = slim.conv2d(preact,depth_bottleneck,[1,1],stride=1,scope="conv1")
        residual = conv2d_same(residual,depth_bottleneck,3,stride,scope="conv2") # padding="SAME"
        residual = slim.conv2d(residual,depth,[1,1],stride=1,
            normalizer_fn=None,activation_fn=None,scope="conv3")
    
        output = shortcut + residual
    
        return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)

def resnet_v2(inputs,blocks,num_classes=None,global_pool=True,
    include_root_block=True,reuse=None,scope=None):
    # This is the main function of ResNet-V2
    # blocks: defined Block class
    # global_pool: whether add the maxpooling layer at the last layer
    # include_root_block: whether add teh 7x7 conv2d and maxpooling layer at first
    # reuse: whether reuse
    
    with tf.variable_scope(scope,"resnet_v2",[inputs],reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + "_end_points"
        # Assign the list of functions outputs_collections as the 'end_points_collection'
        with slim.arg_scope([slim.conv2d,bottleneck,stack_blocks_dense],
            outputs_collections=end_points_collection):
            
            net = inputs
            if include_root_block:
                # n,x,y,z --> n,x/2,y/2,64
                with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
                    net = conv2d_same(net,64,7,stride=2,scope="conv1")
                net = slim.max_pool2d(net,[3,3],stride=2,scope="pool1")
            
            # generate the residual learning block with defined blocks class   
            net = stack_blocks_dense(net,blocks)
            net = slim.batch_norm(net,activation_fn=tf.nn.relu,scope="postnorm")

            if global_pool:
                # global pooling
                net = tf.reduce_mean(net,[1,2],name="pool5",keepdims=True)

            if num_classes is not None:
                net = slim.conv2d(net,num_classes,[1,1],activation_fn=None,
                    normalizer_fn=None,scope="logits")
            
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)    
            
            if num_classes is not None:
                end_points["predictions"] = slim.softmax(net,scope="predictions")

            return net,end_points

def resnet_v2_50(inputs,num_classes=None,global_pool=True,reuse=None,scope="resnet_v2_50"):
    blocks = [
        Block("block1",bottleneck,[(256,64,1)]*2+[(256,64,2)]), 
        Block("block2",bottleneck,[(512,128,1)]*3+[(512,128,2)]),
        Block("block3",bottleneck,[(1024,256,1)]*5+[(1024,256,2)]),
        Block("block4",bottleneck,[(2048,512,1)]*3),]
        # 9+12+18+9 = 48
        # 48 + include_root_block = 48+2 = 50
    return resnet_v2(inputs,blocks,num_classes,global_pool, \
        include_root_block=True,reuse=reuse,scope=scope)


if __name__ == "__main__":
    
    batch_size = 2
    height,width = 224,224
    inputs = tf.random_uniform((batch_size,height,width,3))

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net,end_points = resnet_v2_50(inputs,num_classes=1000)
    
    print(net.op.name,"tensor shape:",net.get_shape().as_list())

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        num_batches = 100
        time_tensorflow_run(sess,net,feed_dict={},info_string="Forward")

