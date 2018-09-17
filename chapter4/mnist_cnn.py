# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variables(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def bias_variables(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

if __name__ == "__main__":
    
    mnist = input_data.read_data_sets("../MNIST_data/",one_hot=True)
    sess = tf.InteractiveSession()
    
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])
    x_image = tf.reshape(x,[-1,28,28,1])

    # Define the CNN's structure
    w_conv1 = weight_variables([5,5,1,32])
    b_conv1 = bias_variables([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variables([5,5,32,64])
    b_conv2 = bias_variables([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_fc1 = weight_variables([7*7*64,1024])
    b_fc1 = bias_variables([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

    dropout_rate = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,dropout_rate)

    w_fc2 = weight_variables([1024,10])
    b_fc2 = bias_variables([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
    
    cross_en = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_en)

    corr_pred = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    acc = tf.reduce_mean(tf.cast(corr_pred,tf.float32))
    
    # training
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_acc = acc.eval(feed_dict={x:batch[0],y_:batch[1],dropout_rate:1.0})
            val_acc = acc.eval(feed_dict={x:mnist.validation.images,y_:mnist.validation.labels,
                dropout_rate:1.})
            print("step %d, training accuracy %g, validation accuracy %g"%(
                i,train_acc,val_acc))
        train_op.run(feed_dict={x:batch[0],y_:batch[1],dropout_rate:0.5})
    
    x_test = mnist.test.images[:5000]
    y_test = mnist.test.labels[:5000]

    print("test acc %g"%acc.eval(feed_dict={x:x_test,y_:y_test,
        dropout_rate:1.}))


    

