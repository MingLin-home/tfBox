import numpy as np
import tensorflow as tf

import tfShallowOptimizer



batch_size = 100

feature_dim=5000
max_iter=1000
tol = 1e-6
var_list = None
init_learning_rate = 0.01
end_learning_rate = 0.01
decay_power = 1.0
decay_steps = 1013
input_device_str = '/cpu:0'
GPU_device_str = '/cpu:0'
num_classes = 10

num_loss_blocks = 100
W_true = np.random.rand(feature_dim,num_classes)

def fetch_batch_fn(options):
    new_X = np.random.randn(batch_size,feature_dim)
    new_Y = new_X.dot(W_true).astype(np.int32)
    return  new_X, new_Y
pass # end def


with tf.Graph().as_default():
    with tf.device(GPU_device_str):
        loss_block_list = []
        feature_block_list = []
        label_block_list = []
        W = tf.Variable(tf.zeros([feature_dim, num_classes]), dtype=tf.float32)
        b = tf.Variable(tf.zeros([1, num_classes]), dtype=tf.float32)
        for i in range(num_loss_blocks):
            X = tf.Variable( np.zeros([batch_size, feature_dim]), dtype=tf.float32)
            Y = tf.Variable( np.zeros([batch_size,num_classes]), dtype=tf.float32)
            feature_block_list.append(X)
            the_loss_block = tf.reduce_mean( tf.square(tf.matmul(X,W) + b - Y))
            loss_block_list.append(the_loss_block)
            feature_block_list.append(X)
            label_block_list.append(Y)
        pass # end for i in range(num_loss_blocks):
    pass # end with tf.device(GPU_device_str):
    var_list = [W,b]
    tfShallowOptimizer.PartialBatchSGD(fetch_batch_fn, batch_size, feature_dim, num_classes, max_iter, tol, loss_block_list, feature_block_list, label_block_list, var_list,
                                       init_learning_rate, end_learning_rate, decay_power, decay_steps, input_device_str, GPU_device_str)
pass # end with tf.Graph

