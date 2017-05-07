import numpy as np
import tensorflow as tf
import time

batch_size = 100*10

feature_dim=5000
max_iter=1000
tol = 1e-6
var_list = None
init_learning_rate = 0.01
end_learning_rate = 0.01
decay_power = 1.0
decay_steps = 1013
input_device_str = '/cpu:0'
#GPU_device_str = '/cpu:0'
num_classes = 10

num_loss_blocks = 100
W_true = np.random.rand(feature_dim,num_classes)/np.sqrt(feature_dim)

def fetch_batch_fn(options=None):
    new_X = np.random.randn(batch_size,feature_dim)/np.sqrt(feature_dim)
    new_Y = new_X.dot(W_true)
    return  new_X, new_Y
pass # end def

with tf.Graph().as_default():
    # with tf.device(GPU_device_str):
    new_X, new_Y = fetch_batch_fn()

    X = tf.placeholder(dtype=tf.float32,shape=[batch_size, feature_dim])
    Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_classes])
    W = tf.Variable(tf.zeros([feature_dim,num_classes]))
    loss = tf.reduce_sum(tf.square(tf.norm(tf.matmul(X,W) - Y)))
    # pass # end

    train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    start_timer = time.time()
    for iter_count in range(max_iter):
        new_X, new_Y = fetch_batch_fn()
        _,epoch_loss = sess.run([train_op,loss],feed_dict={X: new_X, Y: new_Y})
        if iter_count % 100 ==0:
            remaining_time = float((time.time() - start_timer)) / (iter_count + 1) * (max_iter - iter_count) / 3600
            print('iter=%d, epoch_loss=%g, time=%g' % (iter_count, epoch_loss,remaining_time))
