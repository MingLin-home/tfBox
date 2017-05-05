"""
Efficiently optimize shallow models using GPU
"""

import numpy as np
import tensorflow as tf
import time

def PartialBatchSGD(fetch_batch_fn, batch_size, feature_dim, num_classes, max_iter, tol, loss_block_list, feature_block_list, label_block_list, var_list,
                    init_learning_rate, end_learning_rate, decay_power, decay_steps, input_device_str, GPU_device_str, options=0):


    num_vars = len(var_list)
    num_loss_blocks = len(loss_block_list)

    with tf.device(input_device_str):
        input_X = tf.placeholder(dtype=tf.float32, shape=[batch_size,feature_dim])
        input_Y = tf.placeholder(dtype=tf.float32, shape=[batch_size,num_classes])
    pass # end with tf.device

    with tf.device(GPU_device_str):
        block_grads_list = [tf.gradients(loss_block,var_list) for loss_block in loss_block_list]
        loss = 0
        for the_loss_block in loss_block_list:
            loss += the_loss_block
        pass  # end for the_loss_block in loss_block_list:
        loss /= num_loss_blocks


        grad_norm2 = 0

        for block_grads in block_grads_list:
            for the_block_grad in block_grads:
                grad_norm2 += tf.square(tf.norm(the_block_grad))
            pass  # end for the_block_grad in block_grads:
        pass  # end for block_grads in block_grads_list:
        grad_norm2 /= num_loss_blocks

        update_X_ops_list = [tf.assign( feature_block_list[i], input_X ) for i in range(num_loss_blocks)]
        update_Y_ops_list = [tf.assign( label_block_list[i], input_Y ) for i in range(num_loss_blocks)]
    pass # end with tf.device(GPU_device_str):

    global_step = tf.Variable(0, tf.int64,)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    # assign training features/labels to the initial graph
    for i in range(num_loss_blocks):
        new_X, new_Y = fetch_batch_fn(options)
        sess.run([update_X_ops_list[i], update_Y_ops_list[i]], feed_dict={input_X: new_X, input_Y: new_Y})
    pass # end for i in range(num_loss_blocks):

    # comppute the first gradient
    init_grad_norm = sess.run(grad_norm2)
    print('init_grad_norm=%g' % init_grad_norm, flush=True)
    actual_init_learning_rate = init_learning_rate/(init_grad_norm + 1.0)
    actual_end_learning_rate = end_learning_rate/( init_grad_norm + 1.0)

    lr_gpu = tf.train.polynomial_decay(actual_init_learning_rate, global_step=global_step, decay_steps=decay_steps,
                                       power=decay_power, end_learning_rate=actual_end_learning_rate)

    train_ops = tf.train.GradientDescentOptimizer(lr_gpu).minimize(loss, global_step=global_step, var_list=var_list)

    start_timer = time.time()

    for iter_count in range(max_iter):

        new_X, new_Y = fetch_batch_fn(options)
        random_block_index = np.random.randint(num_loss_blocks)
        sess.run([update_X_ops_list[random_block_index], update_Y_ops_list[random_block_index]], feed_dict={input_X: new_X, input_Y: new_Y})
        sess.run([train_ops])
        if iter_count % 100 == 0:
            [epoch_loss, epoch_lr] = sess.run([loss, lr_gpu])
            remaining_time = float((time.time() - start_timer)) / (iter_count + 1) * (max_iter - iter_count) / 3600
            print('iter=%d, epoch_loss=%g, epoch_lr=%g, remain_time=%g' % (iter_count, epoch_loss, epoch_lr, remaining_time))

    pass # end for iter_count in range(max_iter):

pass # end def