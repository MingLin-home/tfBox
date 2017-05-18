"""
Efficiently optimize shallow models using GPU
"""

import numpy as np
import tensorflow as tf
import time
import os
import shutil

def PartialBatchSGD(fetch_batch_fn, loss, feature_var, label_var, var_list, feature_dim, num_classes, GPU_buffer_size, tinyblock_size, max_iter,
                    init_learning_rate, end_learning_rate, decay_power, decay_steps, moving_average_rate,
                    input_device_str, GPU_device_str, log_steps, tol=np.inf, check_point_filename='model_checkpoint.ckpt', check_point_steps=1000, options=0,):
    

    with tf.device(input_device_str):
        new_X, new_Y = fetch_batch_fn(options)
    pass # end with tf.device
    
    with tf.device(GPU_device_str):
        grad_list = tf.gradients(loss,var_list)
        grad_norm2_list = [ tf.square(tf.norm(the_grad)) for the_grad in grad_list ]
        grad_norm2 = tf.add_n(grad_norm2_list)
        update_tinyblock_cycle_gpu = tf.Variable(-1,tf.int32)
        update_tinyblock_cycle_gpu_plus1_ops = tf.assign(update_tinyblock_cycle_gpu,
                                                         tf.mod(update_tinyblock_cycle_gpu + 1, int(GPU_buffer_size/tinyblock_size+0.1) ) ).op
        update_tinyblock_indices_base =  tf.cast(tf.linspace(0.0,tinyblock_size+0.1,tinyblock_size), dtype=tf.int32)
        update_tinyblock_indices_base = tf.reshape(update_tinyblock_indices_base,[-1,1])
        # update_tinyblock_indices = tf.random_uniform([tinyblock_size,1],minval=0,maxval=GPU_buffer_size,dtype=tf.int64)
        
        with tf.control_dependencies([update_tinyblock_cycle_gpu_plus1_ops,]):
            update_tinyblock_indices = update_tinyblock_cycle_gpu * update_tinyblock_indices_base
            with tf.control_dependencies([update_tinyblock_indices,]):
                update_X_ops = tf.scatter_nd_update(feature_var,update_tinyblock_indices, new_X).op
                update_Y_ops = tf.scatter_nd_update(label_var, update_tinyblock_indices, new_Y).op
    pass # end with tf.device(GPU_device_str):

    global_step = tf.Variable(0, tf.int64,)
    actual_init_learning_rate = options.learning_rate
    actual_end_learning_rate = options.end_learning_rate
    lr_gpu = tf.train.polynomial_decay(actual_init_learning_rate, global_step=global_step, decay_steps=decay_steps,
                                       power=decay_power, end_learning_rate=actual_end_learning_rate)

    opt_op = tf.train.GradientDescentOptimizer(lr_gpu).minimize(loss, global_step=global_step, var_list=var_list)

    ema = tf.train.ExponentialMovingAverage(decay=moving_average_rate)
    maintain_averages_op = ema.apply(var_list)
    with tf.control_dependencies([opt_op,]):
        training_op = tf.group(maintain_averages_op)

    var_to_save_list = []
    var_to_save_list += (var_list)
    var_to_save_list += [global_step, ]
    saver = tf.train.Saver(var_list=var_to_save_list, )
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    checkpoint_dir = os.path.dirname(check_point_filename)
    latest_checkpoint_filename = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir,)
    if latest_checkpoint_filename is not None:
        print('*** restore from %s' % latest_checkpoint_filename, flush=True)
        saver.restore(sess, latest_checkpoint_filename)

    # assign training features/labels to the initial graph
    print('*** loading initial batch', end='', flush=True)
    for i in range(int(GPU_buffer_size/tinyblock_size+0.1)):
        print('+',end='')
        sess.run([update_X_ops,update_Y_ops])
    pass # end for i in range(num_loss_blocks):
    print('done!',flush=True)

    # comppute the first gradient
    init_grad_norm, last_loss = sess.run([grad_norm2, loss])
    print('init_grad_norm=%g' % init_grad_norm, flush=True)
    actual_init_learning_rate = init_learning_rate/(init_grad_norm + 1.0)
    actual_end_learning_rate = end_learning_rate/( init_grad_norm + 1.0)

    start_timer = time.time()
    
    start_iter_count = sess.run(global_step)

    for iter_count in range(start_iter_count,max_iter):
        # sess.run([update_X_ops_list[random_block_index], update_Y_ops_list[random_block_index]],)
        sess.run([training_op, ])
        sess.run([update_X_ops, update_Y_ops])
        
        if iter_count % log_steps == 0:
            epoch_loss, epoch_lr,  = sess.run([loss, lr_gpu, ])
            remaining_time = float((time.time() - start_timer)) / (iter_count + 1) * (max_iter - iter_count) / 3600
            print('iter=%d, epoch_loss=%g, epoch_lr=%g, remain_time=%g' % (iter_count, epoch_loss, epoch_lr, remaining_time))
            if not epoch_loss < tol:
                print('*** stop because epoch_loss > %g' % tol)
                break
            pass  # end if epoch_loss > tol:
        pass # end if
        
        if iter_count % check_point_steps == 0:
            saver.save(sess, check_point_filename, global_step=iter_count, write_meta_graph=False)
        pass # end if iter_count % check_point_steps == 0:
    pass # end for iter_count in range(max_iter):

    return_var_list = sess.run(var_list)

    coord.request_stop()
    coord.join(threads)
    sess.close()
    return return_var_list
pass # end def


def BatchGD_with_LineSearch(loss, var_list, new_loss, update_var_ops, learning_rate_list, init_lr, grad_list, max_iter, sess,
                            check_point_filename=None, check_point_steps=100, no_stop_steps=100, arg_feed_dict=None, options=0, min_learning_rate=0.01):
    
    num_vars = len(var_list)
    grad_norm2_list = [tf.square(tf.norm(grad_list[i])) for i in range(len(var_list))]
    # grad_norm2 = tf.add_n(grad_norm2_list)
    
    increase_step_size_count = 0
    
    feed_dict = {}
    if arg_feed_dict is not None: feed_dict.update(arg_feed_dict)
    for lr_gpu in learning_rate_list:
        feed_dict[lr_gpu] = float(1.0)
    
    [init_grad_norm2_cpu,] = sess.run([grad_norm2_list,], feed_dict=feed_dict)
    # print('init_grad_norm2=%g' % init_grad_norm2_cpu)
    learning_rate_cpu_list = np.zeros([num_vars,])
    learning_rate_cpu_list += float(init_lr) / (1e-6 + np.asarray(init_grad_norm2_cpu))
    increase_step_size_count = np.zeros([num_vars,], dtype=np.int32)
    start_iter = 0
    
    last_loss = sess.run(loss)
    stopping_cri_last_loss = last_loss
    
    """
    python variables to save:
    last_loss, learning_rate_cpu_list, increase_step_size_count
    """
    latest_checkpoint_filename = None
    if check_point_filename is not None:
        saver = tf.train.Saver(var_list=var_list, )
        checkpoint_dir = os.path.dirname(check_point_filename)
        
        try:
            latest_checkpoint_filename = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir, )
        except:
            print('**** error in latest_checkpoint=%s, delete the checkpoints. You probably move the checkpoint files, which is not yet supported by Tensorflow!' % checkpoint_dir)
            shutil.rmtree(checkpoint_dir)
            os.mkdir(checkpoint_dir)
        pass # end try

        py_latest_checkpoint_filename = ''
        
        while latest_checkpoint_filename is not None and len(latest_checkpoint_filename)>0:
            try:
                if latest_checkpoint_filename is not None:
                    print('*** restore from %s' % latest_checkpoint_filename, flush=True)
                    saver.restore(sess, latest_checkpoint_filename)
                    # restore python variables
                    py_latest_checkpoint_filename = latest_checkpoint_filename+'.py.npz'
                    print('**** debug: create saver')
                    py_var_loader = np.load(py_latest_checkpoint_filename)
                    last_loss = py_var_loader['last_loss']
                    learning_rate_cpu_list = py_var_loader['learning_rate_cpu_list']
                    increase_step_size_count = py_var_loader['increase_step_size_count']
                    start_iter = py_var_loader['start_iter']
                    stopping_cri_last_loss = py_var_loader['stopping_cri_last_loss']
                    break
                pass # end if
            except:
                if os.path.isfile(latest_checkpoint_filename):
                    os.remove(latest_checkpoint_filename)
                if os.path.isfile(py_latest_checkpoint_filename):
                    os.remove(py_latest_checkpoint_filename)
                pass # end if
            pass # end try
            latest_checkpoint_filename = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir, )
        pass # end while
            
    pass # end if check_point_filename

    start_timer = time.time()
    
    for iter_count in range(start_iter, max_iter):
        for var_count in range(num_vars):
            
            feed_dict = {}
            if arg_feed_dict is not None: feed_dict.update(arg_feed_dict)
            grad_norm2_cpu = sess.run(grad_norm2_list[var_count], feed_dict=feed_dict)
            
            is_line_search_fail = True
            num_line_search = 10
            if learning_rate_cpu_list[var_count] < min_learning_rate:
                learning_rate_cpu_list[var_count] = min_learning_rate
                num_line_search = 1
            for line_search_count in range(num_line_search):
                feed_dict = {}
                if arg_feed_dict is not None: feed_dict.update(arg_feed_dict)
                for lr_count, lr_gpu in enumerate(learning_rate_list):
                    if lr_count==var_count:
                        feed_dict[lr_gpu] = learning_rate_cpu_list[lr_count]
                    else:
                        feed_dict[lr_gpu] = 0.0
                pass  # end for
        
                new_loss_cpu = sess.run(new_loss, feed_dict=feed_dict)
                if last_loss - new_loss_cpu < learning_rate_cpu_list[var_count] * grad_norm2_cpu * 0.01:
                    # learning rate too large, reduce it and retry
                    learning_rate_cpu_list[var_count] /= 2.0
                    increase_step_size_count[var_count] /= 2
                    print('-',end='')
                    if learning_rate_cpu_list[var_count]<min_learning_rate:
                        break
                    pass # end if learning_rate_cpu_list[var_count]<min_learning_rate:
                else:
                    is_line_search_fail = False
                    if line_search_count == 0:
                        increase_step_size_count[var_count] += 1
                    break
            pass
    
            if not is_line_search_fail:
                if increase_step_size_count[var_count] >= 10:
                    increase_step_size_count[var_count] = 0
                    learning_rate_cpu_list[var_count] *= 1.2
                pass  # end if increase_step_size_count >=100:
         
                if iter_count % options.log_steps == 0:
                    remaining_time = float((time.time() - start_timer)) / (iter_count + 1) * (max_iter - iter_count) / 3600
                    print('iter=%d, new_loss=%g, last_loss=%g, delta=%g, var_count=%d, lr=%g, remain_time=%g' % (iter_count, new_loss_cpu, last_loss, last_loss - new_loss_cpu,
                                                                                                   var_count, learning_rate_cpu_list[var_count], remaining_time))
        
                sess.run(update_var_ops[var_count], feed_dict=feed_dict)
                last_loss = new_loss_cpu
            pass # end if not is_line_search_fail
        pass # end for var_count in range(num_vars):

        if iter_count >= no_stop_steps and stopping_cri_last_loss - last_loss < options.tol * last_loss:
            print('*** early stop because last_loss - new_loss_cpu < options.tol')
            break

        stopping_cri_last_loss = last_loss
        
        if check_point_filename is not None and iter_count % check_point_steps == 0:
            saver.save(sess, check_point_filename, global_step=iter_count, write_meta_graph=False)
            latest_checkpoint_filename = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir, )
            if latest_checkpoint_filename is not None:
                py_latest_checkpoint_filename = latest_checkpoint_filename + '.py'
                np.savez_compressed(py_latest_checkpoint_filename,
                                    last_loss=last_loss,learning_rate_cpu_list=learning_rate_cpu_list,
                                    increase_step_size_count=increase_step_size_count,start_iter=iter_count+1,
                                    stopping_cri_last_loss=stopping_cri_last_loss,)
                
            pass  # end if
        pass # end if check_point_filename is not None and iter_count % check_point_steps == 0:
        

    pass  # end for
    
    return sess.run(var_list)
pass  # end def