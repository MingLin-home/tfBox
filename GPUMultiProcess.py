import multiprocessing



def decode_job_queu_and_run(para):
    par_do_func, task_parameters = para
    print('debug: in decode_job_queu_and_run!')
    the_free_device_str = decode_job_queu_and_run.queue.get()
    print('debug: get device %s' % the_free_device_str)
    par_do_func(the_free_device_str, task_parameters)
    decode_job_queu_and_run.queue.put(the_free_device_str)
    
    
def init_decode_job_queu_and_run(the_queue):
    decode_job_queu_and_run.queue = the_queue

def parallel_GPU_CPU_do(num_gpus, num_cpus, par_do_func, task_list):
    """
    
    :param par_todo_func: must accept par_do_func(device_str='/gpu:0', task_list[i])
    :param task_list: list of tasks.
    """

    num_jobs = num_gpus + num_cpus
    free_device_queue = multiprocessing.Queue()
    
    worker_pool = multiprocessing.Pool(num_jobs, init_decode_job_queu_and_run,[free_device_queue,])
    
    for i in range(num_gpus):
        free_device_queue.put('/gpu:%d' % i)
    pass # end for
    
    for i in range(num_cpus):
        free_device_queue.put('/cpu:%d' % i)
    pass # end for

    worker_pool.map(decode_job_queu_and_run,[(par_do_func,task_parameters) for task_parameters in task_list] )
    
    worker_pool.close()
    worker_pool.join()


if __name__ == '__main__':
    # multiprocessing.freeze_support()
    pass