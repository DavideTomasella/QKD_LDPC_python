
from multiprocessing import Process, Lock
from test_error_correction import run_test
from test_cascade import run_test_cascade
import numpy as np
import time

if __name__ == '__main__':
    lock = Lock()
    nrun = np.random.randint(0, 1000)

    n = 1e4  # 10000 - 1944 4000
    f_start = 1.0  # initial efficiency of decoding
    qber_start0 = 0
    range = 0.03
    qber_step = 0.00025  # 0.0001 - 0.001 # range of QBERs
    n_tries = 200  # 200 - 20  # number of keys proccessed for each QBER value
    nthreads = 5

    # passes = 4
    # konst = 0.73
    # threads = []
    # for num in np.arange(nthreads):
    #     qber_start = qber_start0+range*num
    #     qber_end = qber_start+range
    #     print("STARTING [%f,%f]" % (qber_start, qber_end))
    #     threads.append(Process(target=run_test_cascade,
    #                            args=(n, qber_start, qber_end, qber_step,
    #                                  n_tries, passes, konst, lock, num)))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()
    # time.sleep(120)
    passes = 16
    konst = 1
    threads = []
    for num in np.arange(nthreads):
        qber_start = qber_start0+range*num
        qber_end = qber_start+range
        print("STARTING [%f,%f]" % (qber_start, qber_end))
        threads.append(Process(target=run_test_cascade,
                               args=(n, qber_start, qber_end, qber_step,
                                     n_tries, passes, konst, lock, num)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # time.sleep(120)
    # passes = 4
    # konst = 0.1
    # threads = []
    # for num in np.arange(nthreads):
    #     qber_start = qber_start0+range*num
    #     qber_end = qber_start+range
    #     print("STARTING [%f,%f]" % (qber_start, qber_end))
    #     threads.append(Process(target=run_test_cascade,
    #                            args=(n, qber_start, qber_end, qber_step,
    #                                  n_tries, passes, konst, lock, num)))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()
    # threads = []
    # for num in np.arange(nthreads):
    #     qber_start = qber_start0+range*num
    #     qber_end = qber_start+range
    #     print("STARTING [%f,%f]" % (qber_start, qber_end))
    #     threads.append(Process(target=run_test,
    #                            args=(n, f_start, qber_start, qber_end,
    #                                  qber_step, n_tries, None, lock, num)))
    # for thread in threads:
    #     thread.start()
