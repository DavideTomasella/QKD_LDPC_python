
from multiprocessing import Process, Lock
from test_error_correction import run_test
import numpy as np

if __name__ == '__main__':
    lock = Lock()
    nrun = np.random.randint(0, 1000)

    n = 1944  # 1944 4000
    f_start = 1.0  # initial efficiency of decoding
    qber_start0 = 0
    range = 0.03
    qber_step = 0.001  # range of QBERs
    n_tries = 20  # number of keys proccessed for each QBER value
    nthreads = 5

    threads = []
    qber_start = 0.026
    qber_end = 0.05
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.8333, lock, nrun+1)))
    qber_start = 0.009
    qber_end = 0.028
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, lock, nrun+2)))
    qber_start = 0.046
    qber_end = 0.07
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, lock, nrun+3)))
    qber_start = 0.029
    qber_end = 0.048
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6667, lock, nrun+4)))
    qber_start = 0.078
    qber_end = 0.01
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6667, lock, nrun+5)))
    qber_start = 0.069
    qber_end = 0.08
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.5, lock, nrun+6)))
    for thread in threads:
        thread.start()

    threads[-1].join()

#####################
    n = 4000
    threads = []
    qber_start = 0.01
    qber_end = 0.03
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.9, lock, nrun+11)))
    qber_start = 0.004
    qber_end = 0.01
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.85, lock, nrun+12)))
    qber_start = 0.02
    qber_end = 0.035
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.85, lock, nrun+13)))
    qber_start = 0.009
    qber_end = 0.02
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.8, lock, nrun+14)))
    qber_start = 0.03
    qber_end = 0.04
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.8, lock, nrun+15)))
    qber_start = 0.024
    qber_end = 0.03
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, lock, nrun+16)))
    for thread in threads:
        thread.start()

    threads[-1].join()
    threads = []
    qber_start = 0.045  # ccccccccccccccccccccccccccccccccccc
    qber_end = 0.055
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, lock, nrun+17)))
    qber_start = 0.034
    qber_end = 0.045
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.7, lock, nrun+18)))
    qber_start = 0.055
    qber_end = 0.065
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.7, lock, nrun+19)))
    qber_start = 0.045
    qber_end = 0.055
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.65, lock, nrun+20)))
    qber_start = 0.07
    qber_end = 0.08
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.65, lock, nrun+21)))
    for thread in threads:
        thread.start()

    threads[-1].join()
    threads = []
    qber_start = 0.06
    qber_end = 0.07
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6, lock, nrun+22)))
    qber_start = 0.085
    qber_end = 0.095
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6, lock, nrun+23)))
    qber_start = 0.075
    qber_end = 0.085
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.55, lock, nrun+24)))
    qber_start = 0.1
    qber_end = 0.11
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.55, lock, nrun+25)))
    qber_start = 0.09
    qber_end = 0.1
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.5, lock, nrun+26)))
    for thread in threads:
        thread.start()

    threads[-1].join()
