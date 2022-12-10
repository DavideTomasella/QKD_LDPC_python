
from multiprocessing import Process, Lock
from test_error_correction import run_test
import numpy as np

if __name__ == '__main__':
    lock = Lock()
    nrun = np.random.randint(0, 1000)

    n = 1944  # 1944 4000
    f_start = 1.0  # initial efficiency of decoding
    qber_start0 = 0.05
    range = 0.01
    qber_step = 0.001  # range of QBERs
    n_tries = 20  # number of keys proccessed for each QBER value
    nthreads = 5

    threads = []
    for num in np.arange(nthreads):
        qber_start = qber_start0+range*num
        qber_end = qber_start+range
        print("STARTING [%f,%f]" % (qber_start, qber_end))
        threads.append(Process(target=run_test,
                               args=(n, f_start, qber_start, qber_end,
                                     qber_step, n_tries, lock, num)))
    for thread in threads:
        thread.start()
