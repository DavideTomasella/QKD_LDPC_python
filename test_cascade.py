import cascade_correction_lib as cs
import numpy as np
from file_utils import codes_from_file
import sys
import os
from multiprocessing import Process, Lock
sys.path.append(os.path.dirname(__file__))


def run_test_cascade(n, qber_start, qber_end, qber_step, n_tries, passes, konst, lock: Lock, process_number):
    # Choose of the codes pool:
    # codes = codes_from_file('codes_4000.txt'); n = 4000

    fname = 'coutput.txt'  # file name for the output

    lock.acquire()
    if not os.path.exists(fname):
        print(os.path)
        print("fuck")
        with open(fname, 'w') as file_output:
            file_output.write(
                "code_n, n_tries,          qber,        f_mean,  c_iters_mean,  n_iters_mean,             R,  pass/s_n,  kost/p_n,   p_n_max,   correct/k_n,   discl_n,           FER \n")
    lock.release()

    if qber_end is not None:
        for qber in qber_step+np.arange(qber_start, qber_end-1e-6, qber_step):
            # Here we test the error correction for a given QBER
            # Output: f_mean= mean efficiency of decoding, com_iters_mean= mean number of communication iterations,
            # R= rate of the code, s_n= number of shortened bits, p_n= number of punctured bits, p_n_max= maximal number of punctured bits,
            # k_n= n-s_n-p_n, discl_n= number of disclosed bits, FER= frame error rate
            f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER = cs.test_cascade(
                qber, n, n_tries, passes, konst, show=0, max_iter=100500)
            # print(f"qber: {qber}, f_mean: {f_mean}, com_iters_mean: {com_iters_mean}, R: {R}, s_n: {s_n}, p_n: {p_n}, p_n_max: {p_n_max}, k_n: {k_n}, discl_n: {discl_n}, FER: {FER}")
            lock.acquire()
            try:
                with open(fname, 'a') as file_output:
                    file_output.write('%d,%10d,%14.4f,%14.8f,%14.8f,%14.8f,%14.8f,%10d,%10d,%10d,%14d,%10d,%14.8f\n' %
                                      (n, n_tries, qber, f_mean, com_iters_mean, n_iters_mean, R, passes, konst*1e6, 0, k_n, discl_n, FER))
            except:
                pass
            lock.release()
    else:
        f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER = cs.test_cascade(
            qber_start, n, n_tries, passes, konst, show=2, max_iter=100500)
        # print(f"qber: {qber}, f_mean: {f_mean}, com_iters_mean: {com_iters_mean}, R: {R}, s_n: {s_n}, p_n: {p_n}, p_n_max: {p_n_max}, k_n: {k_n}, discl_n: {discl_n}, FER: {FER}")
        lock.acquire()
        with open(fname, 'a') as file_output:
            file_output.write('%d,%10d,%14.4f,%14.8f,%14.8f,%14.8f,%14.8f,%10d,%10d,%10d,%14d,%10d,%14.8f\n' %
                              (n, n_tries, qber_start, f_mean, com_iters_mean, n_iters_mean, R, passes, konst*1000, 0, k_n, discl_n, FER))
        lock.release()
    # Pypy


if __name__ == '__main__':

    n = 32
    f_start = 1.0  # initial efficiency of decoding
    qber_start = 0.02
    qber_end = 0.021
    qber_step = 0.001  # range of QBERs
    n_tries = 1  # number of keys proccessed for each QBER value
    passes = 4  # 16
    konst = 1  # 1
    run_test_cascade(n, qber_start, None, qber_step,
                     n_tries, passes, konst,
                     Lock(), np.random.randint(0, 1000))
