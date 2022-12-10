import error_correction_lib as ec
import numpy as np
from file_utils import codes_from_file
import sys
import os
from multiprocessing import Process, Lock
sys.path.append(os.path.dirname(__file__))


def run_test(n, f_start, qber_start, qber_end, qber_step, n_tries, lock: Lock, process_number):
    # Choose of the codes pool:
    #codes = codes_from_file('codes_4000.txt'); n = 4000
    codes = codes_from_file('codes_'+str(n)+'.txt')

    # Computing the range of rates for given codes
    R_range = []
    for code in codes:
        R_range.append(code[0])
    print(f"R range is: {np.sort(R_range)}")

    fname = 'output.txt'  # file name for the output

    lock.acquire()
    if not os.path.exists(fname):
        print(os.path)
        print("fuck")
        with open(fname, 'w') as file_output:
            file_output.write(
                "code_n, n_tries,          qber,        f_mean,  c_iters_mean,  n_iters_mean,             R,       s_n,       p_n,   p_n_max,           k_n,   discl_n,           FER \n")
    lock.release()

    for qber in qber_step+np.arange(qber_start, qber_end-1e-6, qber_step):
        # Here we test the error correction for a given QBER
        # Output: f_mean= mean efficiency of decoding, com_iters_mean= mean number of communication iterations,
        # R= rate of the code, s_n= number of shortened bits, p_n= number of punctured bits, p_n_max= maximal number of punctured bits,
        # k_n= n-s_n-p_n, discl_n= number of disclosed bits, FER= frame error rate
        f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER = ec.test_ec(
            qber, R_range, codes, n, n_tries, f_start=f_start, show=1, discl_k=1, max_iter=100500)
        # print(f"qber: {qber}, f_mean: {f_mean}, com_iters_mean: {com_iters_mean}, R: {R}, s_n: {s_n}, p_n: {p_n}, p_n_max: {p_n_max}, k_n: {k_n}, discl_n: {discl_n}, FER: {FER}")
        lock.acquire()
        with open(fname, 'a') as file_output:
            file_output.write('%d,%10d,%14.4f,%14.8f,%14.8f,%14.8f,%14.8f,%10d,%10d,%10d,%14d,%10d,%14.8f\n' %
                              (n, n_tries, qber, f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER))
        lock.release()
    # Pypy


if __name__ == '__main__':

    n = 1944
    f_start = 1.0  # initial efficiency of decoding
    qber_start = 0.046
    qber_end = 0.047
    qber_step = 0.001  # range of QBERs
    n_tries = 20  # number of keys proccessed for each QBER value
    run_test(n, f_start, qber_start, qber_end,
             qber_step, n_tries, Lock(), np.random.randint(0, 1000))
