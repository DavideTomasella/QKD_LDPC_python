import error_correction_lib as ec
import numpy as np
from file_utils import codes_from_file
import sys
import os
from multiprocessing import Process, Lock
sys.path.append(os.path.dirname(__file__))


def run_test(n, f_start, qber_start, qber_end, qber_step, n_tries, range, mys_n, myp_n, lock: Lock, process_number):
    # Choose of the codes pool:
    # codes = codes_from_file('codes_4000.txt'); n = 4000
    codes = codes_from_file('codes_'+str(n)+'.txt')

    # Computing the range of rates for given codes
    R_range = []
    my_s_p = None
    for code in codes:
        R_range.append(code[0])
    print(f"R range is: {np.sort(R_range)}")
    if range is not None:
        R_range = [range]
        if mys_n is not None and myp_n is not None:
            my_s_p = [range, mys_n[0], myp_n[0]]
            print(my_s_p)

    fname = 'output.txt'  # file name for the output

    lock.acquire()
    if not os.path.exists(fname):
        print(os.path)
        print("fuck")
        with open(fname, 'w') as file_output:
            file_output.write(
                "code_n, n_tries,          qber,        f_mean,  c_iters_mean,  n_iters_mean,             R,       s_n,       p_n,   p_n_max,           k_n,   discl_n,           FER \n")
    lock.release()

    if qber_end is not None:
        for qber in qber_step+np.arange(qber_start, qber_end-1e-6, qber_step):
            # Here we test the error correction for a given QBER
            # Output: f_mean= mean efficiency of decoding, com_iters_mean= mean number of communication iterations,
            # R= rate of the code, s_n= number of shortened bits, p_n= number of punctured bits, p_n_max= maximal number of punctured bits,
            # k_n= n-s_n-p_n, discl_n= number of disclosed bits, FER= frame error rate
            f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER = ec.test_ec(
                qber, R_range, codes, n, n_tries, f_start=f_start, show=1, discl_k=1, max_iter=100500, my_s_p=my_s_p)
            # print(f"qber: {qber}, f_mean: {f_mean}, com_iters_mean: {com_iters_mean}, R: {R}, s_n: {s_n}, p_n: {p_n}, p_n_max: {p_n_max}, k_n: {k_n}, discl_n: {discl_n}, FER: {FER}")
            lock.acquire()
            with open(fname, 'a') as file_output:
                file_output.write('%d,%10d,%14.4f,%14.8f,%14.8f,%14.8f,%14.8f,%10d,%10d,%10d,%14d,%10d,%14.8f\n' %
                                  (n, n_tries, qber, f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER))
            lock.release()
    elif mys_n is not None and myp_n is not None:
        for ss_n in mys_n:
            for pp_n in myp_n:
                if ss_n+pp_n < n:
                    f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER = ec.test_ec(
                        qber_start, R_range, codes, n, n_tries, f_start=f_start, show=1, discl_k=1, max_iter=1000, my_s_p=[range, ss_n, pp_n])
                    # print(f"qber: {qber}, f_mean: {f_mean}, com_iters_mean: {com_iters_mean}, R: {R}, s_n: {s_n}, p_n: {p_n}, p_n_max: {p_n_max}, k_n: {k_n}, discl_n: {discl_n}, FER: {FER}")
                    lock.acquire()
                    with open(fname, 'a') as file_output:
                        file_output.write('%d,%10d,%14.4f,%14.8f,%14.8f,%14.8f,%14.8f,%10d,%10d,%10d,%14d,%10d,%14.8f\n' %
                                          (n, n_tries, qber_start, f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER))
                    lock.release()
    else:
        f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER = ec.test_ec(
            qber_start, R_range, codes, n, n_tries, f_start=f_start, show=2, discl_k=1, max_iter=100500, my_s_p=my_s_p)
        # print(f"qber: {qber}, f_mean: {f_mean}, com_iters_mean: {com_iters_mean}, R: {R}, s_n: {s_n}, p_n: {p_n}, p_n_max: {p_n_max}, k_n: {k_n}, discl_n: {discl_n}, FER: {FER}")
        lock.acquire()
        with open(fname, 'a') as file_output:
            file_output.write('%d,%10d,%14.4f,%14.8f,%14.8f,%14.8f,%14.8f,%10d,%10d,%10d,%14d,%10d,%14.8f\n' %
                              (n, n_tries, qber_start, f_mean, com_iters_mean, n_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER))
        lock.release()
    # Pypy


if __name__ == '__main__':

    n = 1944
    f_start = 1.0  # initial efficiency of decoding
    qber_start = 0.02
    qber_end = 0.021
    qber_step = 0.001  # range of QBERs
    n_tries = 20  # number of keys proccessed for each QBER value
    s_n = np.int32(np.linspace(0, n*0.8333/2, 50))
    p_n = np.int32(np.linspace(0, 152, 10))  # 152 227 302 439
    run_test(n, f_start, qber_start, None,
             qber_step, n_tries, 0.8333, s_n, p_n,
             Lock(), np.random.randint(0, 1000))
