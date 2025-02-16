
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

# # ############## multiprocess
#     n = 1944
#     threads = []
#     for num in np.arange(nthreads):
#         qber_start = qber_start0+range*num
#         qber_end = qber_start+range
#         print("STARTING [%f,%f]" % (qber_start, qber_end))
#         threads.append(Process(target=run_test,
#                                args=(n, f_start, qber_start, qber_end,
#                                      qber_step, n_tries, None, None, None,
#                                      lock, num)))
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()
#     n = 4000
#     threads = []
#     for num in np.arange(nthreads):
#         qber_start = qber_start0+range*num
#         qber_end = qber_start+range
#         print("STARTING [%f,%f]" % (qber_start, qber_end))
#         threads.append(Process(target=run_test,
#                                args=(n, f_start, qber_start, qber_end,
#                                      qber_step, n_tries, None, None, None,
#                                      lock, num)))
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()

# ########################

#     n = 1944
#     threads = []
#     qber_start = 0.02
#     s_n = np.int32(np.linspace(0, n*0.8333/2, 50))
#     p_n = [0]  # 152 227 302 439
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.8333, s_n, p_n,
#                                  lock, nrun+51)))
#     s_n = [0]
#     p_n = np.int32(np.linspace(0, 152, 10))  # 152 227 302 439
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.8333, s_n, p_n,
#                                  lock, nrun+52)))
#     qber_start = 0.1
#     s_n = np.int32(np.linspace(0, n*0.5/2, 30))
#     p_n = [0]  # 152 227 302 439
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.5, s_n, p_n,
#                                  lock, nrun+53)))
#     s_n = [0]
#     p_n = np.int32(np.linspace(0, 439, 30))  # 152 227 302 439
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.5, s_n, p_n,
#                                  lock, nrun+54)))
#     for thread in threads:
#         thread.start()
#     threads[-1].join()

#     n = 4000
#     threads = []
#     qber_start = 0.02
#     s_n = np.int32(np.linspace(0, n*0.85/2, 70))
#     p_n = [0]  # 266 864
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.85, s_n, p_n,
#                                  lock, nrun+55)))
#     s_n = [0]
#     p_n = np.int32(np.linspace(0, 266, 10))  # 266 864
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.85, s_n, p_n,
#                                  lock, nrun+56)))
#     qber_start = 0.1
#     s_n = np.int32(np.linspace(0, n*0.5/2, 40))
#     p_n = [0]  # 266 864
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.5, s_n, p_n,
#                                  lock, nrun+57)))
#     s_n = [0]
#     p_n = np.int32(np.linspace(0, 864, 30))  # 266 864
#     threads.append(Process(target=run_test,
#                            args=(n, f_start, qber_start, None,
#                                  qber_step, n_tries, 0.5, s_n, p_n,
#                                  lock, nrun+58)))
#     for thread in threads:
#         thread.start()

#     for thread in threads:
#         thread.join()

#     exit(0)
##########################################
##########################################

    n = 1944
    threads = []
    # qber_start = 0.026
    # qber_end = 0.05
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.8333, None, None,
    #                              lock, nrun+1)))
    qber_start = 0
    qber_end = 0.009
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, None, None,
                                 lock, nrun+2)))
    # qber_start = 0.046
    # qber_end = 0.07
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.75, None, None,
    #                              lock, nrun+3)))
    qber_start = 0.000
    qber_end = 0.029
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6667, None, None,
                                 lock, nrun+4)))
    # qber_start = 0.078
    # qber_end = 0.1
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.6667, None, None,
    #                              lock, nrun+5)))
    qber_start = 0.0
    qber_end = 0.069
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.5, None, None,
                                 lock, nrun+6)))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()

#####################
    n = 4000
    # threads = []
    # qber_start = 0.01
    # qber_end = 0.03
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.9, None, None,
    #                              lock, nrun+11)))
    # qber_start = 0.004
    # qber_end = 0.01
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.85, None, None,
    #                              lock, nrun+12)))
    # qber_start = 0.02
    # qber_end = 0.035
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.85, None, None,
    #                              lock, nrun+13)))
    # qber_start = 0.009
    # qber_end = 0.02
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.8, None, None,
    #                              lock, nrun+14)))
    # qber_start = 0.03
    # qber_end = 0.04
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.8, None, None,
    #                              lock, nrun+15)))
    qber_start = 0.014
    qber_end = 0.024
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, None, None,
                                 lock, nrun+16)))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()

    # threads = []
    # qber_start = 0.045  # ccccccccccccccccccccccccccccccccccc
    # qber_end = 0.055
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.75, None, None,
    #                              lock, nrun+17)))
    qber_start = 0.019
    qber_end = 0.034
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.7, None, None,
                                 lock, nrun+18)))
    # qber_start = 0.055
    # qber_end = 0.065
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.7, None, None,
    #                              lock, nrun+19)))
    qber_start = 0.029
    qber_end = 0.044
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.65, None, None,
                                 lock, nrun+20)))
    # qber_start = 0.07
    # qber_end = 0.08
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.65, None, None,
    #                              lock, nrun+21)))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()

    # threads = []
    qber_start = 0.039
    qber_end = 0.059
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6, None, None,
                                 lock, nrun+22)))
    # qber_start = 0.085
    # qber_end = 0.095
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.6, None, None,
    #                              lock, nrun+23)))
    qber_start = 0.049
    qber_end = 0.074
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.55, None, None,
                                 lock, nrun+24)))
    # qber_start = 0.1
    # qber_end = 0.11
    # print("STARTING [%f,%f]" % (qber_start, qber_end))
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, qber_end,
    #                              qber_step, n_tries, 0.55, None, None,
    #                              lock, nrun+25)))
    qber_start = 0.059
    qber_end = 0.09
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.5, None, None,
                                 lock, nrun+26)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    exit(0)

    ########
    n = 1944
    threads = []
    qber_start = 0.026
    qber_end = 0.05
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.8333, None, None,
                                 lock, nrun+31)))
    qber_start = 0.009
    qber_end = 0.028
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, None, None,
                                 lock, nrun+32)))
    qber_start = 0.046
    qber_end = 0.07
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, None, None,
                                 lock, nrun+33)))
    qber_start = 0.029
    qber_end = 0.048
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6667, None, None,
                                 lock, nrun+34)))
    qber_start = 0.078
    qber_end = 0.1
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.6667, None, None,
                                 lock, nrun+35)))
    qber_start = 0.069
    qber_end = 0.08
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.5, None, None,
                                 lock, nrun+36)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    n = 4000
    threads = []
    qber_start = 0.01
    qber_end = 0.015
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.85, None, None,
                                 lock, nrun+42)))
    qber_start = 0.02
    qber_end = 0.022
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.8, None, None,
                                 lock, nrun+43)))
    qber_start = 0.03
    qber_end = 0.035
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.75, None, None,
                                 lock, nrun+44)))
    qber_start = 0.055
    qber_end = 0.057
    print("STARTING [%f,%f]" % (qber_start, qber_end))
    threads.append(Process(target=run_test,
                           args=(n, f_start, qber_start, qber_end,
                                 qber_step, n_tries, 0.65, None, None,
                                 lock, nrun+45)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # threads = []
    # qber_start = 0.02
    # s_n = np.int32(np.linspace(0, n*0.8333/2, 50))
    # p_n = np.int32(np.linspace(0, 152, 10))  # 152 227 302 439
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, None,
    #                              qber_step, n_tries, 0.8333, s_n, p_n,
    #                              lock, nrun+1)))
    # qber_start = 0.1
    # s_n = np.int32(np.linspace(0, n*0.5/2, 30))
    # p_n = np.int32(np.linspace(0, 439, 30))  # 152 227 302 439
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, None,
    #                              qber_step, n_tries, 0.5, s_n, p_n,
    #                              lock, nrun+2)))

    # n = 4000
    # qber_start = 0.02
    # s_n = np.int32(np.linspace(0, n*0.85/2, 70))
    # p_n = np.int32(np.linspace(0, 266, 10))  # 266 864
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, None,
    #                              qber_step, n_tries, 0.85, s_n, p_n,
    #                              lock, nrun+3)))
    # qber_start = 0.1
    # s_n = np.int32(np.linspace(0, n*0.5/2/2, 20))
    # p_n = np.int32(np.linspace(0, 864, 30))  # 266 864
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, None,
    #                              qber_step, n_tries, 0.5, s_n, p_n,
    #                              lock, nrun+4)))
    # s_n = np.int32(np.linspace(n*0.5/2/2, n*0.5/2, 20))
    # p_n = np.int32(np.linspace(0, 864, 30))  # 266 864
    # threads.append(Process(target=run_test,
    #                        args=(n, f_start, qber_start, None,
    #                              qber_step, n_tries, 0.5, s_n, p_n,
    #                              lock, nrun+5)))
    # for thread in threads:
    #     thread.start()

    # threads[-1].join()
