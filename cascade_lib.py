import numpy as np
from scipy.sparse import dok_matrix as sparse_matrix
from scipy.sparse import find as sparse_find
from time import time
import random
from numpy import zeros, ceil, floor, copy, mean, sign


def generate_key(length):
    """
    Generate random key of length 'length'
    """
    return np.random.randint(0, 2, (1, length))[0]


def generate_key_zeros(length):
    """
    Generate key with zeors only of length 'length'
    """
    return np.zeros(length, dtype=np.int)


def add_errors(a, error_prob):
    """
    Flip some values (1->0, 0->1) in 'a' with probability 'error_prob'
    """
    error_mask = np.random.choice(
        2, size=a.shape, p=[1.0-error_prob, error_prob])
    return np.where(error_mask, ~a+2, a)


def add_errors_prec(a, error_prob):
    """
    Add precisely 'error_prob'*length('a') errors in key 'a' 
    """
    len_a = len(a)
    n_er = int(round(len_a*error_prob))
    list1 = list(range(0, len_a))
    list2 = random.sample(list1, n_er)
    K_cor = a.copy()
    for i in list2:
        K_cor[i] = 1-K_cor[i]
    return K_cor


def choose_len(qber, k_i=0, n=1e4, konst=0.73):
    '''
    Choose appropriate len of blocks
    '''
    len = konst/qber
    if k_i > 0:
        len *= (k_i+1)
    roundlen = n/(n//len)
    return roundlen


def h_b(x):
    """
    Binary entropy function of 'x'
    """
    if x > 0:
        return -x*np.log2(x)-(1-x)*np.log2(1-x)
    elif x == 0:
        return 0
    else:
        print("Incorrect argument in binary entropy function")


def split_in_blocks(x, y, block_len):
    # TODO random split
    x_split = [x[i:i+block_len] for i in range(0, len(x), block_len)]
    y_split = [y[i:i+block_len] for i in range(0, len(x), block_len)]
    indexes = [[i, i+block_len] for i in range(0, len(x), block_len)]
    return x_split, y_split, indexes


def recursion_cascade(x_blocks, y_blocks, x_parities, y_parities):  # must be np array
    #x_parities = [sum(block) % 2 for block in x_blocks]
    #y_parities = [sum(block) % 2 for block in y_blocks]
    my_add_info = len(x_blocks)
    if sum((x_parities + y_parities) % 2) == 0:
        return x_blocks, y_blocks, 0, 1, 1

    # correct single length blocks
    tocorrect = (x_parities+y_parities) % 2 + \
        [len(block) <= 1 for block in x_blocks]
    x_blocks = (x_blocks+tocorrect*x_blocks) % 2
    # still wrongs
    wrong_x_blocks = x_blocks[(x_parities+y_parities) % 2]
    wrong_y_blocks = y_blocks[(x_parities+y_parities) % 2]
    correct_x_blocks = x_blocks[1 - (x_parities+y_parities) % 2]
    correct_y_blocks = y_blocks[1 - (x_parities+y_parities) % 2]

    #split and merge
    left_x = [block[:len(block)//2] for block in wrong_x_blocks]
    left_y = [block[:len(block)//2] for block in wrong_y_blocks]
    right_x = [block[len(block)//2:] for block in wrong_x_blocks]
    right_y = [block[len(block)//2:] for block in wrong_y_blocks]

    left_x_parities = [sum(block) % 2 for block in left_x]
    left_y_parities = [sum(block) % 2 for block in left_y]
    right_x_parities = [1 - par for par in left_x_parities]
    right_y_parities = [1 - par for par in left_y_parities]
    select_left = (left_x_parities + left_y_parities) % 2
    select_right = [1-sel for sel in select_left]
    x_blocks = left_x[select_left].append(right_x[select_right])
    y_blocks = left_y[select_left].append(right_y[select_right])

    x_parities = left_x_parities[select_left].append(
        right_x_parities[select_right])
    y_parities = left_y_parities[select_left].append(
        right_y_parities[select_right])

    new_x_blocks, new_y_blocks, add_info, com_iters, n_iters = recursion_cascade(
        correct_x_blocks.append(x_blocks), correct_y_blocks.append(y_blocks),
        zeros(len(correct_x_blocks), 1).append(x_parities), zeros(len(correct_x_blocks), 1).append(y_parities))
    return new_x_blocks, new_y_blocks, add_info+my_add_info, com_iters+1, n_iters+1


def decode_cascade(x, y, qber_est, k_i, n, konst=0.73, show=1, max_iters=100500):
    block_len = choose_len(qber_est, k_i, n, konst)

    x_blocks, y_blocks, indexes = split_in_blocks(x, y, block_len)
    # e_pat_in = generate_key_zeros(n)

    x_parities = [sum(block) % 2 for block in x_blocks]
    y_parities = [sum(block) % 2 for block in y_blocks]

    # add_info = 0
    # com_iters = 0
    # n_iters = 0
    # while sum((x_parities + y_parities) % 2) > 0 and n_iters < max_iters:
    #     add_info += len(x_parities)
    #     com_iters += 1
    #     n_iters += 1

    #     wrong_x_blocks = x_blocks[(x_parities+y_parities) % 2]
    #     wrong_y_blocks = y_blocks[(x_parities+y_parities) % 2]
    #     wrong_x_blocks = [block for block in wrong_x_blocks if len(block > 1)]
    #     wrong_y_blocks = [block for block in wrong_x_blocks if len(block > 1)]
    #     wrong_x_parities = x_parities[(x_parities+y_parities) % 2]
    #     wrong_y_parities = y_parities[(x_parities+y_parities) % 2]

    #     left_x = [block[:len(block)//2] for block in wrong_x_blocks]
    #     left_y = [block[:len(block)//2] for block in wrong_y_blocks]
    #     right_x = [block[len(block)//2:] for block in wrong_x_blocks]
    #     right_y = [block[len(block)//2:] for block in wrong_y_blocks]

    #     left_x_parities = [sum(block) % 2 for block in left_x]
    #     left_y_parities = [sum(block) % 2 for block in left_y]
    #     right_x_parities = [1 - par for par in left_x_parities]
    #     right_y_parities = [1 - par for par in left_y_parities]
    #     select_left = (left_x_parities + left_y_parities) % 2
    #     select_right = [1-sel for sel in select_left]
    #     x_blocks = left_x[select_left].append(right_x[select_right])
    #     y_blocks = left_y[select_left].append(right_y[select_right])

    #     x_parities = left_x_parities[select_left].append(
    #         right_x_parities[select_right])
    #     y_parities = left_y_parities[select_left].append(
    #         right_y_parities[select_right])

    fract_x_blocks, fract_y_blocks, add_info, com_iters, n_iters = recursion_cascade(
        x_blocks, y_blocks, x_parities, y_parities)

    x_dec = np.array([])
    [x_dec.append(xx) for xx in fract_x_blocks]
    # TODO if random split we must recover here or use fract_y_blocks
    ver_check = (x_dec == y).all()
    if show > 1:
        print("Pass ", k_i, ' in ', com_iters, " iters, using ",
              add_info, " bits, matched bits:", sum(x_dec == y), "/", n)

    return add_info, com_iters, n_iters, ver_check, x_dec


def perform_cascade(x, y, qber_est, passes=4, konst=0.73, show=1, max_iter=100500):
    n = len(x)
    m = len(y)

    # s_pos, p_pos, k_pos = generate_sp(s_n, p_n, n-s_n-p_n, p_list=punct_list)

    # x_ext = extend_sp(x, s_pos, p_pos, k_pos)
    # y_ext = extend_sp(y, s_pos, p_pos, k_pos)

    # k_pos_in = copy(k_pos)  # For final exclusion

    # s_x = encode_syndrome(x_ext, s_y_joins)
    # s_y = encode_syndrome(y_ext, s_y_joins)

    # s_d = (s_x+s_y) % 2
    key_sum = (x+y) % 2
    ladd_info, lcom_iters, ln_iters = [0, 0, 0]
    x_dec = x
    for k_i in range(passes):
        add_info, com_iters, n_iters, ver_check, x_dec = decode_cascade(
            x_dec, y, qber_est, k_i, n, konst, show, max_iter)
        ladd_info += add_info
        lcom_iters += com_iters
        ln_iters += n_iters
    e_pat = (x+x_dec) % 2
    ver_check = (x_dec == y).all()
    if not ver_check:
        print("VERIFICATION ERROR")
        # print '\nInitial error pattern:\n', np.nonzero((x_ext+y_ext)%2),'\nFinal error pattern:\n', np.nonzero(e_pat)

    return ladd_info, lcom_iters, e_pat, ver_check, ln_iters


def test_cascade(qber, n, n_tries, passes=4, konst=0.73, show=1, max_iter=100500):
    block_len = choose_len(qber, k_i=0, n=n)
    # if my_s_p is not None:
    #     R, s_n, p_n = my_s_p
    # k_n = n-s_n-p_n
    # m = (1-R)*n
    # code_params = codes[(R, n)]
    # s_y_joins = code_params['s_y_joins']
    # y_s_joins = code_params['y_s_joins']
    # punct_list = code_params['punct_list']
    # syndrome_len = code_params['syndrome_len']
    # p_n_max = len(punct_list)
    # discl_n = int(round(n*(0.0280-0.02*R)*discl_k))
    qber_est = qber
    f_rslt = []
    com_iters_rslt = []
    n_iters_rslt = []
    n_incor = 0

    print("QBER = ", qber, "block len =", block_len)

    for i in range(n_tries):
        print(i, end=' ')
        x = generate_key(n)
        y = add_errors(x, qber)
        add_info, com_iters, x_dec, ver_check, n_iters = perform_cascade(
            x, y, qber_est, passes=passes, konst=konst, show=show, max_iter=max_iter)
        f_cur = float(n-add_info)/(n)/h_b(qber)
        f_rslt.append(f_cur)
        com_iters_rslt.append(com_iters)
        n_iters_rslt.append(n_iters)
        if not ver_check:
            n_incor += 1
        print(i, " Done in ", com_iters, " iters, using ",
              add_info, " bits, matched bits:", sum(x_dec == y), "/", n)

    print('Mean efficiency:', np.mean(f_rslt),
          '\nMean additional communication rounds', np.mean(com_iters_rslt), "Effective R: ", (n-add_info)/(n))
    return np.mean(f_rslt), np.mean(com_iters_rslt), np.mean(n_iters_rslt), 1, 0, 0, 0, 0, 0, float(n_incor)/n_tries
