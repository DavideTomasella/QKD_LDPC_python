import numpy as np
from scipy.sparse import dok_matrix as sparse_matrix
from scipy.sparse import find as sparse_find
from time import time
import random
from numpy import zeros, ceil, floor, copy, mean, sign
from itertools import compress


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
    splitting = int(ceil(n/len))
    return max(splitting, 2)


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


def split_in_blocks(x, y, n_blocks):
    # TODO random split
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    splits = np.int32(np.round(np.linspace(0, len(x), n_blocks+1)))
    x_split = ([x[idx[i:j]] for i, j in zip(splits[:-1], splits[1:])])
    y_split = ([y[idx[i:j]] for i, j in zip(splits[:-1], splits[1:])])
    indexes = ([idx[i:j] for i, j in zip(splits[:-1], splits[1:])])
    return x_split, y_split, indexes


def recursion_cascade(x_blocks, y_blocks, x_parities, y_parities):  # must be np array
    # x_parities = [sum(block) % 2 for block in x_blocks]
    # y_parities = [sum(block) % 2 for block in y_blocks]
    print(x_blocks)
    print(y_blocks)
    if sum((x_parities + y_parities) % 2) == 0:
        return x_blocks, y_blocks, 0, 1, 1

    # correct single length blocks
    tocorrect = np.int32((x_parities+y_parities) % 2 *
                         [len(block) <= 1 for block in x_blocks])
    x_blocks = [(block+flag) % 2 for flag,
                block in zip(tocorrect, x_blocks)]
    x_parities = [(par+flag) % 2 for flag,
                  par in zip(tocorrect, x_parities)]
    # still wrongs
    wrong_x_blocks = [block for block, par in zip(
        x_blocks, (x_parities+y_parities) % 2) if par]
    wrong_y_blocks = [block for block, par in zip(
        y_blocks, (x_parities+y_parities) % 2) if par]
    correct_x_blocks = [block for block, par in zip(
        x_blocks, 1-(x_parities+y_parities) % 2) if par]
    correct_y_blocks = [block for block, par in zip(
        y_blocks, 1-(x_parities+y_parities) % 2) if par]

    # split and merge
    left_x = [block[:len(block)//2] for block in wrong_x_blocks]
    left_y = [block[:len(block)//2] for block in wrong_y_blocks]
    right_x = [block[len(block)//2:] for block in wrong_x_blocks]
    right_y = [block[len(block)//2:] for block in wrong_y_blocks]

    left_x_parities = np.array([sum(block) % 2 for block in left_x])
    left_y_parities = np.array([sum(block) % 2 for block in left_y])
    right_x_parities = np.array([sum(block) % 2 for block in right_x])
    right_y_parities = np.array([sum(block) % 2 for block in right_y])
    select_left = (left_x_parities + left_y_parities) % 2
    select_right = np.array([1-sel for sel in select_left])
    new_wrong_x_blocks = [block for block, par in zip(left_x, select_left) if par] +\
        [block for block, par in zip(right_x, select_right) if par]
    new_wrong_y_blocks = [block for block, par in zip(left_y, select_left) if par] +\
        [block for block, par in zip(right_y, select_right) if par]
    new_correct_x_blocks = [block for block, par in zip(left_x, select_left) if not par] +\
        [block for block, par in zip(right_x, select_right) if not par]
    new_correct_y_blocks = [block for block, par in zip(left_y, select_left) if not par] +\
        [block for block, par in zip(right_y, select_right) if not par]

    x_parities = np.concatenate((np.extract(select_left, left_x_parities),
                                 np.extract(select_right, right_x_parities)))
    y_parities = np.concatenate((np.extract(select_left, left_y_parities),
                                 np.extract(select_right, right_y_parities)))

    # additional bits discosed: for each block, we split and share the parity of the first half
    my_add_info = len(left_x_parities)

    new_corrected_x_blocks, new_corrected_y_blocks, add_info, com_iters, n_iters = recursion_cascade(
        (new_wrong_x_blocks), (new_wrong_y_blocks), x_parities, y_parities)

    return correct_x_blocks+new_correct_x_blocks+new_corrected_x_blocks,\
        correct_y_blocks+new_correct_y_blocks+new_corrected_y_blocks,\
        add_info+my_add_info, com_iters+1, n_iters+1


def decode_cascade(x, y, qber_est, k_i, n, konst=0.73, show=1, max_iters=100500):
    splitting = choose_len(qber_est, k_i, n, konst)
    # to few data for entire process: skip
    if choose_len(qber_est, k_i-1, n, konst) <= 2 and k_i > 0:
        return 0, 0, 0, True, x, y

    x_blocks, y_blocks, indexes = split_in_blocks(x, y, splitting)
    # e_pat_in = generate_key_zeros(n)

    x_parities = np.array([sum(block) % 2 for block in x_blocks])
    y_parities = np.array([sum(block) % 2 for block in y_blocks])
    my_add_info = len(x_parities)

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

    x_dec = np.concatenate((fract_x_blocks))
    y_dec = np.concatenate((fract_y_blocks))
    # TODO if random split we must recover here or use fract_y_blocks
    ver_check = (x_dec == y_dec).all()
    if show > 1:
        print("Pass ", k_i, ' in ', com_iters, " iters, using ",
              my_add_info+add_info, " bits, matched bits:", sum(x_dec == y_dec), "/", n)

    return my_add_info+add_info, com_iters, n_iters, ver_check, x_dec, y_dec


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
    y_dec = y
    for k_i in range(passes):
        add_info, com_iters, n_iters, ver_check, x_dec, y_dec = decode_cascade(
            x_dec, y_dec, qber_est, k_i, n, konst, show, max_iter)  # need y_dec because we reorder bits!
        ladd_info += add_info
        lcom_iters += com_iters
        ln_iters += n_iters
    e_pat = (y_dec+x_dec) % 2
    ver_check = (x_dec == y_dec).all()
    if show > 0:
        if not ver_check:
            print("VERIFICATION ERROR")
            # print '\nInitial error pattern:\n', np.nonzero((x_ext+y_ext)%2),'\nFinal error pattern:\n', np.nonzero(e_pat)

        print("Done in ", lcom_iters, " iters, using ",
              ladd_info, " bits, matched bits:", sum(x_dec == y_dec), "/", n)

    return ladd_info, lcom_iters, e_pat, ver_check, ln_iters, x_dec, y_dec


def test_cascade(qber, n, n_tries, passes=4, konst=0.73, show=1, max_iter=100500):
    n = int(n)
    n_blocks = choose_len(qber, k_i=0, n=n)
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
    add_info_rslt = []
    corrected_rslt = []
    n_incor = 0

    print("QBER = ", qber, "block len =", n//n_blocks)

    for i in range(n_tries):
        if show > 0:
            print(i, end=' ')
        x = generate_key(n)
        y = add_errors_prec(x, qber)
        add_info, com_iters, e_pat, ver_check, n_iters, x_dec, y_dec = perform_cascade(
            x, y, qber_est, passes=passes, konst=konst, show=show, max_iter=max_iter)
        if ver_check:
            f_cur = float(add_info)/(n)/h_b(qber)
            f_rslt.append(f_cur)
            com_iters_rslt.append(com_iters)
            add_info_rslt.append(add_info)
        n_iters_rslt.append(n_iters)
        corrected_rslt.append(sum(x_dec == y_dec))
        if not ver_check:
            n_incor += 1

    print('Mean efficiency:', np.mean(f_rslt),
          '\nMean additional communication rounds', np.mean(com_iters_rslt), "Effective R: ", (n-add_info)/(n))
    if n_incor == n_tries:
        f_rslt = 0
        com_iters_rslt = 0
        add_info_rslt = 0
    return np.mean(f_rslt), np.mean(com_iters_rslt), np.mean(n_iters_rslt), 1, 0, 0, 0, \
        np.mean(corrected_rslt), np.mean(add_info_rslt), float(n_incor)/n_tries
