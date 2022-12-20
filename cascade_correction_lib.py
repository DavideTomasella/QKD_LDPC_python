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

class IdTree:
    def __init__(self, indexes):
        self.indexes = indexes
        self.left = None
        self.right = None
    def create_children(self):
        if len(self.indexes)>1 and self.is_leaf():
            child_idx = splits(self.indexes)
            self.left=IdTree(child_idx[0])
            self.right=IdTree(child_idx[1])
            return 1
        return 0
    def get_parity(self,x,y):
        return sum(x[self.indexes]+y[self.indexes])%2
    def is_onebit(self):
        return len(self.indexes)==1
    def is_leaf(self):
        return self.left is None and self.right is None

def splits(indexes, n_blocks=2):
    np.random.shuffle(indexes)
    splits = np.int32(np.round(np.linspace(0, len(indexes), n_blocks+1)))
    new = [indexes[i:j] for i, j in zip(splits[:-1], splits[1:])]
    return new

def create_trees(length, n_blocks=2):
    trees = []
    for idx in splits(np.arange(length), n_blocks):
        tree=IdTree(idx)
        trees.append(tree)
    return trees

def correct_tree(x,y,tree):
    add_info = 0
    corr = set()
    iters = 0
    # print(x[tree.indexes])
    if tree.is_onebit():
        if tree.get_parity(x,y) != 0:
            x[tree.indexes] = 1 - x[tree.indexes]
            corr = set(tree.indexes)
            iters +=1
    elif tree.get_parity(x,y) !=0:
        add_info += tree.create_children()
        l_corr, l_iters, l_add_info = correct_tree(x,y,tree.left)
        r_corr, r_iters, r_add_info = correct_tree(x,y,tree.right)
        corr |= l_corr | r_corr
        iters += l_iters + r_iters +1
        add_info += l_add_info + r_add_info
    return corr, iters, add_info

def cascade_correction(x,y,tree, cc):
    add_info = 0
    corr = set()
    iters = 0
    if cc in tree.indexes:
        if tree.is_leaf():
            if tree.get_parity(x,y) != 0:
                c_corr, c_iters, c_add_info = correct_tree(x,y,tree)
                corr |= c_corr
                iters += c_iters
                add_info += c_add_info
        else:
            l_corr, l_iters, l_add_info = cascade_correction(x,y,tree.left, cc)
            r_corr, r_iters, r_add_info = cascade_correction(x,y,tree.right, cc)
            corr |= l_corr | r_corr
            iters += l_iters + r_iters
            add_info += l_add_info + r_add_info
    return corr, iters, add_info

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
    # key_sum = (x+y) % 2
    ladd_info, lcom_iters, ln_iters = [0, 0, 0]
    forest = []
    prev_x=x.copy()
    for k_i in range(passes):
        splitting = choose_len(qber_est, k_i, n, konst)
        new_trees = create_trees(len(x), n_blocks=splitting)
        corrected = set()
        com_iters = 0
        # correction
        for tree in new_trees:
            parity = tree.get_parity(x,y)
            ladd_info +=1
            if parity != 0:
                corr, iters, add_info = correct_tree(x,y,tree)
                corrected |= set(corr)
                ln_iters += iters
                ladd_info += add_info
                com_iters = max(com_iters, iters)
        lcom_iters += com_iters
        if show>1:
            print(corrected)
        # cascade effect
        while len(corrected)>0:
            new_corrected = set()
            com_iters = 0
            for cc in corrected:
                for tree in forest:
                    new_corr, iters, add_info = cascade_correction(x,y,tree,cc)
                    new_corrected |= set(new_corr)
                    ln_iters += iters
                    ladd_info += add_info
                    com_iters = max(com_iters, iters)
            lcom_iters += com_iters
            corrected = new_corrected
            if show>1:
                print(corrected)
        forest.extend(new_trees)
    e_pat = (prev_x+x) % 2
    ver_check = (x == y).all()
    if show > 0:
        if not ver_check:
            print("VERIFICATION ERROR")
            # print '\nInitial error pattern:\n', np.nonzero((x_ext+y_ext)%2),'\nFinal error pattern:\n', np.nonzero(e_pat)

        print("Done in ", lcom_iters, " iters, using ",
              ladd_info, " bits, matched bits:", sum(x == y), "/", n)

    return ladd_info, lcom_iters, e_pat, ver_check, ln_iters, x, y


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
    f2_rslt = []
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
        f_cur = float(add_info)/(n)/h_b(qber)
        f2_rslt.append(f_cur)
        if ver_check:
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
    return np.mean(f_rslt), np.mean(com_iters_rslt), np.mean(n_iters_rslt), np.mean(f2_rslt), 0, 0, 0, \
        np.mean(corrected_rslt), np.mean(add_info_rslt), float(n_incor)/n_tries
