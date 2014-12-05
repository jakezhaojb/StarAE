#! /usr/bin/env python
# -*- coding: utf-8
# Author: Junbo Zhao
# Introduction: SparseAE stacked tunning.

import os
import sys
sys.path.append('../..')

from starae import SparseAE
import visualize as vs
import numpy as np
from itertools import product

from starae.utils import activate


def main():
    """Tuning for SparseAE"""
    # First layer
    T = SparseAE(64, 25, optimize_method='cg', max_iter=400,
                 debug=0, verbose=True, tol=1e-8, mini_batch=32)
    X = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=10000)
    T.train(X)
    T.devec_theta()
    # vs.disp_effect(T.w1, '1.jpg')

    # Second layer
    rho = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    beta = [3e-3, 3e-2, 9e-2, 3e-1, 9e-1, 3, 9, 30, 90, 300]
    lamb = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    param = product(rho, beta, lamb)
    param_str = ['rho', 'sparse_beta', 'lamb']
    param = map(lambda x: dict(zip(param_str, x)), param)
    X = activate(np.dot(T.w1, X) + T.b1)
    if not os.path.isdir('./imgs'):
        os.system('mkdir imgs')
    for idx, param_elem in enumerate(param):
        import warnings
        warnings.filterwarnings('error')
        try:
            S = SparseAE(25, 9, optimize_method='cg', max_iter=400,
                         debug=0, verbose=True, tol=1e-8, mini_batch=32,
                         **param_elem)
            S.train(X)
            S.devec_theta()
            fname = 'imgs/' + str(idx) + '.jpg'
            vs.disp_effect(S.w1, fname=fname)
        except:
            fname = 'imgs/' + 'log'
            fid = open(fname, 'w')
            fid.write('Exception: ' + str(idx) + '\n')
            fid.close()


if __name__ == '__main__':
    main()
