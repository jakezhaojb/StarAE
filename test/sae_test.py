#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This script is used to check the gradient computing for SparseAE

import sys
sys.path.append('/Users/zhaojunbo/Projects/StarAE')

from starae import SparseAE
import visualize as vs


def main():
    """test for Sparse AE"""
    '''
    T = SparseAE(64, 25, optimize_method='bfgs', max_iter=200,
                 debug=0, verbose=True, tol=1e-8, mini_batch=32)
    T = SparseAE(64, 25, optimize_method='sgd', max_iter=40,
                 debug=0, verbose=True, tol=1e-8, mini_batch=1024)
    '''
    T = SparseAE(64, 25, optimize_method='cg', max_iter=100,
                 debug=0, verbose=True, tol=1e-8, mini_batch=32)
    X = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=10000)
    T.train(X)
    T.devec_theta()
    vs.disp_effect(T.w1)


if __name__ == '__main__':
    main()
