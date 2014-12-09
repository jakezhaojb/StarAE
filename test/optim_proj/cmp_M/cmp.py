#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This script is used to check the gradient computing for SparseAE

import os
import sys
sys.path.append('../../../')

from starae import SparseAE
import visualize as vs


def main():
    """test for Sparse AE"""
    os.system('rm -rf log')
    T = []
    T1 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.99, alpha=.01, adastep=1,
                  logger='sgd99.csv')
    T2 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd95.csv')
    T3 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.90, alpha=.01, adastep=1,
                  logger='sgd90.csv')
    T4 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.80, alpha=.01, adastep=1,
                  logger='sgd80.csv')
    T5 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.60, alpha=.01, adastep=1,
                  logger='sgd60.csv')
    T6 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.40, alpha=.01, adastep=1,
                  logger='sgd40.csv')
    T7 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.20, alpha=.01, adastep=1,
                  logger='sgd20.csv')
    T8 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=False, momen_beta=0, alpha=.01, adastep=1,
                  logger='sgd0.csv')
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    T.append(T8)
    X = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=40960)

    for i in range(len(T)):
        T[i].train(X)


if __name__ == '__main__':
    main()
