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
                  momentum=False, momen_beta=.95, alpha=.01, adastep=0,
                  logger='sgd_ada.csv')
    T2 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=False, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd_Nada.csv')
    T3 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.95, alpha=.01, adastep=0,
                  logger='sgd_ada95.csv')
    T4 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd_Nada95.csv')
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    X = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=40960)

    for i in range(len(T)):
        T[i].train(X)


if __name__ == '__main__':
    main()
