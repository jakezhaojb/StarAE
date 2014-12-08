#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This script is used to check the gradient computing for SparseAE

import sys
sys.path.append('../../../..')

from starae import SparseAE
import visualize as vs


def main():
    """test for Sparse AE"""
    T = []
    T1 = SparseAE(64, 49, optimize_method='bfgs', max_iter=400,
                  debug=0, verbose=True, tol=1e-8, mini_batch=32,
                  logger='bfgs.csv')
    T2 = SparseAE(64, 49, optimize_method='cg', max_iter=400,
                  debug=0, verbose=True, tol=1e-8, mini_batch=32,
                  logger='cg.csv')
    T3 = SparseAE(64, 49, optimize_method='sgd', max_iter=100,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=True, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd.csv')
    T.append(T1)
    T.append(T2)
    T.append(T3)
    X = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=40960)

    name = ['bfgs.png', 'cg.png', 'sgd.png']
    for i in range(3):
        T[i].train(X)
        T[i].devec_theta()
        vs.disp_effect(T[i].w1, fname=name[i])


if __name__ == '__main__':
    main()
