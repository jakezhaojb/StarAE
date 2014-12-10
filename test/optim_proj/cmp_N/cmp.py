#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This script is used to check the gradient computing for SparseAE

import os
import sys
sys.path.append('../../../')
import numpy as np

from starae import SparseAE
import visualize as vs


def main():
    """test for Sparse AE"""
    os.system('rm -rf log')
    T = []
    T1 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=0, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd_good.csv')
    T2 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=0, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd_ill.csv')
    T3 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=0, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd_worse.csv')
    T4 = SparseAE(64, 49, optimize_method='sgd', max_iter=10,
                  debug=0, verbose=True, tol=1e-8, mini_batch=64,
                  momentum=0, momen_beta=.95, alpha=.01, adastep=1,
                  logger='sgd_worst.csv')
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    
    X = []
    X1 = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=20480)
    X2 = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=20480)
    X3 = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=20480)
    X4 = vs.load_sample('IMAGES.mat', patch_size=8, n_patches=20480)
    idx_jitter = np.random.permutation(64)
    idx_jitter_pos = idx_jitter[: 16]
    idx_jitter_neg = idx_jitter[-16: ]
    for i, j in zip(idx_jitter_pos[:3], idx_jitter_neg[:3]):
        X2[i, :] = np.squeeze(np.random.normal(0.89, 0.01, [20480, ]))
        X2[j, :] = np.squeeze(np.random.normal(0.11, 0.01, [20480, ]))
    for i, j in zip(idx_jitter_pos[:8], idx_jitter_neg[:8]):
        X3[i, :] = np.squeeze(np.random.normal(0.899, 0.0001, [20480, ]))
        X3[j, :] = np.squeeze(np.random.normal(0.101, 0.0001, [20480, ]))
    for i, j in zip(idx_jitter_pos, idx_jitter_neg):
        X4[i, :] = np.squeeze(np.random.normal(0.89999, 0.000001, [20480, ]))
        X4[j, :] = np.squeeze(np.random.normal(0.10001, 0.000001, [20480, ]))
    X.extend([X1, X2, X3, X4])
    print "cond(X1): ", np.linalg.cond(X1)
    print "cond(X2): ", np.linalg.cond(X2)
    print "cond(X3): ", np.linalg.cond(X3)
    print "cond(X4): ", np.linalg.cond(X4)

    for i in range(len(T)):
        try:
            T[i].train(X[i])
        except:
            print 'Training shut down\n'
            pass


if __name__ == '__main__':
    main()
