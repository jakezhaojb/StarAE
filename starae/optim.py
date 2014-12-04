# -*- coding: utf-8 -*-
# Author: Junbo Zhao

from __future__ import division

import numpy as np
from scipy import optimize
from inspect import ismethod
from copy import copy


def bfgs(compute_cost, init_guess, X, compute_grad, args=(),
         maxiter=1000, tol=1.e-7):
    """BFGS revoke"""
    # TODO Notice: input data should be put in args
    assert ismethod(compute_cost)
    assert ismethod(compute_grad)
    assert isinstance(init_guess, np.ndarray)

    args = (X, ) + args
    # TODO bad naming
    opt_bfgs = optimize.fmin_l_bfgs_b(compute_cost, init_guess, compute_grad,
                                      args=args, maxiter=maxiter, pgtol=tol)
    return opt_bfgs[0]


def cg(compute_cost, init_guess, X, compute_grad, args=(),
       maxiter=1000, tol=1.e-7):
    """CG revoke"""
    # TODO Notice: input data should be put in args
    assert ismethod(compute_cost)
    assert ismethod(compute_grad)
    assert isinstance(init_guess, np.ndarray)

    args = (X, ) + args
    # TODO bad naming
    opt_cg = optimize.fmin_cg(compute_cost, init_guess, compute_grad,
                              args=args, maxiter=maxiter, gtol=tol)
    return opt_cg


def sgd(compute_cost, init_guess, X, compute_grad, args=(),
        alpha=0.1, maxiter=1000, tol=1.e-7, mini_batch=0,
        momentum=False, beta=0.95, adastep=False):
    """Stochastic Gradient Descent"""
    assert ismethod(compute_cost)
    assert ismethod(compute_grad)
    assert isinstance(init_guess, np.ndarray)

    theta = init_guess
    # outer-loop
    for oloop in range(maxiter):
        X_idx = np.random.permutation(X.shape[1]).tolist()
        if mini_batch:
            mini_batch_idx = split_list(X_idx, mini_batch)
        else:
            mini_batch_idx = [[X_idx_elem for X_idx_elem in X_idx]]
        # initialize some variables
        ada = np.zeros(shape=init_guess.shape)
        momen = np.zeros(shape=init_guess.shape)
        # inner-loop
        for iloop in mini_batch_idx:
            _args = (theta, X, ) + args
            iter_cost = compute_cost(*_args)
            _args = (theta, X[:, iloop], ) + args
            iter_grad = compute_grad(*_args)
            # adastep
            if adastep:
                ada += iter_grad ** 2
                step_size = alpha / np.sqrt(ada)
                # safeguard
                step_size[step_size > 1] = 1
            else:
                step_size = copy(alpha)
            # momentum
            if momentum:
                momen = beta * momen - iter_grad
            else:
                momen = -iter_grad
            # go downhill
            theta += momen * step_size
        # print 'iter: %d, expirical loss: %f' % (oloop, iter_cost)
        # check tolerance
        if np.linalg.norm(iter_cost) < tol:  # TODO some better criterion?
            break

    # TODO some tricks of SGD??
    return theta


def split_list(x, sec):
    """split a list into multiple sublists"""
    assert isinstance(x, list)
    assert isinstance(sec, int)

    spl = []
    for i in range(len(x) // sec):
        spl.append(x[i: i+sec])
    return spl
