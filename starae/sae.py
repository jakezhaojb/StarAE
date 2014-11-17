# -*- coding: utf-8 -*-
# Author: Junbo (Jake) Zhao

from __future__ import division

import numpy as np

from .utils import *
from .ae import AutoEncoder


class SparseAE(AutoEncoder):
    """docstring for SparseAE"""
    def __init__(self, input_size, hidden_size, activate_fun='sigmoid',
                 optimize_method='sgd', max_iter=1000, tol=1.e-3, alpha=0.1,
                 mini_batch=0, adastep=True, momentum=True, momen_beta=0.95,
                 dpark_enable='False', dpark_run='process',
                 dpark_threads='-p 4', debug='False', lamb=0.0001,
                 rho=0.01, sparse_beta=3):
        super(SparseAE, self).__init__(input_size, hidden_size, activate_fun,
                                       optimize_method, max_iter, tol, alpha,
                                       mini_batch, adastep, momentum,
                                       momen_beta, dpark_enable, dpark_run,
                                       dpark_threads, debug)
        self.lamb = lamb
        self.rho = rho
        self.sparse_beta = sparse_beta

    def compute_cost(self, X, theta=self.theta):
        """SparseAE lost function"""
        # safeguard
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == self.input_size
        assert self.theta.shape == theta.shape
        self.theta = theta  # Update
        # Start go forward
        cost = 0
        if self.dpark_enable:
            # TODO
            print 'Dpark enabled.'
        else:
            # Vectorized solution
            self.feed_forward(X)
            rho_ = np.sum(self.a2, axis=1).reshape(self.a2.shape[0], 1)
            cost += np.linalg.norm(self.a3 - self.ipt)
            cost, rho_ = cost / X.shape[1], rho_ / X.shape[1]
            # KL penalty
            KL_penalty = self.rho * np.log(self.rho / rho_) -\
                (1 - self.rho) * np.log((1 - self.rho) / (1 - rho_))
            cost += self.sparse_beta * np.sum(KL_penalty)
            # weight decay
            cost += self.lamb / 2 * (np.linalg.norm(self.w1) ** 2 +
                                     np.linalg.norm(self.w2) ** 2)
        return cost

    def compute_grad(self, X, theta=self.theta):
        """Back-propagation on SparseAE"""
        # safeguard
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == self.input_size
        assert self.theta.shape == theta.shape
        self.theta = theta  # Update
        # Start iterate
        self.feed_forward(X)
        rho_ = np.sum(self.a2, axis=1).reshape(self.a2.shape[0], 1)
        sigma3 = -(self.a1 - self.a3) * (self.a3 * (1 - self.a3))
        sparse_sigma = -(self.rho / rho_) + (1 - self.rho) / (1 - rho_)
        sigma2 = (np.dot(self.w2.T, sigma3) + self.sparse_beta * sparse_sigma)\
            * (self.a2 * (1 - self.a2))
        # Desired gradients
        w2_grad = np.sum(np.dot(sigma3, self.a2.T), axis=1).\
                         reshape(self.hidden_size, 1)
        b2_grad = np.sum(sigma3, axis=1).reshape(self.input_size, 1)
        w1_grad = np.sum(np.dot(sigma2, self.a1.T), axis=1).\
                         reshape(self.input_size, 1)
        b1_grad = np.sum(sigma2, axis=1).reshape(self.hidden_size, 1)
        # average and weight decay
        w2_grad = w2_grad / X.shape[1] + self.lamb * self.w2
        b2_grad = b2_grad / X.shape[1]
        w1_grad = w1_grad / X.shape[1] + self.lamb * self.w1
        b1_grad = b1_grad / X.shape[1]
        # vectorize
        theta_grad = vectorize(w1_grad, w2_grad, b1_grad, b2_grad)
        return theta_grad  # vector
