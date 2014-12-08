# -*- coding: utf-8 -*-
# Author: Junbo (Jake) Zhao

from __future__ import division
from abc import ABCMeta, abstractmethod
import os
import sys
import numpy as np
from scipy import sparse
from time import time

from .utils import *
from .optim import *
from .base import NeuralNetBase


class AutoEncoder(NeuralNetBase):
    """docstring for AutoEncoder"""
    __metaclass__ = ABCMeta

    def __init__(self, input_size, hidden_size, acti_fun='sigmoid',
                 optimize_method='sgd', max_iter=1000, tol=1.e-3, alpha=0.1,
                 mini_batch=0, adastep=False, momentum=False,
                 momen_beta=0.95, dpark_enable=False, dpark_run='process',
                 dpark_threads='-p 4', debug=False, verbose=False, logger=''):
        # safegruard code TODO

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.acti_fun = acti_fun
        self.optimize_method = optimize_method
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.mini_batch = mini_batch
        self.adastep = adastep
        self.momentum = momentum
        self.momen_beta = momen_beta
        self.dpark_enable = dpark_enable
        self.dpark_run = dpark_run
        self.dpark_threads = dpark_threads
        self.debug = debug
        self.verbose = verbose and logger
        self.logger = logger.partition('.')[0]
        if self.verbose:
            if not os.path.isdir('log'):
                os.mkdir('log')
            # TODO not make sense, for example test code creates three method..
            '''
            if os.path.isdir('log'):
                while 1:
                    print 'Overwrite log?[Y/n]'
                    key = raw_input()
                    if key == 'Y':
                        print 'Overwrite.'
                        os.system('rf -rf log')
                        os.mkdir('log')
                        break
                    elif key == 'n':
                        print 'Not to overwrite'
                        self.verbose = False
                        break
                    else:
                        continue
            else:
                os.mkdir('log')
            '''

    def init_param(self):
        """Initiate parameters of AutoEncoder"""
        # weights and bias are organized in tuples
        # X. Glorot, Y. Bengio, 2010
        r = np.sqrt(6) / np.sqrt(self.hidden_size+self.input_size+1)
        w1 = np.random.rand(self.hidden_size, self.input_size) * 2 * r - r
        w2 = np.random.rand(self.input_size, self.hidden_size) * 2 * r - r
        # w1 = np.random.uniform(-1, 1, (self.hidden_size, self.input_size))
        # w2 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        b1 = np.zeros(self.hidden_size)
        b2 = np.zeros(self.input_size)
        # flag: whether parameters are initiated
        self.init_flag = True
        self.theta = vectorize(w1, w2, b1, b2)

    def feed_forward(self, X):
        """Feed forward. """
        # TODO docstring
        assert self.init_flag
        assert isinstance(X, np.ndarray)
        assert self.input_size == X.shape[0]
        # TODO is it a good strategy to record everything here?
        # TODO is this a good strategy to record things?
        # TODO efficiency
        '''
        s = ((self.hidden_size, self.input_size),
             (self.input_size, self.hidden_size),
             (self.hidden_size, 1), (self.input_size, 1))
        self.w1, self.w2, self.b1, self.b2 = de_vectorize(self.theta, 4, s)
        '''
        self.devec_theta()
        self.a1 = X
        self.z2 = np.dot(self.w1, X) + self.b1
        self.a2 = activate(self.z2, self.acti_fun)
        self.z3 = np.dot(self.w2, self.a2) + self.b2
        self.a3 = activate(self.z3, self.acti_fun)
        self.opt = self.a3

    @abstractmethod
    def compute_cost(self, theta, X):
        """AutoEncoder lost function"""
        return cost

    @abstractmethod
    def compute_grad(self, theta, X):
        """Back-propagation on AutoEncoder"""
        return theta_grad  # vector

    def train(self, X, *args):
        """Optimize weight and bias"""
        # TODO must serielize it? can be slow when training some large nets
        # TODO maybe use a TABLE
        if self.verbose:
            self.timer = time()
            self.time_stamp = 0
        if self.optimize_method == 'bfgs':
            self.theta = bfgs(self.compute_cost, self.theta, X,
                              self.compute_grad, args=args,
                              maxiter=self.max_iter, tol=self.tol)
        elif self.optimize_method == 'cg':
            self.theta = cg(self.compute_cost, self.theta, X,
                            self.compute_grad, args=args,
                            maxiter=self.max_iter, tol=self.tol)
        elif self.optimize_method == 'sgd':
            self.theta = sgd(self.compute_cost, self.theta, X,
                             self.compute_grad, args=args, alpha=0.1,
                             maxiter=self.max_iter, tol=self.tol,
                             mini_batch=self.mini_batch,
                             momentum=self.momentum,
                             beta=self.momen_beta,
                             adastep=self.adastep)
        else:
            print "The optimization method is not supported."
            sys.exit(1)

    def gradient_check(self, tol=1e-9, *args):
        """checking the gradient by numerical comparison"""
        if not self.debug:
            print 'Please use DEBUG mode when checking gradients.'
            return
        theta_ = np.random.rand(*self.theta.shape)
        X = np.random.rand(self.input_size, 300)
        # numerical gradient
        nume_grad = np.zeros(shape=theta_.shape)
        epsilon = np.eye(theta_.shape[0], theta_.shape[0])
        epsilon = epsilon * 1.e-4
        for i in range(theta_.shape[0]):
            nume_grad[i] = self.compute_cost(theta_+epsilon[i, :], X, *args) -\
                           self.compute_cost(theta_-epsilon[i, :], X, *args)
        nume_grad = nume_grad / (2 * 1.e-4)
        grad = self.compute_grad(theta_, X, *args)
        diff = np.linalg.norm(grad - nume_grad) /\
                np.linalg.norm(grad + nume_grad)
        if diff < tol:
            print 'Gradient checked correct'
        else:
            print 'Gradient checked FAILED, diff: %e' % (diff)

    def devec_theta(self):
        s = ((self.hidden_size, self.input_size),
             (self.input_size, self.hidden_size),
             (self.hidden_size, 1), (self.input_size, 1))
        self.w1, self.w2, self.b1, self.b2 = de_vectorize(self.theta, 4, s)

