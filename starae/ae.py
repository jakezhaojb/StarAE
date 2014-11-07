# -*- coding: utf-8 -*-
# Author: Junbo (Jake) Zhao

from __future__ import division
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import sparse

from .utils import *
from .base import NeuralNetBase


class AutoEncoder(NeuralNetBase):
    """docstring for AutoEncoder"""
    __metaclass__ = ABCMeta

    # TODO OPT paras completion
    def __init__(self, input_size, hidden_size, optimize_method='sgd',
                 dpark_enable='True', max_iter=1000, tol=1.e-3, alpha=0.1,
                 debug='False'):
        # safegruard code TODO

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.optimize_method = optimize_method
        self.dpark_enable = dpark_enable
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.debug = debug

    def init_param(self):
        """Initiate parameters of AutoEncoder"""
        # TODO docstring
        # weights and bias are organized in tuples
        self.w = ()
        self.b = ()
        w1 = np.random.uniform(-1, 1, (self.hidden_size, self.input_size))
        w2 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        b1 = np.zeros(self.hidden_size)
        b2 = np.zeros(self.input_size)
        self.w = (w1, w2)
        self.b = (b1, b2)
        # flag: whether parameters are initiated
        self.init_flag = True

    def feed_forward(self, X, acti_fun='sigmoid'):
        """Feed forward. """
        # TODO docstring
        assert self.init_flag
        assert isinstance(X, np.ndarray)
        assert self.input_size == X.shape[0]
        # TODO is it a good strategy to record everything here?
        # TODO is this a good strategy to record things?

    @abstractmethod
    def compute_cost(self):
        """AutoEncoder lost function"""

    @abstractmethod
    def compute_grad(self):
        """Back-propagation on AutoEncoder"""

    def optimize(self):
        """Optimize weight and bias"""
