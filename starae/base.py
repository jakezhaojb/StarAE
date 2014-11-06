# -*- coding: utf-8 -*-
# Author: Junbo (Jake) Zhao <j.zhao@nyu.edu>

"""
Generalized Neural Nets
"""

from abc import ABCMeta, abstractmethod


class NeuralNetBase(object):
    """Base class for Neural Nets"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_param(self):
        """Initiate Neural Nets"""

    @abstractmethod
    def feed_forward(self, X, acti_fun='sigmoid'):
        """Neural Net feed forward"""

    @abstractmethod
    def compute_cost(self):
        """Neural Net lost function"""

    @abstractmethod
    def compute_grad(self):
        """Back-propagation application"""

    @abstractmethod
    def optimize(self):
        """Optimization inteface"""
