# -*- coding: utf-8 -*-
# Author: Junbo (Jake) Zhao <j.zhao@nyu.edu>

"""
Generalized Neural Nets
"""

from __future__ import division
from abc import abstractmethod
import numpy as np
from scipy import sparse
from dpark import DparkContext

from .optim import sgd, bfgs
from .utils import *


class NeuralNetBase(object):
    """Base class for Neural Nets"""

    @abstractmethod
    def init_param(self, layer_num, layer_size):
        """Initiate Neural Nets"""


    def feed_forward(self, acti_fun='sigmoid'):
        """Neural Net feed forward"""

    
    def compute_cost(self):
        """Neural Net lost function"""

    
    def compute_grad(self):
        """Back-propagation application"""


    def optimize(self):
        """Optimization inteface"""
