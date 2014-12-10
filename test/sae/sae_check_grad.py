#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This script is used to check the gradient computing for SparseAE

import sys
sys.path.append('../..')
import numpy as np

from starae import SparseAE


def main():
    """Checking gradient computing."""
    T = SparseAE(16, 9, optimize_method='bfgs', max_iter=200,
                 debug=1, verbose=True)  # TODO Other activation functions.
    X = np.random.rand(16, 128)
    T.gradient_check()
    T.train(X)


if __name__ == '__main__':
    main()
