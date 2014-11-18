#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# Nov. 17
# This script is used to check the gradient computing for SparseAE

import sys
sys.path.append('/Users/zhaojunbo/Projects/StarAE')
import numpy as np

from starae import SparseAE


def main():
    """Checking gradient computing."""
    T = SparseAE(16, 9, optimize_method='sgd', max_iter=200,
                 debug=0, verbose=True)
    X = np.random.rand(16, 128)
    T.gradient_check()
    T.train(X)


if __name__ == '__main__':
    main()
