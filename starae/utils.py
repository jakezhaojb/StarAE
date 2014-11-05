# -*- coding: utf-8 -*-
# Author: Junbo Zhao

from __future__ import division
import numpy as np


def vectorize(*args):  # TODO efficiency of hstack
    """Vectorize parameters, to speed up.

    Parameters
    ----------
    x1, x2, x3... : {np.ndarray}

    Returns
    -------
    X : np.array, shape = (total_length, )
    """

    vec = np.array([])
    for args_elem in args:
        assert isinstance(args_elem, np.ndarray)
        if not vec.size:
            vec = args_elem.reshape(args_elem.size)
        else:
            vec = np.hstack((vec, args_elem.reshape(args_elem.size)))

    return vec


def de_vectorize(vec, n, sizes):
    """De-vectorize weights and bias
    Parameters
    ----------
    vec : np.ndarray, shape = (size, )
    Vectorized weights and bias in 1-D

    n: int
    Number of target de-vectorized matrices

    sizes: tuple, size = n
    Shape of the target matrices

    Returns
    -------
    T : tuple, containing all matrices, ordered as the sizes indicates
    """

    # safeguard code
    assert isinstance(vec, np.ndarray)
    assert isinstance(sizes, tuple)
    assert len(sizes) == n
    num = 0
    for size_elem in sizes:
        assert isinstance(size_elem, tuple)
        assert len(size_elem) == 2
        num += size_elem[0] * size_elem[1]
    assert num == vec.size

    T = ()
    pos = 0
    for size_elem in sizes:
        vec_elem = vec[pos: pos+size_elem[0]*size_elem[1]]
        mat = vec_elem.reshape(size_elem)
        T += (mat, )
        pos += size_elem[0]*size_elem[1]

    return T


def sigmoid(x):
    assert isinstance(x, np.ndarray)
    y = 1.0 / (1 + np.exp(-x))
    return y


def rec_linear(x):
    assert isinstance(x, np.ndarray)
    y = x.copy()
    y[y < 0] = 0
    return y


def tanh(x):
    assert isinstance(x, np.ndarray)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y


def sigmoid_grad(x):
    assert isinstance(x, np.ndarray)
    y = sigmoid(x)
    g = (1 - y) * y
    return g


def rec_linear_grad(x):
    assert isinstance(x, np.ndarray)
    g = np.zeros(shape=x.shape)
    g[x > 0] = 1
    return g


def tanh_grad(x):
    assert isinstance(x, np.ndarray)
    g = 1 - np.power(tanh(x))
    return g
