# -*- coding: utf-8 -*-
# Author: Junbo Zhao

from __future__ import division
import os
import sys
import numpy as np

__all__ = ['vectorize', 'de_vectorize', 'activate', 'activate_grad',
           'write_to_file', 'load_from_file']


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

# Activation functions and their gradients.
################################################################


def sigmoid(x):
    """sigmoid function

    Parameters
    ----------
    x : np.array

    Returns
    -------
    y : np.array, shape = x.shape
    """
    assert isinstance(x, np.ndarray)
    y = 1.0 / (1 + np.exp(-x))
    return y


def rec_linear(x):
    """rectified function

    Parameters
    ----------
    x : np.array

    Returns
    -------
    y : np.array, shape = x.shape
    """
    assert isinstance(x, np.ndarray)
    y = x.copy()
    y[y < 0] = 0
    return y


def tanh(x):
    """Hyperbolic tangent function

    Parameters
    ----------
    x : np.array

    Returns
    -------
    y : np.array, shape = x.shape
    """
    assert isinstance(x, np.ndarray)
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y


def activate(X, function='sigmoid'):
    """Wrapper for activation function

    Parameters
    ----------
    X : np.array
    Input

    function : string, default = 'sigmoid'
    The name of activation function: 'sigmoid', 'rec_linear' or 'tanh'.

    Returns
    -------
    Y : np.array
    Values being activated.
    """
    assert isinstance(X, np.ndarray)
    if function == 'sigmoid':
        return sigmoid(X)
    elif function == 'rec_linear':
        return rec_linear(X)
    elif function == 'tanh':
        return tanh(X)
    else:
        print "The activation function is not supported."
        sys.exit(1)


def sigmoid_grad(x):
    """gradient of sigmoid function

    Parameters
    ----------
    x : np.array

    Returns
    -------
    g : np.array, shape = x.shape
    """
    assert isinstance(x, np.ndarray)
    y = sigmoid(x)
    g = (1 - y) * y
    return g


def rec_linear_grad(x):
    """gradient of rectified linear function

    Parameters
    ----------
    x : np.array

    Returns
    -------
    g : np.array, shape = x.shape
    """
    assert isinstance(x, np.ndarray)
    g = np.zeros(shape=x.shape)
    g[x > 0] = 1
    return g


def tanh_grad(x):
    """gradient of tanh function

    Parameters
    ----------
    x : np.array

    Returns
    -------
    g : np.array, shape = x.shape
    """
    assert isinstance(x, np.ndarray)
    g = 1 - np.power(tanh(x))
    return g


def activate_grad(X, function='sigmoid'):
    """Wrapper for gradient of activation function

    Parameters
    ----------
    X : np.array
    Input

    function : string, default = 'sigmoid'
    The name of activation function: 'sigmoid', 'rec_linear' or 'tanh'.

    Returns
    -------
    Y : np.array
    Gradient with respect to weights.
    """
    assert isinstance(X, np.ndarray)
    if function == 'sigmoid':
        return sigmoid_grad(X)
    elif function == 'rec_linear':
        return rec_linear_grad(X)
    elif function == 'tanh':
        return tanh_grad(X)
    else:
        print "The activation function is not supported."
        sys.exit(1)


def write_to_file(x, file_name, suffix='csv', delim=','):
    """Write array or matrix to a file

    Parameter
    ---------
    x : np.array or np.matrix
    matrices or arrays to save to file

    file_name : string
    The name of the file given

    suffix : string, default = 'csv'
    The extension of file_name

    delim : string, default = ','
    The string used to separate values

    Returns
    -------
    Null
    """
    assert isinstance(x, np.ndarray)
    file_name = file_name + '.' + suffix
    if os.path.isfile(file_name):
        while 1:
            print 'Are you sure to overwrite', file_name, '[Y/n]?'
            key = raw_input()
            if key == 'Y':
                break
            elif key == 'n':
                print 'Overwrite denied.'
                return
            else:
                continue
    np.savetxt(file_name, x, delimiter=delim)


def load_from_file(file_name, delim=','):
    """Load data from a file

    Parameter
    ---------
    file_name: string
    File to be loaded

    delim : string, default = ','
    The string used to separate values

    Returns
    -------
    x : np.array
    """
    assert os.path.isfile(file_name)
    x = np.loadtxt(file_name, delimiter=delim)
    return x
