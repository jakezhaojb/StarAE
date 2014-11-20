# -*- coding : utf-8 -*-
# Author: Junbo Zhao

from __future__ import division

import os
import numpy as np
import scipy.io as sio
from .utils import factor2


def load_sample(sample_name, patch_size=0, n_patches=0):
    """Load samples and patch them"""
    # safe-guard
    assert isinstance(patch_size, int)
    assert isinstance(n_patches, int)
    assert isinstance(sample_name, str)
    load_name = sample_name.split('.')[0]
    sample_name = os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'data')), sample_name)
    assert os.path.isfile(sample_name)
    if not sample_name.endswith('mat'):
        print 'Currently not supported.'
        return  # TODO adapt to other files
    # Load
    imgs = sio.loadmat(sample_name)[load_name]
    assert imgs.shape[0] == imgs.shape[1]
    # Process patch_size and n_patches
    if (not patch_size) or (not n_patches):
        print 'No patching here.'
        patch_size = imgs.shape[0]  # shape[1]
        n_patches = imgs.shape[2]
    # Patching
    patch_all = np.zeros((patch_size**2, n_patches))
    n_patches_img = n_patches // imgs.shape[2]
    n_j, n_k = factor2(n_patches_img)
    x = np.random.permutation(imgs.shape[0]//patch_size)[:n_j] * patch_size
    y = np.random.permutation(imgs.shape[0]//patch_size)[:n_k] * patch_size
    for i in range(imgs.shape[2]):
        img = imgs[:, :, i]
        for j in range(n_j):
            for k in range(n_k):
                patch = img[x[j]: x[j]+patch_size, y[k]: y[k]+patch_size]
                patch = patch.reshape(patch_size**2, 1, order='F')
                patch_all[:, n_patches_img*i + n_k*j + k] = patch.squeeze()
    patch_all = standardize(patch_all)
    return patch_all


def standardize(data, axis=0):
    """Standardize data to [0.1, 0.9]"""
    if axis == 0:
        data_ = data - np.mean(data, axis)
        data_std = 3 * np.std(data_, axis).reshape(1, data.shape[1])
    elif axis == 1:
        data_ = data - np.mean(data, axis)
        data_std = 3 * np.std(data_, axis).reshape(data.shape[1], 1)
    else:
        print 'Argument not accepted.'
        return
    data_ = np.maximum(np.minimum(data_, data_std), -data_std) / data_std
    data_ = (data_ + 1) * 0.4 + 0.1
    return data_


def rescale(data, axis=0):
    """Rescale data"""
    if axis == 0:
        r_max = np.max(data, axis).reshape(1, data.shape[1])
        r_min = np.min(data, axis).reshape(1, data.shape[1])
        data = (data - r_min) / (r_max - r_min)
    elif axis == 1:
        c_max = np.max(data, axis).reshape(data.shape[1], 1)
        c_min = np.min(data, axis).reshape(data.shape[1], 1)
        data = (data - c_min) / (c_max - c_min)
    elif axis == -1:  # element-wise
        e_max = np.max(data)
        e_min = np.min(data)
        data = (data - e_min) / (e_max - e_min)
    else:
        print 'Argument not accepted.'
        return
    return data
