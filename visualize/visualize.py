# -*- coding: utf-8 -*-
# Author: Junbo Zhao

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .utils import factor2
from .sample import rescale


def disp_effect(W, n_disp=100, fname=None):
    """Visualize the weights of trained NN."""
    # safeguard
    assert isinstance(W, np.ndarray)
    if n_disp > 256:
        print 'Too large n_disp.'
        return

    n_disp_, n_pixel = W.shape
    x = np.zeros((n_disp_, n_pixel))
    for i in range(n_disp_):
        '''
        Find the pixels maximally activate hidden units
        '''
        max_acti = W[i, :] / np.linalg.norm(W[i, :])
        x[i, :] = max_acti.squeeze()
    n_disp = n_disp if n_disp < n_disp_ else n_disp_
    idx = np.random.permutation(x.shape[0])[:n_disp]
    x = x[idx, :]
    # displaying
    disp_r, disp_c = factor2(n_disp)
    unit_disp_r, unit_disp_c = factor2(n_pixel)
    visual_grid = np.zeros((disp_r*(unit_disp_r+1)-1,
                            disp_c*(unit_disp_c+1)-1))
    for i in range(disp_r):
        for j in range(disp_c):
            unit_x = x[i*disp_c+j, :].reshape(unit_disp_r, unit_disp_c)
            # For pertain the boundary being black
            unit_x = rescale(unit_x, axis=-1)
            row_pos = i * (unit_disp_r + 1)
            col_pos = j * (unit_disp_c + 1)
            visual_grid[row_pos: row_pos+unit_disp_r,
                        col_pos: col_pos+unit_disp_c] = unit_x
    plt.imshow(visual_grid, cmap=cm.gray)
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
