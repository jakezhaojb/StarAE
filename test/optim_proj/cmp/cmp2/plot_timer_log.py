# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This scripts plot everything in the log/timer

import os
import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
import visualize as vs
from math import ceil


def plot_weight():
    weight_bfgs_file = listdir_sort('log/timer/bfgs')
    weight_cg_file = listdir_sort('log/timer/cg')
    weight_sgd_file = listdir_sort('log/timer/sgd')
    for weight_bfgs_file_elem in weight_bfgs_file:
        _weight_bfgs_file_elem = 'log/timer/bfgs/' + weight_bfgs_file_elem
        weight_bfgs = np.loadtxt(_weight_bfgs_file_elem, delimiter=',')
        file_elem = 'log/timer/bfgs/' + weight_bfgs_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_bfgs, fname=file_elem)
    for weight_cg_file_elem in weight_cg_file:
        _weight_cg_file_elem = 'log/timer/cg/' + weight_cg_file_elem
        weight_cg = np.loadtxt(_weight_cg_file_elem, delimiter=',')
        file_elem = 'log/timer/cg/' + weight_cg_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_cg, fname=file_elem)
    for weight_sgd_file_elem in weight_sgd_file:
        _weight_sgd_file_elem = 'log/timer/sgd/' + weight_sgd_file_elem
        weight_sgd = np.loadtxt(_weight_sgd_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd/' + weight_sgd_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    print 'weights plotting done.'


def plot_loss():
    loss_bfgs = np.loadtxt('log/timer/bfgs.csv', delimiter=',')
    loss_cg = np.loadtxt('log/timer/cg.csv', delimiter=',')
    loss_sgd = np.loadtxt('log/timer/sgd.csv', delimiter=',')
    plt.plot(loss_bfgs[:, 0], loss_bfgs[:, 1], 'r', label='bfgs')
    plt.plot(loss_cg[:, 0], loss_cg[:, 1], 'g', label='cg')
    plt.plot(loss_sgd[:600, 0], loss_sgd[:600, 1], 'b*', label='sgd')
    plt.xlim(0, 200)
    plt.legend(loc='best')
    plt.ylabel('loss function')
    plt.xlabel('time/s')
    plt.savefig('log/timer/graph.png')


def listdir_sort(path):
    fls = os.listdir(path)
    fls = map(lambda x: int(x), fls)
    fls.sort()
    fls = map(lambda x: str(x), fls)
    n_fls = len(fls)
    idx = range(1, n_fls, int(ceil(n_fls / 20.)))
    fls = [fls[i] for i in idx]
    return fls


def main():
    plot_loss()
    # plot_weight()


if __name__ == '__main__':
    main()
