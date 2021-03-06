# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This scripts plot everything in the log/timer

import os
import sys
sys.path.append('../../..')
import numpy as np
import matplotlib as mpi
mpi.use('Agg')
import matplotlib.pyplot as plt
from math import ceil
import visualize as vs


def plot_weight():
    weight_bfgs_good = listdir_sort('log/timer/bfgs_good')
    weight_bfgs_ill = listdir_sort('log/timer/bfgs_ill')
    weight_bfgs_worse = listdir_sort('log/timer/bfgs_worse')
    weight_bfgs_worst = listdir_sort('log/timer/bfgs_worst')
    
    for weight_file_elem in weight_bfgs_good:
        _weight_file_elem = 'log/timer/bfgs_good/' + weight_file_elem
        weight_bfgs = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/bfgs_good/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_bfgs, fname=file_elem)
    for weight_file_elem in weight_bfgs_ill:
        _weight_file_elem = 'log/timer/bfgs_ill/' + weight_file_elem
        weight_bfgs = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/bfgs_ill/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_bfgs, fname=file_elem)
    for weight_file_elem in weight_bfgs_worse:
        _weight_file_elem = 'log/timer/bfgs_worse/' + weight_file_elem
        weight_bfgs = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/bfgs_worse/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_bfgs, fname=file_elem)
    for weight_file_elem in weight_bfgs_worst:
        _weight_file_elem = 'log/timer/bfgs_worst/' + weight_file_elem
        weight_bfgs = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/bfgs_worst/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_bfgs, fname=file_elem)

    print 'weights plotting done.'


def plot_loss(x_max, y_max):
    loss_bfgs_good = np.loadtxt('log/timer/bfgs_good.csv', delimiter=',')
    loss_bfgs_ill = np.loadtxt('log/timer/bfgs_ill.csv', delimiter=',')
    loss_bfgs_worse = np.loadtxt('log/timer/bfgs_worse.csv', delimiter=',')
    loss_bfgs_worst = np.loadtxt('log/timer/bfgs_worst.csv', delimiter=',')
    plt.plot(loss_bfgs_good[:, 0], loss_bfgs_good[:, 1], 'r', label='bfgs_good')
    plt.plot(loss_bfgs_ill[:, 0], loss_bfgs_ill[:, 1], 'g', label='bfgs_ill')
    plt.plot(loss_bfgs_worse[:, 0], loss_bfgs_worse[:, 1], 'b', label='bfgs_worse')
    plt.plot(loss_bfgs_worst[:, 0], loss_bfgs_worst[:, 1], 'y', label='bfgs_worst')
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.legend(loc='best')
    plt.ylabel('loss function')
    plt.xlabel('timer')
    plt.savefig('log/timer/graph_x_'+str(x_max)+'_y_'+str(y_max)+'.png')
    plt.clf()


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
    os.system('find . -name "*.png" | grep timer | xargs rm')
    plot_loss(8, 5)
    plot_loss(80, 5)
    plot_loss(30, 100)
    plot_loss(100, 100)
    # plot_weight()


if __name__ == '__main__':
    main()
