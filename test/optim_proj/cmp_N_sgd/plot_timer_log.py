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
    weight_sgd_good = listdir_sort('log/timer/sgd_good')
    weight_sgd_ill = listdir_sort('log/timer/sgd_ill')
    weight_sgd_worse = listdir_sort('log/timer/sgd_worse')
    weight_sgd_worst = listdir_sort('log/timer/sgd_worst')
    
    for weight_file_elem in weight_sgd_good:
        _weight_file_elem = 'log/timer/sgd_good/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_good/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_ill:
        _weight_file_elem = 'log/timer/sgd_ill/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_ill/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_worse:
        _weight_file_elem = 'log/timer/sgd_worse/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_worse/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_worst:
        _weight_file_elem = 'log/timer/sgd_worst/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_worst/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)

    print 'weights plotting done.'


def plot_loss():
    loss_sgd_good = np.loadtxt('log/timer/sgd_good.csv', delimiter=',')
    loss_sgd_ill = np.loadtxt('log/timer/sgd_ill.csv', delimiter=',')
    loss_sgd_worse = np.loadtxt('log/timer/sgd_worse.csv', delimiter=',')
    loss_sgd_worst = np.loadtxt('log/timer/sgd_worst.csv', delimiter=',')
    plt.plot(loss_sgd_good[:, 0], loss_sgd_good[:, 1], 'r', label='sgd_good')
    plt.plot(loss_sgd_ill[:, 0], loss_sgd_ill[:, 1], 'g', label='sgd_ill')
    plt.plot(loss_sgd_worse[:, 0], loss_sgd_worse[:, 1], 'b', label='sgd_worse')
    plt.plot(loss_sgd_worst[:, 0], loss_sgd_worst[:, 1], 'y', label='sgd_worst')
    plt.xlim(0, 8)
    plt.ylim(0, 5)
    plt.legend(loc='best')
    plt.ylabel('loss function')
    plt.xlabel('timer')
    plt.savefig('log/timer/graph.png')
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
    plot_loss()
    # plot_weight()


if __name__ == '__main__':
    main()
