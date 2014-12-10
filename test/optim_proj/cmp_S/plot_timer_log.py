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


def plot_weight(num):
    weight_bfgs_file = listdir_sort('cmp'+str(num)+'/log/timer/bfgs')
    weight_cg_file = listdir_sort('cmp'+str(num)+'/log/timer/cg')
    weight_sgd_file = listdir_sort('cmp'+str(num)+'/log/timer/sgd')
    for weight_bfgs_file_elem in weight_bfgs_file:
        _weight_bfgs_file_elem = 'cmp'+str(num)+'/log/timer/bfgs/' + weight_bfgs_file_elem
        weight_bfgs = np.loadtxt(_weight_bfgs_file_elem, delimiter=',')
        file_elem = 'cmp'+str(num)+'/log/timer/bfgs/' + weight_bfgs_file_elem.zfill(5) + '.png'
        vs.disp_effect(weight_bfgs, fname=file_elem)
    for weight_cg_file_elem in weight_cg_file:
        _weight_cg_file_elem = 'cmp'+str(num)+'/log/timer/cg/' + weight_cg_file_elem
        weight_cg = np.loadtxt(_weight_cg_file_elem, delimiter=',')
        file_elem = 'cmp'+str(num)+'/log/timer/cg/' + weight_cg_file_elem.zfill(5) + '.png'
        vs.disp_effect(weight_cg, fname=file_elem)
    for weight_sgd_file_elem in weight_sgd_file:
        _weight_sgd_file_elem = 'cmp'+str(num)+'/log/timer/sgd/' + weight_sgd_file_elem
        weight_sgd = np.loadtxt(_weight_sgd_file_elem, delimiter=',')
        file_elem = 'cmp'+str(num)+'/log/timer/sgd/' + weight_sgd_file_elem.zfill(5) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    print 'weights plotting done.'


def plot_loss(num, y_max):
    loss_bfgs = np.loadtxt('cmp'+str(num)+'/log/timer/bfgs.csv', delimiter=',')
    loss_cg = np.loadtxt('cmp'+str(num)+'/log/timer/cg.csv', delimiter=',')
    loss_sgd = np.loadtxt('cmp'+str(num)+'/log/timer/sgd.csv', delimiter=',')
    plt.plot(loss_bfgs[:, 0], loss_bfgs[:, 1], 'r', label='bfgs')
    plt.plot(loss_cg[:, 0], loss_cg[:, 1], 'g', label='cg')
    plt.plot(loss_sgd[:, 0], loss_sgd[:, 1], 'b*', label='sgd')
    plt.xlim(0, 100)
    plt.ylim(0, y_max)
    plt.legend(loc='best')
    plt.ylabel('loss function')
    plt.xlabel('timer')
    plt.savefig('cmp'+str(num)+'/log/timer/graph'+str(y_max)+'.png')
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
    id = range(1, 7)
    for id_elem in id:
        plot_loss(id_elem, 20)
        plot_loss(id_elem, 100)
        # plot_weight(id_elem)


if __name__ == '__main__':
    main()
