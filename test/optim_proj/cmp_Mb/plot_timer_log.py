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
    weight_sgd_mb4096 = listdir_sort('log/timer/sgd_mb4096')
    weight_sgd_mb1024 = listdir_sort('log/timer/sgd_mb1024')
    weight_sgd_mb256 = listdir_sort('log/timer/sgd_mb256')
    weight_sgd_mb64 = listdir_sort('log/timer/sgd_mb64')
    weight_sgd_mb0 = listdir_sort('log/timer/sgd_mb0')
    
    for weight_file_elem in weight_sgd_mb4096:
        _weight_file_elem = 'log/timer/sgd_mb4096/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_mb4096/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb1024:
        _weight_file_elem = 'log/timer/sgd_mb1024/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_mb1024/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb256:
        _weight_file_elem = 'log/timer/sgd_mb256/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_mb256/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb64:
        _weight_file_elem = 'log/timer/sgd_mb64/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_mb64/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb0:
        _weight_file_elem = 'log/timer/sgd_mb0/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd_mb0/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)

    print 'weights plotting done.'


def plot_loss(x_max, y_max):
    loss_sgd_mb4096 = np.loadtxt('log/timer/sgd_mb4096.csv', delimiter=',')
    loss_sgd_mb1024 = np.loadtxt('log/timer/sgd_mb1024.csv', delimiter=',')
    loss_sgd_mb256 = np.loadtxt('log/timer/sgd_mb256.csv', delimiter=',')
    loss_sgd_mb64 = np.loadtxt('log/timer/sgd_mb64.csv', delimiter=',')
    loss_sgd_mb0 = np.loadtxt('log/timer/sgd_mb0.csv', delimiter=',')
    plt.plot(loss_sgd_mb4096[:, 0], loss_sgd_mb4096[:, 1], 'r', label='sgd_mb4096')
    plt.plot(loss_sgd_mb1024[:, 0], loss_sgd_mb1024[:, 1], 'g', label='sgd_mb1024')
    plt.plot(loss_sgd_mb256[:, 0], loss_sgd_mb256[:, 1], 'b', label='sgd_mb256')
    plt.plot(loss_sgd_mb64[:, 0], loss_sgd_mb64[:, 1], 'y', label='sgd_mb64')
    plt.plot(loss_sgd_mb0[:, 0], loss_sgd_mb0[:, 1], 'r*', label='sgd_mb0')
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
    plot_loss(100, 20)
    plot_loss(100, 100)
    plot_loss(20, 5)
    plot_loss(20, 10)
    plot_loss(60, 5)
    plot_loss(60, 10)
    # plot_weight()


if __name__ == '__main__':
    main()
