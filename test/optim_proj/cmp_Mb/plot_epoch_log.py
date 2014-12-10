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
    weight_sgd_mb4096 = listdir_sort('log/epoch/sgd_mb4096')
    weight_sgd_mb1024 = listdir_sort('log/epoch/sgd_mb1024')
    weight_sgd_mb256 = listdir_sort('log/epoch/sgd_mb256')
    weight_sgd_mb64 = listdir_sort('log/epoch/sgd_mb64')
    weight_sgd_mb0 = listdir_sort('log/epoch/sgd_mb0')
    
    for weight_file_elem in weight_sgd_mb4096:
        _weight_file_elem = 'log/epoch/sgd_mb4096/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_mb4096/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb1024:
        _weight_file_elem = 'log/epoch/sgd_mb1024/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_mb1024/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb256:
        _weight_file_elem = 'log/epoch/sgd_mb256/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_mb256/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb64:
        _weight_file_elem = 'log/epoch/sgd_mb64/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_mb64/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_mb0:
        _weight_file_elem = 'log/epoch/sgd_mb0/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_mb0/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)

    print 'weights plotting done.'


def plot_loss():
    loss_sgd_mb4096 = np.loadtxt('log/epoch/sgd_mb4096.csv', delimiter=',')
    loss_sgd_mb1024 = np.loadtxt('log/epoch/sgd_mb1024.csv', delimiter=',')
    loss_sgd_mb256 = np.loadtxt('log/epoch/sgd_mb256.csv', delimiter=',')
    loss_sgd_mb64 = np.loadtxt('log/epoch/sgd_mb64.csv', delimiter=',')
    loss_sgd_mb0 = np.loadtxt('log/epoch/sgd_mb0.csv', delimiter=',')
    plt.plot(range(len(loss_sgd_mb4096)), loss_sgd_mb4096, 'r', label='sgd_mb4096')
    plt.plot(range(len(loss_sgd_mb1024)), loss_sgd_mb1024, 'g', label='sgd_mb1024')
    plt.plot(range(len(loss_sgd_mb256)), loss_sgd_mb256, 'b', label='sgd_mb256')
    plt.plot(range(len(loss_sgd_mb64)), loss_sgd_mb64, 'y', label='sgd_mb64')
    plt.plot(range(len(loss_sgd_mb0)), loss_sgd_mb0, 'r*', label='sgd_mb0')
    plt.xlim(0, 10)
    plt.legend(loc='best')
    plt.ylabel('loss function')
    plt.xlabel('epoch')
    plt.savefig('log/epoch/graph.png')
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
    os.system('find . -name "*.png" | grep epoch | xargs rm')
    plot_loss()
    # plot_weight()


if __name__ == '__main__':
    main()
