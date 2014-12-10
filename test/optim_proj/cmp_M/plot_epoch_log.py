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
    weight_sgd99 = listdir_sort('log/epoch/sgd99')
    weight_sgd95 = listdir_sort('log/epoch/sgd95')
    weight_sgd90 = listdir_sort('log/epoch/sgd90')
    weight_sgd80 = listdir_sort('log/epoch/sgd80')
    weight_sgd60 = listdir_sort('log/epoch/sgd60')
    weight_sgd40 = listdir_sort('log/epoch/sgd40')
    weight_sgd20 = listdir_sort('log/epoch/sgd20')
    weight_sgd0 = listdir_sort('log/epoch/sgd0')
    
    for weight_file_elem in weight_sgd99:
        _weight_file_elem = 'log/epoch/sgd99/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd99/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd95:
        _weight_file_elem = 'log/epoch/sgd95/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd95/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd90:
        _weight_file_elem = 'log/epoch/sgd90/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd90/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd80:
        _weight_file_elem = 'log/epoch/sgd80/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd80/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd60:
        _weight_file_elem = 'log/epoch/sgd60/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd60/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd40:
        _weight_file_elem = 'log/epoch/sgd40/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd40/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd20:
        _weight_file_elem = 'log/epoch/sgd20/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd20/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd0:
        _weight_file_elem = 'log/epoch/sgd0/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd0/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)

    print 'weights plotting done.'


def plot_loss():
    loss_sgd99 = np.loadtxt('log/epoch/sgd99.csv', delimiter=',')
    loss_sgd95 = np.loadtxt('log/epoch/sgd95.csv', delimiter=',')
    loss_sgd90 = np.loadtxt('log/epoch/sgd90.csv', delimiter=',')
    loss_sgd80 = np.loadtxt('log/epoch/sgd80.csv', delimiter=',')
    loss_sgd60 = np.loadtxt('log/epoch/sgd60.csv', delimiter=',')
    loss_sgd40 = np.loadtxt('log/epoch/sgd40.csv', delimiter=',')
    loss_sgd20 = np.loadtxt('log/epoch/sgd20.csv', delimiter=',')
    loss_sgd0 = np.loadtxt('log/epoch/sgd0.csv', delimiter=',')
    plt.plot(range(len(loss_sgd99)), loss_sgd99, 'r', label='sgd99')
    plt.plot(range(len(loss_sgd95)), loss_sgd95, 'g', label='sgd95')
    plt.plot(range(len(loss_sgd90)), loss_sgd90, 'b', label='sgd90')
    plt.plot(range(len(loss_sgd80)), loss_sgd80, 'y', label='sgd80')
    plt.plot(range(len(loss_sgd60)), loss_sgd60, 'r*', label='sgd60')
    plt.plot(range(len(loss_sgd40)), loss_sgd40, 'g*', label='sgd40')
    plt.plot(range(len(loss_sgd20)), loss_sgd20, 'b*', label='sgd20')
    plt.plot(range(len(loss_sgd0)), loss_sgd0, 'y*', label='sgd0')
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
