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
    weight_sgd99 = listdir_sort('/log/timer/sgd99')
    weight_sgd95 = listdir_sort('/log/timer/sgd95')
    weight_sgd90 = listdir_sort('/log/timer/sgd90')
    weight_sgd80 = listdir_sort('/log/timer/sgd80')
    weight_sgd60 = listdir_sort('/log/timer/sgd60')
    weight_sgd40 = listdir_sort('/log/timer/sgd40')
    weight_sgd20 = listdir_sort('/log/timer/sgd20')
    weight_sgd0 = listdir_sort('/log/timer/sgd0')
    
    for weight_file_elem in weight_sgd99:
        _weight_file_elem = 'log/timer/sgd99/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd99/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd95:
        _weight_file_elem = 'log/timer/sgd95/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd95/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd90:
        _weight_file_elem = 'log/timer/sgd90/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd90/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd80:
        _weight_file_elem = 'log/timer/sgd80/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd80/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd60:
        _weight_file_elem = 'log/timer/sgd60/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd60/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd40:
        _weight_file_elem = 'log/timer/sgd40/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd40/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd20:
        _weight_file_elem = 'log/timer/sgd20/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd20/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd0:
        _weight_file_elem = 'log/timer/sgd0/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/timer/sgd0/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)

    print 'weights plotting done.'


def plot_loss():
    loss_sgd99 = np.loadtxt('/log/timer/sgd99.csv', delimiter=',')
    loss_sgd95 = np.loadtxt('/log/timer/sgd95.csv', delimiter=',')
    loss_sgd90 = np.loadtxt('/log/timer/sgd90.csv', delimiter=',')
    loss_sgd80 = np.loadtxt('/log/timer/sgd80.csv', delimiter=',')
    loss_sgd60 = np.loadtxt('/log/timer/sgd60.csv', delimiter=',')
    loss_sgd40 = np.loadtxt('/log/timer/sgd40.csv', delimiter=',')
    loss_sgd20 = np.loadtxt('/log/timer/sgd20.csv', delimiter=',')
    loss_sgd0 = np.loadtxt('/log/timer/sgd0.csv', delimiter=',')
    plt.plot(loss_sgd99[:, 0], loss_sgd99[:, 1], 'r', label='sgd99')
    plt.plot(loss_sgd95[:, 0], loss_sgd95[:, 1], 'g', label='sgd95')
    plt.plot(loss_sgd90[:, 0], loss_sgd90[:, 1], 'b', label='sgd90')
    plt.plot(loss_sgd80[:, 0], loss_sgd80[:, 1], 'y', label='sgd80')
    plt.plot(loss_sgd60[:, 0], loss_sgd60[:, 1], 'r*', label='sgd60')
    plt.plot(loss_sgd40[:, 0], loss_sgd40[:, 1], 'g*', label='sgd40')
    plt.plot(loss_sgd20[:, 0], loss_sgd20[:, 1], 'b*', label='sgd20')
    plt.plot(loss_sgd0[:, 0], loss_sgd0[:, 1], 'y*', label='sgd0')
    plt.xlim(0, 50)
    plt.legend(loc='best')
    plt.ylabel('loss function')
    plt.xlabel('timer')
    plt.savefig('/log/timer/graph.png')
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
    plot_loss(id_elem)
    # plot_weight(id_elem)


if __name__ == '__main__':
    main()
