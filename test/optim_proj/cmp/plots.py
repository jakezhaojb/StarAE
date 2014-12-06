# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This scripts plot everything in the log

import os
import sys
sys.path.append('../../../')
import numpy as np
import matplotlib.pyplot as plt
import visualize as vs


def plot_weight():
    weight_bfgs_file = listdir_sort('log/bfgs')
    weight_cg_file = listdir_sort('log/cg')
    weight_sgd_file = listdir_sort('log/sgd')
    for weight_bfgs_file_elem in weight_bfgs_file:
        _weight_bfgs_file_elem = 'log/bfgs/' + weight_bfgs_file_elem
        weight_bfgs = np.loadtxt(_weight_bfgs_file_elem, delimiter=',')
        file_elem = 'log/bfgs/' + weight_bfgs_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_bfgs, fname=file_elem)
    for weight_cg_file_elem in weight_cg_file:
        _weight_cg_file_elem = 'log/cg/' + weight_cg_file_elem
        weight_cg = np.loadtxt(_weight_cg_file_elem, delimiter=',')
        file_elem = 'log/cg/' + weight_cg_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_cg, fname=file_elem)
    for weight_sgd_file_elem in weight_sgd_file:
        _weight_sgd_file_elem = 'log/sgd/' + weight_sgd_file_elem
        weight_sgd = np.loadtxt(_weight_sgd_file_elem, delimiter=',')
        file_elem = 'log/sgd/' + weight_sgd_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    print 'weights plotting done.'


def plot_loss():
    loss_bfgs = np.loadtxt('log/bfgs.csv', delimiter=',')
    loss_cg = np.loadtxt('log/cg.csv', delimiter=',')
    loss_sgd = np.loadtxt('log/sgd.csv', delimiter=',')
    plt.plot(loss_bfgs[:, 0], loss_bfgs[:, 1], 'r')
    plt.plot(loss_cg[:, 0], loss_cg[:, 1], 'g')
    plt.plot(loss_sgd[:600, 0], loss_sgd[:600, 1], 'b*')
    plt.xlim(0, 20)
    plt.show()


def listdir_sort(path):
    fls = os.listdir(path)
    fls = map(lambda x: int(x), fls)
    fls.sort()
    fls = map(lambda x: str(x), fls)
    return fls


def main():
    plot_loss()
    plot_weight()


if __name__ == '__main__':
    main()
