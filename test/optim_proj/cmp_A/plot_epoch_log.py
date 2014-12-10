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
    weight_sgd_ada = listdir_sort('log/epoch/sgd_ada')
    weight_sgd_Nada = listdir_sort('log/epoch/sgd_Nada')
    weight_sgd_ada95 = listdir_sort('log/epoch/sgd_ada95')
    weight_sgd_Nada95 = listdir_sort('log/epoch/sgd_Nada95')
    
    for weight_file_elem in weight_sgd_ada:
        _weight_file_elem = 'log/epoch/sgd_ada/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_ada/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_Nada:
        _weight_file_elem = 'log/epoch/sgd_Nada/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_Nada/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_ada95:
        _weight_file_elem = 'log/epoch/sgd_ada95/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_ada95/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)
    for weight_file_elem in weight_sgd_Nada95:
        _weight_file_elem = 'log/epoch/sgd_Nada95/' + weight_file_elem
        weight_sgd = np.loadtxt(_weight_file_elem, delimiter=',')
        file_elem = 'log/epoch/sgd_Nada95/' + weight_file_elem.zfill(4) + '.png'
        vs.disp_effect(weight_sgd, fname=file_elem)

    print 'weights plotting done.'


def plot_loss():
    loss_sgd_ada = np.loadtxt('log/epoch/sgd_ada.csv', delimiter=',')
    loss_sgd_Nada = np.loadtxt('log/epoch/sgd_Nada.csv', delimiter=',')
    loss_sgd_ada95 = np.loadtxt('log/epoch/sgd_ada95.csv', delimiter=',')
    loss_sgd_Nada95 = np.loadtxt('log/epoch/sgd_Nada95.csv', delimiter=',')
    plt.plot(range(len(loss_sgd_ada)), loss_sgd_ada, 'r', label='sgd_ada')
    plt.plot(range(len(loss_sgd_Nada)), loss_sgd_Nada, 'g', label='sgd_Nada')
    plt.plot(range(len(loss_sgd_ada95)), loss_sgd_ada95, 'b', label='sgd_ada95')
    plt.plot(range(len(loss_sgd_Nada95)), loss_sgd_Nada95, 'y', label='sgd_Nada95')
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
