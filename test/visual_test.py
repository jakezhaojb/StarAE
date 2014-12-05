#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Junbo Zhao

import sys
sys.path.append('..')
import numpy as np

import visualize as vs


def load_test():
    a = vs.load_sample('IMAGES.mat', patch_size=4, n_patches=400)
    if a.shape == (16, 400):
        print 'Load test testing correct.'
        return
    else:
        print 'Load test testing WRONG.'
        return


def disp_test():
    wgt1 = np.random.rand(49, 81)
    vs.disp_effect(wgt1)
    wgt2 = np.random.rand(48, 150)
    vs.disp_effect(wgt2)


def main():
    load_test()
    disp_test()

if __name__ == '__main__':
    main()
