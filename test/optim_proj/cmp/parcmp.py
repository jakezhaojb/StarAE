# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This script applies dpark to finish BFGS vs CG vs SGD


import os
from dpark import DparkContext

n_cmp = 6


def main():
    for i in range(n_cmp):
        assert os.path.isdir('cmp'+str(i+1))
    dpark_ctx = DparkContext('process')

    # Dpark thread
    def map_iter(i):
        dir_name = 'cmp' + str(i+1)
        # TODO redo or ?
        if os.path.isdir(os.path.join(dir_name, 'log')):
            return
        print "Start running: ", i+1
        os.system('cd cmp' + str(i))
        os.system('python ./cmp.py')

    dpark_ctx.makeRDD(range(n_cmp)).foreach(map_iter)
    print 'Done.'


if __name__ == '__main__':
    main()
