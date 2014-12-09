# -*- coding: utf-8 -*-
# Author: Junbo Zhao
# This script applies dpark to finish BFGS vs CG vs SGD


import os
from dpark import DparkContext

cmp_list = range(6)


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    for i in cmp_list:
        assert os.path.isdir('cmp'+str(i+1))
    dpark_ctx = DparkContext('process')

    # Dpark thread
    def map_iter(i):
        dir_name = 'cmp' + str(i+1)
        logger = os.path.join(dir_name, 'log')
        if os.path.isdir(logger) and os.listdir(logger):
            return
        print "Start running: ", i+1
        os.chdir(os.path.join(current_path, 'cmp') + str(i+1))
        os.system('python ./cmp.py')

    dpark_ctx.makeRDD(cmp_list).foreach(map_iter)
    print 'Done.'


if __name__ == '__main__':
    main()
