# -*- coding: utf-8 -*-
# Author: Junbo Zhao

from __future__ import division
import math


def factor2(n):
    """Factor interger n into a * b"""
    sqr_n = int(math.floor(math.sqrt(n)))
    while sqr_n > 0:
        if n % sqr_n == 0:
            break
        sqr_n -= 1
    return int(n//sqr_n), sqr_n
