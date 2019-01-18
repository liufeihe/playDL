# -*- coding: utf-8 -*-

import collections
from numpy import random
import timeit
import os
import numpy as np


def product_test():
    # a = [[1,3,2],[1,0,0],[1,2,2]]
    a = [[1],[1],[1]]
    a = np.array(a)
    # b = [[0,0,2],[7,5,0],[2,1,1]]
    b = [[2],[0]]
    b = [[2,0]]
    b = np.array(b)
    print b.shape
    # print a*b
    # print np.multiply(a,b)
    # print np.dot(a, b)
    print np.outer(a, b)


def test():
    words = 'asssddsdsdfsdfsd1123'
    print collections.Counter(words)
    words = ['1','1','2','3','1','4','a']
    for word in words:
        print word
    print collections.Counter(words)
    print collections.Counter(words).most_common(2)


def yield_generator():
    for i in range(10):
        yield i


def yield_test():
    for i in yield_generator():
        print i


def str_test():
    print 'i:%d, j:%d'%(1,2)


def os_test():
    print os.path.join('./a/')


def main():
    # test()
    # yield_test()
    # str_test()
    # os_test()
    product_test()

if __name__ == '__main__':
    main()