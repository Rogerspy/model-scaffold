#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2022/06/01 15:16:29
@Author  :   rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import random


def reservoir_sampling(source, k):
    samples = []
    n = 0 
    # try to initial samples 
    while n < k:
        try:
            s = source[n]
        except:
            break 
        samples.append(s)
        n += 1
    n = k
    while True:
        try:
            s = source[n]
        except:
            break 
        t = random.randint(0, n)  # inclusive, choose a number at 1/n prob.  
        if t < k: samples[t] = s  # update when one of the buffer item is chosen
        n += 1         
    return samples 

if __name__ == '__main__':
    s = [[random.randint(1,10) for _ in range(5)] for _ in range(10)]
    print(reservoir_sampling(s, 2))