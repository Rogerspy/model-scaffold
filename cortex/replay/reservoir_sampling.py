#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2022/06/01 15:16:29
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


import random


def ReservoirSampling(source, k):
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


def Main():
    source = Source()
    samples = ReservoirSampling(source, 10)
    print(samples)

if __name__ == '__main__':
    Main()