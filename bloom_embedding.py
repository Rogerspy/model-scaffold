#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bloom_embedding.py
@Time    :   2022/05/18 16:22:26
@Author  :   csong
@Email   :   rogerspy@163.com
@Copyright : rogerspy
'''


import numpy as np
import mmh3


def allocate(n_vectors, n_dimensions):
    table = np.zeros((n_vectors, n_dimensions), dtype='f')
    table += np.random.uniform(-0.1, 0.1, table.size).reshape(table.size)
    return table


def get_vector(table, word):
    hash1 = mmh3.hash(word, seed=0)
    hash2 = mmh3.hash(word, seed=1)
    row1 = hash1 % table.shape[0]
    row2 = hash2 % table.shape[0]
    return table[row1] + table[row2]


def update_vector(table, word, d_vector):
    hash1 = mmh3.hash(word, seed=0)
    hash2 = mmh3.hash(word, seed=1)
    row1 = hash1 % table.shape[0]
    row2 = hash2 % table.shape[0]
    table[row1] -= 0.001 * d_vector
    table[row2] -= 0.001 * d_vector
    return table
