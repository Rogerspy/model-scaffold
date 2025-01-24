#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   model.py
@Time    :   2023/09/26 10:55:11
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


from utils import create_sequence_mask, create_padding_mask


class Model(object):
    def __init__(self, model, optimizer, loss_fn) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def initialize(self, init_fn):
        self.model = init_fn(self.model)

    def __train_step(self, train_data):
        inputs, outputs = train_data
        src_mask = create_padding_mask(inputs)
        tgt_mask = create_sequence_mask(outputs.shape[1])
        pass

    def train(self, epochs, train_iter, dev_iter = None, test_iter = None, verbose = True, dev_frequency = 1000):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def inference(self):
        raise NotImplementedError