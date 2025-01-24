#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   xavier.py
@Time    :   2023/09/26 17:09:36
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch.nn as nn


def xavier_uniform(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model