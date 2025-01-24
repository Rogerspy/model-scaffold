#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ffn.py
@Time    :   2023/09/22 15:56:21
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate: float = 0.1) -> None:
        super().__init__()

        self.in_linear = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_linear = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.in_linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.out_linear(x)
        return out