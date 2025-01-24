#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   positional_encoding.py
@Time    :   2023/09/20 15:13:56
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import math
import torch
import torch.nn as nn

from torch.autograd import Variable


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Attention is all you need 标准位置编码
    pe(pos, 2i) = sin(pos/(base**(2i/d)))
    pe(pos, 2i+1) = cos(pos/(base**(2i/d)))
    """
    def __init__(self, seq_len: int, d_model: int, base: int = 10000, implementation: str | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        
        #                           pos
        # seq_idx:      0,   1,   2,   3,   4,   5
        # embdding:    0.1, 0.2, 0.3, 0.4, 0.5, 0.6   
        # v_size=6     0.2, 0.3, 0.4, 0.5, 0.6, 0.7   i
        # w_dim=3      0.3, 0.4, 0.5, 0.6, 0.7, 0.8

        pe = torch.zeros(seq_len, d_model)
        if implementation == 'simple':  # 直观实现: 效率低
            for pos in range(seq_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = math.sin(pos / (base ** ((2 * i) / d_model)))
                    pe[pos, i + 1] = math.cos(pos / (base ** ((2 * i) / d_model)))
        elif implementation == 'effective':  # 高效实现
            pos = torch.arange(seq_len).unsqueeze(1)
            inv_freq = torch.exp(-(math.log(base) * torch.arange(0, d_model, 2) / d_model))
            pe[:, 0::2] = torch.sin(pos * inv_freq)
            pe[:, 1::2] = torch.cos(pos * inv_freq)
        else:  # 各种框架实现方法
            inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2) / d_model))
            pos = torch.arange(seq_len).type_as(inv_freq)
            sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)  # ixj 矩阵
            pe[:, ::2] = sinusoid_inp.sin()
            pe[:, 1::2] = sinusoid_inp.cos()
        pe = pe.unsqueeze(0)  # 添加一维适应batch_size维
        self.register_buffer('pe', pe)  
        # register_buffer 模型必需，但不参加训练的参数，保存成state_dict的时候会保留
        # 直接用self.pe不会保存到state_dict里面
        # register_parameter和nn.Parameter是等效的

    def forward(self, x: torch.Tensor):
        x = x + Variable(self.pe[:, :x.shape[1]], requires_grad=False)
        return x