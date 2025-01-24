#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   layer_norm.py
@Time    :   2023/09/22 16:16:35
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    为了进一步使得每一层的输入输出范围稳定在一个合理的范围内

    ln(x) = alpha * (x - mu) / sigma + b

    其中 mu 和 sigma 分别代表均值和方差, alpha 和 b 是可学习参数。
    详细信息可参看：https://arxiv.org/abs/1607.06450。
    """
    def __init__(self, d_model, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.alpha, self.b = nn.ParameterList(
            [
                torch.ones(d_model),
                torch.zeros(d_model)
            ]
        )
        self.eps = epsilon  # 防止分母为0
        
    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        out = self.alpha * (x - mu) / (sigma + self.eps) + self.b
        return out