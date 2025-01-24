#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   encode_layer.py
@Time    :   2023/09/25 12:00:14
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch
import torch.nn as nn

from ..attention import MultiHeadAttention
from ..normalization import LayerNorm
from ..ffn import PositionwiseFeedForward


class EncodeLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_ff: int, dropout_rate: float = 0.1) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)
        self.layer_norm_1 = LayerNorm(d_model=d_model)
        self.layer_norm_2 = LayerNorm(d_model=d_model)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor,  mask: torch.Tensor | None = None, residual_method: str = 'pre_norm') -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): _description_
            mask (torch.Tensor | None, optional): _description_. Defaults to None.
            residual_method (str, optional): Defaults to 'pre_norm'.
                - `pre_norm`: x + f(layer_norm(x))
                - `post_norm`: layer_norm(x + f(x))      

        Raises:
            ValueError: _description_

        Returns:
            torch.Tensor: _description_
        """
        if residual_method == 'pre_norm':
            x_norm_1 = self.layer_norm_1(x)
            x_att = self.attention(x_norm_1, x_norm_1, x_norm_1, mask)
            x_att = self.dropout_1(x_att)
            x_res_1 = x + x_att

            x_norm_2 = self.layer_norm_2(x_res_1)
            x_ffn = self.ffn(x_norm_2)
            x_ffn = self.dropout_2(x_ffn)
            out = x_res_1 + x_ffn
        elif residual_method == 'post_norm':
            x_att = self.attention(x, x, x, mask)
            x_att = self.dropout_1(x_att)
            x_res_1 = x + x_att
            x_norm_1 = self.layer_norm_1(x_res_1)

            x_ffn = self.ffn(x_norm_1)
            x_ffn = self.dropout_2(x_ffn)
            x_res_2 = x_norm_1 + x_ffn
            out = self.layer_norm_2(x_res_2)
        else:
            raise ValueError(f'`residual_method` not support {residual_method}, optional [`pre_norm`, `post_norm`].')
        return out