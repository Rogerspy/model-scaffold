#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   decode_layer.py
@Time    :   2023/09/25 14:57:33
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch
import torch.nn as nn

from cortex.layers.attention import MultiHeadAttention
from cortex.layers.ffn import PositionwiseFeedForward
from cortex.layers.normalization import LayerNorm


class DecodeLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_ff: int, dropout_rate: float) -> None:
        super().__init__()

        self.attention_1 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)
        self.attention_2 = MultiHeadAttention(num_heads=num_heads, d_model=d_model, dropout_rate=dropout_rate)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

        self.layer_norm_1 = LayerNorm(d_model=d_model)
        self.layer_norm_2 = LayerNorm(d_model=d_model)
        self.layer_norm_3 = LayerNorm(d_model=d_model)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(
            self, 
            x: torch.Tensor, 
            encode_x: torch.Tensor, 
            src_mask: torch.Tensor, 
            tgt_mask: torch.Tensor,
            residual_method: str = 'pre_norm'
        ) -> torch.Tensor:
        # 第一个sub_layer
        """
        Args:
            x (torch.Tensor): _description_
            encode_x (torch.Tensor): _description_
            src_mask (torch.Tensor): _description_
            tgt_mask (torch.Tensor): _description_
            residual_method (str, optional): Defaults to 'pre_norm'.
                - `pre_norm`: x + f(layer_norm(x))
                - `post_norm`: layer_norm(x + f(x))

        Returns:
            torch.Tensor: _description_
        """
        if residual_method == 'pre_norm':
            x_norm_1 = self.layer_norm_1(x)
            x_att_1 = self.attention_1(x_norm_1, x_norm_1, x_norm_1, tgt_mask)
            x_att_1 = self.dropout_1(x_att_1)
            x_res_1 = x + x_att_1

            x_norm_2 = self.layer_norm_2(x_res_1)
            x_att_2 = self.attention_2(x_norm_2, encode_x, encode_x, src_mask)
            x_att_2 = self.dropout_2(x_att_2)
            x_res_2 = x_res_1 + x_att_2

            x_norm_3 = self.layer_norm_3(x_res_2)
            x_ffn = self.ffn(x_norm_3)
            x_ffn = self.dropout_3(x_ffn)
            out = x_res_2 + x_ffn
        elif residual_method == 'post_norm':
            x_att_1 = self.attention_1(x, x, x, tgt_mask)
            x_att_1 = self.dropout_1(x)
            x_res_1 = x + x_att_1
            x_norm_1 = self.layer_norm_1(x_res_1)

            x_att_2 = self.attention_2(x_norm_1, encode_x, encode_x, src_mask)
            x_att_2 = self.dropout_2(x_att_2)
            x_res_2 = x_norm_1 + x_att_2
            x_norm_2 = self.layer_norm_2(x_res_2)

            x_ffn = self.ffn(x_norm_2)
            x_ffn = self.dropout_3(x_ffn)
            x_res_3 = x_norm_2 + x_ffn
            out = self.layer_norm_3(x_res_3)
        else:
            raise ValueError(f'`residual_method` not support {residual_method}, optional [`pre_norm`, `post_norm`].')
        
        return out