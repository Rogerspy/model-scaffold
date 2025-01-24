#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   multihead_attention.py
@Time    :   2023/09/21 17:55:51
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        # 由于多头注意力是从向量维度进行的切分，所以必须保证向量维度能被整除
        assert d_model % num_heads == 0
        self.num_heads = num_heads 
        self.d_model = d_model
        # 每个头的维度
        self.d_head = d_model // num_heads
        # q, k, v
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        # dropout
        self.dropout = nn.Dropout(dropout_rate)
        # 输出层
        self.out = nn.Linear(d_model, d_model)

    def attention(
            self,
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            mask: torch.Tensor | None = None, 
            dropout: bool = False
        ):
        """
        Z = softmax((q * k) / sqrt(d_k)) * v

        Args:
            q (torch.Tensor): (bs, num_heads, seq_len, h_model)
            k (torch.Tensor): (bs, num_heads, seq_len, h_model)
            v (torch.Tensor): (bs, num_heads, seq_len, h_model)
            mask (torch.Tensor | None, optional): (bs, seq_len). Defaults to None.
            dropout (float, optional): _description_. Defaults to 0.1.
        """
        # 先计算 q * k / sqrt(d_k), 其中d_k是k的张量纬度
        # (bs, seq_len, d_model) x (bs, d_model, seq_len) = (bs, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1]) 

        # mask 掉 pad 的部分
        # 这一步必须在计算softmax之前进行，因为如果在之后再mask的话，会存在不归一的情况
        # 比如，如果mask在softmax之后的话，softmax的结果是：[0.1, 0.2, 0.3, 0.4]
        # 假设最后一位是pad位， 那么进行mask的时候结果会变成：[0.1, 0.2, 0.3, 0.0]
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算 softmax
        scores = F.softmax(scores, dim=-1)

        # dropout随机使一些权重为0，意味着随机使两个token之间的相互依赖关系失效
        if dropout > 0:
            scores = self.dropout(scores)

        # 计算 softmax * v
        # (bs, seq_len, seq_len) x (bs, seq_len, d_model) = (bs, seq_len, d_head)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask: torch.Tensor | None = None):
        bs = q.shape[0]

        if mask is not None:
            mask = mask.squeeze(0)  # 在bs位置添加一维

        # q, k, v 的形状是 (bs, seq_len, d_model), 沿着 d_model 将注意力转换成多头
        # 转换后的形状为 (bs, seq_len, num_heads, h_model)
        # transpose 是将num_heads维与seq_len维进行换位，后续计算注意力是计算 (seq_len, h_model) 进行计算的
        q = self.to_q(q).view(bs, -1, self.num_heads, self.d_head).transpose(1,2)
        k = self.to_k(k).view(bs, -1, self.num_heads, self.d_head).transpose(1,2)
        v = self.to_v(v).view(bs, -1, self.num_heads, self.d_head).transpose(1,2)

        # 计算注意力
        scores = self.attention(q, k, v, mask)

        # 多头合并
        x = scores.transpose(1, 2).view(bs, -1, self.d_model)

        # 线性映射输出
        out = self.out(x)
        return out