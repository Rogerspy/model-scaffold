#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2023/09/25 11:59:38
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch
import torch.nn as nn

from cortex.layers.positional_encoding import SinusoidalPositionalEmbedding
from .encode_layer import EncodeLayer


class Encoder(nn.Module):
    def __init__(
            self, 
            num_layers: int,
            vocab_size: int,
            max_seq_len: int, 
            num_heads: int, 
            d_model: int, 
            d_ff: int,
            dropout_rate: float = 0.1,
            padding_idx: int = 0
        ) -> None:
        super().__init__()
        
        self.num_layers = num_layers

        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        self.position_encoding = SinusoidalPositionalEmbedding(
            seq_len=max_seq_len,
            d_model=d_model
        )
        self.encode_layer = EncodeLayer(
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff, 
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x_emb = self.embed(x)
        x = self.position_encoding(x_emb)
        for _ in range(self.num_layers):
            x = self.encode_layer(x, mask)
        return x