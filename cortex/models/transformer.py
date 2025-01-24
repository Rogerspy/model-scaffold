#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   transformer.py
@Time    :   2023/09/25 16:53:28
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch
import torch.nn as nn

from layers.attention import MultiHeadAttention
from layers.normalization import LayerNorm
from layers.ffn import PositionwiseFeedForward
from layers.positional_encoding import SinusoidalPositionalEmbedding


class EncoderLayer(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            d_model: int, 
            d_ff: int, 
            dropout_rate: float = 0.1
        ) -> None:
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
            out = self.layer_norm_2(x_res_1 + x_ffn)
        else:
            raise ValueError(f'`residual_method` not support {residual_method}, optional [`pre_norm`, `post_norm`].')
        return out


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

        self.embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model, padding_idx=padding_idx)
        self.position_encoding = SinusoidalPositionalEmbedding(seq_len=max_seq_len, d_model=d_model)
        self.encoder_layer = EncoderLayer(num_heads=num_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x_emb = self.embed(x)
        x = self.position_encoding(x_emb)
        for _ in range(self.num_layers):
            x = self.encoder_layer(x, mask)
        return x


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
        

class Decoder(nn.Module):
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
        self.decode_layer = DecodeLayer(
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff, 
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor, encode_x: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x_emb = self.embed(x)
        x = self.position_encoding(x_emb)
        for _ in range(self.num_layers):
            x = self.decode_layer(x, encode_x, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.encoder = Encoder(
            num_layers=args.num_layers,
            vocab_size=args.src_vocab_size,
            max_seq_len=args.max_seq_len,
            num_heads=args.num_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout_rate=args.dropout_rate,
            padding_idx=args.padding_idx
        )
        self.decoder = Decoder(
            num_layers=args.num_layers,
            vocab_size=args.tgt_vocab_size,
            max_seq_len=args.max_seq_len,
            num_heads=args.num_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout_rate=args.dropout_rate,
            padding_idx=args.padding_idx
        )

        self.out = nn.Linear(args.d_model, args.tgt_vocab_size)

    def forward(self, src_x: torch.Tensor, tgt_x: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.encoder(src_x, src_mask)
        x = self.decoder(src_x, tgt_x, src_mask, tgt_mask)
        out = self.out(x)
        return out