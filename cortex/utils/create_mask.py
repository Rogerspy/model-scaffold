#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   create_mask.py
@Time    :   2023/09/26 14:34:14
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import torch


def create_padding_mask(batch_sequences: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
    """
    batch_sequences = [
        [ 1, 2, 3, 4, 0, 0, 0 ],  # token("我是谁?")
        [ 2, 3, 4, 5, 6, 0, 0 ],  # token("我不知道啊")
        [ 4, 5, 6, 0, 0, 0, 0 ]   # token("我爱你")
    ]  # batch_size = 3
    >>> 
    mask = [
        [ 1, 1, 1, 1, 0, 0, 0 ],
        [ 1, 1, 1, 1, 1, 0, 0 ],
        [ 1, 1, 1, 0, 0, 0, 0 ]
    ]

    Args:
        batch_sequences (torch.Tensor): _description_
        padding_idx (int, optional): _description_. Defaults to 0.

    Returns:
        torch.Tensor: (batch_size, seq_len)
    """
    padding_mask = torch.ones_like(batch_sequences)
    padding_mask[batch_sequences == padding_idx] = 0
    return padding_mask


def create_sequence_mask(seq_len: int) -> torch.Tensor:
    """
    mask = [
        #    我   是    谁   pad  pad  pad
        我    1    0    0    0    0    0
        是    1    1    0    0    0    0 
        谁    1    1    1    0    0    0
        pad   1    1    1    1    0    0
        pad   1    1    1    1    1    0
        pad   1    1    1    1    1    1
    ]

    Args:
        seq_len (int): _description_

    Returns:
        torch.Tensor: (seq_len, seq_len)
    """
    seq_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
    return seq_mask