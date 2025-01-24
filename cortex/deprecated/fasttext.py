#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fasttext.py
@Time    :   2022/05/22 19:14:14
@Author  :   Rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, 
                freeze=False
            )
        else:
            self.embedding = nn.Embedding(
                config.n_vocab, 
                config.embed_dim, 
                padding_idx=config.n_vocab-1
            )
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed_dim, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
