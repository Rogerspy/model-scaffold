#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bloom_fasttext.py
@Time    :   2022/05/21 00:08:57
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
                config.nr_hash, 
                config.embed_dim
            )
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed_dim, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
        
        self.nb_keys = config.nb_keys

    def forward(self, x):
        """
        Args:
            x (list): [x__key1, x_key2, x_key3]

        Returns:
            Tensor: _description_
        """
        out = sum([self.embedding(key) for key in x])
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out