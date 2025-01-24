#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   idbr.py
@Time    :   2022/05/26 21:46:59
@Author  :   rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config) -> None:
        super(Model, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.pretrained_path)
        
        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.encoder_size,
                self.config.n_classes
            )
        )
        
    def forward(self, x, mask, token_type_ids):
        encoder_output = self.bert(
            input_ids=x,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = encoder_output.last_hidden_state  # 得到每个 token 的 bert 编码
        bert_embedding = sequence_output[:, 0, :]  # 取 cls

        cls_pred = self.classifier(bert_embedding)
        
        return cls_pred, bert_embedding
