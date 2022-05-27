#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   idbr.py
@Time    :   2022/05/26 21:46:59
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config) -> None:
        super(Model, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.pretrained_path)
        
        self.General_Encoder = nn.Sequential(
            nn.Linear(
                self.config.encoder_size,
                self.config.hidden_size 
            ),
            nn.Tanh()
        )
        
        self.Specific_Encoder = nn.Sequential(
            nn.Linear(
                self.config.encoder_size,
                self.config.hidden_size
            ),
            nn.Tanh()
        )
        
        self.cls_classifier = nn.Sequential(
            nn.Linear(
                self.config.hidden_size * 2,
                self.config.n_class
            )
        )
        
        self.task_classifier = nn.Sequential(
            nn.Linear(
                self.config.hidden_size, 
                self.config.n_tasks
            )
        )
        
    @staticmethod
    def _get_tokens_type(x):
        tokens_type_ids = x.new_ones(x.shape)
        for i in range(tokens_type_ids.size(0)):
            ids = x[i].tolist()
            first_pad = ids.index(102)
            for j in range(first_pad + 1):
                tokens_type_ids[i, j] = 0
        return tokens_type_ids
        
    def forward(self, x, mask):
        token_type_ids = self._get_tokens_type(x)
        encoder_output = self.bert(
            input_ids=x,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = encoder_output.last_hidden_state
        bert_embedding = sequence_output[:, 0:1, :].squeeze(dim=1)
        
        general_features = self.General_Encoder(bert_embedding)
        specific_features = self.Specific_Encoder(bert_embedding)

        task_pred = self.task_classifier(specific_features)

        features = torch.cat([general_features, specific_features], dim=1)
        cls_pred = self.cls_classifier(features)
        
        return general_features, specific_features, cls_pred, task_pred, bert_embedding
        
        
class Predictor(nn.Module):
    def __init__(self, config):
        super(Predictor, self).__init__()
        self.config = config

        self.dis = torch.nn.Sequential(
            torch.nn.Linear(
                self.config.hidden_size, 
                self.config.num_class
            )
        )

    def forward(self, z):
        return self.dis(z)
    

class BaseModel(nn.Module):
    def __init__(self, n_class):
        super(BaseModel, self).__init__()

        self.n_class = n_class
        self.bert = BertModel.from_pretrained(self.config.pretrained_path)
        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.encoder_size,
                self.config.hidden_size 
            )
        )

    def forward(self, x):
        x, _ = self.bert(x)
        x = torch.mean(x, 1)
        logits = self.classifier(x)
        return logits
