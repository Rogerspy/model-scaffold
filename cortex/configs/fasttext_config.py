#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fasttext_config.py
@Time    :   2022/05/22 18:55:32
@Author  :   Rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''

import os
import time
import torch


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding=None):
        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), '../../datasets'
            )
        )
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.model_name = 'fasttext'
        self.train_path = os.path.join(base_dir, dataset, 'train.txt')               # 训练集
        self.dev_path = os.path.join(base_dir, dataset, 'dev.txt')                                    # 验证集
        self.test_path = os.path.join(base_dir, dataset, 'test.txt')                                  # 测试集
        self.class_list = [x.strip() for x in open(
            os.path.join(base_dir, dataset, 'class.txt'), encoding='utf8').readlines()]              # 类别名单
        self.vocab_path = os.path.join(base_dir, dataset, 'vocab.json')                                # 词表
        self.checkpoint_path = os.path.join(base_dir, f'../checkpoints/{self.model_name}.ckpt')        # 模型训练结果
        self.log_path = os.path.join(base_dir, f'../logs/{self.model_name}_{timestamp}.txt')
        self.embedding_pretrained = embedding                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.vocab = None
        self.tokenizer = None
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.max_len = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_dim = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 256
