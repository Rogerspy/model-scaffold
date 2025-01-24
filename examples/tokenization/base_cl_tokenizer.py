#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_cl_tokenizer.py
@Time    :   2022/06/03 07:38:14
@Author  :   rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''

from transformers import BertTokenizer


def tokenizer(config):
    tok = BertTokenizer.from_pretrained(config.pretrained_path)
    return tok