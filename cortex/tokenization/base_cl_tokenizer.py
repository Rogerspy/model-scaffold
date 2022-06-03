#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_cl_tokenizer.py
@Time    :   2022/06/03 07:38:14
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''

from transformers import BertTokenizer


def tokenizer(config):
    tok = BertTokenizer.from_pretrained(config.pretrained_path)
    return tok