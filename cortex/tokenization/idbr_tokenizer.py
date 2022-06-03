#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   idbr_tokenizer.py
@Time    :   2022/06/01 14:55:03
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


from transformers import BertTokenizer


def tokenizer(config):
    tok = BertTokenizer.from_pretrained(config.pretrained_path)
    return tok
