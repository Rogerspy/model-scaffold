#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   vocabulary.py
@Time    :   2023/09/28 11:49:00
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import json
from typing import Callable

from cortex.utils import progress
from cortex.utils import Log


def build_vocab(
    file_path: str, 
    tokenizer: Callable, 
    min_freq: int = 1,
    max_vocab_size: int | None = None,
    unk_token: str = '<UNK>',
    pad_token: str = '<PAD>',
    save: bool = True,
    save_dir: str = '',
    logger: Log | None = None
):
    """
      构建一个词表：
      首先对数据集中的每一行句子按字/空格进行分割, 然后统计所有元素的出现频率
      接下来按照频率从高到低的顺序对所有频率大于 min_freq 的元素进行排序, 取前 max_vocab_size 个元素。
      最后按照频率降序构建字典 vocab_dic: {元素:序号}, vocab_dic 的最后两个元素是'<UNK>'和'<PAD>'
    """
    vocab_dic = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in progress(f, 'Building vocabulary'):
            line = line.strip()
            if not line: 
                continue
            content = line.split('\t')[0]  # 句子和标签通过tab分割, 前面的是句子内容
            for word in tokenizer(content): 
                vocab_dic[word] = vocab_dic.get(word, 0) + 1 # 统计词频或字频
        vocab_list = sorted(
            [w for w in vocab_dic.items() if w[1] >= min_freq], 
            key=lambda x: x[1], 
            reverse=True
        )[:max_vocab_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 添加 unk 和 pad
        vocab_dic.update({unk_token: len(vocab_dic), pad_token: len(vocab_dic) + 1})
        if logger:
            logger.info(f'词表构建完毕, 共 {len(vocab_dic)} 个词。')
    # 保存
    if save:
        with open(save_dir, 'w+', encoding='utf8') as f:
            json.dump(vocab_dic, f, ensure_ascii=False)
    return vocab_dic