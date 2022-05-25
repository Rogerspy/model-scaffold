#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bloom_fasttext_processor.py
@Time    :   2022/05/21 09:56:50
@Author  :   Rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import mmh3
import torch


class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config

    def bloom_keys(self, word):
        return [mmh3.hash(word, i) % self.config.nr_hash for i in range(self.config.nb_keys)]

    def build_dataset(self, fpath):
        """
            加载数据集：
            - 分离内容和标签
            - 分词
            - max_len 补足或截断
            - word to id
            - [(words_ids, int(label), seq_len), ...]
        """
        
        contents = []
        with open(fpath, encoding='utf8') as f:
            for line in f: 
                line = line.strip() 
                if not line: 
                    continue
                content, label = line.split('\t') 
                word_ids = []
                tokens = self.config.tokenizer(content) 
                # padding
                if self.config.max_len: 
                    if len(tokens) < self.config.max_len:
                        tokens.extend([self.config.pad_token] * (self.config.max_len - len(tokens)))
                    else:
                        tokens = tokens[:self.config.max_len] 
                # word to id
                for word in tokens:
                    word_ids.append(self.bloom_keys(word)) # [[key1, key2, key3], [key1, key2, key3], ...]

                contents.append((word_ids, int(label)))
        return contents
    
    def load_data(self):
        train_data = self.build_dataset(self.config.train_path)
        test_data = self.build_dataset(self.config.test_path)
        dev_data = self.build_dataset(self.config.dev_path)
        return train_data, test_data, dev_data


class DatasetIterater(object):
    """
      根据数据集产生batch
      这里需要注意的是, 在_to_tensor()中, 代码把batch中的数据处理成了`(x, seq_len), y`的形式
      其中x是word_ids, seq_len是pad前的长度(超过max_len的设为max_len), y是数据标签
    """
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset  
        self.n_dataset = len(dataset) // self.config.batch_size  # 数据集大小整除batch容量
        self.residue = False  # 记录batch数量是否为整数, false代表可以, true代表不可以
        if len(dataset) % self.n_dataset != 0:
            self.residue = True
        self.index = 0  # 迭代用的索引

    def _to_tensor(self, data):
        """
        将列表中的元素转化成 Tensor

        Args:
            data (list): [([[key1, key2, key3], [key1, key2, key3], ...], y), ...]
                
        Returns:
            tuple: tuple of Tensor
        """
        x_keys = torch.LongTensor([d[0] for d in data])  # nb_sample * max_len * nb_keys
        x = torch.split(x_keys, 1, -1)
        x = tuple([key.view(x_keys.size(0), x_keys.size(1)).to(self.config.device) for key in x])
        
        y = torch.LongTensor([d[1] for d in data]).to(self.config.device)
        return x, y

    def __next__(self):
        # 如果batch外还剩下一点句子, 并且迭代到了最后一个batch
        if self.residue and self.index == self.n_dataset: 
            # 直接拿出剩下的所有数据
            dataset = self.dataset[self.index * self.config.batch_size: len(self.dataset)] 
            self.index += 1
            dataset = self._to_tensor(dataset)
            return dataset  # (x, seq_len), y

        elif self.index >= self.n_dataset:
            self.index = 0
            raise StopIteration
        
        else:  # 迭代器的入口
            dataset = self.dataset[self.index * self.config.batch_size: (self.index + 1) * self.config.batch_size]
            self.index += 1
            dataset = self._to_tensor(dataset)
            return dataset

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_dataset + 1
        else:
            return self.n_dataset
