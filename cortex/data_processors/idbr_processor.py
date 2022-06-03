#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   idbr_processor.py
@Time    :   2022/05/26 22:14:57
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


import json
import torch
import random


class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.label_id = self.label2id()
        
    def label2id(self):
        label2id = {}
        for i, line in enumerate(self.config.class_list):
            label2id[line] = i
        return label2id
    
    def build_dataset(self, fpath):
        contents = []
        with open(fpath, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                content = line['content']
                label = self.label_id[line['label']]
                tokens = self.config.tokenizer.tokenize(content) 
                # max length
                if self.config.max_len: 
                    if len(tokens) > self.config.max_len - 3:
                        tokens = tokens[:self.config.max_len - 3]
                # word to id
                word_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)
                cls_word_ids = self.next_sentence_positive(word_ids)
                mask = [1] * len(word_ids)
                # padding
                padding = [self.config.tokenizer.pad_token_id] * (self.config.max_len - len(word_ids))
                cls_word_ids.extend(padding)
                # mask 
                mask.extend(padding)
                # nsp 数据
                if self.config.nsp:
                    nsp_neg_ids = self.next_sentence_negative(word_ids)
                    nsp_neg_ids.extend(padding)
                    contents.append((cls_word_ids, nsp_neg_ids, mask, int(label)))
                else:
                    contents.append((cls_word_ids, mask, int(label)))
            return contents
        
    def load_data(self):
        train_data, test_data, dev_data = {}, {}, {}
        for task_name in self.config.task_names:
            train_data[task_name] = self.build_dataset(self.config.train_path[task_name])
            test_data[task_name] = self.build_dataset(self.config.test_path[task_name])
            dev_data[task_name] = self.build_dataset(self.config.dev_path[task_name])
        return train_data, test_data, dev_data           
    
    def next_sentence_positive(self, word_ids):
        """
        Next Sentence Prediction 正样本
        
        Args:
            content (list): word ids not padding and not add cls and sep token
        """
        word_len = len(word_ids)
        if word_len == 1:
            cut = 1
        else:
            cut = random.randint(1, word_len)
        word_ids = [
            self.config.tokenizer.cls_token_id,
            *word_ids[:cut], 
            self.config.tokenizer.sep_token_id, 
            *word_ids[cut:],
            self.config.tokenizer.sep_token_id
        ]
        return word_ids
    
    def next_sentence_negative(self, word_ids):
        """
        Next Sentence Prediction 负样本
        """
        word_len = len(word_ids)
        if word_len == 1:
            cut = 1
        else:
            cut = random.randint(1, word_len)
        word_ids = [
            self.config.tokenizer.cls_token_id,
            *word_ids[cut:], 
            self.config.tokenizer.sep_token_id, 
            *word_ids[:cut],
            self.config.tokenizer.sep_token_id
        ]
        return word_ids

    
class TaskDatasetIterater(object):
    """
      根据数据集产生batch
      这里需要注意的是, 在 _to_tensor()中, 代码把batch中的数据处理成了 `(x, mask), y` 的形式
      其中 x 是word_ids, mask 是注意力掩码, y 是数据标签
    """
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset  
        self.n_dataset = len(dataset) // self.config.batch_size  # 数据集大小整除batch容量
        self.residue = False  # 记录batch数量是否为整数, false代表可以, true代表不可以
        if len(dataset) % self.n_dataset != 0:
            self.residue = True
        self.index = 0  # 迭代用的索引

    # def _to_tensor(self, data):
    #     if self.config.nsp:
    #         cls = torch.LongTensor([d[0] for d in data]).to(self.config.device) 
    #         nsp_pos = torch.LongTensor([d[1] for d in data]).to(self.config.device) 
    #         nsp_neg = torch.LongTensor([d[2] for d in data]).to(self.config.device) 
    #         mask = torch.LongTensor([d[3] for d in data]).to(self.config.device) 
    #         y = torch.LongTensor([d[4] for d in data]).to(self.config.device)
    #         return cls, nsp_pos, nsp_neg, mask, y
    #     else:
    #         cls = torch.LongTensor([d[0] for d in data]).to(self.config.device) 
    #         mask = torch.LongTensor([d[3] for d in data]).to(self.config.device) 
    #         y = torch.LongTensor([d[4] for d in data]).to(self.config.device)
    #     return cls, mask, y
    
    def _to_list(self, data):
        if self.config.nsp:
            cls = [d[0] for d in data]
            nsp_neg = [d[2] for d in data] 
            mask = [d[3] for d in data]
            y = [d[4] for d in data]
            return cls, nsp_neg, mask, y
        else:
            cls = [d[0] for d in data] 
            mask = [d[3] for d in data]
            y = [d[4] for d in data]
        return cls, mask, y

    def __next__(self):
        # 如果batch外还剩下一点句子, 并且迭代到了最后一个batch
        if self.residue and self.index == self.n_dataset: 
            # 直接拿出剩下的所有数据
            dataset = self.dataset[self.index * self.config.batch_size: len(self.dataset)] 
            self.index += 1
            dataset = self._to_list(dataset)
            return dataset  # x, y

        elif self.index >= self.n_dataset:
            self.index = 0
            raise StopIteration
        
        else:  # 迭代器的入口
            dataset = self.dataset[self.index * self.config.batch_size: (self.index + 1) * self.config.batch_size]
            self.index += 1
            dataset = self._to_list(dataset)
            return dataset

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_dataset + 1
        else:
            return self.n_dataset


class DatasetIterater(object):
    def __init__(self, dataset, config) -> None:
        """

        Args:
            dataset (dict): 
                {
                    'task_name': [(cls_word_ids, nsp_pos_ids, nsp_neg_ids, mask, int(label)), ...]
                }
            config (dict): 
        """
        self.dataset = dataset
        self.config = config
    
    def __iter__(self):
        data_iter = {}
        for task in self.config.task_names:
            data_iter[task] = TaskDatasetIterater(self.dataset[task], self.config)
        return data_iter
            
    def __len__(self):
        return len(self.dataset.keys())
    