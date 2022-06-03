#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   idbr_dataset.py
@Time    :   2022/05/27 15:34:32
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''

import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DataLoader(object):
    def __init__(self, config) -> None:
        self.config = config
        self.label_id = self.label2id()
        
    def label2id(self):
        label2id = {}
        for i, line in enumerate(self.config.class_list):
            label2id[line] = i
        return label2id

    def train_dataset(self, tasks):
        current_task_path = self.config.train_path[tasks['current_task']]
        replay_task_path = [self.config.replay_path[task] for task in tasks['replay_task']]
        fpath_list = [current_task_path, *replay_task_path]
        contents, labels = [], []
        for fpath in fpath_list:
            with open(fpath, encoding='utf8') as f:
                for line in f:
                    line = json.loads(line)
                    content = line['content']
                    label = self.label_id[line['label']]
                    contents.append(content)
                    labels.append(label)
        labels = torch.LongTensor(labels)
        token_result = self.tokenizer(
            contents, 
            max_length=self.config.max_len, 
            padding='max_length',
            truncation='longest_first',
            return_tensor='pt'
        )
        # shuffle
        index = torch.randperm(len(labels))
        input_ids = token_result['input_ids'][index, :]
        token_type_ids = token_result['token_type_ids'][index, :]
        attention_mask = token_result['attention_mask'][index, :]
        labels = labels[index, :]
        return input_ids, token_type_ids, attention_mask, labels
    
    def test_dataset(self, path_dict):
        """

        Args:
            path_dict (dict): test_path or dev_path.

        Returns:
            _type_: _description_
        """
        contents, labels = {}, {}
        for task in self.config.task_names:
            with open(path_dict[task], encoding='utf8') as f:
                for line in f:
                    line = json.loads(line)
                    content = line['content']
                    label = self.label_id[line['label']]
                    if task not in contents:
                        contents[task] = [content]
                        labels[task] = [label]
                    else:
                        contents[task].append(content)
                        labels[task].append(label)
            
            label_tensor = torch.LongTensor(labels[task])
            token_tensor = self.tokenizer(
                contents[task], 
                max_length=self.config.max_len, 
                padding='max_length',
                truncation='longest_first',
                return_tensor='pt'
            )
            input_ids = token_tensor['input_ids']
            token_type_ids = token_tensor['token_type_ids']
            attention_mask = token_tensor['attention_mask']
            contents[task] = (input_ids, token_type_ids, attention_mask)
            labels[task] = label_tensor
        return contents, labels
    
    # def load_data(self,):
    #     train_data = self.build_dataset(self.config.train_path)
    #     test_data = self.build_dataset(self.config.test_path)
    #     dev_data = self.build_dataset(self.config.dev_path)
    #     return train_data, test_data, dev_data


class DatasetIterater(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, labels, config):
        self.config = config
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.n_samples = input_ids.size(0)
        self.n_batch = self.n_samples // self.config.batch_size  # 数据集大小整除batch容量
        self.residue = False  # 记录batch数量是否为整数, false代表可以, true代表不可以
        if self.n_samples % self.n_batch != 0:
            self.residue = True
        self.index = 0  # 迭代用的索引

    def __next__(self):
        # 如果batch外还剩下一点句子, 并且迭代到了最后一个batch
        if self.residue and self.index == self.n_dataset: 
            # 直接拿出剩下的所有数据
            residue_input_ids = self.input_ids[self.index * self.config.batch_size: len(self.dataset)]
            residue_token_type_ids = self.token_type_ids[self.index * self.config.batch_size: self.n_samples]
            residue_attention_mask = self.attention_mask[self.index * self.config.batch_size: self.n_samples]
            residue_labels = self.labels[self.index * self.config.batch_size: self.n_samples]
            self.index += 1
            return residue_input_ids, residue_token_type_ids, residue_attention_mask, residue_labels

        elif self.index >= self.n_batch:
            self.index = 0
            raise StopIteration
        
        else:  # 迭代器的入口
            batch_input_ids = self.input_ids[
                self.index * self.config.batch_size: (self.index + 1) * self.config.batch_size
            ]
            batch_token_type_ids = self.token_type_ids[
                self.index * self.config.batch_size: (self.index + 1) * self.config.batch_size
            ]
            batch_attention_mask = self.attention_mask[
                self.index * self.config.batch_size: (self.index + 1) * self.config.batch_size
            ]
            batch_labels = self.labels[
                self.index * self.config.batch_size: (self.index + 1) * self.config.batch_size
            ]
            self.index += 1
            return batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_dataset + 1
        else:
            return self.n_dataset


class TaskDatasetIterater(object):
    def __init__(self, dataset, config) -> None:
        """

        Args:
            dataset (dict): 
                {
                    'task_name': (word_ids, mask, token_type_ids, int(label)),
                }
            config (dict): 
        """
        self.dataset = dataset
        self.config = config
    
    def __iter__(self):
        for task in self.config.task_names:
            input_ids, token_type_ids, attention_mask, labels = self.dataset[task]
            data_iter = DatasetIterater(
                input_ids, 
                token_type_ids, 
                attention_mask, 
                labels, 
                self.config
            )
            yield data_iter, task
            
    def __len__(self):
        return len(self.dataset.keys())
