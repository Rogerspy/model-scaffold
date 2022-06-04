#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_cl_args.py
@Time    :   2022/06/02 17:30:43
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


import os
import time
import torch


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), '../../datasets'
            )
        )
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.model_name = 'base_cl'
        self.task_names = [f'task_{i+1}' for i in range(13)]
        self.train_path = {
                task_name: os.path.join(
                    base_dir, dataset, 
                    f'csldcp/tasks/{task_name}/train.json'
                ) for task_name in self.task_names
            }                                                                  
        self.dev_path = {
                task_name: os.path.join(
                    base_dir, dataset, 
                    f'csldcp/tasks/{task_name}/dev.json'
                ) for task_name in self.task_names
            }                                   
        self.test_path = {
                task_name: os.path.join(
                    base_dir, dataset, 
                    f'csldcp/tasks/{task_name}/test.json'
                ) for task_name in self.task_names
            }  
        self.replay_path = {
                task_name: os.path.join(
                    base_dir, dataset, 
                    f'csldcp/replay/{task_name}/train.json'
                ) for task_name in self.task_names
            }                               
        self.class_list = [x.strip() for x in open(
            os.path.join(base_dir, dataset, 'csldcp/labels_all.txt'), encoding='utf8').readlines()]              # 类别名单
        self.checkpoint_path = os.path.join(base_dir, f'../checkpoints/{self.model_name}.ckpt')        # 模型训练结果
        self.pretrained_path = os.path.join(base_dir, f'../pretrained/chinese-bert-wwm')
        self.log_path = os.path.join(base_dir, f'../logs/{self.model_name}_{timestamp}.txt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.tokenizer = None
                                             # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.n_classes = len(self.class_list)   
        self.n_tasks = len(self.task_names)
        self.n_cluster = len(self.class_list)
        self.epochs = 20                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.max_len = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5 
        self.bert_learning_rate = 3e-5  
        self.encoder_size = 768
        self.hidden_size = 128
        self.replay_num = 10
        