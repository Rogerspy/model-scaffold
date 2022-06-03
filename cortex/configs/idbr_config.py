#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   idbr_config.py
@Time    :   2022/05/28 22:45:55
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
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
        self.model_name = 'idbr'
        self.task_names = [f'task_{i+1}' for i in range(13)]
        self.train_path = {
                task_name: os.path.join(
                    base_dir, dataset, 
                    f'clsdcp/tasks/{task_name}/train.txt'
                ) for task_name in self.task_names
            }                                                                  
        self.dev_path = {
                task_name: os.path.join(
                    base_dir, dataset, 
                    f'clsdcp/tasks/{task_name}/dev.txt'
                ) for task_name in self.task_names
            }                                   
        self.test_path = {
                task_name: os.path.join(
                    base_dir, dataset, 
                    f'clsdcp/tasks/{task_name}/test.txt'
                ) for task_name in self.task_names
            }                                
        self.class_list = [x.strip() for x in open(
            os.path.join(base_dir, dataset, 'class.txt'), encoding='utf8').readlines()]              # 类别名单
        self.checkpoint_path = os.path.join(base_dir, f'../checkpoints/{self.model_name}.ckpt')        # 模型训练结果
        self.log_path = os.path.join(base_dir, f'../logs/{self.model_name}_{timestamp}.txt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.tokenizer = None
                                             # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_cls_classes = len(self.class_list)   
        self.num_nsp_classes = 2
        self.n_tasks = len(self.task_names)
        self.n_cluster = len(self.class_list)
        self.epochs = 20                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.max_len = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5 
        self.bert_learning_rate = 3e-5
        self.task_learning_rate = 5e-4    
        self.encoder_size = 768
        self.hidden_size = 128
        self.seed = 0
        self.tskcoe = 1.0
        self.nspcoe = 1.0
        self.dump = True
        
        