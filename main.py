#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/05/21 14:52:07
@Author  :   Rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import os
import json
import argparse
import jieba
import torch

from importlib import import_module


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument(
    '--model_name', 
    type=str, 
    required=True, 
    help="""Choose a model: 
        [fasttext, bloom_fasttext]
    """
)
parser.add_argument(
    '--dataset', 
    type=str, 
    default='THUCNews', 
    help='Dataset name'
)
parser.add_argument(
    '--require_improvement', 
    type=int, 
    default=1000, 
    help='效果没提升, 则提前结束训练'
)
args = parser.parse_args()


if __name__ == '__main__':
    # 搜狗新闻: embedding_SougouNews.npz, 
    # 腾讯: embedding_Tencent.npz, 
    # 随机初始化: random
    # embedding = 'embedding_SougouNews.npz'
    model_name = args.model_name
    dataset = args.dataset
    require_improvement = args.require_improvement
    
    # 导入相应模块
    model_obj = import_module(f'cortex.models.{model_name}')
    config_obj = import_module(f'cortex.configs.{model_name}_config')
    processor_obj = import_module(f'cortex.data_processors.{model_name}_processor')
    trainer_obj = import_module(f'cortex.trainers.{model_name}_trainer')
    
    # 加载模型配置文件
    config = config_obj.Config(dataset)  
    config.require_improvement = require_improvement
    
    # 加载 tokenizer
    config.tokenizer = jieba.lcut
    
    # 加载/生成词表
    if hasattr(config, 'vocab_path'):
        if os.path.exists(config.vocab_path):
            with open(config.vocab_path, encoding='utf8') as f:
                vocab = json.load(f)
        else:
            vocab_obj = import_module(f'cortex.data_processors.vocabulary')
            vocab = vocab_obj.build_vocab(
                file_path=config.train_path,
                tokenizer=jieba.lcut,
                save_dir=config.vocab_path
            )
        config.vocab = vocab
        config.n_vocab = len(vocab)
        
    # 数据处理/加载
    data_loader = processor_obj.DataLoader(config)
    train, test, dev = data_loader.load_data()
    train_iter = processor_obj.DatasetIterater(dataset=train, config=config)
    test_iter = processor_obj.DatasetIterater(dataset=test, config=config)
    dev_iter = processor_obj.DatasetIterater(dataset=dev, config=config)
    
    # 固定随机因子，保证每次结果一样
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  

    # train
    model = model_obj.Model(config).to(config.device)
    trainer = trainer_obj.Trainer(config, model)
    # if model_name != 'Transformer':
    #     init_network(model)
    print(model)
    # trainer.train(train_iter, dev_iter, test_iter)
