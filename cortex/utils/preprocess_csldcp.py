#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess_csldcp.py
@Time    :   2022/05/28 22:52:35
@Author  :   rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import os
import json
import random


def write_to(data, fpath):
    with open(fpath, 'w+', encoding='utf8') as f:
        for line in data:
            # line = json.dumps(line, ensure_ascii=False)
            f.write(line)


def split_data(data, save_dir, test_valid_rate=0.1):
    """
    将数据分成训练集, 测试集和验证集

    Args:
        data (list): _description_
    """
    nb_test_dev = int(len(data) * test_valid_rate)
    dev = random.sample(data, nb_test_dev)
    train = list(set(data) - set(dev))
    test = random.sample(train, nb_test_dev)
    write_to(train, os.path.join(save_dir, 'train.json'))
    write_to(test, os.path.join(save_dir, 'test.json'))
    write_to(dev, os.path.join(save_dir, 'dev.json'))


def split_task(fdir, nb_per_file=1000):
    """
    将 teacher 模型生成的数据分成不同的 task。

    Args:
        fdir (str): 分解后的数据存储路径
        nb_per_file (int, optional): 每个任务的数据量. Defaults to 1000.
    """
    if not os.path.exists(fdir):
        os.mkdir(fdir)
    with open(os.path.join(fdir, '../unlabeled_teacher_prediction.json'), encoding='utf8') as f:
        tmp = []
        i = 0
        for line in f:
            if len(tmp) < nb_per_file:
                tmp.append(line)
            else:
                i += 1
                task_i_dir = os.path.join(fdir, f'task_{i}')
                if not os.path.exists(task_i_dir):
                    os.mkdir(task_i_dir)
                split_data(tmp, task_i_dir)
                tmp = [line]
                
                
if __name__ == '__main__':
    split_task('C:/IDEA/projects/Cortex/datasets/FewCLUE/csldcp/tasks')
    