#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_cl__trainer.py
@Time    :   2022/06/02 12:39:36
@Author  :   rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import json
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from transformers import AdamW, get_constant_schedule_with_warmup

from cortex.utils import Logs
from cortex.replay import reservoir_sampling
from cortex.replay import kmeans
from cortex.replay import task_label
from cortex.data_processors.base_cl_processor import DatasetIterater


class Trainer(object):
    def __init__(self, config, model) -> None:
        self.config = config
        self.__model = model.to(config.device)
        # self.__buffer = Memory()
        # loss function
        self.loss_func = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = AdamW(
            [
                {
                    "params": model.bert.parameters(), 
                    "lr": self.config.bert_learning_rate, 
                    "weight_decay": 0.01
                },
                {
                    "params": model.classifier.parameters(), 
                    "lr": self.config.learning_rate, 
                    "weight_decay": 0.01
                }
            ]
        )
        # scheduler
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, 1000)
        # logger
        self.logger = Logs(self.config.log_path)
          
    def __get_model_features(self, buffer):
        """
        load best model and calculate the features
        """
        self.__model.load_state_dict(
            torch.load(self.config.checkpoint_path)
        )
        features_batches = []
        for tokens, token_types, mask in buffer.get_mini_batch():
            _, features = self.__model(tokens, mask, token_types)
            features_batches.append(features.cpu().numpy())
        return np.concatenate(features_batches, axis=0)
          
    def train_step(self, train_iter, test_iter, dev_iter, task_name):
        self.__model.train() 
        self.__model.zero_grad()
        # buffer = Memory(self.config)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        for epoch in range(self.config.epochs):
            print(f'===== Epoch [{epoch + 1} / {self.config.epochs}] | {task_name} =====')
            # scheduler.step() # 学习率衰减
            buffer = Memory(self.config)
            for train_x, token_type, mask, labels in train_iter:
                buffer.append(train_x, token_type, mask, labels, task_name)
                outputs, embedding = self.__model(train_x, mask, token_type)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if total_batch % 1 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    truth = labels.data.cpu()  # 从cpu tensor中取出标签数据
                    predict = torch.max(outputs.data, 1)[1].cpu()  
                    train_acc = metrics.accuracy_score(truth, predict) 
                    dev_acc, dev_loss = self.evaluate(dev_iter)
                    if dev_loss[task_name].item() < dev_best_loss:  # 使用开发集判断模型性能是否提升
                        dev_best_loss = dev_loss[task_name].item()
                        torch.save(self.__model.state_dict(), self.config.checkpoint_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    msg = f"Iter: {total_batch} === train loss: {loss:>0.3f}, train acc: {train_acc:>0.2f}, val loss: {dev_loss[task_name]:>0.2f}, val acc: {dev_acc[task_name]:>0.2f} {improve}"
                    self.logger.info(msg)
                    self.__model.train()
                total_batch += 1
                if total_batch - last_improve > self.config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    # 开发集loss超过一定数量的batch没下降，则结束训练
                    flag = True
                    self.select_samples_to_store(task_name, categorical='rs')
                    break
            if flag:
                break
        self.test(test_iter)
        return train_acc, loss, dev_acc, dev_loss

    def train(self, train_loader, test_loader, dev_loader):
        """
        按顺序读取数据, 然后训练

        Args:
            train_loader (object): 动态传入数据路径进行读取, 需要构建 DataIterater 实例。
                数据路径格式 {'current_task': '', 'replay_task': []}, 其中 `current_task` 
                表示当前任务名, `replay_task` 表示需要回放的任务名。
            test_loader (dict): {task_name: tensor} 
            dev_loader (dict): {task_name: tensor}
        """
        finish_tasks = []
        acc_track = []
        test_iters = {
            task: DatasetIterater(*test_loader[task], self.config) for task in self.config.task_names
        }
        dev_iters = {
             task: DatasetIterater(*dev_loader[task], self.config) for task in self.config.task_names
        }
        for task in self.config.task_names:
            # load replay data and current data
            # 暂时不考虑 shuffle, replay data 在最后
            train_tasks = {'current_task': task, 'replay_task': finish_tasks}
            train_dataset = train_loader(train_tasks)
            train_iter = DatasetIterater(*train_dataset, self.config)
            train_acc, acc_loss, dev_acc, dev_loss = self.train_step(
                train_iter,
                test_iters,
                dev_iters,
                task
            )
            acc_track.append((train_acc, acc_loss, dev_acc, dev_loss))
            finish_tasks.append(task)
        train_avg_acc = np.mean([x[0] for x in acc_track])
        dev_acc_avg = np.mean([x[2] for x in acc_track])

    def evaluate(self, data_iter, checkpoint_path=None, eval=False):
        if checkpoint_path:
            self.__model.load_state_dict(torch.load(checkpoint_path))
        self.__model.eval()
        loss_total = 0.0
        predict_all, labels_all = {}, {} 
        all_task_acc, all_task_loss = {}, {}
        with torch.no_grad():  
            for task in self.config.task_names:
                task_loss = 0.0
                for texts, token_type, mask, labels in data_iter[task]:  # 对数据集中的每一组数据
                    outputs, _ = self.__model(texts, mask, token_type)  # 使用模型进行预测
                    loss = self.loss_func(outputs, labels)  # 计算模型损失
                    task_loss += loss  # 任务累加损失
                    loss_total += loss  # 累加模型损失
                    labels = labels.data.cpu().numpy()
                    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                    if task not in predict_all:
                        predict_all[task] = np.array([predic], dtype=int)
                        labels_all[task] = np.array([labels], dtype=int)
                    else:
                        labels_all = np.append(labels_all[task], labels)  
                        predict_all = np.append(predict_all[task], predic) 
                # print('===> ',labels_all[task])
                # print('---> ', predict_all[task])
                acc = metrics.accuracy_score(labels_all[task][0], predict_all[task][0]) 
                all_task_acc[task] = acc
                all_task_loss[task] = task_loss / len(data_iter) 
        if eval:
            all_task_reports, all_task_confusions = {}, {}
            for task in self.config.task_names:
                report = metrics.classification_report(
                    labels_all[task], predict_all[task], 
                    target_names=self.config.class_list, 
                    digits=4
                )
                confusion = metrics.confusion_matrix(labels_all[task], predict_all[task])
                all_task_reports[task] = report
                all_task_confusions[task] = confusion
            return all_task_acc, all_task_loss, all_task_reports, all_task_confusions
        return all_task_acc, all_task_loss

    def test(self, test_iter):
        self.__model.eval()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(test_iter, eval=True)
        self.logger.info(f'test loss: {test_loss:>0.2f}, test acc: {test_acc:>0.2f}')
        self.logger.info("===== Precision, Recall and F1-Score =====")
        self.logger.info(test_report)
        self.logger.info("===== Confusion Matrix =====")
        self.logger.info(test_confusion)
    
    def select_samples_to_store(self, buffer, task_name, categorical=None):
        """

        Args:
            task_name (str): 任务名
            categorical (str, optional): 样本采样策略. Defaults to None.
                [
                    rs,  # 蓄水池
                    kmeans,  # kmeans
                    lb,  # label-based 基于文本原始标签
                ]
        """
        if categorical == 'rs':
            with open(self.config.train_path[task_name], encoding='utf8') as f:
                resource = f.readlines()
                replay_sample = reservoir_sampling.reservoir_sampling(resource, self.config.replay_num)
                with open(self.config.replay_path[task_name], 'w+', encoding='utf8') as f:
                    for line in replay_sample:
                        line = json.dumps(line, ensure_ascii=False)
                        f.write(line+'\n')
        elif categorical == 'kmeans':
            # 从最优模型中计算句子特征
            features = self.__get_model_features(buffer)
            # 计算 kmeans
            label_pred, centroids = kmeans.kmeans(features, self.config.n_cluater)
            texts, labels = [], []
            for cluster_id in range(self.config.n_cluster):
                index = [i for i in range(len(label_pred)) if label_pred[i] == cluster_id]
                x_distance = []
                for j in index:
                    dis = np.sqrt(np.sum(np.square(centroids[cluster_id] - features[j])))
                    x_distance.append((dis, j))
                x_dist_sort = sorted(x_distance, key=lambda x: x[0])[:self.config.replay_num]
                texts.extend([buffer.raw_texts[idx] for _, idx in x_dist_sort])
                labels.extend([buffer.labels[idx] for _, idx in x_dist_sort])

            with open(self.config.replay_path[task_name], 'w+', encoding='utf8') as f:
                for idx in range(len(texts)):
                    line = {
                        'content': texts[idx],
                        'label': labels[idx]
                    }
                    line = json.dumps(line, ensure_ascii=False)
                    f.write(line+'\n')
        elif categorical == 'lb':
            # 从最优模型中计算句子特征
            features = self.__get_model_features(buffer)
            # 计算类别中心点    
            centroids = task_label(features, buffer.labels)
            texts, labels = [], []
            for label in buffer.labels:
                index = [i for i in range(len(buffer.labels)) if buffer.labels[i] == label]
                x_distance = []
                for j in index:
                    dis = np.sqrt(np.sum(np.square(centroids[cluster_id] - features[j])))
                    x_distance.append((dis, j))
                x_dist_sort = sorted(x_distance, key=lambda x: x[0])[:self.config.replay_num]
                texts.extend([buffer.raw_texts[idx] for _, idx in x_dist_sort])
                labels.extend([buffer.labels[idx] for _, idx in x_dist_sort])

            with open(self.config.replay_path[task_name], 'w+', encoding='utf8') as f:
                for idx in range(len(texts)):
                    line = {
                        'content': texts[idx],
                        'label': labels[idx]
                    }
                    line = json.dumps(line, ensure_ascii=False)
                    f.write(line+'\n')
            

class Memory(object):
    """
    现在只考虑保存当前任务数据，
    历史数据先直接保存成文件。

    Args:
        object (_type_): _description_
    """
    def __init__(self, config) -> None:
        self.config = config
        self.examples = []
        self.token_type_ids = []
        self.mask = []
        self.raw_texts = []
        self.labels = []
        self.tasks = []
        
    def append(
        self, 
        examples: torch.Tensor, 
        token_type_ids: torch.Tensor, 
        mask: torch.Tensor,
        labels: torch.Tensor, 
        task_name: str
    ):
        self.examples.append(examples)
        self.token_type_ids.append(token_type_ids)
        self.mask.append(mask)
        # 保存原始文本和标签
        texts = self.config.tokenizer.batch_decode(
            self.examples, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        texts = [x.replace(' ', '') for x in examples]
        self.raw_texts.extend(texts)
        labels = labels.cpu().to_list()
        labels = [self.config.id2lable[idx] for idx in labels]
        self.labels.extend(labels)
        self.tasks.append([task_name] * examples.size(0))
        
        
    def get_mini_batch(self):
        for idx in range(len(self.labels)):
            tokens = self.examples[idx]
            token_types = self.token_type_ids[ids]
            mask = self.mask[idx]
            yield tokens, token_types, mask
    
    
