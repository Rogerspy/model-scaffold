#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   base_cl__trainer.py
@Time    :   2022/06/02 12:39:36
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from transformers import AdamW, get_constant_schedule_with_warmup

from cortex.utils import Logs


class BaseCLTrainer(object):
    def __init__(self, model, config) -> None:
        self.config = config
        self.__model = model
        # loss function
        self.loss_func = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = AdamW(
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
        )
        # scheduler
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, 1000)
        # logger
        self.logger = Logs(self.config.log_path)
          
    def train_step(self, train_iter, dev_iter, test_iter):
        self.__model.train() 
        self.__model.zero_grad()
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        for epoch in range(self.config.num_epochs):
            print(f'===== Epoch [{epoch + 1} / {self.config.num_epochs}] =====')
            # scheduler.step() # 学习率衰减
            for train_x, labels in train_iter:
                outputs = self.__model(train_x)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if total_batch % 100 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    truth = labels.data.cpu()  # 从cpu tensor中取出标签数据
                    predic = torch.max(outputs.data, 1)[1].cpu()  
                    train_acc = metrics.accuracy_score(truth, predic) 
                    dev_acc, dev_loss = self.evaluate(dev_iter) 
                    if dev_loss < dev_best_loss:  # 使用开发集判断模型性能是否提升
                        dev_best_loss = dev_loss
                        torch.save(self.__model.state_dict(), self.config.checkpoint_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    msg = f"Iter: {total_batch} === train loss: {loss:>0.3f}, train acc: {train_acc:>0.2f}, val loss: {dev_loss:>0.2f}, val acc: {dev_acc:>0.2f} {improve}"
                    self.logger.info(msg)
                    self.__model.train()
                total_batch += 1
                if total_batch - last_improve > self.config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    # 开发集loss超过一定数量的batch没下降，则结束训练
                    flag = True
                    self.select_samples_to_store(train_iter)
                    break
            if flag:
                break
        self.test(test_iter)
        return train_acc, loss, dev_acc, dev_loss

    def train(self, train_loader, test_loader, dev_loader):
        """
        按顺序读取数据, 然后训练

        Args:
            train_loader (_type_): _description_
            test_loader (_type_): _description_
            dev_loader (_type_): _description_
        """
        finish_tasks = []
        acc_track = []
        for task in self.config.task_names:
            # load replay data and current data
            # 暂时不考虑 shuffle, replay data 在最后
            train_tasks = {'current_task': task, 'replay_task': finish_tasks}
            train_dataset = train_loader([task, *replay_path])
            train_iter = DatasetIterater(train_dataset)
            test_iter = DatasetIterater([task, *finish_tasks])
            dev_iter = DatasetIterater([task, *finish_tasks])
            train_acc, acc_loss, dev_acc, dev_loss = self.train_step(
                train_iter,
                test_iter,
                dev_iter
            )
            acc_track.append((train_acc, acc_loss, dev_acc, dev_loss))
        train_avg_acc = np.mean([x[0] for x in acc_track])
        dev_acc_avg = np.mean([x[2] for x in acc_track])

    def evaluate(self, data_iter, checkpoint_path=None, eval=False):
        if checkpoint_path:
            self.__model.load_state_dict(torch.load(checkpoint_path))
        self.__model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():  # 不追踪梯度
            for texts, labels in data_iter:  # 对数据集中的每一组数据
                outputs = self.__model(texts)  # 使用模型进行预测
                loss = self.loss_func(outputs, labels)  # 计算模型损失
                loss_total += loss  # 累加模型损失
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)  # 记录标签
                predict_all = np.append(predict_all, predic)  # 记录预测结果

        acc = metrics.accuracy_score(labels_all, predict_all)  # 计算分类误差
        if eval:
            report = metrics.classification_report(
                labels_all, predict_all, 
                target_names=self.config.class_list, 
                digits=4
            )
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)

    def test(self, test_iter):
        self.__model.eval()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(test_iter, eval=True)
        self.logger.info(f'test loss: {test_loss:>0.2f}, test acc: {test_acc:>0.2f}')
        self.logger.info("===== Precision, Recall and F1-Score =====")
        self.logger.info(test_report)
        self.logger.info("===== Confusion Matrix =====")
        self.logger.info(test_confusion)
    
    def select_samples_to_store(self, catagorical=None):
        pass
    
