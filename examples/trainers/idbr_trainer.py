#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   idbr_trainer.py
@Time    :   2022/05/30 14:53:05
@Author  :   rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import os
import numpy as np
import torch
from copy import deepcopy
from sklearn.cluster import KMeans
from transformers import AdamW, get_constant_schedule_with_warmup

from cortex.utils import Logs


class Trainer(object):
    def __init__(self, config, model, predictor) -> None:
        self.config = config
        self.__model = model
        self.__predictor = predictor
        self.__buffer = Memory(config)
        # loss function
        self.nsp_loss_func = torch.nn.CrossEntropyLoss()
        self.cls_loss_func = torch.nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = AdamW(
            [
                {"params": model.Bert.parameters(), "lr": self.config.bert_learning_rate, "weight_decay": 0.01},
                {"params": model.General_Encoder.parameters(), "lr": self.config.learning_rate, "weight_decay": 0.01},
                {"params": model.Specific_Encoder.parameters(), "lr": self.config.learning_rate, "weight_decay": 0.01},
                {"params": model.cls_classifier.parameters(), "lr": self.config.learning_rate, "weight_decay": 0.01},
                {"params": model.task_classifier.parameters(), "lr": self.config.task_learning_rate, "weight_decay": 0.01},
            ]
        )
        self.optimizer_P = AdamW(
            [
                {"params": predictor.parameters(), "lr": self.config.learning_rate, "weight_decay": 0.01},
            ]
        )

        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, 1000)
        self.scheduler_P = get_constant_schedule_with_warmup(self.optimizer_P, 1000)
        
        self.logger = Logs(config.log_path)
    
    def select_samples_to_store(self, data_iter, task_id):
        cls_list = []
        nsp_list = []
        mask_list = []
        cls_label_list = []
        feat_list = []
        
        self.__model.eval()
        with torch.no_grad():
            for cls, nsp, mask, y in data_iter:
                cls = cls.to(self.config.device)
                nsp = nsp.to(self.config.device)
                mask = mask.to(self.config.device)
                y = y.to(self.config.device)
                _, _, _, _, bert_emb = self.__model(cls, mask)
                cls_list.append(cls.to("cpu"))
                nsp_list.append(nsp.to("cpu"))
                mask_list.append(mask.to("cpu"))
                cls_label_list.append(y.to("cpu"))
                # Kmeans on bert embedding
                feat_list.append(bert_emb.to("cpu"))
        cls_list = torch.cat(cls_list, dim=0).data.cpu().numpy()
        nsp_list = torch.cat(nsp_list, dim=0).data.cpu().numpy()
        mask_list = torch.cat(mask_list, dim=0).data.cpu().numpy()
        cls_label_list = torch.cat(cls_label_list, dim=0).data.cpu().numpy()
        feat_list = torch.cat(feat_list, dim=0).data.cpu().numpy()
        
        # if use KMeans
        #! 只保留了离中心点最近的一个样本
        if self.config.kmeans:  # todo 改成基于类别的，而不是 kmeans
            estimator = KMeans(n_clusters=self.config.n_cluster, random_state=self.config.seed)
            estimator.fit(feat_list)
            label_pred = estimator.labels_
            centroids = estimator.cluster_centers_
            for clu_id in range(self.config.n_cluster):
                index = [i for i in range(len(label_pred)) if label_pred[i] == clu_id]
                closest = float("inf")
                closest_cls = None
                closest_nsp = None
                closest_mask = None
                closest_y = None
                for j in index:
                    dis = np.sqrt(np.sum(np.square(centroids[clu_id] - feat_list[j])))
                    if dis < closest:
                        closest_cls = cls_list[j]
                        closest_nsp = nsp_list[j]
                        closest_mask = mask_list[j]
                        closest_y = cls_label_list[j]
                        closest = dis

                if closest_cls is not None:
                    self.__buffer.append(closest_cls, closest_nsp, closest_mask, closest_y, task_id)
        else:  # todo 蓄水池算法
            permutations = np.random.permutation(len(cls_list))
            index = permutations[:self.config.n_cluster]
            for j in index:
                self.__buffer.append(cls_list[j], nsp_list[j], mask_list[j], cls_label_list[j], task_id)
        self.logger.info(f"Buffer size:{len(self.__buffer)}")
        self.logger.info(self.__buffer.labels)
        b_lbl = np.unique(self.__buffer.labels)
        for i in b_lbl:
            self.logger.info(f"Label {i} in Buffer: {self.__buffer.labels.count(i)}")

    
    def train_step(
        self, 
        cls_x, 
        nsp_x, 
        mask, 
        cls_y, 
        task_y, 
        x_feat, 
        task_id
    ):
        self.__model.train()
        self.__predictor.train()
        self.__model.zero_grad()
        self.__predictor.zero_grad()
        
        # generate nsp_y
        nsp_p = torch.ones(nsp_x.size(0))
        nsp_n = torch.zeros_like(nsp_p)
        nsp_y = torch.cat([nsp_p, nsp_n])
        x = torch.cat([cls_x, nsp_x], dim=0)
        mask = torch.cat([mask, mask], dim=0)
        cls_y = torch.cat([cls_y, cls_y], dim=0)
        task_y = torch.cat([task_y, task_y], dim=0)
        # shuffle
        index = torch.randperm(x.size(0))
        x = x[index, :]
        mask = mask[index, :]
        cls_y = cls_y[index, :]
        task_y = task_y[index, :]
        nsp_y = nsp_y[index, :]
        # input model        
        general_feat, specific_feat, cls_pred, task_pred, _ = self.__model(x, mask)
        # 取出 nsp 正样本特征
        g_fea = general_feat[torch.where(nsp_y == 1)]
        s_fea = specific_feat[torch.where(nsp_y == 1)]
        
        # calculate classification loss
        _, pred_cls = cls_pred.max(1)
        correct_cls = pred_cls.eq(cls_y.view_as(pred_cls)).sum().item()
        cls_loss = self.cls_loss_func(cls_pred, cls_y)

        task_loss = torch.tensor(0.0).to(self.config.device)
        reg_loss = torch.tensor(0.0).to(self.config.device)
        nsp_loss = torch.tensor(0.0).to(self.config.device)
        
        # Calculate regularization loss
        # 第一个任务没有正则 loss
        if x_feat is not None:
            fea_len = g_fea.size(1)
            old_g_fea = x_feat[:, :fea_len]
            old_s_fea = x_feat[:, fea_len:]

            reg_loss += self.config.regspe * torch.nn.functional.mse_loss(s_fea, old_s_fea) + \
                        self.config.reggen * torch.nn.functional.mse_loss(g_fea, old_g_fea)

            if task_id > 0:
                reg_loss *= self.config.regcoe
            else:
                reg_loss *= 0.0
        
        # Calculate task loss only when in replay batch
        task_pred = task_pred[:, :task_id + 1]
        _, pred_task = task_pred.max(1)
        correct_task = pred_task.eq(task_y.view_as(pred_task)).sum().item()
        if task_id > 0:
            task_loss += self.config.tskcoe * self.cls_loss_func(task_pred, task_y)
            
        # Calculate Next Sentence Prediction loss
        nsp_acc = 0.0
        nsp_output = self.__predictor(general_feat)
        nsp_loss += self.config.nspcoe * self.nsp_loss_func(nsp_output, nsp_y)

        _, nsp_pred = nsp_output.max(1)
        nsp_correct = nsp_pred.eq(nsp_y.view_as(nsp_pred)).sum().item()
        nsp_acc = nsp_correct * 1.0 / (self.config.batch_size * 2.0)
        
        loss = cls_loss + task_loss + reg_loss + nsp_loss

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.optimizer_P.step()
        self.scheduler_P.step()

        return nsp_acc, correct_cls, correct_task, nsp_loss.item(), task_loss.item(), cls_loss.item(), reg_loss.item()
        
    def train(self, train_iter, test_iter, dev_iter):
        """
            
        """
        task_num = len(self.config.task_names)
        for task_id in range(task_num):
            data_loader = train_iter[self.config.task_names[task_id]]
            
            best_acc = 0
            best_model = deepcopy(self.__model.state_dict())
            best_predictor = deepcopy(self.__predictor.state_dict())
            
            # store the features outputted by original model
            self.__buffer.store_features(self.__model)
            acc_track = []

            current_buffer = Memory(self.config)
            self.__model.eval()
            self.logger.info(f'Initialization {self.config.task_names[task_id]}: {task_id} buffer ...')
            with torch.no_grad():
                for cls_batch, nsp_batch, mask_batch, y_batch in data_loader:
                    for i in range(self.config.batch_size):
                        current_buffer.append(
                            cls_batch[i], 
                            nsp_batch[i],
                            mask_batch[i], 
                            y_batch[i], 
                            task_id
                        )
            self.logger.info("Start Storing Features...")
            current_buffer.store_features(self.__model)
            
            length = len(current_buffer)
            
            for epoch in range(self.config.epochs):
                self.logger.info(f'========== Epoch: {epoch / self.config.epochs} | Dataset: {self.config.task_names[task_id]} ==========')
                # Training Loss/Accuracy on replaying batches
                cls_losses, reg_losses, nsp_losses = [], [], []
                tsk_accs, cls_accs, nsp_accs = [], [], []
                
                # Training Loss/Accuracy on batches of current task
                cur_cls_losses, cur_reg_losses, cur_nsp_losses = [], [], []
                cur_tsk_accs, cur_cls_accs, cur_nsp_accs = [], [], []
                
                iteration = 1
                for cls_x, nsp_x, mask, cls_y, task_y, origin_feat in current_buffer.get_mini_batch():
                    if iteration % self.config.replay_freq == 0 and task_id > 0:
                        # total_cls_x, total_nsp_x, total_mask, total_y, total_task, total_feat = x, nsp_pos, nsp_neg, mask, y, task, origin_feat
                        for j in range(task_id):
                            old_x, old_nsp, old_mask, old_y, old_task, old_feat = self.__buffer.get_random_batch(j)
                            total_x = torch.cat([old_x, cls_x], dim=0)
                            total_nsp = torch.cat([old_nsp, nsp_x], dim=0)
                            total_mask = torch.cat([old_mask, mask], dim=0)
                            total_y = torch.cat([old_y, total_y], dim=0)
                            total_task = torch.cat([old_task, total_task], dim=0)
                            total_feat = torch.cat([old_feat, total_feat], dim=0)
                        permutation = np.random.permutation(total_x.shape[0])
                        total_x = total_x[permutation, :]
                        total_nsp = total_nsp[permutation, :]
                        total_mask = total_mask[permutation, :]
                        total_y = total_y[permutation]
                        total_task = total_task[permutation]
                        total_feat = total_feat[permutation, :]
                        for j in range(task_id + 1):
                            x = total_x[j * self.config.batch_size: (j + 1) * self.config.batch_size, :].to(self.config.device)
                            mask = total_mask[j * self.config.batch_size: (j + 1) * self.config.batch_size, :].to(self.config.device)
                            nsp = total_nsp[j * self.config.batch_size: (j + 1) * self.config.batch_size, :].to(self.config.device)
                            y = total_y[j * self.config.batch_size: (j + 1) * self.config.batch_size].to(self.config.device)
                            task = total_task[j * self.config.batch_size: (j + 1) * self.config.batch_size].to(self.config.device)
                            feat = total_feat[j * self.config.batch_size: (j + 1) * self.config.batch_size, :].to(self.config.device)
                            nsp_acc, correct_cls, correct_task, nsp_loss, task_loss, cls_loss, reg_loss, = self.train_step(
                                x, nsp, mask, y, task, feat, task_id
                            )
                            # 记录 loss
                            cls_losses.append(cls_loss)
                            reg_losses.append(reg_loss)
                            nsp_losses.append(nsp_loss)

                            tsk_accs.append(correct_task * 0.5 / x.size(0))
                            cls_accs.append(correct_cls * 0.5 / x.size(0))
                            nsp_accs.append(nsp_acc)   
                        
                    else:
                        cls_x, nsp_x, mask = cls_x.to(self.config.device), nsp_x.to(self.config.device), mask.to(self.config.device)
                        cls_y, task_y, origin_feat = cls_y.to(self.config.device), task_y.to(self.config.device), origin_feat.to(self.config.device)
                        pre_acc, correct_cls, correct_task, pre_loss, task_loss, cls_loss, reg_loss = self.train_step(
                            cls_x, nsp_x, mask, cls_y, task_y, origin_feat, task_id
                        )

                        cur_cls_losses.append(cls_loss)
                        cur_reg_losses.append(reg_loss)
                        cur_nsp_losses.append(pre_loss)

                        cur_tsk_accs.append(correct_task * 1.0 / x.size(0))
                        cur_cls_accs.append(correct_cls * 1.0 / x.size(0))
                        cur_nsp_accs.append(pre_acc)

                    if iteration % 200 == 0:
                        avg_acc, acc_list = self.evaluate(dev_iter, task_id)
                        acc_track.append(acc_list)
                        if avg_acc > best_acc:
                            improve = '*'
                            best_acc = avg_acc
                            best_model = deepcopy(self.__model.state_dict())
                            best_predictor = deepcopy(self.__predictor.state_dict())
                        else:
                            improve = ''
                        self.logger.info(f"Iteration {iteration} === avg acc: {avg_acc} {improve}")
                    iteration += 1 
                    
                if len(reg_losses) > 0:
                    self.logger.info(f"Mean REG Loss: {np.mean(reg_losses)}")
                if len(cls_losses) > 0:
                    self.logger.info(f"Mean CLS Loss: {np.mean(cls_losses)}")
                if len(nsp_losses) > 0:
                    self.logger.info(f"Mean PRE Loss: {np.mean(nsp_losses)}")

                if len(tsk_accs) > 0:
                    self.logger.info(f"Mean TSK Acc: {np.mean(tsk_accs)}")
                if len(cls_accs) > 0:
                    self.logger.info(f"Mean LBL Acc: {np.mean(cls_accs)}")
                if len(nsp_accs) > 0:
                    self.logger.info(f"Mean PRE Acc: {np.mean(nsp_accs)}")

                if len(cur_cls_losses) > 0:
                    self.logger.info(f"Mean Current CLS Loss: {np.mean(cur_cls_losses)}")
                if len(cur_reg_losses) > 0:
                    self.logger.info(f"Mean Current REG Loss: {np.mean(cur_reg_losses)}")
                if len(cur_nsp_losses) > 0:
                    self.logger.info(f"Mean PRE Loss: {np.mean(cur_nsp_losses)}")

                if len(cur_tsk_accs) > 0:
                    self.logger.info(f"Mean Current TSK Acc: {np.mean(cur_tsk_accs)}")
                if len(cur_cls_accs) > 0:
                    self.logger.info(f"Mean Current LBL Acc: {np.mean(cur_cls_accs)}")
                if len(cur_nsp_accs) > 0:
                    self.logger.info(f"Mean Current PRE Acc: {np.mean(cur_nsp_accs)}")

            if len(acc_track) > 0:
                self.logger.info(f"ACC Track: {acc_track}")    
            if self.config.select_best:
                self.__model.load_state_dict(deepcopy(best_model))
                self.__predictor.load_state_dict(deepcopy(best_predictor))
            avg_acc, _ = self.evaluate(test_iter, task_id)
            self.logger.info(f'Best avg acc on task {self.config.task_names[task_id]}: {avg_acc}')            
            
            # save best model
            if self.config.dump is True:
                task_name = self.config.task_names[task_id]
                path = os.path.join(self.config.checkpoint_path, task_name)
                torch.save(self.__model, f'{path}.ckpt')
                
            # select samples and save
            self.select_samples_to_store()
        
    
    def evaluate(self, data_iter, task_id, checkpoint_path=None, eval=False):
        if checkpoint_path:
            self.__model.load_state_dict(torch.load(checkpoint_path))
        self.__model.eval()
        acc_list = []
        with torch.no_grad():
            avg_acc = 0.0
            for i in range(task_id + 1):
                dev_loader = data_iter[self.config.task_names[i]]
                total = 0
                correct = 0
                for cls_batch, nsp_batch, mask_batch, y_batch in dev_loader:
                    x = cls_batch.to(self.config.device)
                    nsp = nsp_batch.to(self.config.device)
                    mask = mask_batch.to(self.config.device)
                    y = y_batch.to(self.config.device)
                    batch_size = x.size(0)
                    g_fea, s_fea, cls_pred, _, _ = self.__model(x, mask)
                    _, pred_cls = cls_pred.max(1)
                    correct += pred_cls.eq(y.view_as(pred_cls)).sum().item()
                    total += batch_size
                self.logger.info(f"Acc on task {self.config.task_names[i]} ({i}) : {correct * 100.0 / total}")
                avg_acc += correct * 100.0 / total
                acc_list.append(correct * 100.0 / total)

        return avg_acc / (task_id + 1), acc_list
    
    def test(self, test_iter):
        pass
    
    
class Memory(object):
    def __init__(self, config) -> None:
        self.config = config
        
        self.cls_examples = []
        self.nsp_examples = []
        self.masks = []
        self.labels = []
        self.tasks = []
        self.features = []
    
    def append(self, cls_example, nsp_example, mask, label, task):
        self.cls_examples.append(cls_example)
        self.nsp_examples.append(nsp_example)
        self.masks.append(mask)
        self.labels.append(label)
        self.tasks.append(task)
    
    def store_features(self, model):
        """
        store previous features before trained on new class

        Args:
            model: The model trained just after previous task
        """
        self.features = []
        length = len(self.labels)
        model.eval()
        with torch.no_grad():
            for i in range(length):
                x = torch.tensor(self.examples[i]).view(1, -1).to(self.config.device)
                mask = torch.tensor(self.masks[i]).view(1, -1).to(self.config.device)
                g_fea, s_fea, _, _, _ = model(x, mask)
                fea = torch.cat([g_fea, s_fea], dim=1).view(-1).data.cpu().numpy()
                self.features.append(fea)
        print(f'Length of features stored: {len(self.features)}')
        print(f'Length of labels stored: {len(self.labels)}')
    
    def get_random_batch(self, task_id=None):
        if task_id is None:
            permutations = np.random.permutation(len(self.labels))
            index = permutations[:self.config.batch_size]
            mini_cls_examples = [self.cls_examples[i] for i in index]
            mini_nsp_examples = [self.nsp_examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            np.random.shuffle(index)
            index = index[:self.config.batch_size]
            mini_cls_examples = [self.cls_examples[i] for i in index]
            mini_nsp_examples = [self.nsp_examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
        batch = (
            torch.tensor(mini_cls_examples),
            torch.tensor(mini_nsp_examples),
            torch.tensor(mini_masks),
            torch.tensor(mini_labels),
            torch.tensor(mini_tasks),
            torch.tensor(mini_features)
        )
        return batch
    
    def get_mini_batch(self):
        length = len(self.labels)
        permutations = np.random.permutation(length)
        for s in range(0, length, self.config.batch_size):
            if s + self.config.batch_size >= length:
                break
            index = permutations[s:s + self.config.batch_size]
            mini_cls_examples = [self.cls_examples[i] for i in index]
            mini_nsp_examples = [self.nsp_examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
            mini_features = [self.features[i] for i in index]
            batch = (
                torch.tensor(mini_cls_examples),
                torch.tensor(mini_nsp_examples),
                torch.tensor(mini_masks),
                torch.tensor(mini_labels),
                torch.tensor(mini_tasks),
                torch.tensor(mini_features)
            )
            yield batch
    
    def __len__(self):
        return len(self.labels)
    