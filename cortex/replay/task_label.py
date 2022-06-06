#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   label.py
@Time    :   2022/06/01 15:35:22
@Author  :   rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import numpy as np


def task_label(features, labels):
    centroids = {}
    for label in labels:
        # 获取同一类别的特征
        x = [features[i] for i in range(features.shape[0]) if labels[i] == label]
        # 计算类别中心点
        centroid = np.mean(x, axis=0)
        centroids[label] = centroid
    return centroids
    
