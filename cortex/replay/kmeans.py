#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   kmeans.py
@Time    :   2022/06/01 15:19:26
@Author  :   csong-idea
@Email   :   songchao@idea.edu.cn
@Copyright : International Digital Economy Academy (IDEA)
'''


from sklearn.cluster import KMeans


def kmeans(features, n_cluster, seed):
    estimator = KMeans(n_clusters=n_cluster, random_state=seed)
    estimator.fit(features)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    return label_pred, centroids
