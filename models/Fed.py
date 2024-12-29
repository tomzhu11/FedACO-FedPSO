#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg
import copy
import torch


def FedAvg_0(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        weighted_sum = torch.zeros_like(w_avg[k])
        for i in range(len(w)):
            weighted_sum += w[i][k]
        w_avg[k] = weighted_sum / len(w)
    return w_avg


def FedAvg(w, data_sizes,w_glob_pre):
    """
    Perform Federated Averaging with weighted average based on local data sizes.

    :param w: List of model weights from each client
    :param data_sizes: List of the number of local data samples for each client
    :return: Averaged model weights
    """
    if len(w) == 0:
        return w_glob_pre
    else:
        w_avg = copy.deepcopy(w[0])
        total_data_size = sum(data_sizes)  # 计算所有客户端的总数据量

        for k in w_avg.keys():
            # 计算加权平均
            weighted_sum = w_avg[k] * data_sizes[0]  # 初始化为第一个客户端的权重
            for i in range(1, len(w)):
                weighted_sum += w[i][k] * data_sizes[i]  # 加权累加
            w_avg[k] = torch.div(weighted_sum, total_data_size)  # 除以总数据量

        return w_avg