import tools
import math
import numpy as np
import torch

import torch


def fedsgd( gradients):
    grads = torch.stack(gradients)  # 将梯度列表堆叠成一个新的张量
    grad = torch.mean(grads, dim=0)  # 计算所有梯度的平均值
    return grad
