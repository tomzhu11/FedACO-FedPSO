import torch
from torch import nn, autograd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

from models import Nets
from models.Nets import MLP, CNNCifar


import numpy as np
from torch.utils.data import Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, shuffle=False):
        self.dataset = dataset
        self.idxs = list(idxs)  # 保证idxs是列表形式
        if shuffle:
            np.random.shuffle(self.idxs)  # 使用numpy来打乱索引

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # 通过打乱（或未打乱）的索引获取数据集中的元素
        image, label = self.dataset[self.idxs[item]]
        return image, label



class LocalUpdate_sgd(object):
    def __init__(self, args, idx1,dataset, idxs, shuffle):
        self.args = args
        self.dataset = DatasetSplit(dataset, idxs,shuffle)
        self.loss_func = nn.CrossEntropyLoss()
        self.local_model = CNNCifar(args=args).to(args.device)
        self.batch_idx = idx1
    def train_sgd(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # 计算起始和结束索引
        start_idx = self.batch_idx * self.args.local_bs
        end_idx = start_idx + self.args.local_bs

        # 获取数据片段
        samples = [self.dataset[i] for i in range(start_idx, min(end_idx, len(self.dataset)))]
        images, labels = zip(*samples)
        images = torch.stack(images).to(self.args.device)
        labels = torch.tensor(labels).to(self.args.device)

        # 训练过程
        net.zero_grad()
        log_probs = net(images)
        loss = self.loss_func(log_probs, labels)
        loss.backward()
        optimizer.step()

        gradients = torch.cat([torch.reshape(param.grad, (-1,)) for param in net.parameters()]).clone().detach()

        return net.state_dict(), loss.item(), gradients
