#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

from models import Nets
from models.Nets import MLP, CNNCifar


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    loss_func: CrossEntropyLoss

    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.local_model = CNNCifar(args=args).to(args.device)
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradients = []
        gradients = torch.cat([torch.reshape(param.grad, (-1,)) for param in net.parameters()]).clone().detach()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradients



    def backdoor(self, net):

        for name, param in net.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)
        pos = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])
        self.local_model.train()
        epoch_loss = []
        for e in range(self.args.local_ep):
            batch_loss = []
            for batch_id, batch in enumerate(self.ldr_train):
                data, target = batch
                for k in range(8):
                    img = data[k].numpy()
                    for i in range(0, len(pos)):
                        img[0][pos[i][0]][pos[i][1]] = 1.0
                        img[1][pos[i][0]][pos[i][1]] = 0
                        img[2][pos[i][0]][pos[i][1]] = 0
                    target[k] = 2
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)

                class_loss = torch.nn.functional.cross_entropy(output, target)
                dist_loss = Nets.model_norm(self.local_model, net)
                loss = 0.5 * class_loss + 0.5 * dist_loss
                loss.backward()
                batch_loss.append(loss.item())
                optimizer.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = 2 * (data - net.state_dict()[name]) + net.state_dict()[name]
        net.load_state_dict(diff)
        return net.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def train_grad(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradients = []

        for param in net.parameters():
            if param.requires_grad:
                # Clone the gradients to avoid any potential in-place changes
                gradients.append(param.grad.clone())

        return gradients
    # def train(self, net):
    #     net.train()
    #     # train and update
    #     optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    #
    #     epoch_loss = []
    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #             images, labels = images.to(self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             log_probs = net(images)
    #             loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             if self.args.verbose and batch_idx % 10 == 0:
    #                 print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     iter, batch_idx * len(images), len(self.ldr_train.dataset),
    #                            100. * batch_idx / len(self.ldr_train), loss.item()))
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #     gradients = []
    #
    #     for param in net.parameters():
    #         if param.requires_grad:
    #             # Clone the gradients to avoid any potential in-place changes
    #             gradients.append(param.grad.clone())
    #
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradients
    def train_agnews(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        gradients = []
        gradients = torch.cat([torch.reshape(param.grad, (-1,)) for param in net.parameters()]).clone().detach()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), gradients