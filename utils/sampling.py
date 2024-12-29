#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import torch
from torchvision import datasets, transforms


def sample_dirichlet(num_classes, alpha):
    """从迪利克雷分布中采样"""
    return np.random.dirichlet([alpha] * num_classes)


def mnist_noniid_ld(dataset, num_users, alpha):
    num_classes = 10  # MNIST有10个类别
    num_items_per_user = len(dataset) // num_users  # 每个用户分配的样本数量
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 记录每个类别的索引
    class_indices = {i: np.where(dataset.train_labels.numpy() == i)[0] for i in range(num_classes)}

    # 为每个用户分配样本
    for user in range(num_users):
        # 为每个用户生成独立的标签分布
        label_distribution = sample_dirichlet(num_classes, alpha)
        samples_per_class = (num_items_per_user * label_distribution).astype(int)

        for class_id in range(num_classes):
            # 从每个类别中随机选择样本
            if samples_per_class[class_id] > 0:  # 确保样本数量大于0
                selected_indices = np.random.choice(class_indices[class_id], samples_per_class[class_id], replace=False)
                dict_users[user] = np.concatenate((dict_users[user], selected_indices), axis=0)

    return dict_users

def cifar_noniid_ld(dataset, num_users, alpha):
    num_classes = 10  # CIFAR10有10个类别
    num_items_per_user = len(dataset) // num_users  # 每个用户分配的样本数量
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 记录每个类别的索引
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)}

    # 为每个用户分配样本
    for user in range(num_users):
        # 为每个用户生成独立的标签分布
        label_distribution = sample_dirichlet(num_classes, alpha)
        samples_per_class = (num_items_per_user * label_distribution).astype(int)

        for class_id in range(num_classes):
            # 从每个类别中随机选择样本
            if samples_per_class[class_id] > 0:  # 确保样本数量大于0
                selected_indices = np.random.choice(class_indices[class_id], samples_per_class[class_id], replace=False)
                dict_users[user] = np.concatenate((dict_users[user], selected_indices), axis=0)

    return dict_users

def fmnist_noniid_ld(dataset, num_users, alpha):
    num_classes = 10  # Fashion MNIST有10个类别
    num_items_per_user = len(dataset) // num_users  # 每个用户分配的样本数量
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 记录每个类别的索引
    class_indices = {i: np.where(dataset.train_labels.numpy() == i)[0] for i in range(num_classes)}

    # 为每个用户分配样本
    for user in range(num_users):
        # 为每个用户生成独立的标签分布
        label_distribution = sample_dirichlet(num_classes, alpha)
        samples_per_class = (num_items_per_user * label_distribution).astype(int)

        for class_id in range(num_classes):
            # 从每个类别中随机选择样本
            if samples_per_class[class_id] > 0:  # 确保样本数量大于0
                selected_indices = np.random.choice(class_indices[class_id], samples_per_class[class_id], replace=False)
                dict_users[user] = np.concatenate((dict_users[user], selected_indices), axis=0)

    return dict_users

def cifar100_noniid_ld(dataset, num_users, alpha):
    num_classes = 100  # CIFAR100有100个类别
    num_items_per_user = len(dataset) // num_users  # 每个用户分配的样本数量
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 记录每个类别的索引
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)}

    # 为每个用户分配样本
    for user in range(num_users):
        # 为每个用户生成独立的标签分布
        label_distribution = sample_dirichlet(num_classes, alpha)
        samples_per_class = (num_items_per_user * label_distribution).astype(int)

        for class_id in range(num_classes):
            # 从每个类别中随机选择样本
            if samples_per_class[class_id] > 0:  # 确保样本数量大于0
                selected_indices = np.random.choice(class_indices[class_id], samples_per_class[class_id], replace=False)
                dict_users[user] = np.concatenate((dict_users[user], selected_indices), axis=0)

    return dict_users


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def mnist_noniid_lq(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_classes = 10  # MNIST有10个类别
    num_parties = 10  # 将客户端分为10个parties
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 记录每个类别的索引
    class_indices = {i: np.where(dataset.train_labels.numpy() == i)[0] for i in range(num_classes)}

    # 将客户端随机分为10个parties
    parties = np.array_split(np.random.permutation(num_users), num_parties)

    # 为每个party随机选择一个标签
    for party in parties:
        selected_label = np.random.choice(num_classes)  # 随机选择一个标签
        indices = class_indices[selected_label]  # 获取该标签的所有索引

        # 将该标签的数据均分到party内的客户端
        num_clients_in_party = len(party)
        num_samples_per_client = len(indices) // num_clients_in_party  # 每个客户端分配的样本数量

        for client in party:
            # 从该标签的索引中随机选择样本
            selected_indices = np.random.choice(indices, num_samples_per_client, replace=False)
            dict_users[client] = np.concatenate((dict_users[client], selected_indices), axis=0)

            # 更新 indices，确保不重复选择
            indices = list(set(indices) - set(selected_indices))

    return dict_users


def mnist_noniid_q(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_classes = 10  # MNIST有10个类别
    num_parties = 10  # 将客户端分为10个parties
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 记录每个类别的索引
    class_indices = {i: np.where(dataset.train_labels.numpy() == i)[0] for i in range(num_classes)}

    # 找出最少类别的样本数量
    min_samples = min(len(class_indices[i]) for i in range(num_classes))

    # 将客户端随机分为10个parties
    parties = np.array_split(np.random.permutation(num_users), num_parties)

    # 为每个party随机选择一个标签
    for party in parties:
        selected_label = np.random.choice(num_classes)  # 随机选择一个标签
        indices = class_indices[selected_label]  # 获取该标签的所有索引

        # 确保从每个类别中选择的样本数量不超过最小样本数量
        num_samples_per_client = min(min_samples // len(party), len(indices) // len(party))  # 每个客户端分配的样本数量

        # 从该标签的索引中随机选择样本
        selected_indices = np.random.choice(indices, num_samples_per_client * len(party), replace=False)

        # 将样本均匀分配到party内的客户端
        for client in party:
            client_indices = selected_indices[:num_samples_per_client]
            dict_users[client] = np.concatenate((dict_users[client], client_indices), axis=0)
            selected_indices = selected_indices[num_samples_per_client:]  # 更新剩余样本

    return dict_users
# def mnist_noniid_lq(dataset, num_users, alpha=0.1):
#     num_classes = 10  # MNIST有10个类别
#     total_samples = len(dataset)  # 数据集总样本数量
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     # 记录每个类别的索引
#     class_indices = {i: np.where(dataset.train_labels.numpy() == i)[0] for i in range(num_classes)}
#
#     # 生成每个客户端的数据量占比
#     data_distribution = sample_dirichlet(num_users, alpha)  # 生成占比
#     samples_per_user = (total_samples * data_distribution).astype(int)  # 计算每个客户端的样本数量
#
#     # 确保每个客户端至少分配一个样本
#     for i in range(num_users):
#         if samples_per_user[i] == 0:
#             samples_per_user[i] = 1  # 将样本数量设置为1
#             # 从其他客户端中减少样本数量以保持总样本数量不变
#             for j in range(num_users):
#                 if j != i and samples_per_user[j] > 0:
#                     samples_per_user[j] -= 1
#                     break
#
#     # 为每个用户分配样本
#     for user in range(num_users):
#         remaining_samples = samples_per_user[user]  # 当前用户应分配的样本数量
#
#         for class_id in range(num_classes):
#             # 从每个类别中随机选择样本
#             if remaining_samples > 0 and len(class_indices[class_id]) > 0:  # 确保样本数量大于0且类别索引不为空
#                 # 计算当前类别可以分配的样本数量
#                 available_samples = min(remaining_samples, len(class_indices[class_id]))
#                 selected_indices = np.random.choice(class_indices[class_id], available_samples, replace=False)
#                 dict_users[user] = np.concatenate((dict_users[user], selected_indices), axis=0)
#
#                 # 更新剩余样本数量
#                 remaining_samples -= available_samples
#
#                 # 更新类别索引，确保不重复选择
#                 class_indices[class_id] = list(set(class_indices[class_id]) - set(selected_indices))
#
#     return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    # labels = np.array(dataset.train_labels)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cifar_noniid_lq(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_classes = 10  # CIFAR10有10个类别
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # 记录每个类别的索引
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)}

    # 将客户端随机分为10个parties
    parties = np.array_split(np.random.permutation(num_users), 10)

    # 为每个party随机选择一个标签
    for party in parties:
        selected_label = np.random.choice(num_classes)  # 随机选择一个标签
        indices = class_indices[selected_label]  # 获取该标签的所有索引

        # 将该标签的数据均分到party内的客户端
        num_clients_in_party = len(party)
        num_samples_per_client = len(indices) // num_clients_in_party  # 每个客户端分配的样本数量

        for client in party:
            # 从该标签的索引中随机选择样本
            selected_indices = np.random.choice(indices, num_samples_per_client, replace=False)
            dict_users[client] = np.concatenate((dict_users[client], selected_indices), axis=0)

            # 更新 indices，确保不重复选择
            indices = list(set(indices) - set(selected_indices))

    return dict_users

def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
# def cifar100_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 250
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     # labels = dataset.train_labels.numpy()
#     # labels = np.array(dataset.train_labels)
#     labels = np.array(dataset.targets)
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
#     return dict_users
def fmnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
def fmnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from Fashion_MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users
def emnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def agnews_iid(dataset, num_users):
    """
    Sample I.I.D. client data from AG_NEWS dataset
    :param dataset: Dataset object containing AG_NEWS data
    :param num_users: Total number of clients
    :return: dict of indices for each client
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_iid_noise_2(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset and add label noise to selected clients
    with varying degrees of noise levels.
    :param dataset: MNIST dataset
    :param num_users: Total number of clients
    :return: dict of image index for each client
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    # Define noise levels for clients
    noise_levels = {

        3: 0.5,

        6: 0.7,

        9: 0.9,

    }

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # Add noise to selected clients
    for client, noise_level in noise_levels.items():
        indices = list(dict_users[client])  # -1 because client IDs start from 1, but list index from 0

        num_noisy_labels = int(len(indices) * noise_level)

        # Randomly select indices to modify
        noisy_indices = np.random.choice(indices, num_noisy_labels, replace=False)

        for idx in noisy_indices:
            # Randomly change the label


            new_label = np.random.randint(0, 10)
            # Ensure the new label is different from the original

            dataset.targets[idx] = new_label

    return dict_users

def add_noise_to_client(dataset, dict_users, client):
    """
    Add label noise to a selected client's data with a certain probability
    :param dataset: MNIST dataset
    :param dict_users: dict of image index for each client
    :param client: the client ID to add noise
    :param probability: probability of adding noise in this round
    :param noise_ratio: ratio of data to be noised for the client
    """
    # Check if the client is selected based on the probability

    # indices = list(dict_users[client])
    # num_noisy_items = int(len(indices) * 0.3)
    # noisy_indices = np.random.choice(indices, num_noisy_items, replace=False)
    #
    # for idx in noisy_indices:
    #     # Randomly change the label
    #     dataset.targets[idx] = np.random.randint(0, 10)  # Assuming 10 classes in MNIST
    indices = list(dict_users[client])

    for idx in indices:
        # Change the label to 9 - k
        current_label = dataset.targets[int(idx)]
        new_label = 9 - current_label
        dataset.targets[int(idx)] = new_label

def add_noise_to_client_shuffle(dataset, dict_users, client):
    indices = list(dict_users[client])
    # 获取当前客户端的所有标签
    current_labels = [dataset.targets[int(idx)] for idx in indices]
    # 随机打乱标签
    shuffled_labels = np.random.permutation(current_labels)
    # 将打乱的标签分配回数据
    for idx, new_label in zip(indices, shuffled_labels):
        dataset.targets[int(idx)] = new_label

def add_gaussian_noise(dataset, dict_users, client):
    """
    Add Gaussian noise to the dataset of a specific client.

    :param dataset: The entire dataset.
    :param dict_users: A dictionary where keys are client IDs and values are lists of data indices.
    :param client: The client ID to which noise should be added.
    :return: The modified dataset with noise added to the specified client's data.
    """
    new_train_dataset = []
    indices = list(dict_users[client])

    for idx in range(len(dataset)):
        feature, label = dataset[idx]

        # Check if the current index belongs to the specified client
        if idx in indices:
            noise = torch.tensor(np.random.normal(0, 5, feature.shape))
            noise = noise.to(torch.float32)
            new_data = feature + noise
            clip_data = torch.clamp(new_data, -1, 1)
            new_train_dataset.append((clip_data, label))
        else:
            # If the index does not belong to the specified client, keep the data unchanged
            new_train_dataset.append((feature, label))

    return new_train_dataset




def add_noise_to_client_2(dataset, dict_users, client):
    """
    Add label noise to a selected client's data with a certain probability
    :param dataset: MNIST dataset
    :param dict_users: dict of image index for each client
    :param client: the client ID to add noise
    :param probability: probability of adding noise in this round
    :param noise_ratio: ratio of data to be noised for the client
    """
    # Check if the client is selected based on the probability

    # indices = list(dict_users[client])
    # num_noisy_items = int(len(indices) * 0.3)
    # noisy_indices = np.random.choice(indices, num_noisy_items, replace=False)
    #
    # for idx in noisy_indices:
    #     # Randomly change the label
    #     dataset.targets[idx] = np.random.randint(0, 10)  # Assuming 10 classes in MNIST
    indices = list(dict_users[client])

    for idx in indices:
        # Change the label to 9 - k
        current_label = dataset.targets[idx]
        new_label = 99 - current_label
        dataset.targets[idx] = new_label



def mnist_iid_noise_30(dataset, num_users, num_noisy_users=20):
    """
    Sample I.I.D. client data from MNIST dataset and add label noise to random selected clients
    :param dataset: MNIST dataset
    :param num_users: Total number of clients
    :param num_noisy_users: Number of clients to add noise
    :return: dict of image index for each client and a list of noisy clients
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # Randomly select clients to add noise
    noisy_clients = np.random.choice(range(num_users), num_noisy_users, replace=False)

    for client in noisy_clients:
        # Access the data indices for the selected noisy client
        indices = list(dict_users[client])
        # Determine the number of items to noise

        num_noisy_items = int(len(indices) * 0.3)  # 30% of the client's data

        # Randomly select 10% of the data indices for this client to add noise
        noisy_indices = np.random.choice(indices, num_noisy_items, replace=False)

        # For each selected index, modify the label in the dataset
        for idx in noisy_indices:
            # Randomly change the label
            dataset.targets[idx] = np.random.randint(0, 10)  # Assuming 10 classes in MNIST

    return dict_users, noisy_clients

def mnist_iid_noise_20(dataset, num_users, num_noisy_users=20):
    """
    Sample I.I.D. client data from MNIST dataset and add label noise to random selected clients
    :param dataset: MNIST dataset
    :param num_users: Total number of clients
    :param num_noisy_users: Number of clients to add noise
    :return: dict of image index for each client and a list of noisy clients
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # Randomly select clients to add noise
    noisy_clients = np.random.choice(range(num_users), num_noisy_users, replace=False)

    for client in noisy_clients:
        # Access the data indices for the selected noisy client
        indices = list(dict_users[client])
        # Determine the number of items to noise

        num_noisy_items = int(len(indices) * 0.2)  # 20% of the client's data

        # Randomly select 10% of the data indices for this client to add noise
        noisy_indices = np.random.choice(indices, num_noisy_items, replace=False)

        # For each selected index, modify the label in the dataset
        for idx in noisy_indices:
            # Randomly change the label
            dataset.targets[idx] = np.random.randint(0, 10)  # Assuming 10 classes in MNIST

    return dict_users, noisy_clients

def mnist_iid_noise_10(dataset, num_users, num_noisy_users=20):
    """
    Sample I.I.D. client data from MNIST dataset and add label noise to random selected clients
    :param dataset: MNIST dataset
    :param num_users: Total number of clients
    :param num_noisy_users: Number of clients to add noise
    :return: dict of image index for each client and a list of noisy clients
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    # Randomly select clients to add noise
    noisy_clients = np.random.choice(range(num_users), num_noisy_users, replace=False)

    for client in noisy_clients:
        # Access the data indices for the selected noisy client
        indices = list(dict_users[client])
        # Determine the number of items to noise

        num_noisy_items = int(len(indices) * 0.1)  # 10% of the client's data

        # Randomly select 10% of the data indices for this client to add noise
        noisy_indices = np.random.choice(indices, num_noisy_items, replace=False)

        # For each selected index, modify the label in the dataset
        for idx in noisy_indices:
            # Randomly change the label
            dataset.targets[idx] = np.random.randint(0, 10)  # Assuming 10 classes in MNIST

    return dict_users, noisy_clients



# def mnist_iid_noise(dataset, num_users, num_noisy_users=3):
#     """
#     Sample I.I.D. client data from MNIST dataset and add label noise to random selected clients
#     using specified mappings.
#     :param dataset: MNIST dataset
#     :param num_users: Total number of clients
#     :param num_noisy_users: Number of clients to add noise
#     :return: dict of image index for each client and a list of noisy clients
#     """
#     num_items = int(len(dataset) / num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#
#     # Define the label mappings for noisy clients
#     mappings = [
#         {1: 0, 5: 2, 3: 4, 9: 6, 7: 8},
#         {1: 2, 3: 0, 4: 6, 5: 9, 7: 8},
#         {1: 5, 2: 4, 3: 6, 9: 7, 8: 0}
#     ]
#
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#
#     # Randomly select clients to add noise
#     noisy_clients = np.random.choice(range(num_users), num_noisy_users, replace=False)
#
#     for i, client in enumerate(noisy_clients):
#         # Access the data indices for the selected noisy client
#         indices = list(dict_users[client])
#
#         # Get the mapping for the current noisy client
#         mapping = mappings[i % len(mappings)]
#
#         # For each index, modify the label in the dataset according to the mapping
#         for idx in indices:
#             original_label = dataset.targets[idx]
#             # Change the label based on the mapping if the original label is in the mapping keys
#             if original_label in mapping:
#                 dataset.targets[idx] = mapping[original_label]
#
#     return dict_users, noisy_clients

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    # d = mnist_noniid(dataset_train, num)
