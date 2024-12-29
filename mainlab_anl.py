import os
import time
from utils.geo_med import geometric_median_list_of_array
from utils.sampling2 import FashionMnist_noniid
import math
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import collections
from models.Uapattack import UAPAttack
from models.bulyan import Bulyan
from models.byzmean_attack import byzMean_attack
from models.fedsgd import fedsgd

from models.krum import krum
from models.lie_attack import little_is_enough_attack
from models.median import simple_median
from models.trimmed_mean import trimmed_mean
from models.updatesgd import LocalUpdate_sgd
from utils import tool
from utils.data_loader import get_dataset
from utils.dataset2 import ShakeSpeare, FEMNIST
from utils.running import byzantineWorker, benignWorker
from utils.shapley import Shapley
from utils.tools import avgSV_weights, arms_selection

os.environ["OMP_NUM_THREADS"] = '1'

import matplotlib
from utils.w_noise import add_gaussian_noise_to_weights, add_noise_to_layers_weights, random_weights

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import Dataset
from collections import Counter
import copy
import numpy as np
import random
from torchvision import datasets, transforms

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_iid_noise_30, mnist_iid_noise_20, \
    mnist_iid_noise_10, add_noise_to_client, fmnist_iid, agnews_iid, emnist_iid, add_noise_to_client_2, fmnist_noniid, \
    cifar_noniid, cifar100_iid, add_gaussian_noise
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar, FashionMNISTCNN, CharLSTM, CNNFemnist, ResnetEmnist, ResNet, resnet18
from models.Fed import FedAvg, FedAvg_0
from models.test import test_img

#from torchtext.datasets import AG_NEWS
#标准差
import  os
import torch
import random
import numpy as np

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_transition_prob(tau, weights, alpha, beta, visited, current_node, neighbors):
    """
    计算从当前节点到每个邻居节点的转移概率。

    Args:
        tau (numpy.ndarray): 信息素矩阵。
        weights (numpy.ndarray): 权重矩阵。
        alpha (float): 信息素的重要性系数。
        beta (float): 权重的重要性系数。
        visited (set): 已访问节点集合。
        current_node (int): 当前节点。
        neighbors (set): 当前节点的邻居节点集合。

    Returns:
        dict: 从当前节点到每个邻居节点的转移概率。
    """
    prob = {}
    for j in neighbors:
        if j not in visited and weights[current_node, j] > 0:  # 检查权重是否大于 0
            prob[j] = (tau[current_node, j] ** alpha) * (weights[current_node, j] ** beta)

    total_prob = sum(prob.values())
    if total_prob > 0:  # 避免归一化时除以 0
        for j in prob:
            prob[j] /= total_prob
    else:
        prob = {j: 0 for j in neighbors}  # 若总概率为 0，则所有邻居概率设置为 0

    return prob


def sigmoid(delta, kappa=1.0):
    return 2 / (1 + math.exp(-kappa * delta)) - 1

if __name__ == '__main__':
    # parse args
    seed = 5
    setup_seed(seed)
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)


        if args.method == 'fedaoc' or 'fedpso':
            dataset_test, valid_dataset = torch.utils.data.random_split(dataset_test, [8000, 2000])

        if args.mnist_iid_noise_30:
            dataset_train_noise = copy.deepcopy(dataset_train)
            dict_users, noisy_clients = mnist_iid_noise_30(dataset_train, args.num_users)
            print("Noisy clients:", noisy_clients)
        elif args.mnist_iid_noise_20:
            dataset_train_noise = copy.deepcopy(dataset_train)
            dict_users, noisy_clients = mnist_iid_noise_20(dataset_train, args.num_users)
            print("Noisy clients:", noisy_clients)
        elif args.mnist_iid_noise_10:
            dataset_train_noise = copy.deepcopy(dataset_train)
            dict_users, noisy_clients = mnist_iid_noise_10(dataset_train, args.num_users)
            print("Noisy clients:", noisy_clients)
        else:
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)


        if args.method == 'fedaoc' or 'fedpso':
            dataset_test, valid_dataset = torch.utils.data.random_split(dataset_test, [8000, 2000])

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)

    elif args.dataset == 'fmnist':
        dataset_train = datasets.FashionMNIST('data', train=True, download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                              ]))
        dataset_test = datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))


        if args.method == 'fedaoc' or 'fedpso':
            dataset_test, valid_dataset = torch.utils.data.random_split(dataset_test, [8000, 2000])

        if args.iid:
            dict_users = fmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = FashionMnist_noniid(dataset_train, args.num_users)

    # 模型初始化
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar100':
        net_glob = resnet18().to(args.device)
    elif args.dataset == 'fmnist' and args.model == 'cnn':
        net_glob = FashionMNISTCNN(args=args).to(args.device)
    elif args.dataset == 'emnist' and args.model == 'resnet':
        net_glob = ResnetEmnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        from models.Nets import MLP

        img_size = dataset_train[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

    else:
        print('Error: unrecognized model')
        exit()


    def euclidean_distance(local_weights, global_weights):
        distance = 0
        for key in global_weights.keys():
            distance += torch.pow(local_weights[key] - global_weights[key], 2).sum()
        distance = torch.sqrt(distance)
        return distance.item()

    print(net_glob)
    net_glob.train()

    setup_seed(42)

    #蚂蚁的参数
    max_aco_iters = 10
    num_ants = 20
    alpha, beta = 1.0, 2.0
    rho = 0.1
    eta = 1.0
    tau0 = 0.1

    # 初始化粒子群参数
    num_particles = 20  # 粒子数量
    num_clients = args.num_users  # 客户端数量
    max_iters = 10  # 最大迭代次数
    w = 0.5  # 惯性权重
    c1, c2 = 1.5, 1.5  # 个体学习因子和群体学习因子

    total_time=[]
    w_glob = net_glob.state_dict()

    global_acctotal = []
    global_losses = []
    control_chart_params = {}
    excluded_clients = []
    distances_matrix = [[] for _ in range(args.num_users)]
    normal_dis = [[] for _ in range(args.num_users)]
    standard_dis = []
    std = []
    client_means = []
    r = []
    p = []
    o = 1
    ratio = []

    save_dir = f"./si_{args.method}_{args.dataset}_{args.attack_pattern}_{args.peer_round}"
    if not args.iid:
        save_dir = f"./si_{args.method}_{args.dataset}_{args.attack_pattern}_{args.peer_round}_noniid"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.method == 'fedaoc':
        tau = np.full((args.num_users, args.num_users), tau0)
        weights = np.random.rand(args.num_users, args.num_users)

    seduce = []
    if args.attack:
        noisy_clients = np.random.choice(range(args.num_users), int(args.peer_round*args.num_users), replace=False)
        print("noisy_clients:", noisy_clients)

    for iters in range(args.epochs):
        normal_id = []
        w_locals = []
        local_grads = []
        loss_locals = []
        loss_tests = []
        normal_clients_dis = []
        normal_clients_dis_mean = []
        normal_std = []
        excluded = []
        distances_matrix_this_round = []
        noisy_this_round = []
        attacker = []
        grads = []
        benign_grad = []
        ra=0
        byz_grad = []

        # if args.attack_pattern == 'byzmean':
        #     w_locals = train_parallel(args, net_glob, train_loader, optimizer, noisy_clients)


        if args.attack_pattern == 'lie':
            for idx in range(args.num_users):
                if idx in noisy_clients:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    byz_grad.append(grad)
                    local_model = copy.deepcopy(net_glob).to(args.device)
                    local_model.load_state_dict(w)
                    w_locals.append(copy.deepcopy(w))
                    continue
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))
                benign_grad.append(grad)
                local_model = copy.deepcopy(net_glob).to(args.device)
                local_model.load_state_dict(w)
                w_locals.append(copy.deepcopy(w))
                loss_test = test_img(local_model, dataset_test, args)[1]
                loss_locals.append(copy.deepcopy(loss))
                loss_tests.append(loss_test)

            byz_grads = little_is_enough_attack(byz_grad, benign_grad)

            for idx in range(args.num_users):
                if idx in noisy_clients:
                    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
                    local_model = copy.deepcopy(net_glob)
                    tool.set_gradient_values(local_model, byz_grads[0])
                    optimizer.step()
                    w = local_model.state_dict()
                    w_locals[idx] = w

        else:

            idxs_users = [i for i in range(args.num_users)]
            for idx in idxs_users:
                if args.attack_pattern == 'lf' and np.random.rand() < args.attack_pos:
                    if idx in noisy_clients:
                        fakedata = copy.deepcopy(dataset_train)
                        if args.dataset == 'cifar100':
                            add_noise_to_client_2(fakedata, dict_users, idx)
                        else:
                            add_noise_to_client(fakedata,dict_users,idx)
                        local = LocalUpdate(args=args, dataset=fakedata, idxs=dict_users[idx])
                        w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))
                        local_model = copy.deepcopy(net_glob).to(args.device)
                        local_model.load_state_dict(w)
                        w_locals.append(copy.deepcopy(w))
                        noisy_this_round.append(idx)
                        continue

                if args.attack_pattern == 'w_noise' and np.random.rand() < args.attack_pos:
                    if idx in noisy_clients:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                        w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))
                        local_model = copy.deepcopy(net_glob).to(args.device)
                        local_model.load_state_dict(w)
                        # fake_w = copy.deepcopy(w)
                        # add_gaussian_noise_to_weights(fake_w)
                        fake_w = add_noise_to_layers_weights(local_model,args)
                        w_locals.append(copy.deepcopy(fake_w))
                        # 记录这一轮添加了噪声的客户端
                        noisy_this_round.append(idx)

                        continue

                if args.attack_pattern == 'gn' and np.random.rand() < args.attack_pos:
                    if idx in noisy_clients:
                        dataset_train_2 = add_gaussian_noise(dataset_train,dict_users,idx)
                        local = LocalUpdate(args=args, dataset=dataset_train_2, idxs=dict_users[idx])
                        w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))
                        local_model = copy.deepcopy(net_glob).to(args.device)
                        local_model.load_state_dict(w)
                        # fake_w = copy.deepcopy(w)
                        # add_gaussian_noise_to_weights(fake_w)
                        w_locals.append(copy.deepcopy(w))
                        # 记录这一轮添加了噪声的客户端
                        noisy_this_round.append(idx)

                        continue

                if args.attack_pattern == 'rw' and np.random.rand() < args.attack_pos:
                    if idx in noisy_clients:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                        w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))
                        local_model = copy.deepcopy(net_glob).to(args.device)
                        local_model.load_state_dict(w)
                        # fake_w = copy.deepcopy(w)
                        # add_gaussian_noise_to_weights(fake_w)
                        fake_w = random_weights(args,w )
                        w_locals.append(copy.deepcopy(fake_w))
                        # 记录这一轮添加了噪声的客户端
                        noisy_this_round.append(idx)

                        continue

                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))


                local_model = copy.deepcopy(net_glob).to(args.device)
                local_model.load_state_dict(w)
                w_locals.append(copy.deepcopy(w))

                loss_test = test_img(local_model, dataset_test, args)[1]
                loss_locals.append(copy.deepcopy(loss))
                loss_tests.append(loss_test)

            print(len(noisy_this_round))

        if args.method == 'fedaoc':

            # 记录当前轮次开始时间
            round_start_time = time.time()

            # 初始化信息素矩阵和权重矩阵
            tau = np.full((args.num_users, args.num_users), tau0)
            weights = np.ones((args.num_users, args.num_users))  # 初始化权重矩阵

            # 存储所有蚁群循环中最优蚂蚁的信息
            best_ant_overall = None
            best_test_acc_overall = -float('inf')

            # 蚁群优化选择客户端
            for iter in range(max_aco_iters):
                ants = []
                best_ant_in_iter = None
                best_test_acc_in_iter = -float('inf')

                for m in range(num_ants):
                    current_node = random.choice(range(args.num_users))
                    visited = {current_node}
                    path = [current_node]

                    for step in range(args.num_users - 1):  # 模拟蚂蚁路径
                        neighbors = set(range(args.num_users)) - visited
                        if not neighbors:
                            break

                        # 计算转移概率
                        prob = compute_transition_prob(tau, weights, alpha, beta, visited, current_node, neighbors)

                        # 检查 prob 是否为空或所有值是否为零
                        if not prob or sum(prob.values()) == 0:
                            break  # 如果没有有效的转移概率，跳过当前路径选择

                        # 根据概率选择下一节点
                        next_node = random.choices(list(prob.keys()), weights=list(prob.values()))[0]
                        visited.add(next_node)
                        path.append(next_node)

                        # 计算精度变化并更新 weights
                        net_glob.load_state_dict(w_locals[current_node])
                        acc_i, _ = test_img(net_glob, valid_dataset, args)

                        net_glob.load_state_dict(w_locals[next_node])
                        acc_j, _ = test_img(net_glob, valid_dataset, args)

                        # 计算精度变化并通过 sigmoid 转换
                        delta_acc = acc_j - acc_i
                        fitness = sigmoid(delta_acc, kappa=1.0)

                        # 更新权重矩阵
                        weights[current_node, next_node] = fitness

                        # 更新当前节点
                        current_node = next_node

                    # 如果路径没有形成有效结果，跳过该蚂蚁
                    if len(path) <= 1:
                        continue

                    # 聚合路径中的模型
                    local_models = [w_locals[client] for client in path]
                    aggregated_model = FedAvg_0(local_models)
                    net_glob.load_state_dict(aggregated_model)
                    test_acc, _ = test_img(net_glob, dataset_test, args)  # 在测试集上测试

                    # 保存当前蚂蚁的信息
                    ants.append({"path": path, "test_acc": test_acc})

                    # 更新本轮最优蚂蚁
                    if test_acc > best_test_acc_in_iter:
                        best_test_acc_in_iter = test_acc
                        best_ant_in_iter = {"path": path, "test_acc": test_acc}

                # 保留本轮最优蚂蚁路径
                if best_ant_in_iter and best_test_acc_in_iter > best_test_acc_overall:
                    best_test_acc_overall = best_test_acc_in_iter
                    best_ant_overall = best_ant_in_iter

                # 信息素更新逻辑
                for ant in ants:
                    path = ant["path"]
                    fitness = ant["test_acc"]  # 使用测试集精度作为信息素更新依据
                    for i in range(len(path) - 1):
                        tau[path[i], path[i + 1]] += eta * fitness
                tau *= (1 - rho)  # 信息素挥发

            # 选取全局最优蚂蚁的路径作为 selected_clients
            if best_ant_overall:
                selected_clients = best_ant_overall["path"]
            else:
                selected_clients = []

            # 根据选择的客户端进行联邦学习
            if selected_clients:
                for client in selected_clients:
                    w_locals_selected = w_locals[client]
                    w_locals.append(copy.deepcopy(w_locals_selected))

            # 聚合模型更新
            w_glob = FedAvg_0(w_locals)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)

            # 记录当前轮次结束时间并计算用时
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time
            # 输出轮次时间和结果
            print(
                f'Round {iters:3d}, Time: {round_duration:.3f}s, Global Acc: {global_acc:.3f}, Global Loss: {global_loss:.3f}')
            print(selected_clients)

        if args.method == 'fedpso':

            w = 0.5

            # 记录当前轮次开始时间
            round_start_time = time.time()

            # 初始化粒子位置 (0 或 1) 和速度矩阵 (OrderedDict)
            x = np.random.randint(2, size=(num_particles, num_clients))  # 粒子位置矩阵
            v = [collections.OrderedDict((key, np.random.rand()) for key in net_glob.state_dict().keys())
                 for _ in range(num_particles)]  # 每个粒子的速度矩阵

            # 存储局部最优和全局最优
            p_best = np.copy(x)  # 局部最优位置
            g_best = np.zeros(num_clients)  # 全局最优位置
            fitness_p = np.zeros(num_particles)  # 局部最优适应度
            fitness_g = -float('inf')  # 全局最优适应度

            # 初始化局部和全局最优
            for i in range(num_particles):
                selected_clients = [j for j in range(num_clients) if x[i, j] == 1]
                if not selected_clients:
                    fitness_p[i] = -float('inf')  # 如果未选择客户端，适应度为负无穷
                else:
                    local_models = [w_locals[j] for j in selected_clients]
                    aggregated_model = FedAvg_0(local_models)
                    net_glob.load_state_dict(aggregated_model)
                    acc, _ = test_img(net_glob, valid_dataset, args)  # 验证集精度
                    fitness_p[i] = acc
                    if acc > fitness_g:  # 更新全局最优
                        fitness_g = acc
                        g_best = np.copy(x[i])

            # PSO 主循环
            for iter in range(max_iters):
                for i in range(num_particles):
                    # 更新粒子速度
                    for key in v[i].keys():  # 逐元素更新速度
                        r1, r2 = np.random.rand(), np.random.rand()  # 随机因子
                        v[i][key] = (
                                w * v[i][key]  # 惯性项
                                + c1 * r1 * (p_best[i].mean() - x[i].mean())  # 个体学习项
                                + c2 * r2 * (g_best.mean() - x[i].mean())  # 社会学习项
                        )

                    # 更新粒子位置（使用 sigmoid 函数映射到 0 或 1）
                    sigmoid_v = np.zeros(num_clients)  # 初始化与 num_clients 一致的数组
                    for j, key in enumerate(v[i].keys()):
                        sigmoid_v[j] = 1 / (1 + np.exp(-v[i][key]))  # 对每个客户端的速度应用 sigmoid

                    x[i] = np.random.rand(num_clients) < sigmoid_v

                    # 计算适应度
                    selected_clients = [j for j in range(num_clients) if x[i, j] == 1]
                    if not selected_clients:
                        fitness = -float('inf')  # 如果未选择客户端，适应度为负无穷
                    else:
                        local_models = [w_locals[j] for j in selected_clients]
                        aggregated_model = FedAvg_0(local_models)
                        net_glob.load_state_dict(aggregated_model)
                        acc, _ = test_img(net_glob, valid_dataset, args)  # 验证集精度
                        fitness = acc

                    # 更新局部最优
                    if fitness > fitness_p[i]:
                        fitness_p[i] = fitness
                        p_best[i] = np.copy(x[i])

                    # 更新全局最优
                    if fitness > fitness_g:
                        fitness_g = fitness
                        g_best = np.copy(x[i])

            # 选取全局最优粒子的客户端路径
            selected_clients = [j for j in range(num_clients) if g_best[j] == 1]

            # 根据选择的客户端进行联邦学习
            aggregated_weights = collections.OrderedDict()
            for key in net_glob.state_dict().keys():
                aggregated_weights[key] = sum(w_locals[client][key] for client in selected_clients) / len(
                    selected_clients)

            # 聚合模型更新
            net_glob.load_state_dict(aggregated_weights)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)

            # 记录当前轮次结束时间并计算用时
            round_end_time = time.time()
            round_duration = round_end_time - round_start_time

            # 输出轮次时间和结果
            print(
                f'Round {iters:3d}, Time: {round_duration:.3f}s, Global Acc: {global_acc:.3f}, Global Loss: {global_loss:.3f}')
            print(selected_clients)

        if args.method == 'median':
            w_glob = simple_median(w_locals)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)
            # print("loss_locals:", loss_tests)
            print('Round {:3d}, global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, global loss {:.3f}'.format(iters, global_loss))

        if args.method == 'krum' :
            w_glob, _ = krum(w_locals, args)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)
            # print("loss_locals:", loss_tests)
            print('Round {:3d}, global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, global loss {:.3f}'.format(iters, global_loss))

            o += 1

        if args.method == 'trimmed_mean':
            w_glob = trimmed_mean(w_locals, 0.2)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)
            # print("loss_locals:", loss_tests)
            print('Round {:3d}, global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, global loss {:.3f}'.format(iters, global_loss))


        if args.method == 'fedavg':
            w_glob = FedAvg_0(w_locals)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_losses.append(global_loss)
            global_acctotal.append(float(global_acc))
            print('Round {:3d}, Global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, Global loss {:.3f}'.format(iters, global_loss))

    with open(os.path.join(save_dir, 'loss_trainfed.txt'), 'w') as f:
        for item in global_losses:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'global_acc.txt'), 'w') as f:
        for item in global_acctotal:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'time.txt'), 'w') as f:
        for item in total_time:
            f.write("%s\n" % item)
    #         f.write("%s\n" % item)

    plt.figure()
    plt.figure()
    plt.plot(range(len(global_acctotal)), global_acctotal)
    plt.ylabel('global acc')
    plt.savefig(os.path.join(save_dir, f'fedacc_{args.dataset}_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}.png'))

    plt.figure()
    plt.figure()
    plt.plot(range(len(global_losses)), global_losses)
    plt.ylabel('global loss')
    plt.savefig(
        os.path.join(save_dir, f'fedloss_{args.dataset}_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}.png'))
    print("global loss:",global_losses[-1])
    print("global acc:",global_acctotal[-1])