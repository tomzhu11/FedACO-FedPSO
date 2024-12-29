#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments

    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--ccepochs', type=int, default=100, help="rounds of initial control chart training")
    parser.add_argument('--L', type=float, default=3, help="width of the bound")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--mnist_iid_noise_30', action='store_true',default=False, help="Apply IID noise sampling for MNIST dataset")
    parser.add_argument('--mnist_iid_noise_20', action='store_true', default=False,
                        help="Apply IID noise sampling for MNIST dataset")
    parser.add_argument('--mnist_iid_noise_10', action='store_true', default=False,
                        help="Apply IID noise sampling for MNIST dataset")
    parser.add_argument('--mnist_iid_noise_2', action='store_true',default=False, help="Apply IID noise2 sampling for MNIST dataset")
    parser.add_argument('--w_noise', action='store_true', help="Apply w noise for model")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid_ld', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid_lq', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid_q', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--noniid_2', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--alpha', type=float, default=0.5, help="distribution parameter")
    parser.add_argument('--method', type=str, default='fedavg', help="Algorithm")
    parser.add_argument('--attack', action='store_true',default=True, help='attack')
    parser.add_argument('--attack_pattern', type=str, default='w_noise', help="name of dataset")
    parser.add_argument('--attack_pos', type=float, default=0.7, help="attack posibility")
    parser.add_argument('--peer_round', type=float, default=0.4, help="attack number peer round")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--savedir', type=str, default='saver', help="files path")
    args = parser.parse_args()
    return args
