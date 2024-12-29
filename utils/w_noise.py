import numpy as np
import torch


def add_gaussian_noise_to_weights(weights):
    with torch.no_grad():  # 确保这些操作不会影响梯度计算
        # std_dev = np.random.uniform(0.001, 0.0015)
        std_dev = 0.5
        for key in weights.keys():
            noise = torch.randn_like(weights[key]) * std_dev  # 为权重生成同形状的高斯噪声
            weights[key].add_(noise)  # 原地为权重添加噪声


def add_noise_to_layers_weights(model,args):
    lamda = 2.5
    for name, param in model.named_parameters():
        if 'bias' in name or 'bn' in name:
            # 不对偏置和BatchNorm的参数添加噪声
            continue
        std = lamda * param.data.std()
        noise = torch.normal(0, std, size=param.size()).cuda(args.device)
        param.data.add_(noise)
    return model.state_dict()

def random_weights(args, w):

    for key in w.keys():
        noise = torch.tensor(np.random.normal(0, 0.01, w[key].shape))
        noise = noise.to(torch.float32)
        noise = noise.to(args.device)
        w[key] = noise
    return w