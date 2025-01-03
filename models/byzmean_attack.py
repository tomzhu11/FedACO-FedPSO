import math
import torch


def byzMean_attack(byz_grads, begin_grad):
    num_byzs = len(byz_grads)
    num_benign = len(begin_grad)
    if num_byzs == 0:
        return list()
    device = byz_grads[0][0].device
    est_grads1 = torch.stack(byz_grads + begin_grad)
    mu = torch.mean(est_grads1, dim=0)
    sigma = torch.std(est_grads1, dim=0)
    z = 0.5
    byz_grad = mu - z * sigma
    m1 = int(0.5 * num_byzs)
    m2 = num_byzs - m1
    byz_grad1 = [byz_grad]*m1
    byz_grad2 = ((num_benign-m1)*byz_grad-torch.sum(est_grads1, dim=0))/m2
    byz_grads2 = [byz_grad2] * m2

    return byz_grad1 , byz_grads2