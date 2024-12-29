import numpy as np
import sklearn.metrics.pairwise as smp
import torch
import time


def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i != j and maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - np.max(cs, axis=1)
    wv = np.clip(wv, 0, 1)
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv = np.clip(wv, 0, 1)
    return wv


class FoolsGold(object):
    def __init__(self, args):
        self.memory = None
        self.wv_history = []
        self.args = args

    def aggregate_gradients(self, client_grads):
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((num_clients, grad_len))

        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))


        self.memory += grads
        wv = foolsgold(self.memory)
        # else:
        #     wv = foolsgold(grads)
        # self.wv_history.append(wv)

        agg_grads = []
        for i in range(len(client_grads[0])):
            temp = wv[0] * client_grads[0][i].cpu().clone()
            for c, client_grad in enumerate(client_grads):
                if c > 0:
                    temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        return agg_grads

def aggregate_weights(args, w_locals, net_glob, fg):

    net_glob.train()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer.zero_grad()
    agg_grads = fg.aggregate_gradients(w_locals)
    for i, (name, params) in enumerate(net_glob.named_parameters()):
        if params.requires_grad:
            params.grad = agg_grads[i].cuda()
    optimizer.step()

