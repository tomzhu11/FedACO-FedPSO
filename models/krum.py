import copy

import torch

def euclid(v1, v2):
    diff = v1 - v2
    # Ensuring diff is a 2D tensor before transposing
    if diff.dim() == 1:
        diff = diff.unsqueeze(0)
    return torch.matmul(diff, diff.T)

def multi_vectorization(w_locals, args):
    vectors = copy.deepcopy(w_locals)

    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(args.device)
        vectors[i] = torch.cat(list(v.values()))

    return vectors
def pairwise_distance(w_locals, args):
    vectors = multi_vectorization(w_locals, args)
    distance = torch.zeros([len(vectors), len(vectors)]).to(args.device)

    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)

    return distance


def krum(w_locals, args):
    # 计算每个模型与其他所有模型之间的距离
    n = 50
    distance = pairwise_distance(w_locals, args)

    # 对每个模型的距离进行求和，然后排序获取前n个模型的索引
    sorted_idx = distance.sum(dim=0).argsort()[:n]

    # 计算选择的模型的平均权重
    w_avg = copy.deepcopy(w_locals[sorted_idx[0]])
    for key in w_avg.keys():
        for i in range(1, len(sorted_idx)):
            w_avg[key] += w_locals[sorted_idx[i]][key]
        w_avg[key] = torch.div(w_avg[key], n)

    return w_avg, sorted_idx.tolist()