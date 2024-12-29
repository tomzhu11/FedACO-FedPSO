import numpy as np
import torch
from sklearn.cluster import KMeans

def lbfgs_torch(args, S_k_list, Y_k_list, v):
    curr_S_k = torch.stack(S_k_list)
    curr_S_k = curr_S_k.transpose(0, 1).cpu()  # (10,xxxxxx)
    # print('------------------------')
    # print('curr_S_k.shape', curr_S_k.shape)
    curr_Y_k = torch.stack(Y_k_list)
    curr_Y_k = curr_Y_k.transpose(0, 1).cpu()  # (10,xxxxxx)
    S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k
    S_k_time_Y_k = S_k_time_Y_k.cpu()

    S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k
    S_k_time_S_k = S_k_time_S_k.cpu()
    # print('S_k_time_S_k.shape', S_k_time_S_k.shape)
    R_k = np.triu(S_k_time_Y_k.numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
    sigma_k = Y_k_list[-1].view(-1, 1).transpose(0, 1) @ S_k_list[-1].view(-1, 1) / (
                S_k_list[-1].view(-1, 1).transpose(0, 1) @ S_k_list[-1].view(-1, 1))
    sigma_k = sigma_k.cpu()

    D_k_diag = S_k_time_Y_k.diagonal()
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = mat.inverse()
    # print('mat_inv.shape',mat_inv.shape)
    v = v.view(-1, 1).cpu()

    approx_prod = sigma_k * v
    # print('approx_prod.shape',approx_prod.shape)
    # print('v.shape',v.shape)
    # print('sigma_k.shape',sigma_k.shape)
    # print('sigma_k',sigma_k)
    p_mat = torch.cat([curr_S_k.transpose(0, 1) @ (sigma_k * v), curr_Y_k.transpose(0, 1) @ v], dim=0)

    approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat
    # print('approx_prod.shape',approx_prod.shape)
    # print('approx_prod.shape',approx_prod.shape)
    # print('approx_prod.shape.T',approx_prod.T.shape)

    return (approx_prod.T)


def fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp):
    pred_update = []
    distance = []
    for i in range(len(old_update_list)):
        pred_update.append((old_update_list[i] + hvp).view(-1))

    pred_update = torch.stack(pred_update)
    local_update_list = torch.stack(local_update_list)
    old_update_list = torch.stack(old_update_list)

    distance = torch.norm((old_update_list - local_update_list), dim=1)
    # print('defense line219 distance(old_update_list - local_update_list):',distance)
    # auc1 = roc_auc_score(pred_update.numpy(), distance)
    # distance = torch.norm((pred_update - local_update_list), dim=1).numpy()
    # auc2 = roc_auc_score(pred_update.numpy(), distance)
    # print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

    # print('defence line 211 pred_update.shape:', pred_update.shape)
    distance = torch.norm((pred_update - local_update_list), dim=1)
    # print('defence line 211 distance.shape:', distance.shape)
    # distance = nn.functional.norm((pred_update - local_update_list), dim=0).numpy()
    distance = distance / torch.sum(distance)
    return distance


def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_

    if np.mean(score[label_pred == 0]) < np.mean(score[label_pred == 1]):
        # 0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label = np.ones(100)
    real_label[:nobyz] = 0
    acc = len(label_pred[label_pred == real_label]) / 100
    recall = 1 - np.sum(label_pred[:nobyz]) / nobyz
    fpr = 1 - np.sum(label_pred[nobyz:]) / (100 - nobyz)
    fnr = np.sum(label_pred[:nobyz]) / nobyz
    # print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    # print(silhouette_score(score.reshape(-1, 1), label_pred))
    # print('defence.py line233 label_pred (0 = malicious pred)', label_pred)
    return label_pred


def detection1(score):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min) / (max - min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m] - center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m] - center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    # print('defense line278 gapDiff:', gapDiff)
    select_k = 2  # default detect attacks
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i + 1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1