import copy

import numpy as np
import torch
import random

from scipy.spatial import distance


def no_byz(v, f):
    return v


def scaling_attack(v, f, epsilon=0.01):
    scaling_factor = len(v)
    for param_id in f:
        for idx in range(len(v[param_id])):
            v[param_id][idx] = v[param_id][idx] * scaling_factor
    return v


def attack_krum(network, local_grads, mal_index, param_index, lower_bound=1e-8, upper_bound=1e-3):
    # print("输出1个客户端权重")
    # print(local_grads[76][0])
    # local_grads_save = copy.deepcopy(local_grads)
    # for i in range(len(local_grads_save)):
    #     for ii in range(len(local_grads_save[i])):
    #         local_grads_save[i][ii] = local_grads_save[i][ii].cpu()
    # np.save('local_grads_save.npy',local_grads_save)

    def krum_(samples, f):
        size = len(samples)
        size_ = size - f - 2
        metric = []

        for idx in range(size):
            sample = samples[idx]
            samples_ = samples.copy()
            del samples_[idx]
            for idx in range(len(samples_)):
                samples_[idx] = np.array(samples_[idx])
            dis = np.array([np.linalg.norm(sample - sample_) for sample_ in samples_])
            metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
        return metric

    def krum(samples, f):
        metric = krum_(samples, f)
        index = np.argmin(metric)
        return samples[index], index

    # local_param = copy.deepcopy(local_grads)
    # for i in sorted(mal_index, reverse=True):
    #     del local_param[i]

    local_param = []
    for i in range(len(local_grads)):
        if i not in range(len(mal_index)):
            local_param.append(local_grads[i])

    m = len(local_grads)

    average_sign = np.zeros(list(network.parameters())[param_index].data.shape)
    benign_max = np.zeros(list(network.parameters())[param_index].data.shape)

    for c in range(len(local_param)):
        average_sign += np.array(local_param[c][param_index].cpu())
    average_sign = np.sign(average_sign)
    min_dis = np.inf
    max_dis = -np.inf
    for i in range(m):
        if i in mal_index:
            continue
        else:
            temp_min_dis = 0
            temp_max_dis = 0
            for j in range(m):
                if j in mal_index or j == i:
                    continue
                else:
                    # print(local_grads[i][param_index])
                    # print(local_grads[j][param_index])
                    # print(i,j,param_index)
                    # print(local_grads[i][param_index].flatten().cpu())
                    # print(local_grads[j][param_index].flatten().cpu())
                    # print(temp_min_dis)
                    temp_min_dis += distance.euclidean(local_grads[i][param_index].flatten().cpu(),
                                                       local_grads[j][param_index].flatten().cpu())
        temp_max_dis += distance.euclidean(local_grads[i][param_index].flatten().cpu(), benign_max.flatten())

        if temp_min_dis < min_dis:
            min_dis = temp_min_dis
        if temp_max_dis > max_dis:
            max_dis = temp_max_dis

    upper_bound = 1.0
    lambda1 = upper_bound

    if upper_bound < lower_bound:
        print('Wrong lower bound!')

    average_sign = torch.from_numpy(average_sign)
    while True:
        krum_local = []
        for kk in range(len(local_grads)):
            krum_local.append(np.array(local_grads[kk][param_index].cpu()))
        for kk in mal_index:
            krum_local[kk] = -lambda1 * average_sign
        _, choose_index = krum(krum_local, f=49)
        if choose_index in mal_index:
            print('found a lambda')
            break
        elif lambda1 < lower_bound:
            print(choose_index, 'Failed to find a proper lambda!')
            break
        else:
            lambda1 /= 2.0

    for kk in mal_index:
        local_grads[kk][param_index] = -lambda1 * average_sign
    return local_grads



def attack_trimmedmean(network, local_grads, mal_index, b=1):
    benign_max = []
    benign_min = []
    average_sign = []
    mal_param = []
    local_param = copy.deepcopy(local_grads)
    for i in sorted(mal_index, reverse=True):
        del local_param[i]

    local_param = []
    for i in mal_index:
        local_param.append(local_grads[i])

    for p in list(network.parameters()):
        benign_max.append(np.zeros(p.data.shape))
        benign_min.append(np.zeros(p.data.shape))
        average_sign.append(np.zeros(p.data.shape))
        mal_param.append(np.zeros(p.data.shape))
    for idx, p in enumerate(average_sign):
        for c in range(len(local_param)):
            average_sign[idx] += np.array(local_param[c][idx].cpu())
        # 生成扰动方向，保留原始样本的符号
        average_sign[idx] = np.sign(average_sign[idx])
    for idx, p in enumerate(network.parameters()):
        temp = []
        for c in range(len(local_param)):
            local_param[c][idx] = p.data.cpu().numpy() - np.array(local_param[c][idx].cpu())
            temp.append(local_param[c][idx])
        temp = np.array(temp)
        benign_max[idx] = np.amax(temp, axis=0)
        benign_min[idx] = np.amin(temp, axis=0)

    for idx, p in enumerate(average_sign):
        for aver_sign, b_max, b_min, mal_p in np.nditer([p, benign_max[idx], benign_min[idx], mal_param[idx]],
                                                        op_flags=['readwrite']):
            if aver_sign < 0:
                if b_min > 0:
                    mal_p[...] = random.uniform(b_min / b, b_min)
                else:
                    mal_p[...] = random.uniform(b_min * b, b_min)
            else:
                if b_max > 0:
                    mal_p[...] = random.uniform(b_max, b_max * b)
                else:
                    mal_p[...] = random.uniform(b_max, b_max / b)
    for c in mal_index:
        for idx, p in enumerate(network.parameters()):
            local_grads[c][idx] = torch.from_numpy(-mal_param[idx] + p.data.cpu().numpy())
    return local_grads


def attack_innerProduct(local_grads, weight, choices, mal_index):
    attack_vec = []
    for idx, pp in enumerate(local_grads[0]):
        tmp = torch.zeros_like(pp)
        for idxs, j in enumerate(choices):
            if idxs not in mal_index:
                tmp += local_grads[idxs][idx]
        attack_vec.append((-weight) * tmp / len(choices))
    for i in range(len(mal_index)):
        local_grads[i] = attack_vec
    return local_grads


def attack_sign_flipping(local_grads, choice_id, candidate_malicious_id, attack_value):
    weights_modified = copy.deepcopy(local_grads)
    for k,_ in enumerate(candidate_malicious_id):
        for kk in range(len(weights_modified[k])):
            weights_modified[k][kk] = weights_modified[k][kk] * attack_value
    return weights_modified

def attack_amplify(local_grads, choice_id, mal_id, attack_value):
    weights_modified = copy.deepcopy(local_grads)
    for k in range(len(mal_id)):
        for kk in range(len(weights_modified[k])):
            weights_modified[k][kk] = weights_modified[k][kk] * attack_value
    return weights_modified

def attack_additive_noise(local_grads,noise_scale, choice_id, candidate_malicious_id):
    weights_modified = copy.deepcopy(local_grads)
    # for k in candidate_malicious_id:
    #     for kk in range(len(weights_modified[k])):
    #         noise = torch.from_numpy(copy.deepcopy(np.random.normal(scale=0.3, size=weights_modified[k][kk].shape))).to(
    #             'cuda:0')
    #         weights_modified[k][kk] = weights_modified[k][kk] + noise
    for idx,k in enumerate(candidate_malicious_id):
        for kk in range(len(weights_modified[idx])):
            noise = torch.from_numpy(copy.deepcopy(np.random.normal(scale=noise_scale, size=weights_modified[idx][kk].shape)))
            weights_modified[idx][kk] = weights_modified[idx][kk] + noise
    return weights_modified
