import copy
import math

import torch
import torch.nn

# 其他模块
import numpy as np
import torchvision.transforms.functional as TF
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.metrics import davies_bouldin_score


class Server(object):
    def __init__(self, conf, helper, test_dataloader, model):
        self.conf = conf
        self.helper = helper
        self.global_model = model
        self.arr = None
        self.record_mal = [0 for _ in range(self.conf['client_number'])]
        self.choice_id = None
        self.malicious_id = None

    def evaluate_accuracy(self, model, test_dataloader):
        total_loss = 0.0
        accuracy = 0
        dataset_size = 0
        model = model.to(self.conf['device'])
        model.eval()
        for batch_id, batch in enumerate(test_dataloader):
            data, target = batch
            dataset_size += data.size()[0]
            data = data.to(self.conf['device'])
            target = target.to(self.conf['device'])
            # sum up batch loss
            output = model(data)
            pred = output.data.max(1)[1]
            # accuracy += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            accuracy += (pred == target).sum()  # 计算预测正确的个数

        # 一轮训练完后计算损失率和正确率
        loss = total_loss / len(test_dataloader.sampler)  # 当前轮的总体平均损失值
        acc = float(accuracy) / len(test_dataloader.sampler)  # 当前轮的总正确率
        return acc, loss

    def evaluate_backdoor_accuracy(self, model, test_dataloader):
        total_loss = 0.0
        accuracy = 0
        dataset_size = 0
        model = model.to(self.conf['device'])
        model.eval()
        for batch_id, batch in enumerate(test_dataloader):
            data, target = batch
            data = data.to(self.conf['device'])
            target = target.to(self.conf['device'])

            data = (TF.erase(data, 0, 0, 5, 5, 0).to(self.conf['device']))
            target = torch.zeros(len(target), dtype=torch.long).to(self.conf['device'])
            dataset_size += data.size()[0]

            # sum up batch loss
            output = model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            accuracy += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = float(accuracy) / float(dataset_size)
        total_l = total_loss / dataset_size
        return acc, total_l

    def evaluate_label_flip_accuracy(self, model, test_dataloader):
        class_number = 10
        origin_correct = [0 for _ in range(10)]
        attack_correct = [0 for _ in range(10)]
        other_correct = [0 for _ in range(10)]
        total = [0 for i in range(10)]

        model = model.to(self.conf['device'])
        model.eval()

        for batch_id, batch in enumerate(test_dataloader):
            data, target = batch
            data = data.to(self.conf['device'])
            target = target.to(self.conf['device'])

            # sum up batch loss
            output = model(data)
            # total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]

            x = target.cpu().numpy()
            for i in [j for j in range(class_number)]:
                idxs = np.argwhere(x == i)

                for idx in idxs:
                    if pred[idx] == i:
                        origin_correct[i] += 1
                    elif pred[idx] == (i + 1) % class_number:
                        attack_correct[i] += 1
                    else:
                        other_correct[i] += 1
                total[i] += len(idxs)
        # print(origin_correct)
        # print(attack_correct)
        # print(other_correct)
        # print(total)
        # sum = 0
        # for i in range(class_number):
        #     print(origin_correct[i] / total[i])
        #     sum += origin_correct[i] / total[i]
        # print(sum / 10)

        acc = float(origin_correct[0]) / float(total[0])
        asr = float(attack_correct[0]) / float(total[0])

        return acc, asr

    def fedavg(self, grads_list):

        # 平均梯度
        grads_average = []
        for p in list(self.global_model.parameters()):
            grads_average.append(torch.from_numpy(np.zeros(p.data.shape)))

        # 聚合权重
        alpha = [1 / len(grads_list) for _ in range(len(grads_list))]

        # 梯度平均
        for i, _ in enumerate(grads_list):
            for ii, _ in enumerate(grads_list[i]):
                grads_average[ii] += alpha[i] * grads_list[i][ii]

        # 更新平均值放到GPU上
        for i, _ in enumerate(grads_list):
            for ii, _ in enumerate(grads_list[i]):
                grads_average[ii] = grads_average[ii].to(self.conf['device'])

        # 更新模型
        for i, p in enumerate(list(self.global_model.parameters())):
            p.data += grads_average[i]

        return grads_average

    def trimmedmean(self, grads_list):
        local_grads_save = copy.deepcopy(grads_list)
        for i in range(len(local_grads_save)):
            for ii in range(len(local_grads_save[i])):
                local_grads_save[i][ii] = local_grads_save[i][ii].cpu()

        # np.save('local_grads_server_save.npy',local_grads_save)

        def trimmed_mean(samples, beta=0.1):
            samples = np.array(samples)
            # average_grad_tmp = np.zeros(samples[0].shape)
            size = samples.shape[0]
            beyond_choose = int(size * beta)
            samples = np.sort(samples, axis=0)
            samples = samples[beyond_choose:size - beyond_choose]
            average_grad_tmp = np.average(samples, axis=0)
            return average_grad_tmp

        average_grad = []
        for p in grads_list[0]:
            average_grad.append(np.zeros(p.shape))
        for idx, _ in enumerate(average_grad):
            trimmedmean_local = []
            for c in range(len(grads_list)):
                trimmedmean_local.append(np.array(grads_list[c][idx].cpu()))
            average_grad[idx] = torch.from_numpy(trimmed_mean(trimmedmean_local))

        # 更新平均值放到GPU上
        for i, _ in enumerate(grads_list):
            for ii, _ in enumerate(grads_list[i]):
                average_grad[ii] = average_grad[ii].to(self.conf['device'])

        # 更新模型
        for i, p in enumerate(list(self.global_model.parameters())):
            p.data += average_grad[i]
        return average_grad

    def trimmedmean_(self, grads_list):
        local_grads_save = copy.deepcopy(grads_list)
        for i in range(len(local_grads_save)):
            for ii in range(len(local_grads_save[i])):
                local_grads_save[i][ii] = local_grads_save[i][ii].cpu()

        # np.save('local_grads_server_save.npy',local_grads_save)

        def trimmed_mean(samples, beta=0.3):
            samples = np.array(samples)
            # average_grad_tmp = np.zeros(samples[0].shape)
            size = samples.shape[0]
            beyond_choose = int(size * beta)
            samples = np.sort(samples, axis=0)
            samples = samples[beyond_choose:size - beyond_choose]
            average_grad_tmp = np.average(samples, axis=0)
            return average_grad_tmp

        average_grad = []
        for p in grads_list[0]:
            average_grad.append(np.zeros(p.shape))
        for idx, _ in enumerate(average_grad):
            trimmedmean_local = []
            for c in range(len(grads_list)):
                trimmedmean_local.append(np.array(grads_list[c][idx].cpu()))
            average_grad[idx] = torch.from_numpy(trimmed_mean(trimmedmean_local))

        return average_grad

    def cluster_(self, grads_list, grads_list_fc):
        from sklearn.cluster import KMeans
        import torch.nn.functional as F

        # 拉平
        # grad_list_flattens = []
        # for grad_list_fc in grads_list_fc:
        #     grad_list_flatten = []
        #     for layer in grad_list_fc:
        #         grad_list_flatten.append(layer.flatten())
        #     # print(type(grad_list_flatten))
        #     # grad_list_flatten = torch.concat(grad_list_flatten)
        #     # grad_list_flatten = np.array(grad_list_flatten)
        #     grad_list_flattens.append(grad_list_flatten)

        grads_trimmedmean = self.trimmedmean_(grads_list_fc)
        # for i in range(len(grads_trimmedmean)):
        #     grads_trimmedmean[i] = torch.from_numpy(grads_trimmedmean[i])

        tmp_grad = []
        for i, v in enumerate(grads_list[0]):
            tmp_grad.append(torch.zeros_like(v))

        tmp_grads_list_fc = []
        tmp_grads_trimmedmean = []
        norms = []
        coses = []

        grads_list_fc_flattens = []
        for i in range(len(grads_list_fc)):
            grads_list_fc_flatten = []
            for id in range(len(grads_list_fc[i])):
                grads_list_fc_flatten.append(grads_list_fc[i][id].flatten())
            grads_list_fc_flatten = torch.concat(grads_list_fc_flatten, dim=0)
            print(type(grads_list_fc_flatten))
            grads_list_fc_flattens.append(grads_list_fc_flatten)
        grads_trimmedmean_flatten = []
        for id in range(len(grads_trimmedmean[i])):
            grads_trimmedmean_flatten.append(grads_trimmedmean[id].flatten())
        grads_trimmedmean_flatten = torch.concat(grads_trimmedmean_flatten, dim=0)
        for i in range(len(grads_list_fc)):
            norm = torch.norm(grads_list_fc_flattens[i] - grads_trimmedmean_flatten)
            cose = self.helper.cosine_similarity(grads_list_fc_flattens[i] - grads_trimmedmean_flatten)
            norms.append(norm)
            coses.append(cose)

        print(norms)
        print("========================")
        print(coses)
        #
        # merged_array = np.column_stack((coses, norms))
        #
        #
        # k = 2
        # X = merged_array
        # kmeans = KMeans(n_clusters=k, init='k-means++', n_init=2, random_state=42)  # 设置聚类簇的数量和初始化方法
        # kmeans.fit(X)
        # # 获取聚类标签
        # labels = kmeans.labels_
        # print(labels)
        # clusters_number = len(np.unique(labels))
        # indices_dict = {}
        # # indices_list = []
        # for i in range(clusters_number):
        #     indices_dict[i] = np.where(labels == i)[0]
        #     # indices_list.append(np.where(labels == i)[0]
        #     print(f"cluster {i} :{indices_dict[i]}")
        #
        # max_number_cluster = 0
        # max_number_cluster_label = -1
        # for i in range(clusters_number):
        #     if max_number_cluster < len(indices_dict):
        #         max_number_cluster = len(indices_dict)
        #         max_number_cluster_label = i

        # 平均梯度
        grads_average = []
        for p in list(self.global_model.parameters()):
            grads_average.append(torch.from_numpy(np.zeros(p.data.shape)))

        # 聚合权重
        alpha = [1 / len(grads_list) for _ in range(len(grads_list))]

        # 梯度平均
        for i, _ in enumerate(grads_list):
            for ii, _ in enumerate(grads_list[i]):
                grads_average[ii] += alpha[i] * grads_list[i][ii]

        # 更新平均值放到GPU上
        for i, _ in enumerate(grads_list):
            for ii, _ in enumerate(grads_list[i]):
                grads_average[ii] = grads_average[ii].to(self.conf['device'])

        # 更新模型
        for i, p in enumerate(list(self.global_model.parameters())):
            p.data += grads_average[i]
        return grads_average

    def cluster_new(self, grads_list, grads_list_fc):
        # 获取方向标记 ########################################################################
        grad_trimmedmean = self.trimmedmean_(grads_list_fc)

        # 拉平 ############################################################################
        grads_list_fc_flattens = []
        for i in range(len(grads_list_fc)):
            grads_list_fc_flatten = []
            for ii in range(len(grads_list_fc[i])):
                grads_list_fc_flatten.append(grads_list_fc[i][ii].flatten())
            grads_list_fc_flatten = torch.concat(grads_list_fc_flatten, dim=0)
            grads_list_fc_flattens.append(grads_list_fc_flatten)

        grad_trimmedmean_flatten = []
        for ii in range(len(grad_trimmedmean)):
            grad_trimmedmean_flatten.append(grad_trimmedmean[ii].flatten())
        grad_trimmedmean_flatten = torch.concat(grad_trimmedmean_flatten, dim=0)

        # top-k ############################################################################
        k = int(len(grad_trimmedmean_flatten))
        print(f"查找top-{k}个最大值")
        topk_values, topk_indices = torch.topk(grad_trimmedmean_flatten, k)
        top_grad_trimmedmean_flatten = grad_trimmedmean_flatten[topk_indices]

        top_grads_list_fc_flattens = []
        for i in range(len(grads_list_fc_flattens)):
            top_grads_list_fc_flatten = []
            top_grads_list_fc_flattens.append(grads_list_fc_flattens[i][topk_indices])

        # t = int(20)
        # topk_indices_list = []
        # for i in range(len(grads_list_fc_flattens)):
        #     topk_values, topk_indices = torch.topk(grads_list_fc_flattens[i], t)
        #     topk_indices_list.append(topk_indices)
        # print(topk_indices_list)
        # 计算范数 ############################################################################
        norms = []
        for i in range(len(grads_list_fc_flattens)):
            norm = torch.norm(grads_list_fc_flattens[i]).item()
            norms.append([round(norm, 2)])
        print("梯度范数==========================================")
        print(norms)

        # 计算余弦相似度 #######################################################################
        cose_similiary = []
        # direction_zero = torch.zeros((grads_list_fc_flattens[0].shape))
        # direction_vector = torch.randn(grads_list_fc_flattens[0].shape)

        for i in range(len(grads_list_fc_flattens)):
            cose = torch.cosine_similarity(top_grads_list_fc_flattens[i], top_grad_trimmedmean_flatten, dim=0).item()
            cose = (cose + 1) / 2
            cose_similiary.append([round(cose, 2)])
        print("余弦相似度==========================================")
        print(cose_similiary)

        shield = []
        # for i in range(len(cose_similiary)):
        #     if cose_similiary[i][0] < 0:
        #         shield.append(i)

        cose_similiary_norms = []
        for i in range(len(norms)):
            cose_similiary_norms.append([norms[i][0] * cose_similiary[i][0]])

        # k-means ###########################################################################
        cose_similiary_norms = np.array(cose_similiary_norms)
        # cose_similiary = self.helper.min_max_normalize__(cose_similiary)
        # cose_similiary_norms = self.helper.min_max_normalize__(cose_similiary_norms)

        # X = np.column_stack((cose_similiary_norms, norms))
        # X = np.column_stack((cose_similiary_norms, cose_similiary, norms))
        # X = np.array(norms)
        # print(X)
        X = np.column_stack((norms, cose_similiary))

        dbis = {}
        clusters_average_distances = {}
        clusters_norms = {}
        clusters_cos_similiary = {}
        indices_dicts = {}
        cluster_labels = {}
        for k in self.conf['cluster_number']:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=2, random_state=42)  # 设置聚类簇的数量和初始化方法
            kmeans.fit(X)
            # 获取聚类标签
            labels = kmeans.labels_
            cluster_labels[k] = labels
            print(f"{k}-means:{labels}")
            clusters_number = len(np.unique(labels))
            indices_dict = {}
            # indices_list = []

            cluster_average_distances = {}
            cluster_norms = {}
            cluster_cos_similiary = {}
            for i in range(clusters_number):
                indices_dict[i] = np.where(labels == i)[0]
                # indices_list.append(np.where(labels == i)[0]
                # print(f"cluster {i} :\033[1;31m {indices_dict[i]}\033[0m")
                # print(f"cluster {i} :{sum(np.array(norms)[indices_dict[i]]) / len(indices_dict[i])}")
                # print(f"cluster {i} :{sum(np.array(cose_similiary)[indices_dict[i]]) / len(indices_dict[i])}")
                total_distance = 0
                for ii in range(len(X[indices_dict[i]])):
                    for jj in range(ii + 1, len(X[indices_dict[i]])):
                        distance = euclidean(X[indices_dict[i]][ii], X[indices_dict[i]][jj])
                        total_distance += distance
                if len(X[indices_dict[i]]) == 1:
                    average_distance = 1
                else:
                    average_distance = total_distance / (len(X[indices_dict[i]]) * (len(X[indices_dict[i]]) - 1) / 2)
                print(f"cluster {i} 簇内距离是：{average_distance}")
                dbi = davies_bouldin_score(X, labels)
                cluster_average_distances[i] = average_distance
                cluster_norms[i] = sum(np.array(norms)[indices_dict[i]]) / len(indices_dict[i])
                cluster_cos_similiary[i] = sum(np.array(cose_similiary)[indices_dict[i]]) / len(indices_dict[i])

            indices_dicts[k] = indices_dict
            dbis[k] = dbi
            print(f"{k}-means DBI距离是：{dbi}")

            clusters_average_distances[k] = cluster_average_distances
            clusters_norms[k] = cluster_norms
            clusters_cos_similiary[k] = cluster_cos_similiary
        min_dbi = math.inf
        choice_cluster = -1
        for ii in dbis.keys():
            if min_dbi > dbis[ii]:
                min_dbi = dbis[ii]
                choice_cluster = ii
        # dis_sums = []
        # for idx,k in enumerate(self.conf['cluster_number']):
        #     dis_cum = 0
        #     for i in clusters_average_distances[k]:
        #         dis_cum += i
        #     dis_sums.append(dis_cum)
        # min_dis = math.inf
        # for idx,k in  enumerate(self.conf['cluster_number']):
        #     if min_dis > dis_sums[idx]:
        #         min_dis = dis_sums[idx]
        #         choice_cluster = k
        print(f"====================")

        print(f"选择的聚类数量是{choice_cluster}")
        cluster_cos = clusters_cos_similiary[choice_cluster]
        cluster_norm = clusters_norms[choice_cluster]
        cluster_distances = clusters_average_distances[choice_cluster]
        cluster_indices_dict = indices_dicts[choice_cluster]

        indices_dict = cluster_indices_dict
        labels = cluster_labels[choice_cluster]
        for i in range(choice_cluster):
            indices_dict[i] = np.where(labels == i)[0]
            # indices_list.append(np.where(labels == i)[0]
            print(f"cluster {i} :\033[1;31m {indices_dict[i]}\033[0m")
            print(f"cluster {i} :{sum(np.array(norms)[indices_dict[i]]) / len(indices_dict[i])}")
            print(f"cluster {i} :{sum(np.array(cose_similiary)[indices_dict[i]]) / len(indices_dict[i])}")
            print(f"cluster {i} 簇内距离是：{cluster_distances[i]}")
        print(labels)
        # print(cluster_cos)
        # print(cluster_norm)
        # print(cluster_distances)

        cluster_cos_list = []
        cluster_norm_list = []
        cluster_distances_list = []
        dis_filter = []
        print(f"过滤阈值是{self.conf['threshod']}")
        for k in range(choice_cluster):
            # print(len(cluster_indices_dict[k]))
            if len(cluster_indices_dict[k]) > self.conf['threshod']:

                cluster_cos_list.append(cluster_cos[k])
                cluster_norm_list.append(cluster_norm[k])
                cluster_distances_list.append(cluster_distances[k])
            else:
                dis_filter.append(k)

        # # # 计算均值和标准差
        # mean = np.mean(cluster_distances_list)
        # std = np.std(cluster_distances_list)
        # #
        # # 定义阈值
        # mean_clus_dis = 0
        # for i in cluster_distances_list:
        #     mean_clus_dis+=i
        # mean_clus_dis /=len(cluster_distances_list)
        # threshold = mean_clus_dis + 0.1  # 可根据实际情况进行调整
        # print(threshold)
        # # 根据阈值判断异常值
        # outliers = []
        # for value in cluster_distances_list:
        #     z_score = (value - mean) / std
        #     print(z_score)
        #     if abs(z_score) > threshold:
        #         outliers.append(value)

        # 计算差值
        # differences = []
        # for i in range(len(cluster_distances_list)):
        #     for j in range(i + 1, len(cluster_distances_list)):
        #         difference = cluster_distances_list[i] - cluster_distances_list[j]
        #         differences.append(difference/(len(cluster_distances_list)-1))
        #
        #
        # cluster_distances_list_norm = np.array(cluster_distances_list)
        # cluster_distances_list_norm = self.helper.mean_nomalize(cluster_distances_list_norm)
        # # 计算箱线图
        # q1 = np.percentile(cluster_distances_list_norm, 10)
        # q3 = np.percentile(cluster_distances_list_norm, 90)
        # iqr = q3 - q1
        # print(f"q1,q3,iqr is {q1, q3, iqr}")
        # # 阈值（可根据实际情况调整）
        # threshold = 0.5
        #
        # # 标识异常值
        # # outliers = cluster_distances_list[
        # #     (cluster_distances_list < q1 - threshold * iqr) | (cluster_distances_list > q3 + threshold * iqr)]
        # outliers = cluster_distances_list_norm[
        #     (cluster_distances_list_norm < q1) | (cluster_distances_list_norm > q3)]
        #
        # print(f"原始值是:{cluster_distances_list_norm}")
        # print(f"异常值为{outliers}")
        # dis_filter = []
        # for idx,i in enumerate(cluster_distances_list_norm):
        #     if i in outliers:
        #         dis_filter.append(idx)
        # dis_filter = np.where(cluster_distances_list == outliers)[0]
        # print(f"过滤的值为{dis_filter}")
        # dis_filter_new = []
        # for idx in dis_filter:
        #     dis_filter_new.append(idx)
        # print(f"过滤的簇为{dis_filter_new}")
        print(f"分簇后的簇内距离是:{cluster_distances_list}")
        print(f"簇内平均范数是:{cluster_norm_list}")

        # cluster_norm_list_sum = 0
        # for i in range(len(cluster_distances_list)):
        #     cluster_norm_list_sum += cluster_norm_list[i][0]
        # cluster_norm_list_mean = cluster_norm_list_sum / len(cluster_distances_list)

        cluster_norm_list_tmp = []
        for i in range(len(cluster_norm_list)):
            cluster_norm_list_tmp.append([cluster_norm_list[i][0]])
        cluster_norm_list = np.array(cluster_norm_list_tmp)

        cluster_dis_list_tmp = []
        for i in range(len(cluster_distances_list)):
            cluster_dis_list_tmp.append([cluster_distances_list[i]])
        cluster_dis_list_tmp = np.array(cluster_dis_list_tmp)

        cluster_cos_list_tmp = []
        for i in range(len(cluster_cos_list)):
            cluster_cos_list_tmp.append([cluster_cos_list[i][0]])
        cluster_cos_list_tmp = np.array(cluster_cos_list_tmp)

        # Z-Score 标准化
        from sklearn.preprocessing import StandardScaler
        zscore_scaler = StandardScaler()
        data_zscore_dis = zscore_scaler.fit_transform(cluster_dis_list_tmp)

        from sklearn.preprocessing import StandardScaler
        zscore_scaler = StandardScaler()
        data_zscore_norm = zscore_scaler.fit_transform(cluster_norm_list)

        from sklearn.preprocessing import StandardScaler
        zscore_scaler = StandardScaler()
        cluster_cos_list = zscore_scaler.fit_transform(cluster_cos_list_tmp)

        # cluster_distances_list_tmp = []
        # for i in range(len(cluster_distances_list)):
        #     cluster_distances_list_tmp.append([data_zscore_dis[i]])
        # cluster_distances_list = cluster_distances_list_tmp
        #
        # cluster_norm_list_tmp = []
        # for i in range(len(cluster_norm_list)):
        #     cluster_norm_list_tmp.append([data_zscore_norm[i]])
        # cluster_norm_list = cluster_norm_list_tmp
        # X = np.column_stack((np.abs(data_zscore_dis), np.abs(data_zscore_norm)))
        X = np.column_stack((cluster_distances_list, cluster_norm_list, cluster_cos_list))
        new_x = []
        for i in range(len(X)):
            if i not in dis_filter:
                new_x.append(X[i])

        print(X)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=2, random_state=42)  # 设置聚类簇的数量和初始化方法
        kmeans.fit(X)
        # 获取聚类标签
        labels = kmeans.labels_
        # 创建 HDBSCAN 对象
        # import hdbscan
        #
        # min_cluster_size = 2
        # min_samples = 1
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        # # 拟合数据并获取标签
        # labels = clusterer.fit_predict(X)

        # from sklearn.cluster import DBSCAN
        # # 创建 DBSCAN 对象
        # eps = 0.3
        # min_samples = 2
        # dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # 拟合数据并获取标签
        # labels = dbscan.fit_predict(X)

        # 输出聚类标签
        print(f"{choice_cluster}-means:{labels}")
        print(f"filter is {dis_filter}")
        cluster_indices_0 = []
        cluster_indices_1 = []

        for i in dis_filter:
            labels = np.insert(labels, i, -1)

        print(f"labels {labels}")
        for idx, i in enumerate(labels):
            if i == 0:
                for ii in cluster_indices_dict[idx]:
                    cluster_indices_0.append(ii)
            elif i == 1:
                for ii in cluster_indices_dict[idx]:
                    cluster_indices_1.append(ii)

        # self.conf['threshod'] = 10
        # print(f"过滤阈值是{self.conf['threshod']}")

        filter_list_first = []
        for i in dis_filter:
            for index in indices_dict[i]:
                filter_list_first.append(index)
        print(f"第一次过滤数量为{len(filter_list_first)}坐标为{sorted(filter_list_first)}")
        filter_list = []
        for k in range(choice_cluster):
            if len(cluster_indices_dict[k]) < self.conf['threshod']:
                for i in cluster_indices_dict[k]:
                    filter_list.append(i)
        print(f"第二次过滤数量为{len(filter_list)}坐标为{sorted(filter_list)}")

        for i in filter_list_first:
            filter_list.append(i)
        filter_list = sorted(filter_list)

        if len(cluster_indices_0) > len(cluster_indices_1):
            agg_tmp = cluster_indices_0
            mal_tmp = cluster_indices_1
        else:
            agg_tmp = cluster_indices_1
            mal_tmp = cluster_indices_0

        print(f"多数簇数量为{len(agg_tmp)}为{sorted(agg_tmp)}")
        print(f"少数簇数量为{len(mal_tmp)}为{sorted(mal_tmp)}")

        agg = []
        for i in agg_tmp:
            if i not in filter_list:
                agg.append(i)
        agg = sorted(agg)
        agg_norms = np.array(norms)[agg]
        # 计算均值和标准差
        mean = np.mean(agg_norms)
        std = np.std(agg_norms)
        median = np.median(agg_norms)
        # 定义阈值
        threshold = 2  # 可根据实际情况进行调整

        # 根据阈值判断异常值
        outliers = []
        z_scores = []
        for value in agg_norms:
            z_score = (value - mean) / std
            z_scores.append(z_score)
            if abs(z_score) > threshold:
                outliers.append(value[0])
        # print("原始数据:", agg_norms)
        # print("异常值:", outliers)
        print("=============")
        print(agg)
        print(agg_norms)
        print(f"socore is {z_scores}")
        index = np.argwhere(agg_norms[:] == outliers)
        filter_index = [idx[0] for idx in index]
        filter_index_indices = []
        agg_list = []
        for idx, i in enumerate(agg):
            if idx not in filter_index:
                agg_list.append(i)
            else:
                filter_index_indices.append(i)
        agg = sorted(agg_list)
        print(f"第三次过滤的坐标为{filter_index}")
        print(f"最终聚合列表{agg}")

        # =========================================================================================================

        # 平均梯度 #######################################################################
        grads_average = []
        for p in list(self.global_model.parameters()):
            grads_average.append(torch.from_numpy(np.zeros(p.data.shape)))

        # threshod = 5
        # for idx, i in enumerate(self.choice_id):
        #     if idx not in agg:
        #         self.record_mal[i] += 1
        #     elif self.record_mal[i] > threshod:
        #         index = np.argwhere(agg[:] == idx)
        #         print(f"删除{agg[index]}")
        #         np.delete(agg, (index), axis=0)
        # print(f"聚合索引是{agg}")
        # print(f"记录{self.record_mal}")

        for i in range(self.conf['client_number']):
            if i not in agg:
                self.record_mal[i] += 1
            # elif self.record_mal[i] > threshod:
            #     index = np.argwhere(agg[:] == i)[0]
            #     print(f"删除{agg[index]}")
            #     np.delete(agg, (index), axis=0)
        print(self.record_mal)
        # 聚合权重
        alpha = [0 for _ in range(len(grads_list))]
        for i in agg:
            alpha[i] = 1 / len(agg)

        print(alpha)

        # 梯度平均
        for i, _ in enumerate(grads_list):
            if i in agg:
                for ii, _ in enumerate(grads_list[i]):
                    grads_average[ii] += alpha[i] * grads_list[i][ii]

        # 更新平均值放到GPU上
        for i, _ in enumerate(grads_list):
            for ii, _ in enumerate(grads_list[i]):
                grads_average[ii] = grads_average[ii].to(self.conf['device'])

        # 更新模型
        for i, p in enumerate(list(self.global_model.parameters())):
            p.data += grads_average[i]
        return grads_average
