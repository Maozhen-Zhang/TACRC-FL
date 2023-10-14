import math

import numpy as np
import torch
from sklearn.metrics import pairwise_distances


class Helper:
    def __init__(self, conf):
        self.conf = conf


    @staticmethod
    def model_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def grad_norm(grad_list):
        squared_sum = 0
        for layer in enumerate(grad_list):
            squared_sum += torch.sum(torch.pow(layer, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def grad_layer_norm(layer):
        squared_sum = 0
        squared_sum += torch.sum(torch.pow(layer, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def cosine_similarity(A, B):

        norm_A = np.array(torch.sum(torch.pow(A, 2)))
        norm_B = np.array(torch.sum(torch.pow(B, 2)))

        A = np.array(A)
        B = np.array(B)
        cosine_similarity = np.dot(A, B) / (norm_A * norm_B)
        return cosine_similarity
    @staticmethod
    def grad_layer_cosine_similarity(A, B, norm_A, norm_B):
        cosine_similarity = np.dot(A, B) / (norm_A * norm_B)
        return cosine_similarity

    @staticmethod
    def hook_fn(module, input, output):
        """
        # 钩子函数，用于捕获特定层的特征图

        :param module:
        :param input:
        :param output:
        :return:
        """
        global target_layer_features
        target_layer_features = output
    @staticmethod
    # Min-Max 归一化
    def min_max_normalize(data):
        """
        字典数据类型的输入
        :param data: dict()
        :return:
        """

        min_val = min(data.values())
        max_val = max(data.values())

        # print(f"归一化中，最大最小值，{max_val, min_val}")

        normalized_data = {k: (v - min_val) / (max_val - min_val) for k, v in data.items()}
        return normalized_data

    @staticmethod
    def min_max_normalize__(data):
        """
        字典数据类型的输入
        :param data: dict()
        :return:
        """
        tmp = []
        for i in data:
            tmp.append(i[0])
        data = tmp
        print(data)
        min_val = min(data)
        max_val = max(data)

        # print(f"归一化中，最大最小值，{max_val, min_val}")
        normalized_data = [(v - min_val) / (max_val - min_val) for v in data]
        return normalized_data
    @staticmethod
    def min_max_normalize___(data):
        """
        字典数据类型的输入
        :param data: dict()
        :return:
        """
        tmp = []
        for i in data:
            tmp.append(i[0])
        data = tmp
        min_val = min(data)
        max_val = max(data)

        # print(f"归一化中，最大最小值，{max_val, min_val}")
        normalized_data = [[(v - min_val) / (max_val - min_val)] for v in data]
        return normalized_data

    @staticmethod
    # Min-Max 归一化
    def min_max_normalize_(data):
        """
        字典数据类型的输入
        :param data: dict()
        :return:
        """

        min_val = min(data)
        max_val = max(data)

        # print(f"归一化中，最大最小值，{max_val, min_val}")

        normalized_data = [ (v - min_val) / (max_val - min_val) for v in data]
        return normalized_data

    @staticmethod
    def mean_nomalize(data):
        # 计算平均值
        mean = np.mean(data)
        min_val = min(data)
        max_val = max(data)

        # 计算每个数据点与平均值的绝对距离
        distances = np.abs(data - mean)

        # 计算最大距离
        max_distance = np.max(distances)

        sign_distances = data - mean

        # 进行归一化
        normalized_data = (sign_distances / max_distance)
        return normalized_data