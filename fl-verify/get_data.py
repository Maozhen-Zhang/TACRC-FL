import random

from collections import defaultdict
from torchvision import datasets, transforms

# 其他模块
import numpy as np


def print_distribution(dataset_slices):
    # 打印分布情况
    len_datasets = 0
    # for i in dataset_slices:
        # print(type(i))
    for idx, dataset_slice in enumerate(dataset_slices):
        label_counts = [0 for i in range(11)]
        label_sum = 0
        for group in dataset_slice:
            label_counts[group[1]] += 1
            label_sum += 1
        label_counts[-1] = label_sum
        len_datasets += label_sum
        print(f"|---Client {idx} datasets distribute is {label_counts}")
    print(f"|---Sum datasets lenth is {len_datasets}")


def get_dataset(conf):
    dataset = conf['datasets']
    path = conf['datasets_path']
    if dataset == 'mnist':
        path = path
        # 因为resnet18输入的CHW是(3, 224, 224)，而mnist中单张图片CHW是(1, 28, 28)，所以需要对MNIST数据集进行预处理。

        # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor(),
        #                                 transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), ])

        # # 定义数据预处理操作
        transform = transforms.Compose([
            # 将图像调整为固定的尺寸，如28x28
            transforms.Resize((28, 28)),
            # 将图像转换为Tensor格式，并进行归一化，将像素值缩放到 [0, 1] 的范围
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.ToTensor(),
        #                                 transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), ])

        train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
    elif dataset == 'fashionmnist':
        # path = path + '/FashionMNIST'
        transform = transforms.Compose([
            # 将图像调整为固定的尺寸，如28x28
            transforms.Resize((28, 28)),
            # 将图像转换为Tensor格式，并进行归一化，将像素值缩放到 [0, 1] 的范围
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(path, train=False, download=True,
                                             transform=transform)

    elif dataset == 'cifar' or dataset == 'cifar10':
        path = path + '/CIFAR-10'

        # transforms.RandomCrop： 切割中心点的位置随机选取
        # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化

        # transform_train = transforms.Compose(
        #     [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        #      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        #
        # transform_test = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


        transform_train = transforms.Compose([
            transforms.ToTensor()
            , transforms.RandomCrop(32, padding=4)  # 先四周填充0，在吧图像随机裁剪成32*32
            , transforms.RandomHorizontalFlip(p=0.5)  # 随机水平翻转 选择一个概率概率
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        train_dataset = datasets.CIFAR10(path, train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(path, train=False, download=True,
                                        transform=transform_test)
    else:
        print("Error!!The name is Error!")
        assert (1 == 2)
    return train_dataset, test_dataset


def get_iid(train_dataset, idxs):
    users = len(idxs)
    # train_dataset, test_dataset = get_dataset(conf)
    all_range = list(range(len(train_dataset)))
    data_len = int(len(train_dataset) / users)
    # SubsetRandomSampler(indices)：会根据indices列表从数据集中按照下标取元素
    # 无放回地按照给定的索引列表采样样本元素。
    dataset_slices = []
    for id in range(len(idxs)):
        client_dataset_slice = []
        train_indices_per_client = all_range[id * data_len:(id + 1) * data_len]
        for i in train_indices_per_client:
            client_dataset_slice.append(train_dataset[i])
        dataset_slices.append(client_dataset_slice)
    return dataset_slices


def get_sampling_dirichlet(conf, train_dataset, idxs):
    users = len(idxs)
    train_indices_all_client = sample_dirichlet_train_data(conf, train_dataset, users, conf['dirichlet_alpha'])
    dataset_slices = []
    for id in range(len(idxs)):
        client_dataset_slice = []
        train_indices_per_client = train_indices_all_client[id]
        for i in train_indices_per_client:
            client_dataset_slice.append(train_dataset[i])
        dataset_slices.append(client_dataset_slice)
    return dataset_slices


def sample_dirichlet_train_data(conf, train_dataset, client_number, alpha):
    cifar_classes = {}
    for indx, x in enumerate(train_dataset):
        _, target = x
        if target in cifar_classes:
            cifar_classes[target].append(indx)
        else:
            cifar_classes[target] = [indx]

    class_size = len(cifar_classes[0])
    list_per_client = defaultdict(list)
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(client_number * [alpha]))
        for user in range(client_number):
            number_of_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), number_of_imgs)]
            list_per_client[user].extend(sampled_list)

            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), number_of_imgs):]

    return list_per_client

def mal_data_setting(conf, train_dataset_slices, malicious_id):
    if conf['attack'] == 'label_flip':
        for i in range(conf['client_number']):
            if i in malicious_id:
                train_dataset = train_dataset_slices[i]
                # index_of_label_0 = [index for index, data_point in enumerate(train_dataset) if data_point[1] == 0]
                # # print(index_of_label_0)
                # # 修改标签为1的数据点
                # for index in index_of_label_0:
                #     # print(type((train_dataset[index][0],1)))
                #     train_dataset[index] = (train_dataset[index][0],1)

                index_of_labels = []
                #要反转的标签
                flip_label = [0]
                for o in flip_label:
                    index_of_label = [index for index, data_point in enumerate(train_dataset) if data_point[1] == o]
                    index_of_labels.append(index_of_label)
                for o in flip_label:
                    o_l = (o + 1) % 10
                    for index in index_of_labels[o]:
                        train_dataset[index] = (train_dataset[index][0], o_l)
        # # #验证
        # for i in range(conf['client_number']):
        #     if i in malicious_id:
        #         flip_label = [0]
        #         # for o in flip_label:
        #         #     train_dataset = train_dataset_slices[i]
        #             # index_of_labels = [index for index, data_point in enumerate(train_dataset) if data_point[1] == o]
        #         for index in index_of_labels[o]:
        #             print(train_dataset[index][1])

    return train_dataset_slices
