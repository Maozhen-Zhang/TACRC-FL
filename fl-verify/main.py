import copy
import logging
import math
import random

import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset
import datetime

import byzantine
from helper import Helper
from client import Client
from get_conf import get_conf
from get_data import get_dataset, get_sampling_dirichlet, print_distribution, get_iid, mal_data_setting
from get_model import get_model
from mal_client import MalClient
from server import Server

logger = logging.getLogger("logger")
#############################################
current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
# 训练模型所需参数
# 用于记录损失值未发生变化batch数
counter = 0
# 学习率
Lr = 0.001
#############################################
conf = get_conf(current_time)
helper = Helper(conf)

# 【固定】设置良性、恶意客户端
candidate_id = [i for i in range(conf['client_number'])]
malicious_all_id = [i for i in range(conf['mal_number'])]
benign_all_id = [i for i in range(conf['mal_number'], conf['client_number'])]
# 数据集处理
train_dataset, test_dataset = get_dataset(conf)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32)
# train_dataset_slices = get_iid(train_dataset,idxs)
train_dataset_slices = get_sampling_dirichlet(conf, train_dataset, candidate_id)
train_dataset_slices_clean = train_dataset_slices
if conf['attack'] == "label_flip":
    train_dataset_tmp = copy.deepcopy(train_dataset)
    train_dataset_slices_mal = get_sampling_dirichlet(conf, train_dataset_tmp, candidate_id)
    train_dataset_slices_mal = mal_data_setting(conf, train_dataset_slices_mal, malicious_all_id)

print_distribution(train_dataset_slices)

# 模型获取
model = get_model(conf)
model = model.to(conf['device'])

# 初始化对象
server = Server(conf, helper, test_dataloader, model)
benign_client = Client(conf)
mal_client = MalClient(conf)
progress_e = tqdm(total=conf["global_epoch"], desc="|===\033[1;31m FL Epoch is \033[0m=========", leave=True,
                  position=0, dynamic_ncols=True)
# 记录当前最小损失值
valid_loss_min = np.Inf
for e in range(conf['global_epoch']):
    progress_e.update(1)

    ##随机选择的模式
    # candidate_id = [i for i in range(conf['client_number'])]
    if conf['defense'] != 'no_defnse':
        benign_ratio = conf['sample_ratio'] * (1 - conf['mal_number'] / conf['client_number'])
        mal_ratio = conf['sample_ratio'] * conf['mal_number'] / conf['client_number']
    else:
        benign_ratio = conf['sample_ratio']
        mal_ratio = 0
    benign_id = sorted(random.sample(benign_all_id, int(benign_ratio * conf['client_number'])))
    malicious_id = sorted(random.sample(malicious_all_id, int(mal_ratio * conf['client_number'])))
    choice_id = malicious_id + benign_id

    server.choice_id = choice_id
    server.malicious_id = malicious_id
    # malicious_id = []

    print(f" |===Setting=====================================")
    print(f"|---Dataset: {conf['datasets']}")
    print(f"|---model: {conf['model']}")
    print(f"|---client num: {conf['client_number']}")
    print(f"|---sample ratio: {conf['sample_ratio']}")
    print(f"|---attack: {conf['attack']}")
    print(f"|---defense: {conf['defense']}")
    print(f"|---Lr is {Lr}")
    logger.info(f"|---This epoch client is {choice_id}")
    logger.info(f"|---The malicious client is {malicious_id}")

    # 初始化存储
    grads_list = []
    grads_list_fc = []
    progress_c = tqdm(total=len(choice_id), desc=f'In {e} epoch , {len(choice_id)} clients will train', leave=False,
                      position=1, dynamic_ncols=True)
    if conf["attack"] == "label_flip":
        attack_epoch = [att_e for att_e in range(0, conf['global_epoch'])]

        if e in attack_epoch:
            train_dataset_slices = train_dataset_slices_mal
        else:
            train_dataset_slices = train_dataset_slices_clean
    for i in choice_id:
        progress_c.update(1)
        if i in malicious_id:
            client = mal_client
        else:
            client = benign_client

        client.local_model = copy.deepcopy(server.global_model)

        client.train_dataset = train_dataset_slices[i]
        client.train_dataloader = DataLoader(dataset=client.train_dataset, batch_size=conf["batch_size"])
        # 动态调整学习率
        # if counter / 10 == 1:
        #     counter = 0
        #     Lr = Lr * 0.5
        optimizer = torch.optim.SGD(client.local_model.parameters(), lr=Lr, momentum=0.9, weight_decay=5e-4)

        # optimizer = torch.optim.SGD(client.local_model.parameters(), lr=0.01,
        #                             momentum=0.0005, weight_decay=0.0001)
        criterion = torch.nn.functional.cross_entropy
        client.optimizer = optimizer
        client.criterion = criterion
        client.train()


        # 计算更新梯度
        grad_list = []
        grad_list_fc = []
        for idx, name in enumerate(client.local_model.state_dict()):
            grad_list.append((client.local_model.state_dict()[name] - server.global_model.state_dict()[name]).cpu())
            if 'fc' in name:  # 假设全连接层名称包含 'fc'
                grad_list_fc.append(
                    (client.local_model.state_dict()[name] - server.global_model.state_dict()[name]).cpu())
            if conf['model'] == "lr":
                grad_list_fc = grad_list
        grads_list.append(grad_list)
        grads_list_fc.append(grad_list_fc)

    progress_c.close()
    # if i % 10 == 0:
    #     acc, loss = server.evaluate_accuracy(test_dataloader, client.local_model)
    #     logger.info(f"|===local agents=================")
    #     logger.info(f"|---Client {i} is traing ...")
    #     logger.info(f"|---Client {i}'s acc is:{acc:.2f}, loss is : {loss:.2f}")
    #     logger.info(f"|--------------------------------")
    #             logger.info(f"|---Grads is  {grads_list[i]}...")
    #             logger.info(f"|--------------------------------")

    if conf['attack'] == 'additive_noise':
        grads_list = byzantine.attack_additive_noise(grads_list,0.6, choice_id, malicious_id)
        grads_list_fc = byzantine.attack_additive_noise(grads_list_fc,0.6, choice_id, malicious_id)
    elif conf['attack'] == 'attack_trimmedmean':
        grads_list = byzantine.attack_trimmedmean(model, grads_list, malicious_id, b=1.5)
        grads_list_fc = byzantine.attack_trimmedmean(model, grads_list_fc, malicious_id, b=1.5)

    elif conf['attack'] == 'attack_krum':
        for idx, _ in enumerate(grads_list[0]):
            grads_list = byzantine.attack_krum(model, grads_list, malicious_id, idx)
            grads_list_fc = byzantine.attack_krum(model, grads_list_fc, malicious_id, idx)
    elif conf['attack'] == 'inner_product':
        grads_list = byzantine.attack_innerProduct(grads_list, 1, choice_id, malicious_id)
        grads_list_fc = byzantine.attack_innerProduct(grads_list_fc, 1, choice_id, malicious_id)
    elif conf['attack'] == 'label_flip' and e in attack_epoch:
        attack_value = conf['attack_value']
        grads_list = byzantine.attack_amplify(grads_list, choice_id, malicious_id, attack_value)
        grads_list_fc = byzantine.attack_amplify(grads_list_fc, choice_id, malicious_id, attack_value)
    elif conf['attack'] == 'gradient_rise':
        grads_list = byzantine.attack_sign_flipping(grads_list, choice_id, malicious_id, attack_value=-3)
        grads_list_fc = byzantine.attack_sign_flipping(grads_list_fc, choice_id, malicious_id, attack_value=-3)
    elif conf['attack'] == 'mutil':
        grads_list_fc = byzantine.attack_trimmedmean(model, grads_list_fc, [i for i in range(10)], b=1.5)
        grads_list_fc = byzantine.attack_krum(model, grads_list_fc, [i for i in range(10,20)], idx)
        grads_list_fc = byzantine.attack_innerProduct(grads_list_fc, 1, choice_id, [i for i in range(20,30)])
        grads_list_fc = byzantine.attack_amplify(grads_list_fc, choice_id, [i for i in range(30,40)], 3)



    if conf['defense'] == "no_defense":
        server.fedavg(grads_list)
    elif conf['defense'] == "confidence":
        server.confidence
    elif conf['defense'] == 'trimmedmean':
        global_weight = server.trimmedmean(grads_list)
    elif conf['defense'] == "cluster":
        server.cluster_new(grads_list, grads_list_fc)


    if e % 1 == 0:
        YELLOW = '\033[38;5;220m'
        RED = '\033[1;31m'
        RESET = '\033[0m'
        acc, loss = server.evaluate_accuracy(server.global_model, test_dataloader)

        print(f" |===Test========================================")
        print(f"|---Test loss is : {loss:.4f} [{counter}], Test acc is:\033[1;31m{acc * 100:.2f}\033[0m")
        # print('|---Test Loss: {:.4f}[{}] Acc: {:.4f}'.format(loss, counter, acc*100))

        if conf['attack'] == "how_to_backdoor":
            asr, asr_loss = server.evaluate_backdoor_accuracy(server.global_model, test_dataloader)
            print(f"|===Test backdoor============================")
            print(f"|------------Test asr is:\033[1;31m {asr * 100:.2f}\033[0m")
        elif conf['attack'] == "label_flip":
            acc_label, asr = server.evaluate_label_flip_accuracy(server.global_model, test_dataloader)
            print(
                f"|===Test label flip sigle label acc is \033[1;31m{acc_label * 100:.2f}\033[0m, asr is \033[1;31m{asr * 100:.2f}\033[0m")
        print(f"|===============================================")

    # 存储
    if loss < valid_loss_min:
        best_acc = acc
        # 保存当前模型
        # best_model_wts = copy.deepcopy(model.state_dict())
        state = {
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        # 只保存最近2次的训练结果

        save_name = '{}{}.pth'.format(conf['logPath'], current_time)
        torch.save(state, save_name)  # \033[1;31m 字体颜色：红色\033[0m
        print(
            "|---已保存最优模型，准确率:\033[38;5;220m {:.2f}%\033[0m，文件名：{}".format(best_acc * 100, save_name))

        valid_loss_min = loss
        counter = 0
    else:
        counter += 1
    # 累计训练信息
    print(f"|=============================================================")
    print(" ")
    print(" ")
    print(" ")
progress_e.close()
