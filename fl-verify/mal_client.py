import copy as copy

import torch

from client import Client
import torchvision.transforms.functional as TF


class MalClient(Client):
    def __init__(self, conf):
        super().__init__(conf)
    def train(self):
        epoch = self.conf['local_epoch']
        if self.conf['attack'] == "no_attack":
            for e in range(epoch):
                for idx, batch in enumerate(self.train_dataloader):
                    img, label = batch
                    img = img.to(self.conf['device'])
                    label = label.to(self.conf['device'])

                    self.optimizer.zero_grad()
                    output = self.local_model(img)
                    loss = self.criterion(output, label)
                    loss.backward()
                    self.optimizer.step()
        if self.conf['attack'] == "how_to_backdoor":
            for e in range(epoch):
                for idx, batch in enumerate(self.train_dataloader):
                    img, label = batch
                    batch_size = img.shape[0]
                    attack_img = (TF.erase(img, 0, 0, 5, 5, 0).to(self.conf['device']))
                    attack_target = torch.zeros(batch_size, dtype=torch.long).to(self.conf['device'])
                    self.optimizer.zero_grad()
                    output = self.local_model(attack_img)
                    loss = self.criterion(output, attack_target)
                    loss.backward()
                    self.optimizer.step()
        else:
            for e in range(epoch):
                for idx, batch in enumerate(self.train_dataloader):
                    img, label = batch
                    img = img.to(self.conf['device'])
                    label = label.to(self.conf['device'])

                    self.optimizer.zero_grad()
                    output = self.local_model(img)
                    loss = self.criterion(output, label)
                    loss.backward()
                    self.optimizer.step()

    def train_(self):
        start_model = []
        for p in list(self.local_model.parameters()):
            start_model.append(copy.deepcopy(p.data.cpu()))
        self.local_model.train()
        epoch = self.conf['local_epoch']

        if self.conf['attack'] == "no_attack":
            for e in range(epoch):
                for idx, batch in enumerate(self.train_dataloader):
                    img, label = batch
                    img = img.to(self.conf['device'])
                    label = label.to(self.conf['device'])

                    self.optimizer.zero_grad()
                    output = self.local_model(img)
                    loss = self.criterion(output, label)
                    loss.backward()
                    self.optimizer.step()
        if self.conf['attack'] == "how_to_backdoor":
            for e in range(epoch):
                for idx, batch in enumerate(self.train_dataloader):
                    img, label = batch
                    batch_size = img.shape[0]
                    attack_img = (TF.erase(img, 0, 0, 5, 5, 0).to(self.conf['device']))
                    attack_target = torch.zeros(batch_size, dtype=torch.long).to(self.conf['device'])
                    self.optimizer.zero_grad()
                    output = self.local_model(attack_img)
                    loss = self.criterion(output, attack_target)
                    loss.backward()
                    self.optimizer.step()
        else:
            for e in range(epoch):
                for idx, batch in enumerate(self.train_dataloader):
                    img, label = batch
                    img = img.to(self.conf['device'])
                    label = label.to(self.conf['device'])

                    self.optimizer.zero_grad()
                    output = self.local_model(img)
                    loss = self.criterion(output, label)
                    loss.backward()
                    self.optimizer.step()
        grad_list = []
        end_model = []
        for p in list(self.local_model.parameters()):
            end_model.append(copy.deepcopy(p.data.cpu()))
        for (p1, p2) in zip(end_model, start_model):
            grad_list.append(p1 - p2)

        # 还原
        for idx, p in enumerate(list(self.local_model.parameters())):
            p.data = copy.deepcopy(start_model[idx].to(self.conf['device']))
        return grad_list
