import copy as copy


class Client(object):
    def __init__(self, conf):
        self.conf = conf
        self.local_model = None
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.optimizer = None
        self.criterion = None

    def train(self):
        self.local_model.train()
        epoch = self.conf['local_epoch']
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


    #始终用一个模型的版本
    def train_(self):
        start_model = []
        for p in list(self.local_model.parameters()):
            start_model.append(copy.deepcopy(p.data.cpu()))
        self.local_model.train()
        epoch = self.conf['local_epoch']

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
