from models.init_model import MLP, LogisticRegression, CNNMnist, Cifar10Net, SimpleCNN
from models.resnet import ResNet18


def get_model(conf):
    name = conf["model"]
    if name == "mlp":
        model = MLP()
    elif name == 'lr':
        input_size = 784
        num_classes = 10
        model = LogisticRegression(input_size, num_classes)
    elif name == "cnn":
        if conf['datasets'] == 'mnist':
            model = CNNMnist()
        elif conf['datasets'] == 'cifar' or conf['datasets'] =='cifar10':
            # model = CNNCifar()
            model = Cifar10Net()
        elif conf['datasets'] == 'fashionmnist':
            print("使用femnist")
            model = SimpleCNN()
    elif name == "resnet18":
        model = ResNet18()
    elif name == "resnet34":
        model = ResNet34()
    else:
        print("识别不到模型,您要的模型是：" + name)
        assert (2 == 1)

    return model