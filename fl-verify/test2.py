import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from models.init_model import CNNMnist

# 示例的卷积神经网络


model = CNNMnist()

# 输入数据
input_data = torch.randn(1, 1, 28, 28)  # 假设输入尺寸为 64x64 的 RGB 图像

# 用于存储特定层的特征图的变量
target_layer_features = None


# 钩子函数，用于捕获特定层的特征图
def hook_fn(module, input, output):
    global target_layer_features
    target_layer_features = output


# 选择特定的卷积层
target_layer = model.conv1  # 假设要获取第一个卷积层的特征图

# 注册钩子到特定层
hook = target_layer.register_forward_hook(hook_fn)

# 前向传播
output = model(input_data)
# 获取特定层的特征图
feature_map = target_layer_features.detach().squeeze().cpu()

print(feature_map.shape)
# 移除钩子
hook.remove()

# 使用灰度颜色映射将多通道特征图转换为单通道
converted_feature_map = torch.sum(feature_map, dim=0, keepdim=True)

# 将特征图保存为图像文件
feature_map = transforms.ToPILImage()(converted_feature_map)
feature_map.save("feature_map.png")

print("Feature map saved as feature_map.png")
