import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义编码器类
class Encoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# 定义解码器类
class Decoder(nn.Module):
    def __init__(self, encoding_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

# 设置参数
input_dim = 784  # 输入数据维度（MNIST图像像素数）
encoding_dim = 32  # 编码维度
output_dim = 784  # 解码器输出维度

# 创建编码器和解码器实例
encoder = Encoder(input_dim, encoding_dim)
decoder = Decoder(encoding_dim, output_dim)

# 定义自动编码器
autoencoder = nn.Sequential(
    encoder,
    decoder
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 准备数据（示例使用随机数据）
num_samples = 1000
data = torch.rand(num_samples, input_dim)

# 开始训练
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = autoencoder(data)
    loss = criterion(outputs, data)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用编码器和解码器进行重构
encoded_data = encoder(data)
decoded_data = decoder(encoded_data)
