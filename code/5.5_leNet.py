# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:52:38 2020

@author: Administrator
"""


import torch
import torchvision
import torch.nn as nn
import d2lzh_pytorch as d2l
import torch.optim as optim


# 定义模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # convolutional layer， samples * channels * height * width
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2)
            )
        # fc = full connection
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
            )
        
    def forward(self, img): # img is 4-dimension [num_sample,channels,h,w]
        feature = self.conv(img)
        output = self.fc(feature.view(feature.shape[0], -1)) # 进入全连接层之前要flatten一下
        return output

net = LeNet()

# 获取Mnist数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置参数
num_epochs, lr= 5, 0.01
# 设置优化器
optimizer = optim.Adam(net.parameters(), lr)

# 训练模型
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)





