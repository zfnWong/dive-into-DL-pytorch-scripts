# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 00:41:10 2020

@author: Administrator
"""


import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch as d2l


# 定义模型

# 定义参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 定义网络
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs), 
        )

# 初始化模型参数(但其实初始化网络时，参数已经初始化)
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

# 获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer) # lr已经包含在optimizer中
