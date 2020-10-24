    # -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:30:53 2020

@author: Administrator
"""


import torch
import torchvision
import sys
import numpy as np
import d2lzh_pytorch as d2l
from torch import nn


# 定义模型-含隐藏层、激活函数的多层感知机

# 初始化参数
num_inputs, num_hidden, num_outputs = 784, 256, 10 # 隐藏层单元个数是超参数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hidden)), dtype=torch.float)
b1 = torch.zeros(num_hidden, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hidden, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
# 合并，方便处理
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)
    
# 定义激活函数
def relu(X):
    # max(x, 0)
    return torch.max(input=X, other=torch.tensor(0.0))

# 定义模型
def net(X):
    X = X.view(-1, num_inputs) # flatten一下
    H = relu(torch.mm(X, W1) + b1)
    return torch.mm(H, W2) + b2

# 定义损失函数
# PyTorch提供了一个包括softmax运算的交叉熵损失函数
loss = nn.CrossEntropyLoss()


# 获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 训练模型
num_epochs, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)



    
    



    
