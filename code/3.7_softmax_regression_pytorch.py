# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:12:33 2020

@author: Administrator
"""


import torch
import torchvision
import numpy as np
import sys
from torch import nn
from collections import OrderedDict
from torch.nn import init
import d2lzh_pytorch as d2l

# 定义模型
# 神经网络,用来求导，计算梯度
num_inputs, num_outputs = 784, 10
net = nn.Sequential()
net.add_module('flatten', d2l.FlattenLayer())
net.add_module('linear', nn.Linear(num_inputs, num_outputs))


# 初始化参数(但其实初始化网络时，参数已经初始化)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)


# 定义损失函数
# PyTorch提供了一个包括softmax运算的交叉熵损失函数
loss = nn.CrossEntropyLoss()


# 定义最优化算法,用来更新参数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# 初始化参数
batch_size, num_epochs = 256, 5

# 获取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 训练模型
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# 预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())

# 绘制
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
for i in range(2):
    d2l.show_fashion_mnist(X[(5 * i):(5 * (i + 1))], titles[(5 * i):(5 * (i + 1))])
    
    
    
    