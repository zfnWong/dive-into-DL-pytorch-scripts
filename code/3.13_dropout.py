# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:35:22 2020

@author: Administrator
"""

import torch
import numpy as np
import d2lzh_pytorch as d2l
import torch.nn as nn


# 从零开始
# 定义丢弃函数
def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1 # 确保丢弃率在 [0,1] 之间
    keep_prob = 1 - drop_prob

    # 因为keep_prob要作为分母，当keep_prob=0时把全部元素都丢弃，
    if keep_prob == 0:
        return torch.zeros_like(X)

    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob


# 初始化
num_inputs, num_hidden1, num_hidden2, num_outputs = 784, 256, 256, 10
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hidden1)),
                                                dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hidden1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hidden1, num_hidden2)),
                                                dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hidden2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_hidden2, num_outputs)),
                                                dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)
# 设置变量存储合并后的参数集， 方便梯度清零和更新
params = [W1, b1, W2, b2, W3, b3]


# 使用丢弃法
drop_prob1, drop_prob2 = 0.2, 0.5
# 定义模型
def net(X, is_training=True):
    X = X.view(-1, num_inputs)  
    H1 = (torch.mm(X, W1) + b1).relu()

    # 训练状态下使用丢弃法
    if is_training:
        H1 = dropout(H1, drop_prob1)
    H2 = (torch.mm(H1, W2) + b2).relu()

    if is_training:
        H2 = dropout(H2, drop_prob2)

    return torch.mm(H2, W3) + b3


# 损失函数
loss = torch.nn.CrossEntropyLoss()


# 设置参数
num_epochs, lr, batch_size = 5, 0.5, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 训练
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)




# 简洁实现
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hidden1),
        nn.ReLU(),
        nn.Dropout(drop_prob1),
        nn.Linear(num_hidden1, num_hidden2),
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(num_hidden2, 10)
        )

# 初始化参数
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)
    
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

