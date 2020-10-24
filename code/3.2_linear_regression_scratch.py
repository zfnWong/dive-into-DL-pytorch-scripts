# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import d2lzh_pytorch as d2l


# 生成数据集
num_examples, num_inputs = 1000, 2
true_w, true_b= [2, -3.4], 4.2
features = torch.randn(num_examples, num_inputs)  # feature : shape (num_examples, num_inputs)
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b # labels : shape (num_examples)
# 引入噪声
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.shape), dtype=torch.float)


# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float)
b = torch.zeros(1, dtype=torch.float)
# 记录梯度
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义损失函数：平方损失函数
loss = d2l.squared_loss


# 定义模型，即前向计算
net = d2l.linreg


# 模型参数：学习率，迭代次数，批量大小
lr, num_epoches, batch_size = 0.03, 3, 10


# 训练模型
for epoch in range(num_epoches):

    for X, y in d2l.data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # 计算批量损失，已平均
        l.backward()  # 代价函数求导，得到梯度
        d2l.sgd([w, b], lr)  # 更新梯度的值
        w.grad.data.zero_()  # 清零梯度，否则梯度会一直累加之前的梯度
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels) # 计算总损失
    print('epoch %d, loss %f' % (epoch+1, train_l.mean().item()))

