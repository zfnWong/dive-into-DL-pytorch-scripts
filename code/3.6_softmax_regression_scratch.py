# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:24:27 2020

@author: Administrator
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import torch.utils.data as Data
import d2lzh_pytorch as d2l

    
# 定义softmax函数
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)  # keepdim可以保持原有维度，否则只剩一维
    return X_exp / partition  # 这里的partition应用了广播机制


# 初始化模型参数
num_inputs, num_outputs = 784, 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
# 记录梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 
# 定义模型前向计算
def net(X):
    return softmax(torch.mm(X.view(-1, num_inputs), W) + b)


# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    # -1 表示该维度大小由其他维度决定
    # gather(dim, index) 可以理解为映射, index（即y）是一个tensor，它的dim和必须和y_hat一样
    # 以y的值为下标，从y_hat中抽取对应的值
    # 比如 torch.gather(t, 1, torch.tensor([[0,0],[1,0]])) 的结果为 tensor([[1, 1], [4, 3]])
    return -torch.log(y_hat.gather(1, y.view(-1, 1))).sum() / y_hat.shape[0]



# 设置参数
batch_size, num_epochs, lr = 256, 3, 0.1

# 获取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 训练模型
d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())

# 绘制
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
for i in range(2):
    # test[:] 记住冒号右边的下标的元素不会包含在目标列表中！
    d2l.show_fashion_mnist(X[(5 * i):(5 * (i + 1))], titles[(5 * i):(5 * (i + 1))])
    

    
