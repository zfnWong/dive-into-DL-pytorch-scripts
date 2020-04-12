# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:01:42 2020

@author: Administrator
"""

import torch
import numpy as np
# 由于data常用作变量名，所以用Data代替
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


# 生成数据集
num_examples, num_inputs = 1000, 2
true_w, true_b = [2, -3.4], 4.2
features = torch.randn(num_examples, num_inputs)  # feature : shape (num_examples, num_inputs)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b # labels : shape (num_examples)
# 引入噪声
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.shape), dtype=torch.float)


# 读取数据
batch_size = 10  # 批量大小
# 将 features 和 labels 组合组合后转化成 TensorDataset 类型
dataset = Data.TensorDataset(features, labels)
# 将 TensorDataset 类型再转化迭代器Iteration，可以打乱顺序(shuffle)后读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# 示例
# for X, y in data_iter:
#     print(X,y)
#     break


# 定义模型
net = nn.Sequential()    
net.add_module('linear', nn.Linear(num_inputs, 1))  # 添加线性回归模块
# print(net) # 输出模型
# for param in net.parameters(): # 输出模型参数 (模型在创建的时候，其实参数就已经随机初始化了
#     print(param)

# 前面已经说过，模型在创建的时候，其实参数就已经随机初始化了
# 但这里可以自定义初始化方式来初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)


# 定义损失函数
loss = nn.MSELoss()


# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)

# 分别指定学习率
# optimizer = optim.SGD([
#         {'params': net.subnet1.parameters()}, # lr=0.03
#         {'params': net.subnet2.parameters(), lr: 0.01}
#     ], lr=0.03) #没有指定学习率的就为外层的默认学习率
# print(optimizer)


# 训练模型
num_epoch = 3

for epoch in range(num_epoch):

    train_l_sum, num_batches = 0.0, 0
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(output.size())) # 计算损失,已平均

        train_l_sum += l.item()
        num_batches += 1 # 统计一共有多少个小批量

        optimizer.zero_grad() # 梯度清零
        l.backward()  # 损失函数求导
        optimizer.step()  # 参数更新

    print('epoch %d, loss %f' % (epoch+1, train_l_sum / num_batches))

