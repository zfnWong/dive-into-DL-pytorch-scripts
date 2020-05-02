# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:01:34 2020

@author: Administrator
"""


import torch
import torch.nn as nn


# 读写Tensor

# 存储Tensor
x = torch.ones(2,4)
torch.save(x, 'x.pt') # 存储的文件后缀最好是pt或pth

# 读取Tensor
x2 = torch.load('x.pt')



# 读写模型(推荐)
# 有两种方式：1、存储参数(推荐)  2、直接存储模型


# 定义模块
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
        
    def forward(self, X):
        a = self.act(self.hidden(X))
        return self.output(a)
# 通过state_dict(),以字典形式获得模块参数
net = MLP()
print(net.state_dict()) # state_dict()是一个从参数名称映射到参数Tensor的对象

# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # momentum是动量
# 通过state_dict(),以字典形式获得模块参数
print(optimizer.state_dict())

# 1、通过state_dict()读写模型参数
X = torch.rand(2,3)
y1 = net(X)
torch.save(net.state_dict(), './net.pt') # 存储模型参数
net2 = MLP()  # 创建相同模型
net2.load_state_dict(torch.load('./net.pt')) # 读取模型参数
y2 = net2(X)
print(y1 == y2) # True


# 2、直接存储模型
y1 = net(X)
torch.save(net, './model.pt')
net2 = torch.load('./model.pt')
y2 = net2(X)
print(y1 == y2) # True







