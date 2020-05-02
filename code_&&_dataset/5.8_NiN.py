# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:44:37 2020

@author: Administrator
"""

import torch
import torchvision
from torch import nn, optim 
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# NiN 思想：以小网络构建深度网络

# 定义NiN模块-网络中的网络
# 一个NiN模块由一个卷积层和两个充当全连接层的1*1卷积层构成
def  nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
        )
    return blk

# 定义NiN模型
# 不使用全连接层，通过卷积层是输出通道数等于类别标签数
# 减小模型参数尺寸，缓解过拟合，但有时会造成训练时间增加
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),  # (224-11+4-1)/4=54
    nn.MaxPool2d(kernel_size=3, stride=2),  # (54-1)/2=26
    
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),  # 不变
    nn.MaxPool2d(kernel_size=3, stride=2),  # (26-1)/2=12
    
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),  # 不变
    nn.MaxPool2d(kernel_size=3, stride=2),  # (12-1)/2=5
    
    nn.Dropout(0.5), # 丢弃法
    
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),  # 类别标签数是10
    d2l.GlobalAvgPool2d(),  # 全局平均池化层 (batch_size, 10, 1, 1)
    d2l.FlattenLayer()  # 维度转化 (batch_size, 10)
    )
# 测试
# x = torch.rand(1,1,224,224)
# for name, blk in net.named_children(): # name_children()返回每一模块的名字和对象
#     x = blk(x)
#     print(name, 'output shape: ', x.shape)
    

# 参数
lr, num_epochs, batch_size = 0.002, 5, 64
# 数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 优化器
optimizer = optim.Adam(net.parameters(), lr=lr)
# 训练
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
