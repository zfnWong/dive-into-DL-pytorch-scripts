# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:32:19 2020

@author: Administrator
"""

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义Inception块--4条并行线路
# 输入输出形状相同
class Inception(nn.Module):
    # c1-c4为每条线路里的层的输出通道数, in_c为输入通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        
        # 线路1，单 1*1 卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        
        # 线路2，1*1卷积层后接3*3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1) # 减少输入通道数
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        
        # 线路3, 1*1卷积层后接5*5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1) # 减少输入通道数
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        
        # 线路4, 3*3最大池化层后接1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # 形状不变
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
        
    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维(dim=1)上连结
    
   
# 定义GoogLeNet模型

# 模块1
# 一个卷积层        
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 长宽减半
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化层，长宽减半
    )

# 模块2
# 两个卷积层
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),  # 形状不变
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 形状不变
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化层，长宽减半
    )

# 模块3
# 两个Inception块
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),  # out_channels 64+128+32+32=256
    Inception(256, 128, (128, 192), (32, 96), 64),  # 128+192+96+64=480
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化层
    )

# 模块4
# 5个Inception块
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16,48), 64),  # 192+208+48+64=512
    Inception(512, 160, (112, 224), (24, 64), 64),  # 160+224+64+64=512
    Inception(512, 128, (128, 256), (24, 64), 64),  # 128+256+64+64=512
    Inception(512, 112, (144, 288), (32, 64), 64),  # 112+288+64+64=528
    Inception(528, 256, (160, 320), (32, 128), 128),  # 256+320+128+128=832
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化层 长宽减半
    )

# 模块5
# 2个Inception块接全局平均池化层
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),  # 256+320+128+128=832
    Inception(832, 384, (192, 384), (48, 128), 128),  # 384+384+128+128=1024
    d2l.GlobalAvgPool2d()
    )

# 总模型
# 接一个维度转换和全连接层
net = nn.Sequential(b1, b2, b3, b4, b5, d2l.FlattenLayer(), nn.Linear(1024, 10))

# 测试
# GoogLeNet计算复杂，而且不如vgg那样便于修改通道数
# 将输入的高宽从224降到96
# x = torch.rand(1,1,96,96)
# for blk in net.children():
#     x = blk(x)
#     print(x.shape)


# 参数
lr, num_epochs, batch_size = 0.001, 5, 128
# 数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)  # 为减小数据规模，resize减小为96
# 优化器
optimizer = optim.Adam(net.parameters(), lr=lr)
# 训练
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


