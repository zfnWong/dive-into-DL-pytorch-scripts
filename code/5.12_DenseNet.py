# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:00:20 2020

@author: Administrator
"""

import time
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 改良版卷积块
# 保持宽高不变，只改变通道数
def conv_block(in_channels, out_channels):
    # 批量归一化、激活函数、卷积层组合结构
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    return blk


# 稠密块, 沿用了重复的思想，改造了残差块的架构，形状不变
# 通道数变化情况： in -> in+out -> (in+out)+out -> ((in+out)+out)+out -> ...
# 跨越的数据通道，一次跨越一层卷积层，在稠密层中多次跳跃
class DenseBlock(nn.Module):
    
    # 卷积层数目，输入通道，输出通道
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        
        # 模块组成
        net = []
        for i in range(num_convs):
            tmp_c = in_channels + i * out_channels
            # 添加改良版卷积块，每个块拼接前的输出 channels 都是 out_channels
            net.append(conv_block(tmp_c, out_channels))
            
        self.net = nn.ModuleList(net) # list转成ModuleList，才会自动记录梯度
        self.out_channels = in_channels + num_convs * out_channels # 最终输出通道数
        
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 第1维是通道数，需要连结
        return X
# 测试
# blk =  DenseBlock(2, 3, 10)
# X = torch.rand(4, 3, 8, 8)
# Y = blk(X)
# print(Y.shape) # [4, 23, 8, 8]

 

# 过渡块：降低复杂度
# 因为稠密块不断连结得到的 channels 太大了
# 稠密块中没有使用池化层，而是在过渡块中使用
def TransitionBlock(in_channels, out_channels):
    
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 使用1*1卷积层减少通道数
        nn.AvgPool2d(kernel_size=2)  # 使用平均池化层减半高宽
        )
    
    return blk
# 测试
# blk = TransitionBlock(23, 10)
# Y = blk(Y)
# print(Y.shape)
    


# 定义 DenseNet 模型
# 首先，类似 ResNet 模块开头
net = nn.Sequential(
    # 前两层与GoogLeNet类似,在卷积层后多了批量归一化层
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


# 然后，添加稠密块和过渡块
num_channels, growth_rate = 64, 32 # **当前通道数**(稠密块和过渡块的channels的中介)、增长率
num_convs_in_dense_block = [4, 4, 4, 4] # 稠密块的跨越线路“跳跃”次数

for i, num_convs in enumerate(num_convs_in_dense_block):
    
    # 稠密块
    DB = DenseBlock(num_convs, num_channels, growth_rate)  # growth_rate即out_channels
    net.add_module("DenseBlock_%d" % (i+1), DB)
    
    # 过渡块
    num_channels = DB.out_channels  # 过渡块的输入通道数等于稠密块的输出通道数
    # 过渡块只在稠密块*之间*加入
    if i != len(num_convs_in_dense_block)-1:
        # 过渡块使得通道数、高、宽减半
        net.add_module("TransitionBlock_%d" % (i+1), TransitionBlock(num_channels, num_channels // 2))
        num_channels = num_channels // 2


# 最后，同 ResNet 结上全局池化层和全连接层
net.add_module("BN", nn.BatchNorm2d(num_channels))
net.add_module("relu", nn.ReLU())
net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_channels, 10)))
# print(num_channels)

# print(net)
# 测试
# X = torch.rand(1, 1, 96, 96)
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape: ', X.shape)



# 训练参数
lr, num_epochs, batch_size = 0.001, 5, 128
# 数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# 优化器
optimizer = optim.Adam(net.parameters(), lr=lr)
# 训练
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)