# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:52:55 2020

@author: Administrator
"""

import torch
import torchvision
from torch import nn, optim
import d2lzh_pytorch as d2l


# 构造vgg块
# 对于给定的感受野，采用堆积的小卷积核优于采用大的卷积核，因为可以增加网络深度来学习更复杂的模式，而代价差不多
# 例如，vgg中用3个3*3的卷积核替代一个7*7的卷积核，用2个3*3的卷积核替代一个5*5的卷积核
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 池化层使宽高减半
    return nn.Sequential(*blk)

# 实现vgg-11网络
# 11=8+3，8为卷积层层数，3为全连接层层数
def vgg(conv_arch, fc_features, fc_hidden_unit=4096):
    net = nn.Sequential()
    # 卷积层部分
    # enumerate()用于将一个可遍历的数据对象(如列表、元组或字符串)
    # 组合为一个索引序列，同时列出数据和数据下标
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_blk_"+str(i+1), vgg_block(num_convs, in_channels, out_channels))
    
    # 全连接层部分
    net.add_module(
        "fc", nn.Sequential(
            d2l.FlattenLayer(),  # 四维转二维
            nn.Linear(fc_features, fc_hidden_unit),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_unit, fc_hidden_unit),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_unit, 10)
            )
        )
    return net
# 测试模型
# conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# fc_features = 512 * 7 * 7 
# fc_hidden_units = 4096
#
# testnet = vgg(conv_arch, fc_features, fc_hidden_units)
# X = torch.rand(1,1,224,224)
# for name, blk in testnet.named_children():
#     X = blk(X)
#     print(name, 'output: shape', X.shape)



# 设置模型参数：降低参数规模
ratio = 8
small_conv_arch = ((1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),
                                            (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio))
# 经过5个vgg_block，宽高减半5次，从224变成7
small_fc_features = 512 // ratio * 7 * 7
small_fc_hidden_unit = 4096 // ratio


# 参数
lr, num_epochs, batch_size = 0.001, 5, 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 模型
net = vgg(small_conv_arch, small_fc_features, small_fc_hidden_unit)
# print(net)
# 优化器
optimizer = optim.Adam(net.parameters(), lr=lr)
# 训练
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    
    