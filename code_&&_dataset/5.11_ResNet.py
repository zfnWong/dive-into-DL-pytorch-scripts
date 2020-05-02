# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:44:16 2020

@author: Administrator
"""

import time
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 模型定义保存在 d2lzh_pytorch中
net = d2l.resnet18(output=10, in_channels=1)
# 测试
# X = torch.rand(1,1,224,224)
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, 'output shape: ', X.shape)



# 参数
lr, num_epochs, batch_size = 0.001, 5, 256
# 数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# 优化器
optimizer = optim.Adam(net.parameters(), lr=lr)
# 训练
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

        


    
