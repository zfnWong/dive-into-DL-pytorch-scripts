# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:53:43 2020

@author: Administrator
"""

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 从零开始实现批量归一化
# 批量归一化*单次*运算
# 分训练和预测模式、卷积层和全连接层模式
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    
    # 判断当前是训练模式还是预测模式
    # Module实例的training属性默认为true
    # 在 evaluate_accuracy() 中调用.eval()后设成false
    if not is_training:
        # 预测模式，直接使用整个训练集的移动平均所得的均值和方差做标准化
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        
    else:
        # 训练模式
        assert len(X.shape) in (2, 4) # X的维度数为2或4
        if len(X.shape) == 2:
            # 全连接层，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 二维卷积层
            # 计算通道维、宽、高上的均值和方差,因为我们将每个通道视作一个特征
            # 此均值和方差针对小批量，是一个临时变量
            # 保持X的形状以便做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        # 标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)  # eps防止分母为0
        # 更新移动平均的均值和方差
        # momentum 是移动平均常数
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
        
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var  # 保存移动平均的均值和方差以最终进行预测


# 实现批量归一化层--跟在*卷积层*和*全连接层*后，激活层前
# 对于卷积层来说，num_features是输出通道数, num_dims是4
# 对于全连接层来说，num_features是输出数,num_dims是2
class BatchNorm(nn.Module):
    
    def __init__(self, num_features, num_dims):
        
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features) # 全连接层
        else:
            shape = (1, num_features, 1, 1) # 卷积层, 将1个通道视作一个特征
        
        # 参与求梯度和迭代的拉伸和偏移参数，即可学习参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        
        # 不参与求梯度和迭代的变量，初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
        
    def forward(self, X):
        
        # 将moving_mean和moving_var复制到X所在内存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
            
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, X, self.gamma, self.beta,
                                                self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y
    

# 定义模型
# 将批量归一化应用到LeNet        
net = nn.Sequential(
    
    # 第一卷积层
    nn.Conv2d(1, 6, 5), 
    BatchNorm(num_features=6, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    
    # 第二卷积层
    nn.Conv2d(6, 16, 5),
    BatchNorm(num_features=16, num_dims=4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    
    # 维度转换
    d2l.FlattenLayer(),
    
    # 第一全连接层
    nn.Linear(16*4*4, 120),
    BatchNorm(num_features=120, num_dims=2),
    nn.Sigmoid(),
    
    # 第二全连接层
    nn.Linear(120, 84),
    BatchNorm(num_features=84, num_dims=2),
    nn.Sigmoid(),
    
    # 第三全连接层
    # softmax包含在交叉熵loss损失函数里了
    nn.Linear(84, 10)
    )        


# 参数
lr, num_epochs, batch_size = 0.001, 5, 256
# 数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 优化器
optimizer = optim.Adam(net.parameters(), lr=lr)
# 训练
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
# 查看第一个批量归一化层学习到的拉伸参数gamma和偏移参数beta
print(net[1].gamma.view(1, -1), net[1].beta.view(1, -1))



# 简洁实现
# 定义模型
net2 = nn.Sequential(
    
    # 第一卷积层
    nn.Conv2d(1, 6, 5),  # (in_channels, out_channels, kernel_size)
    nn.BatchNorm2d(6),  # 简洁实现不需要指定层的维度
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    
    # 第二卷积层
    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    
    # 维度转换
    d2l.FlattenLayer(),
    
    # 第一全连接层
    nn.Linear(16*4*4, 120),
    nn.BatchNorm2d(120),
    nn.Sigmoid(),
    
    # 第二全连接层
    nn.Linear(120, 84),
    nn.BatchNorm2d(84),
    nn.Sigmoid(),
    
    # 第三全连接层
    # softmax包含在交叉熵loss损失函数里了
    nn.Linear(84, 10)
    )

# d2l.train_ch5(net2, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
