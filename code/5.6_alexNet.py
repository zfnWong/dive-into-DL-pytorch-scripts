# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 22:08:06 2020

@author: Administrator
"""

# =============================================================================
# # AlexNet与LeNet对比
# =============================================================================
# 1、AlexNet包括8层变换，其中5层卷积层，3层全连接层（2层隐藏层，1层输出层）。
#    卷积窗口依次为 11*11、5*5、3*3、3*3....通道数为LeNet的数十倍
#    第1、2、5层后使用了3*3步长为2的池化层
# 2、AlexNet将激活函数改成了更简单的ReLU
#   （单侧抑制 & 稀疏激活，即把小于0的参数直接仍掉，十分简单粗暴）
#   然而经实践证明，训练后的网络完全具备适度的稀疏性。
#   LeNet使用sigmoid（将参数惩罚至接近0，从而产生稀疏数据）
# 3、AlexNet使用丢弃法（Dropout）控制全连接层的模型复杂度，LeNet没有使用
# 4、AlexNet引入了大量的图像增广，如翻转、裁剪和颜色变化，进一步扩大数据集防止过拟合。
# =============================================================================


# 简化版 AlexNet
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            
            # 使用较大的11 x 11窗口来捕获物体。同时使用步幅4来较大幅度减小输出高和宽。
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3,2), # kernel_size, stride
                                # 若剩下的元素不够凑出一个kernel_size，则舍去
            
            # 减小卷积窗口，使用填充为2来使输入和输出的形状一致，同时增大通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            
            # 连续3个卷积层，且使用更小的卷积窗口。通道数由增加到减少。
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
            )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
            )
        
    def forward(self, img):
        features = self.conv(img) # features = [num_samples, channels, h, w]
        out = self.fc(features.view(features.shape[0], -1))
        return out
    
# batch_size 越大越占显存，但训练会快一点
# 参数
lr, num_epochs, batch_size = 0.001, 5, 128 # 显存不够，就把128换成64之类较小的值
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 模型
net = AlexNet()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=lr)

d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


# 如果是在console中运行程序的，程序运行完后，显存不会被释放，参数仍然可以访问
# 所以如果要再次运行程序的话，如果电脑的显存不是很充裕，就会报显存不足
# 如果是在命令行或者ide（除了spyder，因为spyder在console中运行程序）中运行程序
# 则运行完毕后会自动释放显存

# 清空gpu缓存，GPU内存起码要有3G才够呀
# torch.cuda.empty_cache()






