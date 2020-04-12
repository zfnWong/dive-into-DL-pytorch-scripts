# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:36:40 2020

@author: Administrator
"""


# =============================================================================
# 微调
# 1.在源数据集（如ImageNet数据集）上预训练一个神经网络模型，即源模型。
# 2.创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。
#   我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。
#   我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
# 3.为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
# 4.在目标数据集（如椅子数据集）上训练目标模型。我们将从头训练输出层，而其余层的参数都是
#   基于源模型的参数微调得到的。
# =============================================================================

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = './Datasets'


# 在使用预训练模型时，一定要和预训练时作同样的预处理
# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 训练集增广
train_augs = transforms.Compose([
    # 先从图像中裁剪出随机大小和随机高宽比的一块随机区域，
    # 然后将该区域缩放为高和宽均为224像素的输入
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])

# 测试集增广
test_augs = transforms.Compose([
    # 将图像的高和宽均缩放为256像素，然后从中裁剪出高和宽均为224像素的中心区域作为输入
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
    ])

# 默认都会将预训练好的模型参数下载到你的home目录下.torch文件夹
pretrained_net = models.resnet18(pretrained=True)
# print(pretrained_net)

# 修改输出层
pretrained_net.fc = nn.Linear(512, 2)

# 为使输出层参数和预训练参数使用不同的学习率，需要将两组参数分开
output_params = list(map(id, pretrained_net.fc.parameters()))
features_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
lr = 0.01
optimizer = torch.optim.SGD([{'params': features_params},
                            {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                            lr=lr, weight_decay=0.001)

# 训练模型
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):

    # 创建两个ImageFolder实例来分别读取训练数据集和测试数据集中的所有图像文件
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'),
                                    transform=train_augs), batch_size, shuffle=True)

    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'),
                                                     transform=test_augs), batch_size)

    loss = nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


# 预训练的模型
train_fine_tuning(pretrained_net, optimizer)


# 从头训练的模型
# scratch_net = models.resnet18(pretrained=False)
# lr = 0.1
# optimizer = optim.SGD(scratch_net.parameters(), lr)
# train_fine_tuning(scratch_net, optimizer)

