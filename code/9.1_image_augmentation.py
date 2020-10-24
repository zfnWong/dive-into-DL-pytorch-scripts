# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:49:08 2020

@author: Administrator
"""

import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import d2lzh_pytorch as d2l
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 对输入图像img多次运行图像增广方法aug并展示所有的结果
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale)

img = Image.open('./small_dataset/cat1.jpg')

# RH_aug = torchvision.transforms.RandomHorizontalFlip()
# apply(img, RH_aug) # 左右颠倒

# RV_aug = torchvision.transforms.RandomVerticalFlip()
# apply(img, RV_aug) # 上下颠倒

# 形状变化（裁剪） parameter-(长宽，scale-面积占比, ratio-区域宽高比)
# shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

# 颜色变化  brightness-亮度，contrast-对比度， saturation-饱和度 hue-色调
# color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# 叠加多个图像增广方法
# augs = torchvision.transforms.Compose([RH_aug, shape_aug, color_aug])
# apply(img, aug=augs)



# 在torchvision的datasets中用aug不会改变样本的数量，即用变换后得到的样本替换原来样本
# 训练数据增广
flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])
# 预测数据增广
no_aug = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


# load数据集，返回迭代器
num_workers = 0 if sys.platform.startswith('win32') else 4
# 将图像增广运用在训练集上；预测集上为保证预测的准确，不使用图像增广
def load_cifar10(is_train, augs, batch_size, root="./Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


# 训练模型
def train_with_data_aug(train_augs, test_augs, lr=0.001):

    # 数据
    batch_size = 256
    train_iter = load_cifar10(True, train_augs, batch_size) # 训练数据的迭代器
    test_iter = load_cifar10(False, test_augs, batch_size) # 预测数据的迭代器

    net = d2l.resnet18(10) # 模型，输出10种分类概率

    num_epochs = 10 # 参数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # 优化器
    loss = torch.nn.CrossEntropyLoss() # 损失函数

    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs) # 训练


train_with_data_aug(flip_aug, no_aug)


