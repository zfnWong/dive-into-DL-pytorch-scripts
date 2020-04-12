# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:59:20 2020

@author: Administrator
"""

import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import sys
import torch.utils.data as Data
from torch import nn, optim
from IPython import display
import random
import time
import math
from tqdm import tqdm
import collections
import torchtext.vocab as vocab
import os
from PIL import Image
import numpy as np
from collections import namedtuple
import json


# ==========================================================================
# ch3
# ==========================================================================

# 梯度下降法：更新参数
def sgd(params, lr):
    for param in params:
        param.data -= lr * param.grad  # 注意这里更改param时用的param.data


# 线性回归从零开始-计算预测结果
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 线性回归从零开始-定义损失函数
def squared_loss(y_hat, y):
    return ((y_hat - y.view(y_hat.size())) ** 2).sum() / (2 * y_hat.shape[0])  # 有个2是为了抵消平方求导后的2


# 使用矢量图表示
def use_svg_display():
    display.set_matplotlib_formats('svg')


# 设置矢量图的大小
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # figsize 决定将图像的宽高划分为多少段
    # dpi决定每一段用多少像素表示
    plt.rcParams['figure.figsize'] = figsize


# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个 batch_size
        # LongTensor是因为 index_select 函数的参数指定类型
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        # 0指第一个维度
        yield features.index_select(0, j), labels.index_select(0, j)


# 将数值标签转化为字符标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                   'coat', 'sandal', 'shirt', 'sneaker',
                   'bag', 'ankle boot']
    return [text_labels[i] for i in labels]


# 在一张画布里画出多张图像和对应字符标签
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 定义画布里的子图个数和每张子图的长宽
    # 这里的 '_' 表示忽略这个变量，
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, label in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(label)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 得到 fashion-mnist 数据集
def load_data_fashion_mnist(batch_size, resize=None, root='./Datasets/FashionMNIST'):
    # 改变图像高和宽
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    # 下载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=transform, download=True)

    # 查看数据集的类型
    # print(type(mnist_train))
    # print(len(mnist_train), len(mnist_test))

    # 访问一个样本
    # feature, label = mnist_train[0]
    # 因为是灰度图，所以 feature 第一维为 1
    # print(feature.shape, label) #Channel * Height * Width

    # 读取数据, 转换成 Iteration 类
    # 如果使用本地的机器(win系统)跑数据，不使用多进程
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    # 使用多进程
    else:
        num_workers = 4
    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


# 定义分类准确率-增强版-适应丢弃法、支持GPU计算
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没有指定 device，就用 net 的 device
        device = list(net.parameters())[0].device
    acc_num, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式，会关闭 dropout()
                # 加法，cpu上完成
                acc_num += (net(X.to(device)).argmax(dim=1) == y.to(device)) \
                    .float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义模型 3.13节后不会用到，不考虑GPU
                if ('is_training' in net.__code__.co_varnames):  # 如果有这个参数
                    # 将其设置为False
                    acc_num += (net(X, is_training=False).argmax(dim=1) == y) \
                        .float().sum().item()
                else:  # 没有这个参数，不执行丢弃法
                    acc_num += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_num / n


# 训练模型
# params为None是因为参数可能包含在nn.net里了
# lr为None是因为可能包含在optimizer里了
# optimizer为None是因为可能直接用的d2l里的sgd函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:

            # 参数迭代
            # 损失函数
            y_hat = net(X)
            l = loss(y_hat, y)
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            # 求导
            l.backward()
            # 更新梯度
            # 如果优化器已经定义，优化器必然已经读取了参数和学习率
            if optimizer is not None:
                optimizer.step()
            else:
                # 没有定义优化器，则默认使用梯度下降法
                # 代入参数和优化率
                sgd(params, lr)

            # 结果统计
            train_l_sum += l.item() * y_hat.shape[0]
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            # n += batch_size 是不对的因为最后一次的批处理可能达不到batch_size的数量
            n += y_hat.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc %.3f, test_acc %3f' %
              (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# ===================================================================================
# ch5
# ===================================================================================

# 形状转换功能
# 通常用于网络的首层格式化数据
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch_size, channels, height, width)
        return x.view(x.shape[0], -1)  # to shape: (batch_size, features)


# 绘制误差曲线
# semilogy 又是 plt 的函数，看起来跟 plot 的用法差不多
# 这里定义的 semilogy 其实用法也差不多，就是加了一些自定义的东西
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


# 二维互相关运算
# 步长为1
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 训练模型
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    # 切换 device
    net = net.to(device)
    print("training on ", device)
    # 使用交叉熵损失函数，包含softmax运算，损失已平均
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # 参数初始化
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)  # 转移 device
            y = y.to(device)  # 转移 device
            # 预测 在gpu上 
            y_hat = net(X)
            # 计算损失 在gpu上
            l = loss(y_hat, y)
            # 梯度清零
            optimizer.zero_grad()
            # 求导 在gpu上
            l.backward()
            # 更新梯度
            optimizer.step()
            # 加法在cpu上，矩阵计算才去gpu，累加小批量的平均损失
            train_l_sum += l.cpu().item()
            # 加法在cpu上，累加训练集的预测准确数
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]  # 累加总训练数,本来就在cpu上
            batch_count += 1  # 累加批次，本来就在cpu上
        test_acc = evaluate_accuracy(test_iter, net)  # 预测准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


# 定义全局平均池化层
class GlobalAvgPool2d(nn.Module):
    # 全局池化层的池化窗口形状为输入的高和宽
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, X):
        return F.avg_pool2d(X, kernel_size=X.size()[2:])  # X的第3，4维度即H、W


# Residual块定义
# 跨越的数据线路跨过前面的层，与主线路在模块末尾的ReLU激活函数前相加
# 两条线路的输出形状相同
class Residual(nn.Module):
    # 主线路指定 stride 改变主线路输出形状
    # 而跨越线路也可以通过1x1的卷积层改变形状
    def __init__(self, in_channels, out_channels, use_1x1_conv=False, stride=1):
        super(Residual, self).__init__()

        # 两个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 两条线路的的输出形状、通道数也要相同，要想改变形状和通道数
        # 则要在跨越线路加上一个1x1的卷积层
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride)
        else:
            self.conv3 = None

        # 两个卷积层的批量归一化层，需要分开求梯度的
        # 先批量归一化再到激活函数
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):

        # 主线路：3*3_conv -> b_n -> activate -> 3*3_conv -> b_n
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        # 跨越线路,判断是否使用1*1卷积层改变形状和通道数
        if self.conv3:
            X = self.conv3(X)

        # 线路相加
        return F.relu(Y + X)

# # 测试
# # 输入输出形状一致
# blk = Residual(3, 3) 
# X = torch.rand(4, 3, 6, 6)
# print(blk(X).shape)
# # 输入输出形状不一致
# blk = Residual(3, 6, use_1x1_conv=True, stride=2)
# print(blk(X).shape)


# 定义 ResNet 重复模块，由多个residual模块组成，借鉴了vgg的模块思想
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):

    if first_block:
        assert in_channels == out_channels

    blk = []
    for i in range(num_residuals):
        # 每个重复模块的第一层都要将通道数翻倍，高和宽减半，除非是第一个重复模块
        # 因为第一个重复模块之前已经使用了步幅为2的最大池化层，所以第一层无须减小样本的高和宽
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1_conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))

    return nn.Sequential(*blk)


# 残差网络
def resnet18(output=10, in_channels=3):
    net = nn.Sequential(
        # 前两层与GoogLeNet类似,在卷积层后多了批量归一化层
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # 形状不变
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 长宽减半
    )

    # 接着为 ResNet 加入四个模块， 每个模块使用两个 Residual 层
    # 每个模块将通道数翻倍，长宽减半（除了第一个模块）
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))

    # 最后，与GoogLeNet一样加入全局平均池化层,接维度转换和全连接层后输出
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

    return net


# ===============================================================================
# ch6
# ===============================================================================

# 随机批量采样
# corpus_indices 为字符下标列表， example_len 为样本串长度,即 num_steps         
def data_iter_random(corpus_indices, batch_size, example_len, device=None):
    # 减1是因为 y 是 x 的预测，即 x 的下一个字符
    num_examples = (len(corpus_indices) - 1) // example_len  # 样本数
    num_batches = num_examples // batch_size  # 批量个数
    example_indices = list(range(num_examples))  # 样本下标
    random.shuffle(example_indices)  # 打乱样本下标

    # 返回下标从pos开始的样本串
    def _data(pos):
        return corpus_indices[pos:pos + example_len]  # 记得右边是开区间

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(num_batches):
        # 每次返回 batch_size 个随机样本
        i = i * batch_size
        batch_indices = example_indices[i:i + batch_size]  # 样本编号区间
        X = [_data(j * example_len) for j in batch_indices]  # 批量样本串
        Y = [_data(j * example_len + 1) for j in batch_indices]  # 批量预测串
        # yield专门用于批量输出
        yield torch.tensor(X, dtype=torch.float32, device=device), \
              torch.tensor(Y, dtype=torch.float32, device=device)


# 测试
# test_sq = list(range(30))                     
# for X, Y in data_iter_random(test_sq, 2, 6):
#     print('X:', X, '\n', 'Y:', Y, '\n')


# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, example_len, device=None):
    # 将 corpus_indices 转成 tensor 类型，才可以用 view()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)

    # 将 corpus_indices 变形为相邻矩阵 indices ，去掉尾巴
    # column为batch_len， row为batch_size
    # batch_len = example_len * num_batches 
    batch_len = len(corpus_indices) // batch_size
    indices = corpus_indices[0:batch_len * batch_size].view(batch_size, -1)

    # 减1原因同上一个方法
    num_batches = (batch_len - 1) // example_len  # 批量个数

    for i in range(num_batches):
        i = i * example_len
        X = indices[:, i: i + example_len]
        Y = indices[:, i + 1: i + example_len + 1]
        yield X, Y


# 测试
# test_sq = list(range(30))                     
# for X, Y in data_iter_consecutive(test_sq, 2, 6):
#     print('X:', X, '\n', 'Y:', Y, '\n')


# 读取周杰伦歌词数据集
def load_data_jay_lyrics():
    with open('./small_dataset/jaychou_lyrics.txt', encoding='utf-8') as f:  # 编码为utf-8
        corpus_chars = f.read()  # 将文本内容读取到字符数组
    # print(corpus_chars[:40])
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')  # 换行符、tab换成空格
    corpus_chars = corpus_chars[:10000]  # 取前10000个汉字

    # 建立字符索引
    idx_to_char = list(set(corpus_chars))  # 列表用于以下标索引汉字
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])  # 字典用于以汉字索引下标
    vocab_size = len(idx_to_char)  # 词典大小，即不重复的汉字个数
    corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 汉字对应的下标集
    # print(corpus_chars[:40])
    # print(corpus_indices[:40])

    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# x为batch_size*1的indices,将其转换为batch_size*vocab的词向量
def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch, 1), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    # 这个方法比较难理解
    # self.scatter_(dim, index, src)  
    # dim是要update的维度，index是该维度下update的下标, src是update后的值
    # src除了dim外其他维度要和self相同，或者为某一个标量
    res.scatter_(1, x.view(-1, 1), 1)
    return res


# 测试
# x = torch.tensor([[0],[2]])
# print(one_hot(x, 10))


# X shape: batch_size, num_steps
# 输入是一个批量
def to_onehot(X, n_class):
    # 返回结果 shape: num_steps, batch_size, vocab_size  三维
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


# 测试
# X = torch.arange(10).view(2, 5)    
# inputs = to_onehot(X, 10)
# print(len(inputs), inputs[0].shape)


# 循环神经网络从零实现的预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    # 初始化隐藏状态，batch_size为1
    state = init_rnn_state(1, num_hiddens, device)
    # 从第一个字符开始预测
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 取上一个字符作为输入，转化为词向量
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 预测
        (Y, state) = rnn(X, state, params)
        # 判断当前预测字符是否在输入字符串中
        # 是则使用输入字符串的字符
        # 不是则使用预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))

    return ''.join([idx_to_char[i] for i in output])


# 梯度裁剪
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    # 将所有参数拼接成向量
    # 然后求向量的二范式
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    # 若二范式大于设定的边界值theta
    # 则进行裁剪，裁剪比例为theta/norm
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


# 循环神经网络从零实现的训练函数
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    # 判断是随机批量采样
    # 还是相邻批量采样
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    # 初始化参数
    params = get_params()
    # 交叉熵损失函数
    loss = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(num_epochs):
        # 如果是相邻批量采样
        # 则只需在迭代的开始初始化隐藏状态
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, = 0.0, 0
        # 这里踩坑了
        # data_iter要放在迭代里面，不能放外面
        # 因为每次迭代都要获取一次data_iter
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            # 随机批量采样在每次批量都要初始化隐藏状态
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
            # 相邻采样在每次批量后 要将隐藏状态从计算图分离
            # 因为H的计算依赖于模型参数
            # 不分离的话会有许多层的梯度要求
            # 分离的话就只有num_steps-1层梯度
            else:
                if isinstance(state, tuple):
                    for s in state:
                        s.detach_()
                else:
                    state.detach_()

            inputs = to_onehot(X, vocab_size)
            (outputs, state) = rnn(inputs, state, params)
            # 结果拼接
            # 将一个列表中的tensor在第0维上拼接
            outputs = torch.cat(outputs, dim=0)
            # 将Y转换成向量形式
            # transpose会造成tensor在内存中不连续
            # contiguous能将在内存中不连续的tensor变成连续的
            # 在内存中连续的tensor才能用view
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)

            l = loss(outputs, y.long())

            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr)

            # 累加求总，便于最后统计模型误差
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        # 每pred_period进行一次预测
        if (epoch + 1) % pred_period == 0:
            # 迭代次数，困惑度，单次迭代时间
            print('epoch %d, perplexity %f, time %.2f sec' %
                  (epoch + 1, math.exp(l_sum / n), time.time() - start))
            start = time.time()
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                        num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


# RNN 的 pytorch 模型
class RNN_model(nn.Module):
    # 传入参数：rnn层，字典大小
    def __init__(self, rnn_layer, vocab_size):
        super(RNN_model, self).__init__()
        self.rnn = rnn_layer
        # bidirectional 判断是否为双向循环神经网络
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        # 输出层
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        # 不需要对state初始化
        self.state = None

    def forward(self, inputs, state):
        X = to_onehot(inputs, self.vocab_size)  # X 是一个tensor的list，len为num_step
        # stack 将 len(list)作为一个新的维度, 拼接list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 先将Y变形成(num_steps * batch_size, num_hiddens)
        # 再输出成(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


# RNN的 pytorch 实现的预测函数
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    # 存储字符下标
    output = [char_to_idx[prefix[0]]]

    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([[output[-1]]], device=device)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))

    # 将下标转换成字符输出
    return ''.join([idx_to_char[i] for i in output])


# RNN的pytorh实现的训练+预测函数
# 直接使用相邻批量采样
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    # 参数初始化
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None

    start = time.time()
    for epoch in range(num_epochs):
        l_sum, n = 0.0, 0
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        for X, Y in data_iter:
            # 相邻采样在每次批量后 要将隐藏状态从计算图分离
            # 因为H的计算依赖于模型参数
            # 不分离的话会有许多层的梯度要求
            # 分离的话就只有num_steps-1层梯度
            if state is not None:
                if isinstance(state, tuple):
                    # LSTM state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            # 计算误差
            (output, state) = model(X, state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            # 梯度清零
            optimizer.zero_grad()
            # 求导
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            # 更新梯度
            optimizer.step()

            # 统计误差
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        # 不知道为什么要溢出处理，不溢出处理会报错
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % ( \
                epoch + 1, perplexity, time.time() - start))
            start = time.time()
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))


# ==============================================================================
# ch9
# ===============================================================================

# 显示多张图片
def show_images(imgs, num_rows, num_cols, scale=2):
    # 设置**画布**大小 (len_x，len_y) * dpi
    # dpi = pixels / inch
    figsize = (num_cols * scale, num_rows * scale)
    # 第一个参数是画布指针，第二个是子图指针
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            # 取消坐标显示
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)

    return axes  # 每个子图的指针


# 训练模型
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            batch_count += 1
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


# =============================================================================
# ch10
# =============================================================================

# 读取文件
# 返回形如 [text, label] 的数据集
def read_imdb(folder='train', data_root="./Datasets/aclImdb"):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        # tqdm 用来显示进度条，通过封装迭代器实现
        for file in tqdm(os.listdir(folder_name)):
            # rb 以二进制形式读取文件
            with open(os.path.join(folder_name, file), 'rb') as f:
                # utf-8进行解码
                # 用空格替换换行，方便后面以空格作为分割符
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


# 测试
# train_data, test_data = read_imdb('train'), read_imdb('test')


# 对文本进行分割
# 返回切割后文本list
def get_tokenized_imdb(data):
    # 分割器
    def tokenizer(text):
        # 以空格作为分割符并小写化
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


# 构造词典
# stoi: 类似 token_to_idx
# itos: 类似 idx_to_token
# vectors: 词向量
def get_vocab_imdb(data):
    # 先分割，后统计
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 通过包装 Counter 类构造字典，并将词频小于5的词删除
    return vocab.Vocab(counter, min_freq=5)


#
# vocab = get_vocab_imdb(train_data)


# 将数据预处理为常规的 features, labels
# features: shape (num_files, num_tokens)
# labels: shape (num_files)
def preprocess_imdb(data, vocab):
    # 文本最大长度
    max_l = 500

    def padding(x):
        # 将文本修改为定长，超出则切断，不够就填充0
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    # 以文本单词的idx作为特征
    features = torch.tensor([padding([vocab.stoi[word] for word in words]) for \
                             words in tokenized_data])
    labels = torch.tensor([label for _, label in data])
    return features, labels


# 装载预训练的词向量，预训练的词量>当前数据集上的词量
def load_pretrained_embedding(words, pretrain_vocab):
    # 初始化
    embed = torch.zeros(len(words), pretrain_vocab.vectors[0].shape[0])
    # 统计目标数据集上有而预训练词典上没有的词量
    # 这其中包括标点符号，口语化的单词、符号等
    oov_count = 0
    for i, word in enumerate(words):
        try:
            idx = pretrain_vocab.stoi[word]
            embed[i, :] = pretrain_vocab.vectors[idx]
        # 如果预训练词典上没有该词
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print('There are %d oov words' % oov_count)
    return embed


# 情感预测
def predict_sentiment(net, vocab, sentence):
    device = list(net.parameters())[0].device
    # sentence已经将句子tokenization
    # 将sentence转成tensor并转移设备
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    # 结果取概率最大的一方
    label = torch.argmax(net(sentence.view(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
