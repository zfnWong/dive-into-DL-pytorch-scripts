# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:59:19 2020

@author: Administrator
"""


import torch
import torchvision
import numpy as np
import d2lzh_pytorch as d2l


# 参数引起过拟合
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05


# 生成数据
features = torch.randn(n_train+n_test, num_inputs)
labels = torch.mm(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, labels.size()), dtype=torch.float)


# 从零开始实现
# 初始化参数
def init_params():
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), 
                     dtype=torch.float, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚项
def l2_penalty(w):
    return (w**2).sum()/2 # 除以2是为了抵消平方求导的2

# 定义训练和测试
# 设置参数
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

# 读取训练数据
dataset = torch.utils.data.TensorDataset(features[:n_train, :], labels[:n_train, :])
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# 训练
def fit_and_plot(lamb):
    w, b = init_params()
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in data_iter:
            # 添了L2范数惩罚项
            l = loss(net(X, w, b), y) + lamb * l2_penalty(w)
            if w.grad is not None: # 第一次求导前参数是没梯度的
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward() # 求导
            d2l.sgd([w, b], lr) # 更新参数
        train_ls.append(loss(net(features[:n_train, :], w, b), labels[:n_train]).item())
        test_ls.append(loss(net(features[n_train:, :], w, b), labels[n_train:]).item())
    print('final epoch: train_loss ', train_ls[-1], 'test_loss ', test_ls[-1])
    print('L2 norm of w', w.norm().item())
    # 绘制误差曲线
    d2l.semilogy(range(1, num_epochs+1), train_ls, 'epoch', 'loss',
                     range(1, num_epochs+1), test_ls, ['train', 'test'])       
                
# fit_and_plot(3.0)
    


# 简洁实现
def fit_and_plot_pytorch(wd): # wd = weight_dency
    net = torch.nn.Linear(num_inputs, 1)
    # 构造不同优化器对权重和偏差不同处理
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X,y in data_iter:
            # 因为优化器已说明是否进行 regularization
            # 所以这里就不需要了
            l = loss(net(X), y)
            optimizer_b.zero_grad()
            optimizer_w.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(features[:n_train, :]), labels[:n_train]).item())
        test_ls.append(loss(net(features[n_train:, :]), labels[n_train:]).item())
    print('final epoch: train_loss ', train_ls[-1], 'test_loss ', test_ls[-1])
    print('L2 norm of w', net.weight.data.norm())
    # 绘制误差曲线
    d2l.semilogy(range(1, num_epochs+1), train_ls, 'epoch', 'loss',
                     range(1, num_epochs+1), test_ls, ['train', 'test']) 
    
# fit_and_plot_pytorch(3.0)
    
    
    