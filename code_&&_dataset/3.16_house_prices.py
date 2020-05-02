# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:31:39 2020

@author: Administrator
"""


import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data
import d2lzh_pytorch as d2l


# 读取数据
# 得到 pandas 主类型的数据，看起来跟 numpy, tensor 差不多，但是访问方式有区别
train_data = pd.read_csv('./Datasets/kaggle_house/train.csv')
test_data = pd.read_csv('./Datasets/kaggle_house/test.csv')
# 首列是id，不作为训练特征; 末列是label，不作为特征；测试集没有末列的label
# 行合并便于预处理数据
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) # 用 data.iloc[] 按数字下标访问*列*


# 预处理数据
# numeric_features 可以理解为过滤器，用于访问特定的**列**
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # 过滤掉非数值型的特征列

# 对数值列标准化
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean()) / x.std())

# 标准化后，数值特征的平均值变为0，可以用来填补缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散值转化为指示特征：比如color特征有red、blue、green等元素，get_dummies后变成red:{0,1}、blue:{0,1}
# dummy_na=True 为缺失值也创建合法的指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)


# 格式转换
n_train = train_data.shape[0] # 训练集和测试集的分界线
# 通过 values 属性得到 numpy 类型，再转换为 torch 类型
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float) # 直接用下标访问的是行
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1) # SalePrice列为label



# 定义模型：线性回归模型
def get_net(num_inputs):
    net = nn.Linear(num_inputs, 1)
    for param in net.parameters():
        nn.init.normal_(param, 0, 0.01)
    return net

# 损失函数
loss = nn.MSELoss()

# 误差计算函数，不作为损失函数求导
# 为什么不对rmse求梯度，没必要，最优解相同, 对MSE求梯度即可
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的结果设成1，求对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse



# 训练模型
def train(net, train_features, train_labels, test_features, test_labels,
                                num_epochs, learning_rate, weight_decay, batch_size):

    # 批量数据读取器
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    # 优化器-使用了adam优化算法
    # 相比小批量随机梯度下降，对学习率相对不那么敏感
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 迭代
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X,y in train_iter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        # 如果处于交叉验证下，则记录验证集的误差
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls



# 返回第k折交叉验证时所需的训练和验证数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # '//' 是向下取整的除法
    X_train, y_train = None, None
    X_val, y_val = None, None
    for j in range(k):
        idx = slice(fold_size * j, fold_size * (j+1))
        X_part, y_part = X[idx,:], y[idx]
        if j == i:
            X_val, y_val = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_val, y_val       



# k次训练求平均：用于得到合适的模型超参数
def k_fold_train(k, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, val_l_sum = 0.0, 0.0
    for i in range(k):
        # data：(train_features, train_labels, test_features, test_labels)
        data = get_k_fold_data(k, i, train_features, train_labels)
        net = get_net(train_features.shape[1])

        # *data 可以一次将数据传给四个参数，想一下指针
        train_ls, val_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1] # 累加每一折的误差，最后计算平均值
        val_l_sum += val_ls[-1] # 同上
        print('fold %d, train rmse %f, valid rmse %f' % (i+1, train_ls[-1], val_ls[-1]))

        if i == 0:
            # 取k折中的1折，绘制 epoch-rmse 图观察模型训练效果
            d2l.semilogy(range(1, num_epochs+1), train_ls, 'epoch', 'rmse',
                                range(1, num_epochs+1), val_ls, ['train', 'valid'])

    return train_l_sum / k, val_l_sum / k



# 得到合适的模型超参数后，使用完整的训练集来训练模型，并在测试集上得出测试结果
def train_and_pred(train_features, train_labels, test_features, test_data, 
                            num_epochs, learning_rate, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    # 训练模型
    train_ls, _ = train(net, train_features, train_labels, None, None, 
                                num_epochs, learning_rate, weight_decay, batch_size)
    print('train rmse %f' % (train_ls[-1]))
    d2l.semilogy(range(1, num_epochs+1), train_ls, 'epoch', 'rmse')

    # 得到结果并保存
    # detach()切断反向传播，即返回一个不具有梯度grad的复制变量，但仍指向原变量的存储位置
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(-1)) # 将结果拼接到原数据上
    submission = pd.concat((test_data['Id'], test_data['SalePrice']), axis=1) # 抽取Id和SalePrice列合并
    submission.to_csv('./Datasets/kaggle_house/submission.csv', index=False) # index(bool): Write row names (index).
    


# 调试模型超参数 
k, num_epochs, lr, wd, batch_size = 5, 100, 5, 0, 64
# train_l, val_l = k_fold_train(k, train_features, train_labels, num_epochs, lr, wd, batch_size)
# print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, val_l))

# 训练模型与提交
train_and_pred(train_features, train_labels, test_features, test_data, num_epochs, lr, wd, batch_size)
        

                





