# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:06:54 2020

@author: Administrator
"""

import time
import torch
import math
import numpy as np
from torch import optim, nn
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取数据
corpus_indices, char_to_idx, idx_to_char, vocab_size = d2l.load_data_jay_lyrics()
# 设置参数
num_hiddens, num_steps = 256, 35



#初始化参数，返回参数列表
def get_params():
    def _one(shape):
        param = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float)
        return nn.Parameter(param, requires_grad=True)
    
    # 隐藏层参数
    W_xh = _one((vocab_size, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    
    # 输出层参数
    W_hq = _one((num_hiddens, vocab_size))
    b_q = nn.Parameter(torch.zeros(vocab_size, device=device, requires_grad=True))
    
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


# 初始化隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    # return (torch.zeros((batch_size, num_hiddens), device=device),)
    return torch.zeros((batch_size, num_hiddens), device=device)


# 定义模型
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # b_h使用了广播机制
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, H
# 测试
# X = torch.arange(10).view(2, 5)
# state = init_rnn_state(X.shape[0], num_hiddens, device)
# inputs = d2l.to_onehot(X.to(device), vocab_size)
# params = get_params()
# outputs, state_new = rnn(inputs, state, params)
# print(len(outputs), outputs[0].shape, state_new.shape)



num_epochs, pred_period, clipping_theta, lr, batch_size = 300, 50, 1e-2, 1e2, 32
prefixes, pred_len = ['分开', '不分开'], 50
is_random_iter = False

d2l.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes)