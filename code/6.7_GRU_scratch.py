# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:00:50 2020

@author: Administrator
"""


import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

# 设置参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


# 初始化所有参数
def get_params():
    
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return nn.Parameter(ts, requires_grad=True)
    
    def _three():
        return (_one((num_inputs, num_hiddens)), # 输入参数
                _one((num_hiddens, num_hiddens)), # 隐藏状态参数
                nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float), requires_grad=True)
                )
    
    W_xz, W_hz, b_z = _three() # 更新门参数
    W_xr, W_hr, b_r = _three() # 重置门参数
    W_xh, W_hh, b_h = _three() # 隐藏状态参数
    
    # 输出参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


# 初始化隐藏状态
def init_gru_state(batch_size, num_hiddens, device):
    return torch.zeros((batch_size, num_hiddens), device=device)


# 定义模型
def gru(inputs, state, params):
    
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    
    for X in inputs:
        # 更新门计算
        Z = torch.sigmoid(torch.mm(X, W_xz) + torch.mm(H, W_hz) + b_z)
        # 重置门计算
        R = torch.sigmoid(torch.mm(X, W_xr) + torch.mm(H, W_hr) + b_r)
        # 候选隐藏状态
        H_tilda = torch.tanh(torch.mm(X, W_xh) + torch.mm(R * H, W_hh) + b_h)
        # 隐藏状态
        H = Z * H + (1-Z) * H_tilda # 注意 (1-Z)乘的是候选隐藏状态
        # 输出
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
        
    return outputs, H


# 设置参数
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2

pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)


    
    

    
    
