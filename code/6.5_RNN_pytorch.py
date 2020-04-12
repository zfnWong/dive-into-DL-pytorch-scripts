# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:39:16 2020

@author: Administrator
"""


import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()



# =============================================================================
# rnn_layer作为nn.RNN实例，在前向计算后会分别返回输出和隐藏状态h
# 1.输出指的是隐藏层在**各个时间步上**计算并输出的隐藏状态，它们通常作为后续输出层的输入
# 该“输出”本身**并不涉及输出层计算**，形状为(时间步数, 批量大小, 隐藏单元个数)
# 2.而nn.RNN实例在前向计算返回的隐藏状态指的是隐藏层在最后时间步的隐藏状态
# 当隐藏层有多层时，每一层的隐藏状态都会记录在该变量中，形状为(层数, 批量大小, 隐藏单元个数)
# =============================================================================
# 模型参数
num_hiddens, num_steps = 256, 35
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNN_model(rnn_layer, vocab_size)
model = model.to(device)

# 训练参数
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
