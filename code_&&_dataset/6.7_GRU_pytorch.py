# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:35:26 2020

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

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2

pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']


# 注意调整学习率
lr = 1e-2
gru_layer = nn.GRU(input_size=num_inputs, hidden_size=num_hiddens)
model = d2l.RNN_model(gru_layer, vocab_size).to(device)
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)