# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:38:13 2020

@author: Administrator
"""


import collections
import os
import io
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data
import d2lzh_pytorch as d2l

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>' # 填充符、起始符、结束符
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 将一个序列中所有的词记录在all_tokens中以便之后构造词典
# 然后在该序列后面添加PAD直到序列长度变为max_seq_len，然后将序列保存在all_seqs中
# 传入的 seq_tokens 已分割好
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    # 补充结束符和填充符直到最大长度
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor
def build_data(all_tokens, all_seqs):
    vocab = Vocab.Vocab(collections.Counter(all_tokens), specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


# 读取文本，限定句子长度
def read_data(max_seq_len):
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('./small_dataset/fr-en-small.txt') as f:
        lines = f.readlines()
    # 统计词语和句子
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        # 如果加上EOS后长于max_seq_len，则忽略掉此样本
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    # 构造字典和下标序列
    in_vocab, in_indices = build_data(in_tokens, in_seqs)
    out_vocab, out_indices = build_data(out_tokens, out_seqs)
    # 返回两个字典和一个用户构造迭代器的Dataset
    return in_vocab, out_vocab, Data.TensorDataset(in_indices, out_indices)

max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)
print(dataset[0])


# 编码器
# nn.GRU实例在前向计算后也会分别返回各个时间步的输出和最终时间步的多层隐藏状态
# 输出不涉及输出层运算，注意力机制将这些输出作为键项和值项（详情看笔记）
class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
         # (batch_size, seq_len) -> (seq_len, batch_size, embed_size)
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)
        # output shape (seq_len, batch_size, num_hiddens)
        # state shape (num_layers, batch_size, num_hiddens)
        return self.rnn(embedding, state)

    def begin_state(self):
        return None # 隐藏态初始化为None时PyTorch会自动初始化为0
#
# encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# output, state = encoder(torch.zeros((4, 7)), encoder.begin_state())
# print(output.shape, state.shape) # GRU的state是h(LSTM的是一个元组(h, c))


# 注意力模型
# 隐藏层的输入是解码器的隐藏状态与编码器在各时间步上隐藏状态的连结
# 其中函数a定义里向量v的长度是一个超参数
def attention_model(input_size, attention_size):
    # 得出每个时间步隐藏状态的权重
    # input_size = enc_num_hiddens + dec_num_hiddens
    model = nn.Sequential(nn.Linear(input_size, attention_size, bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False)
                          )
    return model


# 注意力机制
# 注意力机制的输入包括查询项、键项和值项.设编码器和解码器的隐藏单元个数相同。
# 这里的查询项为解码器在上一时间步的隐藏状态，形状为(批量大小, 隐藏单元个数)；
# 键项和值项均为编码器在所有时间步的隐藏状态，形状为(时间步数(seq_len), 批量大小, 隐藏单元个数)。
# 注意力机制返回当前时间步的背景变量，形状为(批量大小, 隐藏单元个数)。
def attention_forward(model, enc_states, dec_state):
    # 将解码器隐藏状态广播到和编码器隐藏状态数目相同后进行连结
    dec_state = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    enc_and_dec_states = torch.cat((enc_states, dec_state), dim=2)
    e = model(enc_and_dec_states) # 形状为(时间步数, 批量大小, 1)
    alpha = F.softmax(e, dim=0) # 在时间步维度做softmax运算,形状同上
    # 这里也用到了广播，返回 batch_size 个背景向量
    return (alpha * enc_states).sum(dim=0)

# 测试
# seq_len, batch_size, num_hiddens = 10, 4, 8
# model = attention_model(2*num_hiddens, 10) 
# enc_states = torch.zeros((seq_len, batch_size, num_hiddens))
# dec_state = torch.zeros((batch_size, num_hiddens))
# print(attention_forward(model, enc_states, dec_state).shape) # torch.Size([4, 8])


# 解码器
# 将输入通过词嵌入层得到表征，然后和背景向量在特征维连结。
# 我们将连结后的结果与上一时间步的隐藏状态通过门控循环单元计算出当前时间步的输出与隐藏状态。
# 最后，我们将输出通过全连接层变换为有关各个输出词的预测，形状为(批量大小, 输出词典大小)
class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(2 * num_hiddens, attention_size)
        # GRU的输入包括inputs和隐藏状态，inputs由attention输出的c和实际输入组成
        self.rnn = nn.GRU(num_hiddens+embed_size, num_hiddens, num_layers,
                          dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        # 使用注意力机制计算背景向量
        # 查询项为解码器在上一时间步的**最靠近输出**的隐藏状态
        c = attention_forward(self.attention, enc_states, state[-1])
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, num_hiddens+embed_size)
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1,符合模型输入形状要求
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(0)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state


# 实现batch_loss函数计算一个小批量的损失
# 解码器在最初时间步的输入是特殊字符BOS
# 之后，解码器在某时间步的输入为样本输出序列在上一时间步的词，即强制教学
def batch_loss(encoder, decoder, X, Y, loss):

    batch_size = X.shape[0]
    enc_state = encoder.begin_state()

    # enc_outputs 每个时间步最后一个隐藏层的输出
    # enc_state 最后一个时间步的所有state
    enc_outputs, enc_state = encoder(X, enc_state)

    # 初始化解码器的隐藏状态
    # 是否将编码器输出的隐藏状态作为解码器的初始隐藏状态
    # 实验发现，是的话，模型训练的效率较高
    dec_state = decoder.begin_state(enc_state)
    # dec_state = torch.zeros(enc_state.shape) # 重新初始化解码器的初始隐藏状态

    # 解码器在最初时间步的输入是BOS
    dec_input = torch.tensor([out_vocab.stoi[BOS]] * batch_size)

    # 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size), 0
    l = torch.tensor([0.0])

    for y in Y.permute(1,0): # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y # 使用强制教学,用在训练中，不能用在预测中
        num_not_pad_tokens += mask.sum().item()
        # EOS后面全是PAD. 下面一行保证一旦遇到EOS接下来的循环中mask就一直是0
        mask = mask * (y != out_vocab.stoi[EOS]).float()

    return l / num_not_pad_tokens


# 训练函数
def train(encoder, decoder, dataset, lr, batch_size, num_epochs):

    # 优化器
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    # 不对 loss 平均，因为有一些词是用 <pad> 填充的
    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))


# 设置参数
embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers,
                  drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
                  attention_size, drop_prob)
train(encoder, decoder, dataset, lr, batch_size, num_epochs)


# 预测不定长序列
def translate(encoder, decoder, input_seq, max_seq_len):

    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)

    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]]) # batch = 1
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input, enc_state)

    dec_input = torch.tensor([out_vocab.stoi[BOS]])
    dec_state = decoder.begin_state(enc_state)
    # dec_state = torch.zeros(enc_state.shape)
    output_tokens = []

    for i in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[int(pred.item())]
        if pred_token == EOS: # 当任一时间步搜索出EOS时，输出序列即完成
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred

    return output_tokens

input_seq = 'ils regardent .'
print(translate(encoder, decoder, input_seq, max_seq_len))


# 评价翻译结果
# Pn 是预测序列与标签序列匹配词数为n的子序列的数量与预测序列中词数为n的子序列的数量之比，且不可重复匹配
# k是我们**希望**匹配的子序列的最大词数
def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)

    # score 存储 bleu 值
    score = math.exp(min(0, 1 - len_label/len_pred))

    # 如果k过大，会使得一项Pn的值为0，使得总乘积bleu为0
    for n in range(1, k+1):
        num_matches, label_subs = 0, collections.defaultdict(int)

        # 统计标签序列词数为k的各个子序列的个数
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i+n])] += 1

        # 统计预测序列与标签序列匹配词数为n的子序列的数量，且不可重复匹配
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i+n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i+n])] -= 1

        # 加上一个0.001避免一项Pn的值为0，使得总乘积bleu为0，感觉又好像没什么必要
        score *= math.pow((0.001 + num_matches)/(len_pred - n + 1), math.pow(0.5, n))
    return score


# 辅助打印函数
def score(input_seq, label_seq, k):
    # 预测结果tokens
    pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
    # 标签序列tokens
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s' % (bleu(pred_tokens, label_tokens, k),
                                      ' '.join(pred_tokens)))

# 测试
score('ils regardent .', 'they are watching .', k=2)
score('ils sont canadienne .', 'they are canadian .', k=2)







