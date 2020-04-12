# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:53:47 2020

@author: Administrator
"""
import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import d2lzh_pytorch as d2l

# 设置当前使用的GPU设备仅为0号设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = "./Datasets"


# =============================================================================
# 文本分类把一段不定长的文本序列变换为文本的类别,它属于词嵌入的下游应用
# 文本分类的子问题，情感分析，使用文本情感分类来分析文本作者的情绪
# =============================================================================


# 获取数据和词典
train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
vocab = d2l.get_vocab_imdb(train_data)

# 创建数据迭代器
batch_size = 64
train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)
#
# for X,y in train_iter:
#     print('X ', X.shape, ',y ', y.shape)
#     break
# print(len(train_iter))


# 使用循环神经网络的模型
class BiRNN(nn.Module):
    # 参数：词典、嵌入层大小（向量长度）、隐藏层大小、隐藏层层数
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 编码层：双向循环神经网络
        # num_directions=2，隐藏层大小实际 = num_hiddens * 2
        # 因为有两个方向，每一层隐藏层都有200个单元，每个方向100个单元
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens,
                               num_layers=num_layers, bidirectional=True)
        # 解码层：线性计算
        # 取首尾的输出叠加后进行映射
        # 本来的理解是最初（相对而言）时间步100个单元，最终时间步100个单元，加起来200个
        # 因为最初是最终往前传播的最后结果，最终是最初往后传播的最后结果，各有100单元才对
        # 但因为有两个方向，两者的参数和单元都是叠加起来的，所以是各有200个
        self.decoder = nn.Linear(4*num_hiddens, embed_size)

    def forward(self, inputs):
        # inputs的形状为 (批量大小,词数)
        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置
        # 再提取向量特征，输出形状为 (词数，批量大小，向量长度)
        embedding = self.embedding(inputs.permute(1,0))
        # (词数，批量大小，2*隐藏单元个数)
        outputs, _ = self.encoder(embedding)
        # (批量大小，4*隐藏单元个数)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        # (批量大小，向量长度)
        outs = self.decoder(encoding)
        return outs

# 参数设置
embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

# 由于情感分类的训练数据集并不是很大，为应对过拟合
# 我们将直接使用在更大规模语料上预训练的词向量作为每个词的特征向量
# 预训练词向量的维度需要与创建的模型中的嵌入层输出大小embed_size一致
cache_dir = os.path.join(DATA_ROOT, 'glove')
glove_vocab = Vocab.pretrained_aliases["glove.6B.100d"](cache=cache_dir)

# 将预训练的的词向量copy到模型参数上
net.embedding.weight.data.copy_(d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
# 词向量脱离梯度求导
net.embedding.weight.requires_grad = False

# 训练参数
lr, num_epochs = 0.01, 5
# 词向量脱离梯度求导
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# 测试
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))  # positive
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']))  # negative

