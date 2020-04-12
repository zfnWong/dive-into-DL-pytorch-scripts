# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:17:57 2020

@author: Administrator
"""

import os
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import  torch.nn.functional as F
import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = "./Datasets"


# 一维卷积运算
def corr1d(x, k):
    w = k.shape[0]
    y = torch.zeros((x.shape[0] - w + 1))
    for i in range(y.shape[0]):
        y[i] = (x[i:i+w] * k).sum()
    return y
# 测试
# X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
# print(corr1d(X, K))


# 多输入通道一维卷积运算
def corr1d_multi_in(X, K):
    return torch.stack([corr1d(x, k) for x,k in zip(X,K)]).sum(dim=0)
# 测试
# X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
#               [1, 2, 3, 4, 5, 6, 7],
#               [2, 3, 4, 5, 6, 7, 8]])
# K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
# print(corr1d_multi_in(X, K))


# PyTorch没有自带全局的最大池化层(全局：通道长度可变)
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
                return F.max_pool1d(x, kernel_size=x.shape[-1])


# 获取训练、测试数据并转换成迭代器
batch_size = 64
train_data = d2l.read_imdb('train', data_root=os.path.join(DATA_ROOT, "aclImdb"))
test_data = d2l.read_imdb('test', data_root=os.path.join(DATA_ROOT, "aclImdb"))
vocab = d2l.get_vocab_imdb(train_data)
train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)


# =============================================================================
# 为什么用两个嵌入层
# 在静态的词向量中，与"bad"最相近的词竟是"good"，这主要是因为这两个词出现场景以及所在句子的语法结构都颇为类似，
# 而可调整（动态）的词向量找到的词都比较"得体"，与人类的想法也比较接近，甚至连缩写和标点符号都能"揣摩"到其含义。
# 不过在某种程度上这种"过度推断"容易造成过拟合，作者受图像表示的启发，
# 将这两种词向量作为了输入层不同的channel来进行训练，在一定程度上进行了正则化，取得了还不错的效果
# =============================================================================

# 定义模型
class TextCNN(nn.Module):

    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        # 两个嵌入层，一个静态（不进行训练，固定词向量）
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 一个动态（训练，在预训练词向量基础上训练）
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        # 多个卷积核，捕捉到不同个数的相邻词的相关性
        self.convs = nn.ModuleList()
        for c,k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2*embed_size,
                                        out_channels=c, kernel_size=k))
        # 丢弃法
        self.dropout = nn.Dropout(0.5)
        # 全局池化层
        self.pool = GlobalMaxPool1d()
        # 输出层
        self.decoder = nn.Linear(sum(num_channels), 2)

    def forward(self, inputs):
        # shape (batch_size, seq_len, 2*embed_size)
        embeddings = torch.cat((self.embedding(inputs),
                               self.constant_embedding(inputs)), dim=2)
        # shape (batch_size, 2*embed_size, seq_len)
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个卷积核，在时序最大池化层后会得到一个形状为
        # (batch_size, out_channels, 1) 的tensor，使用flatten函数去掉长度为1的维度
        # 然后在通道维上连接，得到 (batch_size, sum(out_channels))
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) \
                              for conv in self.convs], dim=1)
        # 使用丢弃法后进行输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


# 模型参数
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)


# 装载预训练的词向量
cache_dir = os.path.join(DATA_ROOT, 'glove')
glove_vocab = Vocab.pretrained_aliases["glove.6B.100d"](cache=cache_dir)
net.embedding.weight.data.copy_(d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
net.constant_embedding.weight.data.copy_(d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
net.constant_embedding.weight.requires_grad = False


# 设置训练参数
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# 测试
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])) # positive
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])) # negative