# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:08:21 2020

@author: Administrator
"""


import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import d2lzh_pytorch as d2l


# PTB（Penn Tree Bank）是一个常用的小型语料库,它采样自《华尔街日报》的文章
assert 'ptb.train.txt' in os.listdir("./Datasets/ptb/")

with open('./Datasets/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]
    
# for st in raw_dataset[:3]:
#     print('# tokens ', len(st), st[:5])
    

# 为了计算简单，只保留在数据集中至少出现5次的词。
# counter 
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))


# 然后将词映射到整数索引
idx_to_token = [tk for tk,_ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
#
dataset = [[token_to_idx[tk] for tk in st if tk in idx_to_token]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in raw_dataset])


# 二次采样试图尽可能减轻高频词对训练词嵌入模型的影响
def discard(idx):
    # bool
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / (counter[idx_to_token[idx]] / num_tokens))
# discard 作为一个判断条件
subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]


# 测试
# def compare_counts(token):
#     return '# %s: before=%d, after=%d' % (token, 
#         sum([st.count(token_to_idx[token]) for st in dataset]),
#         sum([st.count(token_to_idx[token]) for st in dataset]))


# 提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        # 大于等于2个词才有一组中心-背景词
        if len(st) < 2:
            continue
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            # 窗口下背景词在st中的下标list
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + window_size + 1)))
            # 删除中心词
            indices.remove(center_i)
            # 将该组背景词加入到背景词集合中
            contexts.append([st[idx] for idx in indices])
            # 将中心词加入到中心词集合中
            centers.append(st[center_i]) #
    return centers, contexts

# tiny_dataset = [list(range(7)), list(range(7, 10))]
# print('dataset', tiny_dataset)
# for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
#     print('center', center, 'has contexts', context)
    

# 获取二次采样后的数据集的所有中心词和背景词
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)


# 负采样(跳字模型)
# 
def get_negatives(all_contexts, sampling_weight, K):
    all_negatives, neg_candidates, i = [], [], 0
    
    population = list(range(len(sampling_weight)))
    
    for context in all_contexts:
        negatives = []
        
        # K为每个背景词需要采样的噪声词数量
        while len(negatives) < K * len(context):
            # 如果候选的采样词用完了
            if i == len(neg_candidates):
                # random.choices(下标集合，概率权重，采样个数)
                i, neg_candidates = 0, random.choices(
                    population, sampling_weight, k=int(1e5))
            # 提取一个候选词
            neg, i = neg_candidates[i], i+1
            # 如果该候选词没在窗口内，则采样成功
            # 这里有可能采样到中心词，没问题的
            if neg not in set(context):
                negatives.append(neg)
        # 每组噪声词对应每组背景词    
        all_negatives.append(negatives)
    return all_negatives

# 采样概率为 词频/总词数 再取0.75次幂，因为分母都相同所以省去了
sampling_weight = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weight, 5)


# 由Dataset类才可以用DataLoader，然后得到data_iter
# 可以在 DataLoader 中返回 batch_size 大小的批量
class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives
        
    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])
    
    def __len__(self):
        return len(self.centers)
    


# 指定 DataLoader 中小批量的读取方式
def batchify(data):
    # 获取最大长度以统一长度，不足的用填充项填充
    max_len = max(len(c) + len(n) for _,c,n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    # 
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        # 将背景词和噪声词合并，根据条件概率的定义这样做可以方便向量运算
        # list的加法就是求两个 list 的并
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        # 掩码
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        # 标签向量，背景词对应1，噪声词对应0
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
        
    return (torch.tensor(centers).view(-1,1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))


# 设置参数
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
# 获取数据
dataset = MyDataset(all_centers, all_contexts, all_negatives)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True,
                            num_workers=num_workers,
                            collate_fn=batchify)

# 测试
# for batch in data_iter:
#     for name, data in zip(['center', 'context_negative', 'mask', 'label'], batch):
#         print(name, 'shape:', data.shape)
#     break
        
    

# 输出中的每个元素是中心词向量与背景词向量和噪声词向量的内积
def skip_gram(centers, contexts_negatives, embed_v, embed_u):
    v = embed_v(centers)  # shape (batch_size, 1, embed_size)
    u = embed_u(contexts_negatives) # shape (batch_size, max_len, embed_size)
    pred = torch.bmm(v, u.permute(0, 2, 1)) # shape (batch_size, 1, max_len)
    return pred


# 二元交叉熵损失函数,他妈的就是二分类的交叉熵损失函数啦，被坑了
# 之前的交叉熵损失函数都是用在fashion_mnist集上的多分类
# 与以往理解不同的是，对 y_hat 先进行了 sigmoid
# **损失函数每次计算的是一个context的损失,而不是一个batch的损失**
# 损失函数会按 len(mask) 求平均,但是有一部分是填充项
# 所以真正的损失还要乘 len(mask) 再除以 sum(mask)
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    
    def __init__(self): 
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    
    def forward(self, inputs, targets, mask=None):
        # target 是 label ，1和0分别表示背景词和噪声词
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        
        # weight即mask决定相应位置的预测与标签是否参与损失函数的计算
        res = nn.functional.binary_cross_entropy_with_logits(inputs, 
                                    targets, reduction="none", weight=mask)
            
        return res.mean(dim=1)

loss = SigmoidBinaryCrossEntropyLoss()


# 定义模型
embed_size = 100
net = nn.Sequential(
    # 两个嵌入层（给定下标，返回向量表示）：上面是中心词，下面是背景词
    nn.Embedding(len(idx_to_token), embedding_dim=embed_size),
    nn.Embedding(len(idx_to_token), embedding_dim=embed_size)
    )


# 训练模型
def train(net, lr, num_epochs):
    
    # 设备、模型、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('train on', device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # 迭代
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            # 将所有数据转移到gpu上
            center, context_negative, mask, label = [d.to(device) for d in batch]
            # 
            pred = skip_gram(center, context_negative, net[0], net[1])
            # **损失函数每次计算的是一个context的损失,而不是一个batch的损失**
            # 损失函数会按 len(mask) 求平均,但是有的只是填充项
            # 所以真正的损失还要乘 len(mask) 再除以 sum(mask)
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean()
            # 
            optimizer.zero_grad()
            #
            l.backward()
            # 
            optimizer.step()
            #
            l_sum += l.cpu().item()
            n += 1
        # loss 是平均到每个词(出现在context_negative中的词)的误差
        print('epoch %d, loss %.2f, time %.2fs' % \
                (epoch + 1, l_sum / n, time.time() - start))
            
train(net, 0.01, 10)


# 寻找近义词
def get_similar_tokens(query_token, n, embed):
    W = embed.weight.data
    x = W[token_to_idx[query_token]].view(-1, 1)
    
    # 添加的1e-9是为了数值稳定性
    # 用了sum后会降维,所以先把W*x降维
    cos = torch.mm(W, x).view(-1) / (torch.sum(W * W, dim=1).sqrt() * 
                                        torch.sum(x * x).sqrt() + 1e-9) 
    # torch.topk 返回values和indices
    _, topk = torch.topk(cos, k=n+1) # 包含查找词本身，所以要+1
    topk = topk.cpu().numpy()
    
    for i in topk[1:]:
        print('cosine sim=%.3f: %s' % (cos[i], idx_to_token[i]))
    
get_similar_tokens('chip', 3, net[0]) # 跳字模型用中心词向量