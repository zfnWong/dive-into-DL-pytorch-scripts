# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:45:03 2020

@author: Administrator
"""

import torch
import torchtext.vocab as vocab



# 使用预训练的词向量,在大规模语料上预训练的词向量常常可以应用于下游自然语言处理任务中
# 查看它目前提供的预训练词嵌入的名称
# print(vocab.pretrained_aliases.keys())

# glove考虑了形状相似的词 bag, bags
# 查看glove词嵌入提供了哪些预训练的模型
# 每个模型的词向量维度可能不同，或是在不同数据集上预训练得到的
# 预训练的GloVe模型的命名规范大致是“模型.（数据集.）数据集词数.词向量维度”
# print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])


# 下载词典，单词都是小写化过的
# stoi: 类似 token_to_idx
# itos: 类似 idx_to_token
# vectors: 词向量
cache_dir = "./Datasets/glove/"
glove = vocab.pretrained_aliases["glove.6B.50d"](cache=cache_dir)
# print(len(glove))
# print(glove.stoi['beautiful'], glove.itos[3366])


# 使用余弦相似度来搜索近义词的近邻算法
def knn(W, x, k):
    cos = torch.mm(W, x.view(-1, 1)).view(-1) / (torch.sum(W * W, dim=1).sqrt() * 
                            torch.sum(x * x).sqrt() + 1e-9)
    _, topk = torch.topk(cos, k)
    topk = topk.cpu().numpy()
    # 返回下标、余弦值集合
    return topk, [cos[i].item() for i in topk]


# 获取近义词 beautiful-lovely
def get_similar_tokens(query_token, k, embed):
    # 输入
    topk, tokens = knn(embed.vectors, embed.vectors[embed.stoi[query_token]], k+1)
    # 输出
    for i, c in zip(topk[1:], tokens[1:]):
        print('consine sim=%.3f: %s' % (c, embed.itos[i]))
        
# 测试
# get_similar_tokens('chip', 3, glove)
# get_similar_tokens('baby', 3, glove)
# get_similar_tokens('beautiful', 3, glove)



# 获取类比词  Beijing-China 相当于 Tokyo-?
# 思路：搜索与vec(c)+vec(b)−vec(a)的结果向量最相似的词向量
def get_analogy(token_a, token_b, token_c, embed):
    # 获取三个向量,用数组更方便
    vecs = [embed.vectors[embed.stoi[t]] for t in [token_a, token_b, token_c]]
    # 目标
    index_d, _ = knn(embed.vectors, vecs[1]+vecs[2]-vecs[0], 1)
    print(embed.itos[index_d[0]])

# get_analogy("beijing", "china", "tokyo", glove) # 'japan'
# get_analogy('man', 'woman', 'son', glove) # 'daughter'
# get_analogy('bad', 'worst', 'big', glove) # 'biggest'
# get_analogy('do', 'did', 'go', glove) # 'went'









