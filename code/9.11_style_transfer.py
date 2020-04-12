# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:22:32 2020

@author: Administrator
"""

import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 读取图片
d2l.set_figsize()
content_img = Image.open('./small_dataset/rainier.jpg')
# d2l.plt.imshow(content_img)
style_img = Image.open('./small_dataset/autumn_oak.jpg')



# =============================================================================
# torchvision.transforms模块有大量现成的转换方法，不过需要注意的
# 是有的方法输入的是PIL图像，如Resize；有的方法输入的是tensor，如
# Normalize；而还有的是用于二者转换，如ToTensor将PIL图像转换成tensor
# =============================================================================
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

# 转换尺寸，tensor转换，归一化转换（应用训练好的模型）
def preprocess(PIL_img, image_shape):
    process = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ])
    return process(PIL_img).unsqueeze(dim=0) # (1, 3, H, W)

# 0-1归一化（clamp指定边界），PIL转换
def postprocess(img_tensor):
    inv_normalize = torchvision.transforms.Normalize(
        mean = -rgb_mean / rgb_std, 
        std = 1/rgb_std)
    to_PIL_image = torchvision.transforms.ToPILImage()
    return to_PIL_image(inv_normalize(img_tensor[0].cpu()).clamp(0, 1))


# =============================================================================
# 一般来说，越靠近输入层的输出越容易抽取图像的细节信息，反之则越容易
# 抽取图像的全局信息。为了避免合成图像过多保留内容图像的细节，我们选
# 择VGG较靠近输出的层，也称内容层，来输出图像的内容特征。实验中，我
# 们选择第四卷积块的最后一个卷积层作为内容层，以及每个卷积块的第一个
# 卷积层作为样式层。
# =============================================================================
# 加载预训练模型
pretrained_net = torchvision.models.vgg19(pretrained=True, progress=True)
# print(pretrained)

# 只保留需要用到的VGG的所有层
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
net_list = []  # 在extract_features中被调用
for i in range(max(content_layers + style_layers) + 1):
    net_list.append(pretrained_net.features[i])
net = torch.nn.Sequential(*net_list) # *可以直接取list里的内容，还记得c语言吗
net = net.to(device)


# =============================================================================
# 给定输入X，如果简单调用前向计算net(X)，只能获得最后一层的输出。由
# 于我们还需要中间层的输出，因此这里我们逐层计算，并保留内容层和样式
# 层的输出。
# =============================================================================
# 对合成图像抽取内容和样式特征
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# get_contents函数对内容图像抽取内容特征
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

# get_styles函数则对样式图像抽取样式特征
def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


# =============================================================================
# 样式迁移的损失函数即内容损失、样式损失和总变差损失的加权和。通
# 过调节这些权值超参数，我们可以权衡合成图像在保留内容、迁移样式
# 以及降噪三方面的相对重要性。
# =============================================================================

# 内容损失：平方误差函数
def content_loss(Y_hat, Y):
    return F.mse_loss(Y_hat, Y)

# 样式损失：格拉姆矩阵+平方误差函数
# (bn,c,h,w) -> (c,hw)
# /= chw
def gram(X):
    num_channels, n = X.shape[1], X.shape[2] * X.shape[3]
    X = X.view(num_channels, n)
    return torch.mm(X, X.t()) / (num_channels * n)
def style_loss(Y_hat, gram_Y):
    # 这里假设基于样式图像的格拉姆矩阵gram_Y已经预先计算好了
    return F.mse_loss(gram(Y_hat), gram_Y)

# 总变差损失：总变差降噪
# 假设xi,j表示坐标为(i,j)(i,j)的像素值，降低总变差损失能够尽可
# 能使邻近的像素值相似。
def tv_loss(Y_hat):
    # |xi,j - xi+1,j| + |xi,j - xi,j+1| 求和
    return 0.5 * (F.l1_loss(Y_hat[:,:,1:,:], Y_hat[:,:,:-1,:]) +
                  F.l1_loss(Y_hat[:,:,:,1:], Y_hat[:,:,:,:-1]))

# 通过调节三种损失的权值超参数，我们可以权衡合成图像在保留内容、迁移样式以及降噪三方面的
# 相对重要性
content_weight, style_weight, tv_weight = 1, 1e3, 10
# 样式迁移的损失函数即内容损失、样式损失和总变差损失的加权和。
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算
    contents_l = [content_loss(Y_hat, Y) * content_weight \
                  for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight\
                for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 求和
    l = sum(styles_l) + sum(contents_l) + tv_l
    # 返回多个loss，留着训练时观察各部分的训练误差
    return contents_l, styles_l, tv_l, l


# =============================================================================
# 创建和初始化合成图像
# =============================================================================
# 定义一个简单的模型GeneratedImage，并将合成图像视为模型参数
# 模型的前向计算只需返回模型参数即可
class GenerateedImage(torch.nn.Module):
    def __init__(self, img_shape):
        super(GenerateedImage, self).__init__()
        # Parameter继承自tensor类，可以直接给loss函数计算
        self.weight = torch.nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

# 初始化为图像X
# 样式图像在各个样式层的格拉姆矩阵styles_Y_gram将在训练前预先计算好
def get_inits(X, device, lr, styles_Y):
    gen_img = GenerateedImage(X.shape).to(device)
    gen_img.weight.data = X.data
    optimizer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    # gen_img(), 相当于net(X)，因为不需要输入，直接返回参数
    return gen_img(), styles_Y_gram, optimizer


# =============================================================================
# 训练
# 不断抽取合成图像的内容特征和样式特征，并计算损失函数
# =============================================================================
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    print("training on", device)

    # 初始化得到初始合成图像X，X的样式的格拉姆矩阵，优化器
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)

    # 等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size。
    # 需要注意的是，step通常是指epoch，不要弄成mini-batch的iteration了
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, gamma=0.1)

    start = time.time()
    for i in range(num_epochs):

        # 抽取合成图像的内容特征和样式特征
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        # 计算损失函数
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)

        # 求导，更新参数
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # 每50次迭代输出一次训练结果
        if i % 50 == 0 and i != 0:
            print('epoch %3d, content loss %.2f, style loss %.2f,'
                  'TV loss %.2f, %.2f sec'
                  % (i, sum(contents_l).item(), sum(styles_l).item(), tv_l.item(),
                     time.time()-start))
            start = time.time()

    return X.detach() # 返回X的脱离计算图的副本，后面还要Resize继续训练


# 低画质版本
image_shape = (150, 225)
content_X, contents_Y = get_contents(image_shape, device)
style_X, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.01, 500, 200)
d2l.set_figsize((7, 5))
d2l.plt.imshow(postprocess(output))
d2l.plt.show()  # 在pycharm里要加上这一句才能显示图片


# 较高画质版本
# image_shape = (300, 450)
# _, contents_Y = get_contents(image_shape, device)
# _, styles_Y = get_styles(image_shape, device)
# X = preprocess(postprocess(output), image_shape).to(device) # 在output基础上继续训练
# big_output = train(X, contents_Y, styles_Y, device, 0.01, 500, 200)
# d2l.plt.imshow(postprocess(big_output))


