#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Zhaoyang LI
@Contact: zhaoyang.li@centrale-med.fr
@File: util
@Time: 12/26/24 4:15 PM
"""


import numpy as np
import torch
import torch.nn.functional as F


import torch

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    
    # 将标签值从 [1, 2] 转换为 [0, 1]
    # gold = gold - 1   # 将标签值调整为从 0 开始
    batch_size, num_points = gold.size()  # 获取批量大小和点的数量 gold.size()返回真实标签的形状

    # 创建一个形状为 [batch_size, num_points, n_class] 的全零张量
    n_class = pred.size(2)  # 获取类别数，即pred的第三维大小，也即 output_channels
    one_hot = torch.zeros(batch_size, num_points, n_class, device=gold.device)

    # 使用 scatter_ 方法创建 one-hot 编码
    # scatter(2, gold.unsqueeze(-1), 1)表示将标签对应位置的值设置为1，其他位置保持为0
    # 2表示沿第三维进行
    one_hot = one_hot.scatter(2, gold.unsqueeze(-1), 1)  # gold.unsqueeze(-1) 将 gold 转换为 [batch_size, num_points, 1]
    
    if smoothing:    # 启用标签平滑，目的是在损失计算中让模型不对标签完全过拟合，通过调整目标标签，使得它不完全是1或0
        eps = 0.2
        # 应用标签平滑
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

        # 对输入张量pred沿着类别维度（即dim=2）进行softmax操作，然后对结果取对数，得到每个类别的对数概率，以避免数值不稳定
        # softmax的核心作用是将模型的输出转换为一个概率分布
        log_prb = F.log_softmax(pred, dim=2)

        # 计算平滑后的交叉熵损失
        # -(one_hot * log_prb)是计算每个类别的交叉熵损失
        # sum(dim=2)沿着类别维度对每个点的损失进行求和
        # 对整个批次的所有点的损失求平均
        loss = -(one_hot * log_prb).sum(dim=2).mean()
    else:
        # 计算标准的交叉熵损失
        # pred.view(-1, n_class)将pred展平为二维张量，形状为[batch_size * num_points, n_class]，使得每个点的预测与类别一一对应
        # gold.view(-1)将真实标签展平为一维张量，形状为[batch_size * num_points]
        # 通过reduction='mean'来返回整个批次的平均损失
        loss = F.cross_entropy(pred.view(-1, n_class), gold.view(-1), reduction='mean')

    return loss




class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
