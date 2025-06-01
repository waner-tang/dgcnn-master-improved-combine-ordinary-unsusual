#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Zhaoyang LI
@Contact: zhaoyang.li@centrale-med.fr
@File: util
@Time: 12/26/24 4:15 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):       # x所对应形状为(batch_size,num_dims,num_points)
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)     # # 计算负的欧氏距离平方
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)取Top-K索引（最近的k个点）
    return idx


def get_graph_feature(x, k=20, idx=None):
  
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    # torch.arange(0, batch_size)：生成 [0, 1, ..., batch_size-1]
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points    # 得到每个样本的起始索引偏移量，形状 (batch_size, 1, 1)

    idx = idx + idx_base   # 将不同样本的索引偏移到正确的位置，避免跨样本索引

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]    # 通过索引收集每个点的 k 个邻居特征，形状 (batch_size*num_points*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # 拼接差值和原始特征,并调整形状为 (batch_size, 2*num_dims, num_points, k)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
     
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=2):  # output_channels标签个数
        super(DGCNN, self).__init__()     # 显式调用父类（nn.Module）的构造函数
        self.args = args
        self.k = args.k

        # 定义批量归一化层，使其均值为0、方差为1，用于加速训练并提升稳定性
        self.bn1 = nn.BatchNorm2d(64)   # BatchNorm2d：用于处理4D张量
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)     # BatchNorm1d：用于处理2D或3D张量

        # nn.Sequential按顺序执行这些层，形成一个小型网络模块
        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv_final = nn.Conv1d(args.emb_dims, output_channels, kernel_size=1)
        # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)    # 线性层的作用是将特征从高维空间映射到低维空间
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)    # Dropout层，防止过拟合，随机置零部分神经元，增强模型的泛化能力
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        
        batch_size, _, num_points = x.size()
        x = get_graph_feature(x, k=self.k)    # (batch_size, 2*num_dims, num_points, k)
        x = self.conv1(x)      # (batch_size, 64, num_points, k)
        # dim=-1：指定在张量的最后一个维度上进行最大值计算 keepdim=False：缩减维度后不保留原始维度形状
        # [0] 用来提取最大值，而忽略最大值的索引
        # 沿邻居维度（K）取最大值，聚合局部特征
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)      # (batch_size, 64*2, num_points)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # 拼接各层特征,[batch_size, 512, num_points]

        x = self.conv5(x)  # [batch_size, emb_dims, num_points]
        x = self.conv_final(x)  # [batch_size, output_channels, num_points]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, num_points, output_channels]

        return x
