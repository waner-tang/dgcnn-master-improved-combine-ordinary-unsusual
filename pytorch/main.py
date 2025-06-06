#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Zhaoyang LI
@Contact: zhaoyang.li@centrale-med.fr
@File: util
@Time: 12/26/24 4:15 PM
"""


from __future__ import print_function
import os
import argparse
import shutil
import time
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
from custom_data import create_dataloader
import sklearn.metrics as metrics
import matplotlib.pyplot as plt  # 添加用于可视化的包
import matplotlib.colors as mcolors


def _init_():
    if not os.path.exists('pytorch/checkpoints'):
        os.makedirs('pytorch/checkpoints')
    if not os.path.exists('pytorch/checkpoints/'+args.exp_name):
        os.makedirs('pytorch/checkpoints/'+args.exp_name)
    if not os.path.exists('pytorch/checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('pytorch/checkpoints/'+args.exp_name+'/'+'models')
    # 定义路径
    src_main = 'pytorch/main.py'
    src_model = 'pytorch/model.py'
    src_util = 'pytorch/util.py'
    src_custom_data = 'pytorch/custom_data.py'
    dest_dir = f'pytorch/checkpoints/{args.exp_name}'
    # 确保目标文件夹存在
    os.makedirs(dest_dir, exist_ok=True)
    # 使用 shutil 复制文件
    shutil.copy(src_main, os.path.join(dest_dir, 'main.py.backup'))
    shutil.copy(src_model, os.path.join(dest_dir, 'model.py.backup'))
    shutil.copy(src_util, os.path.join(dest_dir, 'util.py.backup'))
    shutil.copy(src_custom_data, os.path.join(dest_dir, 'custom_data.py.backup'))

def train(args, io):

    _init_()
    # -----------added
    train_loader = create_dataloader('pytorch/dataset/train_set', num_points=args.num_points, batch_size=args.batch_size, partition='train', drop_last=True)
    val_loader = create_dataloader('pytorch/dataset/val_set', num_points=args.num_points, batch_size=args.test_batch_size, partition='test', drop_last=True)
    num_points=1024  # args.num_points
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
# 优化器
# 优化器是用于调整模型参数的核心组件，其作用是通过反向传播计算的梯度信息（梯度指向损失增长最快的方向）
# 动态更新网络参数以最小化损失函数，优化器的选择直接影响训练速度、模型收敛性及最终性能
    # SGD通过随机选取一个样本或小批量样本计算梯度，并沿负梯度方向更新参数
    if args.use_sgd:
        print("Use SGD")
        # 动量通过将过去的梯度加权平均，减少震荡，并加速收敛，帮助加速SGD在相关方向上的收敛，并减少震荡，一般推荐使用 0.9 或类似的值
        # 权重衰减是L2正则化的一种形式，控制模型复杂度，防止过拟合
        # 权重衰减太大：模型可能会过于简单，导致欠拟合；对参数更新产生更多的约束，导致优化过程更为缓慢
        # 权重衰减太小：模型可能会过于复杂，从而容易过拟合训练数据。训练数据表现得很好，但在测试集上表现较差。
        # 一般情况下，设置较小的值（如 1e-4 或 1e-5）能有效防止过拟合，同时对模型的复杂度进行一定的约束
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        # Adam适用于复杂模型、大规模数据的情况下，在简单任务中可能导致快速收敛但泛化性下降
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 这里，在训练过程中CosineAnnealingLR逐渐将学习率从初始值降低到最小值，此为余弦退火，还可以指数衰减
    # 余弦衰减更适合长时间训练，提供平滑的学习率衰减
    # 学习率衰减：通过逐步减小学习率，使得训练在接近最优解时更稳定，避免过度跳跃
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        start_time = time.time()  # 记录开始时间
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            # batch_size 是每次训练中送入模型的样本数量
            # num_points 是每个样本中包含的元素数量（例如，点云中的点数或图像中的像素数）
            # data对应形状为(batch_size,num_points,num_dims) ,num_dims=6
            # label对应形状为(batch_size,num_points，1)->(batch_size,num_points)
            data, label = data.to(device), label.to(device).squeeze()    # squeeze()用于去除单维度
            # print(f"Data shape before permute: {data.shape}")
            # print(f"label shape:{label.shape}")
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()    # 清空优化器梯度，防止梯度累加，确保每个批次独立计算梯度
            # logits 是模型输出的未经处理的原始预测结果，通常是一个张量，形状为 (batch_size, num_points, num_classes)
            # 即对于每个样本（batch），每个点（num_points）都有一个关于各类别的预测得分（num_classes）
            logits = model(data)
            # print(logits.shape)
            loss = criterion(logits, label)
            loss.backward()    # 计算梯度
            opt.step()       # 根据所计算的梯度更新模型的参数
            # 每个点（即维度2）进行 argmax 操作，返回在类别维度（dim=2）上具有最大得分的类别索引
            # preds的形状为(batch_size, num_points)，每个元素表示模型预测的类别索引
            preds = torch.argmax(logits, dim=2)
            count += batch_size * num_points
            train_loss += loss.item() * batch_size * num_points
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        
        train_pred = np.concatenate(train_pred)
        # train_pred = train_pred + 1  # 标签值[01]回到[12]
        
        # print(train_true)
        # print(train_pred)

        # 预测和真实标签被展平为一维数组，以便于比较
        train_true = train_true.flatten()
        train_pred = train_pred.flatten()

        # 计算 LOU
        iou_score = metrics.jaccard_score(train_true, train_pred, average='macro')

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, IoU: %.6f' % (epoch,
                                                                                            train_loss * 1.0 / count,
                                                                                            metrics.accuracy_score(
                                                                                                train_true, train_pred),
                                                                                            metrics.balanced_accuracy_score(
                                                                                                train_true, train_pred),
                                                                                            iou_score)
        io.cprint(outstr)  # 将outstr转换为字符串记录到日志并同步打印到终端

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in val_loader:
            data, label = data.to(device), label.to(device).squeeze()
            #print(f"Data shape before permute: {data.shape}")
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = torch.argmax(logits, dim=2)
            count += batch_size * num_points
            test_loss += loss.item() * batch_size * num_points
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        # test_pred = test_pred + 1  # 标签补回
        test_true = test_true.flatten()
        test_pred = test_pred.flatten()
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time  # 计算当前轮次所用的时间
        # 计算 LOU
        iou_score = metrics.jaccard_score(test_true, test_pred, average='macro')
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, IoU: %.6f, time: %.2f s' % (epoch,
                                                                                                       test_loss * 1.0 / count,
                                                                                                       test_acc,
                                                                                                       avg_per_class_acc,
                                                                                                       iou_score,
                                                                                                       epoch_time)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            # -------------added pytorch/
            torch.save(model.state_dict(), 'pytorch/checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = create_dataloader('data/test_set', num_points=args.num_points, batch_size=args.test_batch_size, partition='test')

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = torch.argmax(logits, dim=2)
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

    
def predict(args,io):
    if not os.path.exists(args.predict_path):
        io.cprint("Predict path does not exist")
        return
    # 获取文件夹中的所有文件（假设文件是文本文件，您可以根据需要更改扩展名）
    # ---------修改以处理嵌套文件夹
    files = glob.glob(os.path.join(args.predict_path, "**", "*.txt"), recursive=True)
    if len(files) == 0:
        io.cprint("No files found in the directory")
        return
    # 创建一个结果文件夹，如果它不存在
    result_dir = 'pytorch/result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    output_folder = partseg_target_path(result_dir)
    text_root = os.path.join(output_folder, "text")
    image_root = os.path.join(output_folder, "image")
    os.makedirs(text_root, exist_ok=True)
    os.makedirs(image_root, exist_ok=True)

    for file in files:
        io.cprint(f"Processing file: {file}")

        # 生成输出文件夹的路径，保持输入文件夹的结构
        relative_path = os.path.relpath(file, args.predict_path)  # 获取相对路径
        subfolder = os.path.dirname(relative_path)  # 获取子文件夹部分
        full_output_folder = os.path.join(output_folder, subfolder)  # 在 partseg 文件夹下创建对应的子文件夹

        # # +++ 添加目录创建 +++
        # if not os.path.exists(full_output_folder):
        #     os.makedirs(full_output_folder, exist_ok=True)  # 递归创建缺失的父目录

        # 在 text_root 下创建对应的子文件夹
        full_text_folder = os.path.join(text_root, subfolder)
        if not os.path.exists(full_text_folder):
            os.makedirs(full_text_folder, exist_ok=True)
        # 在 image_root 下创建对应的子文件夹
        full_image_folder = os.path.join(image_root, subfolder)
        if not os.path.exists(full_image_folder):
            os.makedirs(full_image_folder, exist_ok=True)

        with open(file, 'r') as f:
            lines = f.readlines()
            data = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    io.cprint(f"Skipping malformed line: {line.strip()}")
                    continue
                try:
                    coords_and_features = [float(x) for x in parts[:6]]
                    data.append(coords_and_features)
                except ValueError as e:
                    io.cprint(f"Skipping line with invalid data: {line.strip()}")
                    io.cprint(f"Error: {e}")
                    continue
            data = np.array(data, dtype=np.float32)
            n= data.shape[0]//1024
            data = data[:n*1024,:]
            data_o = data.copy()
            data_o = data_o.reshape(-1,1024,6)

            data_o = torch.from_numpy(data_o).float()
            data_o = data_o.permute(0, 2, 1)
            device = torch.device("cuda" if args.cuda else "cpu")
            model = DGCNN(args).to(device)
            # model = nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_path,weights_only = True))
            model = model.eval()
            data_o = data_o.to(device)
            logits = model(data_o)
            preds = torch.argmax(logits, dim=2)
            # preds = preds + 1
            new_pointcloud = np.hstack((data.reshape(-1,6),preds.cpu().numpy().reshape(-1,1)))
            # 生成预测结果文件名并保存
            # output_filename = os.path.splitext(os.path.basename(file))[0] + '_predicted.txt'
            # output_path = os.path.join(full_output_folder, output_filename)
            # np.savetxt(output_path, new_pointcloud, fmt='%f')
            output_filename = os.path.splitext(os.path.basename(file))[0] + '_predicted.txt'
            output_path = os.path.join(full_text_folder, output_filename)
            np.savetxt(output_path, new_pointcloud, fmt='%f')
            output_image_filename = output_filename.replace('_predicted.txt', '_predicted.jpg')
            output_image_path = os.path.join(full_image_folder, output_image_filename)
            visualize_pointcloud(new_pointcloud, output_image_path)
    # print("已保存: " + output_folder)
    io.cprint("已保存txt结果至: " + os.path.join(output_folder, "text"))
    io.cprint("已保存可视化图片至: " + os.path.join(output_folder, "image"))

def partseg_target_path(base_path):
    # 将每次的结果保存在新文件夹中
    # 确保 result 文件夹存在
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # 获取所有以 'partseg' 开头的子文件夹
    existing_folders = [folder for folder in os.listdir(base_path) if folder.startswith('partseg')]
    # 提取已有子文件夹中的数字，找出最大的数字
    folder_numbers = []
    for folder in existing_folders:
        try:
            # 提取 'partseg' 后面的数字
            folder_number = int(folder.replace('partseg', ''))
            folder_numbers.append(folder_number)
        except ValueError:
            continue
    # 如果没有任何 'partseg' 文件夹，初始化数字为 1
    if folder_numbers:
        new_folder_number = max(folder_numbers) + 1
    else:
        new_folder_number = 1
    # 新文件夹的路径
    new_folder = os.path.join(base_path, f'partseg{new_folder_number}')
    # 创建新的文件夹
    os.makedirs(new_folder)
    return new_folder


def visualize_pointcloud(pointcloud, output_image_path):
    """
    可视化点云数据，并用图例表示类别颜色，确保颜色唯一。
    """
    # 提取坐标和标签
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]
    labels = pointcloud[:, 6].astype(int)  # 确保整数类别

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 方式 1：使用 tab10/tab20 代替 jet，适用于分类任务
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'tab20')

    # 方式 2：手动指定颜色，避免 jet 颜色过于相近
    colors = cmap(np.linspace(0, 1, num_classes))
    color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}

    fig = plt.figure(figsize=(15, 5))

    def scatter_with_legend(ax, x_data, y_data, title, xlabel, ylabel):
        for label in unique_labels:
            mask = labels == label
            ax.scatter(x_data[mask], y_data[mask], color=color_dict[label], s=1, label=f'Class {label}')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', markerscale=5, fontsize=8)

    # 主视图
    ax1 = fig.add_subplot(1, 3, 1)
    scatter_with_legend(ax1, x, y, 'main view', 'X', 'Y')

    # 左视图
    ax2 = fig.add_subplot(1, 3, 2)
    scatter_with_legend(ax2, y, z, 'left view', 'Y', 'Z')

    # 俯视图
    ax3 = fig.add_subplot(1, 3, 3)
    scatter_with_legend(ax3, x, z, 'top view', 'X', 'Z')

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40','custom'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--predict', type=bool, default=False,help='Predict the model')
    parser.add_argument('--predict_path', type=str, default='', metavar='N',help='Path to predict')
    args = parser.parse_args()

    _init_()  # -----------------added

    io = IOStream('pytorch/checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    if args.predict:
        predict(args, io)
    else:
        if not args.eval:
            train(args, io)
        else:
            test(args, io)

        
    