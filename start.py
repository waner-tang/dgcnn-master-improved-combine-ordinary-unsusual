#!/usr/bin/env python
# coding: utf-8


# In[ ]:

import os
import subprocess
import numpy as np
import torch
import glob
from IPython import get_ipython


# In[ ]:


def load_data(self, dir_path):
        data = []
        labels = []

        # 遍历目录中的所有txt文件
        for file_path in glob.glob(os.path.join(dir_path, '*.txt')):
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    # 确保这一行有7个部分 (x, y, z, nx, ny, nz, label)
                    if len(parts) != 7:
                        print(f"Skipping malformed line in {file_path}: {line.strip()}")
                        continue

                    try:
                        coords_and_features = [float(x) for x in parts[:6]]  # 提取坐标和法向量
                        label = int(float(parts[6]))  # 提取标签并转换为int
                        data.append(coords_and_features)
                        labels.append(label)
                    except ValueError as e:
                        print(f"Skipping line with invalid data in {file_path}: {line.strip()}")
                        print(f"Error: {e}")
                        continue

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        return data, labels


# In[ ]:

# get_ipython().system('ls pytorch/dataset/train_set')
# 获取文件夹中的文件列表
# files = os.listdir('pytorch/dataset/train_set')
# print("\n".join(file for file in files))


# In[7]:


# get_ipython().system('python pytorch/main.py --exp_name test_predict --model dgcnn --model_path pytorch/pretrained/model.t7 --predict True --predict_path pytorch/dataset_p/003007.txt')
# get_ipython().system('python pytorch/main.py --exp_name test_predict --model dgcnn --model_path pytorch/pretrained/model.t7 --predict True --predict_path pytorch/dataset_p/003008.txt')
# get_ipython().system('python pytorch/main.py --exp_name test_predict --model dgcnn --model_path pytorch/pretrained/model.t7 --predict True --predict_path pytorch/dataset_p/003009.txt')
# get_ipython().system('python pytorch/main.py --exp_name test_predict --model dgcnn --model_path pytorch/pretrained/model.t7 --predict True --predict_path pytorch/dataset_p/003010.txt')

# 训练
# subprocess.run([
#     'python', 'pytorch/main.py',
#     '--exp_name', 'test_my_data_three_classifications',
#     '--model', 'dgcnn',
#     '--epoch', '250'])
# 测试
subprocess.run([
    'python', 'pytorch/main.py',
    '--exp_name', 'test_my_data_three_classifications',
    '--model', 'dgcnn',
    '--model_path', 'pytorch/checkpoints/test_my_data_three_classifications/models/model.t7',    # 待修改
    '--predict', 'True',
    '--predict_path', 'pytorch/dataset/test_set'])





