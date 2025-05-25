"""
@Author: Zhaoyang LI
@Contact: zhaoyang.li@centrale-med.fr
@File: util
@Time: 12/26/24 4:15 PM
"""

import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dir_path, num_points=1024, partition='train'):
        self.data, self.labels = self.load_data(dir_path)
        self.num_points = num_points
        self.partition = partition
    
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

    def __getitem__(self, idx):
      """
      获取点云数据，将其划分为 shape 为 [n, num_points, 6] 的 minibatch。
      """
      # 获取当前样本的点云数据
      pointcloud = self.data[idx]
      label = self.labels[idx]

      # 计算当前点云的点数
      total_points = pointcloud.shape[0]

      # 划分为 minibatch，每个 batch 包含 num_points 个点
      num_batches = int(np.ceil(total_points / self.num_points))
      padded_pointcloud = np.zeros((num_batches, self.num_points, pointcloud.shape[1]), dtype=np.float32)

      for i in range(num_batches):
          start_idx = i * self.num_points
          end_idx = min(start_idx + self.num_points, total_points)
          batch_points = pointcloud[start_idx:end_idx]
          padded_pointcloud[i, :batch_points.shape[0], :] = batch_points

      if self.partition == 'train':
          # 数据增强和随机打乱
          for i in range(num_batches):
              padded_pointcloud[i] = self.translate_pointcloud(padded_pointcloud[i])
              np.random.shuffle(padded_pointcloud[i])

      return padded_pointcloud, label

    def __len__(self):
        return len(self.data)

    def translate_pointcloud(self, pointcloud):
        # 只对坐标部分进行平移和缩放
        xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])  # 缩放因子
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])         # 平移因子
        translated_coords = np.add(np.multiply(pointcloud[:, :3], xyz1), xyz2).astype('float32')
        
        # 拼接变换后的坐标和原始的法向量
        translated_pointcloud = np.hstack((translated_coords, pointcloud[:, 3:]))
        return translated_pointcloud

def create_dataloader(dir_path, num_points=1024, batch_size=32, partition='train'):
    dataset = CustomDataset(dir_path, num_points, partition)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(partition == 'train'))
    return dataloader
