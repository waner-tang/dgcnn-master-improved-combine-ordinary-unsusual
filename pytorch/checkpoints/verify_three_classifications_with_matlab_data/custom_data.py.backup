import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dir_path, num_points=1024, partition='train'):
        self.num_points = num_points
        self.data, self.labels = self.load_data(dir_path)
        #print(self.data.shape)
        self.partition = partition
    
    def load_data(self, dir_path):
        data = []
        labels = []
        
        # 遍历目录中的所有txt文件
        # ------------修改以遍历目录及其子目录中的所有txt文件
        for file_path in glob.glob(os.path.join(dir_path, '**', '*.txt'), recursive=True):
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
        target_length = (data.shape[0] // self.num_points) * self.num_points
        trimmed_data = data[:target_length]
        trimmed_labels = labels[:target_length]
        data_reshaped = trimmed_data.reshape(-1,1024,6)  # 输入1024个点，6个特征
        labels_reshaped = trimmed_labels.reshape(-1,1024,1)  # 输入1024个点，1个标签
        
        data = data_reshaped
        labels = labels_reshaped
        
        #print(data.shape)
        #print(labels.shape)
        
        return data, labels

    def __getitem__(self, idx):
        # 获取点云和其对应的特征
        
        pointcloud = self.data[idx][:self.num_points]  # 保留前num_points个点
        label = self.labels[idx]
        return pointcloud, label


    def __len__(self):
        return self.data.shape[0]

# ---------- modified
def create_dataloader(dir_path, num_points=1024, batch_size=32, partition='train', drop_last=False):
    mydataset = CustomDataset(dir_path, num_points, partition)
    dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=(partition == 'train'), drop_last=drop_last)
    return dataloader
