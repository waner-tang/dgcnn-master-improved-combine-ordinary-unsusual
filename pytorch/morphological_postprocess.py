"""
形态学后处理模块
用于改善DGCNN点云分割的边界质量
主要功能：去除小噪声区域、填充小空洞、边界平滑
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time

class MorphologicalPostProcessor:
    def __init__(self, k_neighbors=8, min_component_size=10, max_hole_size=5):
        """
        初始化形态学后处理器
        
        Args:
            k_neighbors: 邻域大小，用于构建空间图
            min_component_size: 最小连通分量大小，小于此值的区域会被去除
            max_hole_size: 最大空洞大小，小于此值的空洞会被填充
        """
        self.k_neighbors = k_neighbors
        self.min_component_size = min_component_size
        self.max_hole_size = max_hole_size
        
    def process(self, labels, points, verbose=True):
        """
        执行完整的形态学后处理
        
        Args:
            labels: 预测标签 (N,)
            points: 点云坐标 (N, 3) 或 (N, 6) [x,y,z] 或 [x,y,z,nx,ny,nz]
            verbose: 是否打印处理信息
            
        Returns:
            processed_labels: 处理后的标签
            processing_time: 处理时间（秒）
        """
        start_time = time.time()
        
        if verbose:
            print(f"开始形态学后处理... (点数: {len(labels):,})")
        
        # 确保使用坐标部分 [x, y, z]
        coords = points[:, :3] if points.shape[1] >= 3 else points
        
        # 构建空间邻接图
        neighbors = self._build_spatial_graph(coords, verbose)
        
        # 步骤1: 去除小的噪声连通分量
        cleaned_labels = self._remove_small_components(labels, neighbors, verbose)
        
        # 步骤2: 填充小的空洞
        filled_labels = self._fill_small_holes(cleaned_labels, neighbors, verbose)
        
        # 步骤3: 边界平滑
        smoothed_labels = self._smooth_boundaries(filled_labels, neighbors, verbose)
        
        processing_time = time.time() - start_time
        
        if verbose:
            print(f"形态学后处理完成，用时: {processing_time:.3f}秒")
            self._print_statistics(labels, smoothed_labels)
            
        return smoothed_labels, processing_time
    
    def _build_spatial_graph(self, coords, verbose=True):
        """构建k-近邻空间图"""
        if verbose:
            print(f"  构建{self.k_neighbors}-邻近图...")
            
        # 使用sklearn的高效k-NN实现
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors+1, algorithm='auto')
        nbrs.fit(coords)
        
        # 获取邻居索引（排除自己）
        _, indices = nbrs.kneighbors(coords)
        neighbors = indices[:, 1:]  # 排除第一列（自己）
        
        return neighbors
    
    def _remove_small_components(self, labels, neighbors, verbose=True):
        """去除小的连通分量（噪声区域）"""
        if verbose:
            print(f"  去除小于{self.min_component_size}个点的噪声区域...")
            
        cleaned_labels = labels.copy()
        unique_labels = np.unique(labels)
        
        total_removed = 0
        
        for class_label in unique_labels:
            # 为当前类别构建连通分量
            class_mask = (labels == class_label)
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
                
            # 构建当前类别的邻接矩阵
            adj_matrix = self._build_class_adjacency_matrix(class_indices, neighbors)
            
            # 找连通分量
            n_components, component_labels = connected_components(adj_matrix, directed=False)
            
            # 计算每个连通分量的大小
            for comp_id in range(n_components):
                comp_indices = class_indices[component_labels == comp_id]
                
                if len(comp_indices) < self.min_component_size:
                    # 小连通分量：分配给最常见的邻居类别
                    new_label = self._get_majority_neighbor_label(
                        comp_indices, neighbors, labels, class_label
                    )
                    cleaned_labels[comp_indices] = new_label
                    total_removed += len(comp_indices)
        
        if verbose and total_removed > 0:
            print(f"    已去除 {total_removed} 个噪声点")
            
        return cleaned_labels
    
    def _fill_small_holes(self, labels, neighbors, verbose=True):
        """填充小的空洞"""
        if verbose:
            print(f"  填充小于{self.max_hole_size}个点的空洞...")
            
        filled_labels = labels.copy()
        unique_labels = np.unique(labels)
        
        total_filled = 0
        
        for class_label in unique_labels:
            # 找到当前类别的"反向"连通分量（空洞）
            non_class_mask = (labels != class_label)
            non_class_indices = np.where(non_class_mask)[0]
            
            if len(non_class_indices) == 0:
                continue
            
            # 构建非当前类别的邻接矩阵
            adj_matrix = self._build_class_adjacency_matrix(non_class_indices, neighbors)
            
            # 找连通分量
            n_components, component_labels = connected_components(adj_matrix, directed=False)
            
            # 检查每个连通分量是否被当前类别包围（即空洞）
            for comp_id in range(n_components):
                comp_indices = non_class_indices[component_labels == comp_id]
                
                if len(comp_indices) <= self.max_hole_size:
                    # 检查是否为空洞（被当前类别包围）
                    if self._is_surrounded_by_class(comp_indices, neighbors, labels, class_label):
                        filled_labels[comp_indices] = class_label
                        total_filled += len(comp_indices)
        
        if verbose and total_filled > 0:
            print(f"    已填充 {total_filled} 个空洞点")
            
        return filled_labels
    
    def _smooth_boundaries(self, labels, neighbors, verbose=True):
        """边界平滑处理"""
        if verbose:
            print("  边界平滑处理...")
            
        smoothed_labels = labels.copy()
        
        # 识别边界点
        boundary_mask = self._detect_boundary_points(labels, neighbors)
        boundary_indices = np.where(boundary_mask)[0]
        
        if len(boundary_indices) == 0:
            return smoothed_labels
        
        smoothed_count = 0
        
        # 对边界点进行邻域投票
        for idx in boundary_indices:
            neighbor_indices = neighbors[idx]
            neighbor_labels = labels[neighbor_indices]
            
            # 统计邻居标签
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            
            # 找到最多的邻居标签
            majority_label = unique_labels[np.argmax(counts)]
            max_count = np.max(counts)
            
            # 如果邻域一致性足够强，则更新标签
            consistency_threshold = 0.6  # 60%的邻居需要是同一标签
            if max_count / len(neighbor_labels) >= consistency_threshold:
                if smoothed_labels[idx] != majority_label:
                    smoothed_labels[idx] = majority_label
                    smoothed_count += 1
        
        if verbose and smoothed_count > 0:
            print(f"    已平滑 {smoothed_count} 个边界点")
            
        return smoothed_labels
    
    def _build_class_adjacency_matrix(self, class_indices, neighbors):
        """为特定类别构建邻接矩阵"""
        n_points = len(class_indices)
        index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(class_indices)}
        
        row_indices = []
        col_indices = []
        
        for new_idx, orig_idx in enumerate(class_indices):
            for neighbor_idx in neighbors[orig_idx]:
                if neighbor_idx in index_map:
                    neighbor_new_idx = index_map[neighbor_idx]
                    row_indices.append(new_idx)
                    col_indices.append(neighbor_new_idx)
        
        # 创建稀疏邻接矩阵
        data = np.ones(len(row_indices), dtype=np.int8)
        adj_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_points, n_points))
        
        return adj_matrix
    
    def _get_majority_neighbor_label(self, indices, neighbors, labels, exclude_label):
        """获取邻居中最常见的标签（排除指定标签）"""
        neighbor_labels = []
        
        for idx in indices:
            for neighbor_idx in neighbors[idx]:
                neighbor_label = labels[neighbor_idx]
                if neighbor_label != exclude_label:
                    neighbor_labels.append(neighbor_label)
        
        if not neighbor_labels:
            return exclude_label  # 如果没有其他邻居，保持原标签
            
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
    
    def _is_surrounded_by_class(self, indices, neighbors, labels, target_class):
        """检查一组点是否被特定类别包围"""
        surrounding_labels = []
        
        for idx in indices:
            for neighbor_idx in neighbors[idx]:
                if neighbor_idx not in indices:  # 外部邻居
                    surrounding_labels.append(labels[neighbor_idx])
        
        if not surrounding_labels:
            return False
            
        # 如果80%以上的外部邻居都是目标类别，则认为是被包围的
        target_count = np.sum(np.array(surrounding_labels) == target_class)
        return target_count / len(surrounding_labels) >= 0.8
    
    def _detect_boundary_points(self, labels, neighbors):
        """检测边界点"""
        boundary_mask = np.zeros(len(labels), dtype=bool)
        
        for i in range(len(labels)):
            current_label = labels[i]
            neighbor_labels = labels[neighbors[i]]
            
            # 如果邻居中有不同的标签，则是边界点
            if np.any(neighbor_labels != current_label):
                boundary_mask[i] = True
        
        return boundary_mask
    
    def _print_statistics(self, original_labels, processed_labels):
        """打印处理统计信息"""
        print("  处理统计:")
        
        # 标签变化统计
        changed_points = np.sum(original_labels != processed_labels)
        total_points = len(original_labels)
        change_ratio = changed_points / total_points * 100
        
        print(f"    标签改变: {changed_points:,} / {total_points:,} ({change_ratio:.2f}%)")
        
        # 类别分布变化
        original_dist = np.bincount(original_labels)
        processed_dist = np.bincount(processed_labels)
        
        print("    类别分布变化:")
        for i in range(max(len(original_dist), len(processed_dist))):
            orig_count = original_dist[i] if i < len(original_dist) else 0
            proc_count = processed_dist[i] if i < len(processed_dist) else 0
            
            if orig_count > 0 or proc_count > 0:
                change = proc_count - orig_count
                change_str = f"({change:+d})" if change != 0 else ""
                print(f"      类别 {i}: {orig_count:,} → {proc_count:,} {change_str}")


def apply_morphological_postprocess(labels, points, **kwargs):
    """
    便捷函数：应用形态学后处理
    
    Args:
        labels: 预测标签数组
        points: 点云坐标数组
        **kwargs: 传递给MorphologicalPostProcessor的参数
        
    Returns:
        processed_labels: 处理后的标签
        processing_time: 处理时间
    """
    # 分离构造函数参数和process方法参数
    constructor_params = {}
    process_params = {}
    
    # 构造函数参数
    if 'k_neighbors' in kwargs:
        constructor_params['k_neighbors'] = kwargs['k_neighbors']
    if 'min_component_size' in kwargs:
        constructor_params['min_component_size'] = kwargs['min_component_size']
    if 'max_hole_size' in kwargs:
        constructor_params['max_hole_size'] = kwargs['max_hole_size']
    
    # process方法参数
    if 'verbose' in kwargs:
        process_params['verbose'] = kwargs['verbose']
    
    processor = MorphologicalPostProcessor(**constructor_params)
    return processor.process(labels, points, **process_params)


# 测试函数
def test_morphological_postprocess():
    """测试形态学后处理功能"""
    print("测试形态学后处理...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_points = 5000
    
    # 模拟点云
    points = np.random.randn(n_points, 3).astype(np.float32)
    
    # 模拟有噪声的标签
    labels = np.random.randint(0, 3, n_points)
    
    # 添加一些小噪声区域
    noise_indices = np.random.choice(n_points, 50, replace=False)
    labels[noise_indices] = (labels[noise_indices] + 1) % 3
    
    print(f"原始标签分布: {np.bincount(labels)}")
    
    # 应用后处理
    processed_labels, proc_time = apply_morphological_postprocess(
        labels, points,
        k_neighbors=8,
        min_component_size=5,
        max_hole_size=3
    )
    
    print(f"处理后标签分布: {np.bincount(processed_labels)}")
    print(f"处理时间: {proc_time:.3f}秒")
    
    return processed_labels, proc_time


if __name__ == "__main__":
    test_morphological_postprocess() 