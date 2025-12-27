#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变长序列处理工具模块

提供支持变长序列的工具函数，包括：
1. 掩码生成
2. 变长序列collate函数
3. 掩码感知的池化操作
4. 序列长度统计和分析
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union


def create_padding_mask(lengths: List[int], max_length: Optional[int] = None, device: str = 'cpu') -> torch.Tensor:
    """
    创建填充掩码
    
    Args:
        lengths: 每个序列的实际长度列表
        max_length: 最大长度，如果为None则使用lengths中的最大值
        device: 设备
        
    Returns:
        mask: [batch_size, max_length] 的布尔掩码，True表示有效位置，False表示填充位置
    """
    if max_length is None:
        max_length = max(lengths)
    
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


def create_attention_mask(lengths: List[int], max_length: Optional[int] = None, device: str = 'cpu') -> torch.Tensor:
    """
    创建注意力掩码（用于Transformer）
    
    Args:
        lengths: 每个序列的实际长度列表
        max_length: 最大长度
        device: 设备
        
    Returns:
        mask: [batch_size, max_length, max_length] 的注意力掩码
    """
    if max_length is None:
        max_length = max(lengths)
    
    batch_size = len(lengths)
    # 创建基础掩码
    padding_mask = create_padding_mask(lengths, max_length, device)
    
    # 扩展为注意力掩码 [B, L, L]
    attention_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
    
    return attention_mask


def to_float_array(x):
    """将输入转换为float32数组"""
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif hasattr(x, 'numpy'):
        try:
            # 安全地将tensor转换为CPU，避免CUDA初始化问题
            if hasattr(x, 'is_cuda') and x.is_cuda:
                return x.detach().cpu().numpy().astype(np.float32)
            else:
                return x.detach().numpy().astype(np.float32)
        except Exception as e:
            # 如果tensor操作失败，尝试直接转换
            return np.array(x, dtype=np.float32)
    else:
        return np.array(x, dtype=np.float32)


def pad_sequences_to_max(sequences: List[np.ndarray], max_length: Optional[int] = None) -> Tuple[torch.Tensor, List[int]]:
    """
    将序列列表填充到最大长度
    
    Args:
        sequences: 序列列表
        max_length: 最大长度，如果为None则使用序列中的最大长度
        
    Returns:
        padded_tensor: 填充后的张量 [batch_size, max_length, feature_dim]
        lengths: 原始长度列表
    """
    # 转换为float32数组
    sequences = [to_float_array(seq) for seq in sequences]
    lengths = [len(seq) for seq in sequences]
    
    if max_length is None:
        max_length = max(lengths)
    
    # 获取特征维度
    feature_dim = sequences[0].shape[-1] if len(sequences[0].shape) > 1 else 1
    
    # 创建填充后的张量
    batch_size = len(sequences)
    if len(sequences[0].shape) > 1:
        padded_tensor = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    else:
        padded_tensor = torch.zeros(batch_size, max_length, dtype=torch.float32)
    
    # 填充数据
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length)
        if len(seq.shape) > 1:
            padded_tensor[i, :seq_len] = torch.from_numpy(seq[:seq_len])
        else:
            padded_tensor[i, :seq_len] = torch.from_numpy(seq[:seq_len])
    
    return padded_tensor, lengths


def variable_length_collate_fn(batch: List[Tuple]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    变长序列的批处理整理函数
    
    Args:
        batch: 批次数据，每个元素为 (trajectory_a, trajectory_b) 或 (s_a, a_a, s_b, a_b)
        
    Returns:
        (s_a, a_a): 轨迹A的状态和动作张量
        (s_b, a_b): 轨迹B的状态和动作张量
        mask_a: 轨迹A的填充掩码
        mask_b: 轨迹B的填充掩码
    """
    s_a, a_a, s_b, a_b = [], [], [], []
    
    # 解析批次数据
    for item in batch:
        if isinstance(item, (tuple, list)) and len(item) == 2 \
           and all(isinstance(x, (tuple, list)) and len(x) >= 2 for x in item):
            # 格式: ((s_a, a_a), (s_b, a_b))
            s_a.append(item[0][0])
            a_a.append(item[0][1])
            s_b.append(item[1][0])
            a_b.append(item[1][1])
        elif isinstance(item, (tuple, list)) and len(item) >= 4:
            # 格式: (s_a, a_a, s_b, a_b)
            s_a.append(item[0])
            a_a.append(item[1])
            s_b.append(item[2])
            a_b.append(item[3])
        else:
            raise ValueError(f"不支持的批次数据格式: {type(item)}")
    
    # 提取obs字段（如果是dict）
    s_a = [x["obs"] if isinstance(x, dict) else x for x in s_a]
    s_b = [x["obs"] if isinstance(x, dict) else x for x in s_b]
    
    # 填充序列到最大长度
    s_a_padded, lengths_a = pad_sequences_to_max(s_a)
    a_a_padded, _ = pad_sequences_to_max(a_a)
    s_b_padded, lengths_b = pad_sequences_to_max(s_b)
    a_b_padded, _ = pad_sequences_to_max(a_b)
    
    # 创建填充掩码
    max_len_a = s_a_padded.shape[1]
    max_len_b = s_b_padded.shape[1]
    
    mask_a = create_padding_mask(lengths_a, max_len_a)
    mask_b = create_padding_mask(lengths_b, max_len_b)
    
    return (s_a_padded, a_a_padded), (s_b_padded, a_b_padded), mask_a, mask_b


def masked_mean_pooling(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    掩码感知的平均池化
    
    Args:
        x: 输入张量 [batch_size, seq_len, hidden_dim]
        mask: 填充掩码 [batch_size, seq_len]
        dim: 池化维度
        
    Returns:
        pooled: 池化后的张量 [batch_size, hidden_dim]
    """
    # 将填充位置设为0
    masked_x = x * mask.unsqueeze(-1).float()
    
    # 计算有效长度
    lengths = mask.sum(dim=dim, keepdim=True).float()
    lengths = torch.clamp(lengths, min=1.0)  # 避免除零
    
    # 计算平均值
    pooled = masked_x.sum(dim=dim) / lengths
    
    return pooled


def masked_max_pooling(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    掩码感知的最大池化
    
    Args:
        x: 输入张量 [batch_size, seq_len, hidden_dim]
        mask: 填充掩码 [batch_size, seq_len]
        dim: 池化维度
        
    Returns:
        pooled: 池化后的张量 [batch_size, hidden_dim]
    """
    # 将填充位置设为很小的负数
    masked_x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    
    # 最大池化
    pooled, _ = masked_x.max(dim=dim)
    
    return pooled


def analyze_sequence_lengths(lengths: List[int]) -> Dict[str, float]:
    """
    分析序列长度分布
    
    Args:
        lengths: 序列长度列表
        
    Returns:
        stats: 统计信息字典
    """
    lengths = np.array(lengths)
    
    stats = {
        'count': len(lengths),
        'mean': float(np.mean(lengths)),
        'std': float(np.std(lengths)),
        'min': float(np.min(lengths)),
        'max': float(np.max(lengths)),
        'median': float(np.median(lengths)),
        'q25': float(np.percentile(lengths, 25)),
        'q75': float(np.percentile(lengths, 75)),
    }
    
    return stats


def get_optimal_max_length(lengths: List[int], coverage: float = 0.95) -> int:
    """
    根据序列长度分布获取最优的最大长度
    
    Args:
        lengths: 序列长度列表
        coverage: 覆盖率（例如0.95表示覆盖95%的序列）
        
    Returns:
        optimal_max_length: 最优最大长度
    """
    lengths = np.array(lengths)
    percentile = coverage * 100
    optimal_length = int(np.percentile(lengths, percentile))
    
    return optimal_length


def create_length_based_weights(lengths: List[int], method: str = 'inverse') -> torch.Tensor:
    """
    基于序列长度创建权重
    
    Args:
        lengths: 序列长度列表
        method: 权重计算方法 ('inverse', 'sqrt_inverse', 'log')
        
    Returns:
        weights: 权重张量
    """
    lengths = np.array(lengths, dtype=np.float32)
    
    if method == 'inverse':
        weights = 1.0 / lengths
    elif method == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(lengths)
    elif method == 'log':
        weights = 1.0 / np.log(lengths + 1)
    else:
        raise ValueError(f"不支持的权重计算方法: {method}")
    
    # 归一化权重
    weights = weights / np.sum(weights) * len(weights)
    
    return torch.from_numpy(weights)


class VariableLengthBatchSampler:
    """
    变长序列的批次采样器
    根据序列长度进行分组，减少填充开销
    """
    
    def __init__(self, lengths: List[int], batch_size: int, drop_last: bool = False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # 根据长度排序索引
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
    def __iter__(self):
        # 按长度分组创建批次
        batches = []
        current_batch = []
        
        for idx in self.sorted_indices:
            current_batch.append(idx)
            
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # 处理最后一个不完整的批次
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        # 随机打乱批次顺序
        np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    # 测试代码
    print("测试变长序列处理工具...")
    
    # 创建测试数据
    sequences_a = [
        np.random.randn(50, 10),   # 长度50
        np.random.randn(30, 10),   # 长度30
        np.random.randn(80, 10),   # 长度80
        np.random.randn(20, 10),   # 长度20
    ]
    
    sequences_b = [
        np.random.randn(45, 10),
        np.random.randn(35, 10),
        np.random.randn(75, 10),
        np.random.randn(25, 10),
    ]
    
    # 测试填充和掩码
    padded_a, lengths_a = pad_sequences_to_max(sequences_a)
    mask_a = create_padding_mask(lengths_a)
    
    print(f"填充后张量形状: {padded_a.shape}")
    print(f"原始长度: {lengths_a}")
    print(f"掩码形状: {mask_a.shape}")
    
    # 测试掩码池化
    pooled_mean = masked_mean_pooling(padded_a, mask_a)
    pooled_max = masked_max_pooling(padded_a, mask_a)
    
    print(f"平均池化结果形状: {pooled_mean.shape}")
    print(f"最大池化结果形状: {pooled_max.shape}")
    
    # 测试序列长度分析
    stats = analyze_sequence_lengths(lengths_a)
    print(f"序列长度统计: {stats}")
    
    optimal_length = get_optimal_max_length(lengths_a, coverage=0.9)
    print(f"最优最大长度 (90%覆盖率): {optimal_length}")
    
    print("变长序列处理工具测试完成！")