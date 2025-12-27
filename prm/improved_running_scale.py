#!/usr/bin/env python3
"""
改进的RunningScale实现 - 解决不同奖励尺度下的数值稳定性问题
"""

import torch
import torch.nn as nn
from typing import Optional

class ImprovedRunningScale(nn.Module):
    """
    改进的运行尺度估计器，具有更好的数值稳定性和适应性
    
    主要改进：
    1. 使用更稳定的百分位数估计
    2. 自适应学习率调整
    3. 异常值检测和处理
    4. 多尺度适应机制
    """
    
    def __init__(self, cfg, percentile: float = 95.0, min_scale: float = 1e-6, 
                 max_scale: float = 1e6, adaptive_tau: bool = True):
        super().__init__()
        self.cfg = cfg
        self.percentile = percentile
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.adaptive_tau = adaptive_tau
        
        # 基础学习率
        self.base_tau = getattr(cfg, 'tau', 0.005)
        
        # 注册缓冲区
        self.register_buffer('_value', torch.tensor(1.0))
        self.register_buffer('_momentum', torch.tensor(0.0))
        self.register_buffer('_variance', torch.tensor(1.0))
        self.register_buffer('_update_count', torch.tensor(0))
        
        # 历史统计
        self.register_buffer('_recent_values', torch.zeros(100))  # 保存最近100个值
        self.register_buffer('_buffer_idx', torch.tensor(0))
        
    def _compute_percentile(self, x: torch.Tensor) -> torch.Tensor:
        """计算稳定的百分位数"""
        # 展平张量
        flat_x = x.flatten()
        
        # 移除异常值（超过3个标准差的值）
        if len(flat_x) > 10:
            mean_val = flat_x.mean()
            std_val = flat_x.std()
            mask = torch.abs(flat_x - mean_val) <= 3 * std_val
            if mask.sum() > 0:
                flat_x = flat_x[mask]
        
        # 计算百分位数
        if len(flat_x) == 0:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
            
        k = int((self.percentile / 100.0) * len(flat_x))
        k = max(0, min(k, len(flat_x) - 1))
        
        sorted_x, _ = torch.sort(flat_x)
        return sorted_x[k]
    
    def _adaptive_learning_rate(self, current_scale: torch.Tensor) -> float:
        """自适应学习率调整"""
        if not self.adaptive_tau:
            return self.base_tau
            
        # 基于当前尺度和历史方差调整学习率
        scale_ratio = current_scale / (self._value + 1e-8)
        
        # 如果尺度变化很大，增加学习率
        if scale_ratio > 2.0 or scale_ratio < 0.5:
            return min(self.base_tau * 2.0, 0.1)
        
        # 如果方差很大，减少学习率
        if self._variance > self._value:
            return self.base_tau * 0.5
            
        return self.base_tau
    
    def _update_statistics(self, new_scale: torch.Tensor):
        """更新统计信息"""
        # 更新历史值缓冲区
        idx = self._buffer_idx % 100
        self._recent_values[idx] = new_scale
        self._buffer_idx += 1
        
        # 计算方差（使用最近的值）
        if self._buffer_idx >= 10:
            recent_count = min(self._buffer_idx, 100)
            recent_vals = self._recent_values[:recent_count]
            self._variance = recent_vals.var()
    
    def update(self, x: torch.Tensor):
        """更新尺度估计，支持负数奖励"""
        if x.numel() == 0:
            return
            
        # 对于可能包含负数的奖励，使用绝对值计算尺度
        # 但保留原始值的符号信息用于后续处理
        abs_x = torch.abs(x)
        
        # 过滤掉过小的值以避免噪声
        valid_mask = abs_x > 1e-8
        if valid_mask.sum() == 0:
            # 如果所有值都太小，使用最小尺度
            return
        
        valid_abs_x = abs_x[valid_mask]
        
        # 计算当前尺度
        current_scale = self._compute_percentile(valid_abs_x)
        current_scale = torch.clamp(current_scale, self.min_scale, self.max_scale)
        
        # 自适应学习率
        tau = self._adaptive_learning_rate(current_scale)
        
        # 更新尺度值（指数移动平均）
        if self._update_count == 0:
            self._value.data.copy_(current_scale)
        else:
            # 确保数据类型匹配
            current_scale = current_scale.to(dtype=self._value.dtype)
            self._value.data.lerp_(current_scale, float(tau))
            
        # 更新统计信息
        self._update_statistics(current_scale)
        self._update_count += 1
    
    def __call__(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """应用尺度归一化"""
        if update:
            self.update(x)
        
        # 确保尺度值在合理范围内
        scale_value = torch.clamp(self._value, self.min_scale, self.max_scale)
        
        # 应用归一化
        return x / (scale_value + 1e-8)
    
    @property
    def scale(self) -> torch.Tensor:
        """获取当前尺度值"""
        return self._value
    
    def reset(self):
        """重置尺度估计器"""
        self._value.data.fill_(1.0)
        self._momentum.data.fill_(0.0)
        self._variance.data.fill_(1.0)
        self._update_count.data.fill_(0)
        self._recent_values.data.fill_(0.0)
        self._buffer_idx.data.fill_(0)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'scale': self._value.item(),
            'variance': self._variance.item(),
            'update_count': self._update_count.item(),
            'recent_mean': self._recent_values[:min(self._buffer_idx, 100)].mean().item(),
            'effective_scale_range': f'[{self.min_scale:.2e}, {self.max_scale:.2e}]',
            'percentile': self.percentile,
            'adaptive_tau': self.adaptive_tau
        }


class AdaptiveRewardNormalizer(nn.Module):
    """
    自适应奖励归一化器 - 根据环境奖励尺度动态调整偏好奖励
    """
    
    def __init__(self, cfg, env_scale_percentile: float = 90.0, 
                 pref_scale_percentile: float = 95.0):
        super().__init__()
        self.cfg = cfg
        
        # 环境奖励和偏好奖励的尺度估计器
        self.env_scale = ImprovedRunningScale(cfg, env_scale_percentile)
        self.pref_scale = ImprovedRunningScale(cfg, pref_scale_percentile)
        
        # 自适应权重
        self.register_buffer('_env_weight', torch.tensor(1.0))
        self.register_buffer('_pref_weight', torch.tensor(1.0))
        
    def normalize_rewards(self, env_reward: torch.Tensor, 
                         pref_reward: torch.Tensor) -> tuple:
        """归一化环境奖励和偏好奖励"""
        # 更新尺度估计
        norm_env = self.env_scale(env_reward, update=True)
        norm_pref = self.pref_scale(pref_reward, update=True)
        
        # 自适应权重调整
        env_scale_val = self.env_scale.scale
        pref_scale_val = self.pref_scale.scale
        
        # 根据尺度比例调整权重
        scale_ratio = pref_scale_val / (env_scale_val + 1e-8)
        
        if scale_ratio > 10.0:  # 偏好奖励尺度过大
            self._pref_weight = torch.clamp(self._pref_weight * 0.9, 0.1, 2.0)
        elif scale_ratio < 0.1:  # 偏好奖励尺度过小
            self._pref_weight = torch.clamp(self._pref_weight * 1.1, 0.1, 2.0)
            
        return norm_env, norm_pref * self._pref_weight
    
    def get_stats(self) -> dict:
        """获取归一化器统计信息"""
        return {
            'env_scale_stats': self.env_scale.get_stats(),
            'pref_scale_stats': self.pref_scale.get_stats(),
            'env_weight': self._env_weight.item(),
            'pref_weight': self._pref_weight.item()
        }


# 测试代码已移除以保持生产代码简洁