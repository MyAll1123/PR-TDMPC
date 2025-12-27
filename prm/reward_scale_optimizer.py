#!/usr/bin/env python3
"""
奖励尺度优化系统 - 集成改进的尺度处理和损失函数
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import numpy as np
from collections import deque

class RewardScaleOptimizer(nn.Module):
    """
    奖励尺度优化器 - 统一管理不同类型奖励的尺度问题
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 获取配置参数
        self.scale_percentile = getattr(cfg, 'scale_percentile', 95)
        self.scale_min = getattr(cfg, 'scale_min', 1e-6)
        self.scale_max = getattr(cfg, 'scale_max', 1e6)
        self.tau = getattr(cfg, 'tau', 0.005)
        
        # 偏好奖励相关配置
        self.pref_reward_scale = getattr(cfg, 'pref_reward_scale', 1.0)
        self.adaptive_scaling = getattr(cfg, 'adaptive_reward_scaling', True)
        
        # 初始化尺度估计器
        self._init_scale_estimators()
        
        # 损失函数配置
        self.loss_type = getattr(cfg, 'preference_loss_type', 'adaptive_logistic')
        self.loss_temperature = getattr(cfg, 'preference_loss_temperature', 1.0)
        self.loss_margin = getattr(cfg, 'preference_loss_margin', 1.0)
        
        # 统计信息
        self.stats = {
            'env_reward_scales': deque(maxlen=1000),
            'pref_reward_scales': deque(maxlen=1000),
            'q_value_scales': deque(maxlen=1000),
            'loss_values': deque(maxlen=1000)
        }
    
    def _init_scale_estimators(self):
        """初始化尺度估计器"""
        # 环境奖励尺度估计器
        self.register_buffer('_env_reward_scale', torch.tensor(1.0))
        self.register_buffer('_env_reward_variance', torch.tensor(1.0))
        self.register_buffer('_env_update_count', torch.tensor(0))
        
        # 偏好奖励尺度估计器
        self.register_buffer('_pref_reward_scale', torch.tensor(1.0))
        self.register_buffer('_pref_reward_variance', torch.tensor(1.0))
        self.register_buffer('_pref_update_count', torch.tensor(0))
        
        # Q值尺度估计器
        self.register_buffer('_q_value_scale', torch.tensor(1.0))
        self.register_buffer('_q_variance', torch.tensor(1.0))
        self.register_buffer('_q_update_count', torch.tensor(0))
        
        # 自适应权重
        self.register_buffer('_adaptive_env_weight', torch.tensor(1.0))
        self.register_buffer('_adaptive_pref_weight', torch.tensor(1.0))
    
    def _compute_robust_percentile(self, x: torch.Tensor, percentile: float) -> torch.Tensor:
        """计算鲁棒的百分位数"""
        if x.numel() == 0:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        
        # 展平并移除NaN/Inf
        flat_x = x.flatten()
        valid_mask = torch.isfinite(flat_x)
        if not valid_mask.any():
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        
        flat_x = flat_x[valid_mask]
        
        # 移除极端异常值（超过5个标准差）
        if len(flat_x) > 10:
            mean_val = flat_x.mean()
            std_val = flat_x.std()
            if std_val > 0:
                mask = torch.abs(flat_x - mean_val) <= 5 * std_val
                if mask.sum() > 0:
                    flat_x = flat_x[mask]
        
        # 计算百分位数
        k = int((percentile / 100.0) * len(flat_x))
        k = max(0, min(k, len(flat_x) - 1))
        
        sorted_x, _ = torch.sort(torch.abs(flat_x))
        # 确保clamp的边界值在正确的设备上
        scale_min = torch.tensor(self.scale_min, device=x.device, dtype=x.dtype)
        scale_max = torch.tensor(self.scale_max, device=x.device, dtype=x.dtype)
        return torch.clamp(sorted_x[k], scale_min, scale_max)
    
    def _update_scale_estimator(self, current_scale: torch.Tensor, 
                               scale_buffer: torch.Tensor, 
                               variance_buffer: torch.Tensor,
                               count_buffer: torch.Tensor) -> torch.Tensor:
        """更新尺度估计器"""
        if count_buffer == 0:
            scale_buffer.data.copy_(current_scale)
            variance_buffer.data.copy_(torch.tensor(0.1, device=current_scale.device, dtype=current_scale.dtype))
        else:
            # 自适应学习率
            adaptive_tau = self.tau
            if count_buffer > 100:
                # 基于方差调整学习率
                scale_ratio = current_scale / (scale_buffer + 1e-8)
                if scale_ratio > 2.0 or scale_ratio < 0.5:
                    adaptive_tau = min(self.tau * 2.0, 0.1)
                elif variance_buffer > scale_buffer:
                    adaptive_tau = self.tau * 0.5
            
            # 指数移动平均更新 - 确保数据类型和设备匹配
            adaptive_tau = float(adaptive_tau)
            current_scale = current_scale.to(dtype=scale_buffer.dtype, device=scale_buffer.device)
            scale_buffer.data.lerp_(current_scale, adaptive_tau)
            
            # 更新方差估计
            diff = current_scale - scale_buffer
            diff_squared = diff.pow(2).to(dtype=variance_buffer.dtype, device=variance_buffer.device)
            variance_buffer.data.lerp_(diff_squared, adaptive_tau)
        
        count_buffer.data.add_(1)
        return scale_buffer
    
    def update_env_reward_scale(self, env_rewards: torch.Tensor):
        """更新环境奖励尺度"""
        if env_rewards.numel() == 0:
            return
        
        current_scale = self._compute_robust_percentile(env_rewards, self.scale_percentile)
        self._update_scale_estimator(
            current_scale, self._env_reward_scale, 
            self._env_reward_variance, self._env_update_count
        )
        
        # 记录统计信息
        self.stats['env_reward_scales'].append(current_scale.item())
    
    def update_pref_reward_scale(self, pref_rewards: torch.Tensor):
        """更新偏好奖励尺度"""
        if pref_rewards.numel() == 0:
            return
        
        current_scale = self._compute_robust_percentile(pref_rewards, self.scale_percentile)
        self._update_scale_estimator(
            current_scale, self._pref_reward_scale,
            self._pref_reward_variance, self._pref_update_count
        )
        
        # 记录统计信息
        self.stats['pref_reward_scales'].append(current_scale.item())
    
    def update_q_value_scale(self, q_values: torch.Tensor):
        """更新Q值尺度"""
        if q_values.numel() == 0:
            return
        
        current_scale = self._compute_robust_percentile(q_values, self.scale_percentile)
        self._update_scale_estimator(
            current_scale, self._q_value_scale,
            self._q_variance, self._q_update_count
        )
        
        # 记录统计信息
        self.stats['q_value_scales'].append(current_scale.item())
    
    def normalize_rewards(self, env_rewards: torch.Tensor, 
                         pref_rewards: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """归一化奖励，支持负数奖励"""
        # 更新尺度
        self.update_env_reward_scale(env_rewards)
        
        # 对于环境奖励，保持符号但按尺度归一化
        # 使用符号保持的归一化：sign(x) * |x| / scale
        env_scale = self._env_reward_scale + 1e-8
        normalized_env = env_rewards / env_scale
        
        # 对于极小的奖励值，避免过度放大
        abs_env_rewards = torch.abs(env_rewards)
        small_reward_mask = abs_env_rewards < (env_scale * 0.01)  # 小于尺度1%的奖励
        if small_reward_mask.any():
            # 对小奖励使用更温和的归一化
            normalized_env[small_reward_mask] = env_rewards[small_reward_mask] / (env_scale * 0.1)
        
        normalized_pref = None
        if pref_rewards is not None:
            self.update_pref_reward_scale(pref_rewards)
            pref_scale = self._pref_reward_scale + 1e-8
            
            if self.adaptive_scaling:
                # 自适应权重调整
                scale_ratio = pref_scale / env_scale
                
                if scale_ratio > 10.0:  # 偏好奖励尺度过大
                    self._adaptive_pref_weight = torch.clamp(
                        self._adaptive_pref_weight * 0.95, 0.1, 2.0
                    )
                elif scale_ratio < 0.1:  # 偏好奖励尺度过小
                    self._adaptive_pref_weight = torch.clamp(
                        self._adaptive_pref_weight * 1.05, 0.1, 2.0
                    )
                
                # 应用自适应权重
                normalized_pref = (pref_rewards / pref_scale) * self._adaptive_pref_weight
            else:
                normalized_pref = pref_rewards / pref_scale * self.pref_reward_scale
            
            # 对偏好奖励也应用小值保护
            abs_pref_rewards = torch.abs(pref_rewards)
            small_pref_mask = abs_pref_rewards < (pref_scale * 0.01)
            if small_pref_mask.any():
                normalized_pref[small_pref_mask] = pref_rewards[small_pref_mask] / (pref_scale * 0.1)
        
        return normalized_env, normalized_pref
    
    def compute_preference_loss(self, chosen_q: torch.Tensor, 
                               rejected_q: torch.Tensor) -> torch.Tensor:
        """计算改进的偏好损失"""
        # 更新Q值尺度
        all_q = torch.cat([chosen_q.flatten(), rejected_q.flatten()])
        self.update_q_value_scale(all_q)
        
        # 根据配置选择损失函数
        if self.loss_type == 'relative_hinge':
            loss = self._relative_hinge_loss(chosen_q, rejected_q)
        elif self.loss_type == 'normalized_hinge':
            loss = self._normalized_hinge_loss(chosen_q, rejected_q)
        elif self.loss_type == 'adaptive_logistic':
            loss = self._adaptive_logistic_loss(chosen_q, rejected_q)
        elif self.loss_type == 'bradley_terry':
            loss = self._bradley_terry_loss(chosen_q, rejected_q)
        else:
            # 默认使用自适应logistic损失
            loss = self._adaptive_logistic_loss(chosen_q, rejected_q)
        
        # 记录损失值
        self.stats['loss_values'].append(loss.item())
        
        return loss
    
    def _relative_hinge_loss(self, chosen_q: torch.Tensor, rejected_q: torch.Tensor) -> torch.Tensor:
        """相对hinge损失"""
        q_scale = self._q_value_scale
        relative_margin = self.loss_margin * q_scale
        
        diff = (chosen_q - rejected_q) / self.loss_temperature
        loss = torch.clamp(relative_margin - diff, min=0.0)
        
        return loss.mean()
    
    def _normalized_hinge_loss(self, chosen_q: torch.Tensor, rejected_q: torch.Tensor) -> torch.Tensor:
        """归一化hinge损失"""
        # 使用当前Q值尺度进行归一化
        norm_chosen = chosen_q / (self._q_value_scale + 1e-8)
        norm_rejected = rejected_q / (self._q_value_scale + 1e-8)
        
        diff = norm_chosen - norm_rejected
        loss = torch.clamp(self.loss_margin - diff, min=0.0)
        
        return loss.mean()
    
    def _adaptive_logistic_loss(self, chosen_q: torch.Tensor, rejected_q: torch.Tensor) -> torch.Tensor:
        """自适应logistic损失"""
        # 自适应温度
        q_scale = self._q_value_scale
        adaptive_temp = self.loss_temperature * torch.log(q_scale + 1.0)
        adaptive_temp = torch.clamp(adaptive_temp, min=0.1, max=10.0)
        
        diff = (chosen_q - rejected_q) / adaptive_temp
        loss = torch.nn.functional.softplus(-diff)
        
        return loss.mean()
    
    def _bradley_terry_loss(self, chosen_q: torch.Tensor, rejected_q: torch.Tensor) -> torch.Tensor:
        """Bradley-Terry损失"""
        diff = chosen_q - rejected_q
        
        # 自适应温度
        q_scale = self._q_value_scale
        adaptive_temp = self.loss_temperature * torch.sqrt(q_scale)
        adaptive_temp = torch.clamp(adaptive_temp, min=0.1, max=5.0)
        
        logits = diff / adaptive_temp
        loss = -torch.nn.functional.logsigmoid(logits)
        
        return loss.mean()
    
    def get_scale_info(self) -> Dict[str, Any]:
        """获取尺度信息"""
        return {
            'env_reward_scale': self._env_reward_scale.item(),
            'pref_reward_scale': self._pref_reward_scale.item(),
            'q_value_scale': self._q_value_scale.item(),
            'adaptive_env_weight': self._adaptive_env_weight.item(),
            'adaptive_pref_weight': self._adaptive_pref_weight.item(),
            'env_reward_variance': self._env_reward_variance.item(),
            'pref_reward_variance': self._pref_reward_variance.item(),
            'q_variance': self._q_variance.item(),
            'scale_ratio': (self._pref_reward_scale / (self._env_reward_scale + 1e-8)).item(),
            'adaptive_scaling': self.adaptive_scaling,
            'loss_type': self.loss_type
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        stats = self.get_scale_info()
        
        # 添加历史统计
        for key, values in self.stats.items():
            if values:
                values_tensor = torch.tensor(list(values))
                stats[f'{key}_mean'] = values_tensor.mean().item()
                stats[f'{key}_std'] = values_tensor.std().item()
                stats[f'{key}_min'] = values_tensor.min().item()
                stats[f'{key}_max'] = values_tensor.max().item()
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key].clear()
    
    def save_state(self) -> Dict[str, Any]:
        """保存状态"""
        return {
            'env_reward_scale': self._env_reward_scale.clone(),
            'pref_reward_scale': self._pref_reward_scale.clone(),
            'q_value_scale': self._q_value_scale.clone(),
            'adaptive_env_weight': self._adaptive_env_weight.clone(),
            'adaptive_pref_weight': self._adaptive_pref_weight.clone(),
            'env_reward_variance': self._env_reward_variance.clone(),
            'pref_reward_variance': self._pref_reward_variance.clone(),
            'q_variance': self._q_variance.clone(),
            'env_update_count': self._env_update_count.clone(),
            'pref_update_count': self._pref_update_count.clone(),
            'q_update_count': self._q_update_count.clone()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """加载状态"""
        for key, value in state.items():
            if hasattr(self, f'_{key}'):
                getattr(self, f'_{key}').data.copy_(value)


def create_reward_scale_optimizer(cfg) -> RewardScaleOptimizer:
    """创建奖励尺度优化器的工厂函数"""
    return RewardScaleOptimizer(cfg)