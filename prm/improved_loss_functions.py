#!/usr/bin/env python3
"""
改进的损失函数实现 - 解决不同Q值尺度下的数值稳定性问题
"""

import torch
import torch.nn.functional as F
from typing import Optional

class ImprovedPreferenceLoss:
    """
    改进的偏好损失函数集合，具有更好的数值稳定性
    """
    
    @staticmethod
    def relative_hinge_loss(chosen_q: torch.Tensor, rejected_q: torch.Tensor, 
                           margin: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
        """
        相对hinge损失 - 使用相对差异而非绝对差异
        
        Args:
            chosen_q: 选择动作的Q值
            rejected_q: 拒绝动作的Q值
            margin: 相对边际（相对于Q值尺度）
            temperature: 温度参数
        """
        # 计算Q值的尺度
        q_scale = torch.max(torch.abs(chosen_q).mean(), torch.abs(rejected_q).mean())
        q_scale = torch.clamp(q_scale, min=1e-6)
        
        # 相对边际
        relative_margin = margin * q_scale
        
        # 计算损失
        diff = (chosen_q - rejected_q) / temperature
        loss = torch.clamp(relative_margin - diff, min=0.0)
        
        return loss.mean()
    
    @staticmethod
    def normalized_hinge_loss(chosen_q: torch.Tensor, rejected_q: torch.Tensor,
                             margin: float = 0.1) -> torch.Tensor:
        """
        归一化hinge损失 - 先归一化Q值再计算损失
        """
        # 归一化Q值
        all_q = torch.cat([chosen_q.flatten(), rejected_q.flatten()])
        q_mean = all_q.mean()
        q_std = all_q.std() + 1e-8
        
        norm_chosen = (chosen_q - q_mean) / q_std
        norm_rejected = (rejected_q - q_mean) / q_std
        
        # 计算损失
        diff = norm_chosen - norm_rejected
        loss = torch.clamp(margin - diff, min=0.0)
        
        return loss.mean()
    
    @staticmethod
    def adaptive_logistic_loss(chosen_q: torch.Tensor, rejected_q: torch.Tensor,
                              temperature: float = 1.0, adaptive_temp: bool = True) -> torch.Tensor:
        """
        自适应logistic损失 - 根据Q值尺度调整温度
        """
        if adaptive_temp:
            # 自适应温度调整
            q_scale = torch.max(torch.abs(chosen_q).mean(), torch.abs(rejected_q).mean())
            q_scale = torch.clamp(q_scale, min=1e-6, max=1e6)
            
            # 温度与Q值尺度成正比
            adaptive_temperature = temperature * torch.log(q_scale + 1.0)
            adaptive_temperature = torch.clamp(adaptive_temperature, min=0.1, max=10.0)
        else:
            adaptive_temperature = temperature
        
        # 计算logistic损失
        diff = (chosen_q - rejected_q) / adaptive_temperature
        
        # 使用数值稳定的sigmoid
        loss = F.softplus(-diff)
        
        return loss.mean()
    
    @staticmethod
    def robust_bradley_terry_loss(chosen_q: torch.Tensor, rejected_q: torch.Tensor,
                                 temperature: float = 1.0, label_smoothing: float = 0.0) -> torch.Tensor:
        """
        鲁棒的Bradley-Terry损失，支持标签平滑
        """
        # 计算Q值差异
        diff = chosen_q - rejected_q
        
        # 自适应温度
        q_scale = torch.max(torch.abs(chosen_q).mean(), torch.abs(rejected_q).mean())
        q_scale = torch.clamp(q_scale, min=1e-6)
        adaptive_temp = temperature * torch.sqrt(q_scale)
        adaptive_temp = torch.clamp(adaptive_temp, min=0.1, max=5.0)
        
        # 计算概率
        logits = diff / adaptive_temp
        
        # 数值稳定的log-sigmoid
        log_prob = F.logsigmoid(logits)
        
        # 标签平滑
        if label_smoothing > 0:
            smooth_loss = -log_prob * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
        else:
            smooth_loss = -log_prob
        
        return smooth_loss.mean()
    
    @staticmethod
    def multi_scale_loss(chosen_q: torch.Tensor, rejected_q: torch.Tensor,
                        loss_weights: Optional[dict] = None) -> torch.Tensor:
        """
        多尺度损失 - 结合多种损失函数
        """
        if loss_weights is None:
            loss_weights = {
                'hinge': 0.3,
                'logistic': 0.4,
                'bradley_terry': 0.3
            }
        
        losses = {}
        
        # 相对hinge损失
        if 'hinge' in loss_weights:
            losses['hinge'] = ImprovedPreferenceLoss.relative_hinge_loss(chosen_q, rejected_q)
        
        # 自适应logistic损失
        if 'logistic' in loss_weights:
            losses['logistic'] = ImprovedPreferenceLoss.adaptive_logistic_loss(chosen_q, rejected_q)
        
        # Bradley-Terry损失
        if 'bradley_terry' in loss_weights:
            losses['bradley_terry'] = ImprovedPreferenceLoss.robust_bradley_terry_loss(chosen_q, rejected_q)
        
        # 加权组合
        total_loss = sum(loss_weights[name] * loss for name, loss in losses.items())
        
        return total_loss


class ScaleAwareLossFunction:
    """
    尺度感知的损失函数包装器
    """
    
    def __init__(self, base_loss_fn, scale_adaptation: bool = True):
        self.base_loss_fn = base_loss_fn
        self.scale_adaptation = scale_adaptation
        
        # 尺度统计
        self.q_scale_history = []
        self.loss_scale_history = []
    
    def __call__(self, chosen_q: torch.Tensor, rejected_q: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算尺度感知的损失
        """
        # 记录Q值尺度
        current_q_scale = torch.max(torch.abs(chosen_q).mean(), torch.abs(rejected_q).mean()).item()
        self.q_scale_history.append(current_q_scale)
        
        # 保持历史记录在合理长度
        if len(self.q_scale_history) > 1000:
            self.q_scale_history = self.q_scale_history[-500:]
        
        # 计算基础损失
        loss = self.base_loss_fn(chosen_q, rejected_q, **kwargs)
        
        # 记录损失尺度
        self.loss_scale_history.append(loss.item())
        if len(self.loss_scale_history) > 1000:
            self.loss_scale_history = self.loss_scale_history[-500:]
        
        # 尺度自适应调整
        if self.scale_adaptation and len(self.q_scale_history) > 10:
            recent_q_scales = torch.tensor(self.q_scale_history[-10:], dtype=chosen_q.dtype, device=chosen_q.device)
            q_scale_std = recent_q_scales.std()
            
            # 如果Q值尺度变化很大，调整损失
            if q_scale_std > recent_q_scales.mean() * 0.5:
                # 尺度不稳定时，减小损失以避免梯度爆炸
                loss = loss * 0.8
        
        return loss
    
    def get_scale_stats(self) -> dict:
        """
        获取尺度统计信息
        """
        if not self.q_scale_history:
            return {}
        
        q_scales = torch.tensor(self.q_scale_history, dtype=torch.float32)
        loss_scales = torch.tensor(self.loss_scale_history, dtype=torch.float32) if self.loss_scale_history else torch.tensor([0.0], dtype=torch.float32)
        
        return {
            'q_scale_mean': q_scales.mean().item(),
            'q_scale_std': q_scales.std().item(),
            'q_scale_min': q_scales.min().item(),
            'q_scale_max': q_scales.max().item(),
            'loss_scale_mean': loss_scales.mean().item(),
            'loss_scale_std': loss_scales.std().item()
        }


# 测试代码已移除以保持生产代码简洁