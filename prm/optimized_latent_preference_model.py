#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的潜空间偏好模型 (Optimized Latent Preference Model)

改进点：
1. 简化的Transformer架构替代LSTM+Attention
2. 更好的不确定性估计
3. 改进的训练稳定性
4. 更高效的推理性能

作者：AI Assistant
日期：2025-01-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedLatentPreferenceConfig:
    """优化的潜空间偏好模型配置"""
    # 模型架构参数
    latent_dim: int = 512  # TD-MPC2的潜空间维度
    action_dim: int = 61   # 动作维度
    hidden_dim: int = 256  # 隐藏层维度
    n_transformer_layers: int = 2  # Transformer层数（减少）
    n_attention_heads: int = 4     # 注意力头数（减少）
    dropout: float = 0.1   # Dropout率
    
    # 训练参数
    learning_rate: float = 1e-3  # 提高学习率
    batch_size: int = 64
    max_seq_len: int = 1000  # 最大序列长度
    
    # 损失函数参数
    temperature: float = 1.0  # 温度参数
    label_smoothing: float = 0.05  # 减少标签平滑
    
    # 正则化参数
    weight_decay: float = 1e-4  # 权重衰减
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 不确定性估计
    enable_uncertainty: bool = True  # 启用不确定性估计以提供模型预测的可信度评估
    uncertainty_method: str = "ensemble"  # 不确定性方法："ensemble" 或 "dropout"
    
    # 训练状态控制参数
    min_training_episodes_before_reward: int = 100  # 开始输出偏好奖励前的最小训练轮次
    enable_training_state_check: bool = True  # 是否启用训练状态检查

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class OptimizedLatentPreferenceModel(nn.Module):
    """优化的潜空间偏好模型
    
    使用简化的Transformer架构，提供更好的训练稳定性和推理效率
    """
    
    def __init__(self, config: OptimizedLatentPreferenceConfig):
        super().__init__()
        self.config = config
        
        # 训练状态跟踪
        self.training_episodes_completed = 0  # 已完成的训练轮次
        self.is_trained = False  # 是否已经训练过
        
        # 输入投影层：简化版本
        input_dim = config.latent_dim + config.action_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_seq_len)
        
        # Transformer编码器（简化版本）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_attention_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_transformer_layers
        )
        
        # 输出层：分数头（使用原始输出，不进行硬编码缩放）
        self.score_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # 动态归一化统计
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_std', torch.ones(1))
        self.register_buffer('reward_count', torch.zeros(1))
        self.momentum = 0.99  # 动量系数
        
        # 不确定性估计头
        if config.enable_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 4, 1),
                nn.Sigmoid()  # 输出0-1之间的不确定性
            )
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # logger.info(f"[优化潜空间偏好模型] 初始化完成")
        # logger.info(f"  - 潜空间维度: {config.latent_dim}")
        # logger.info(f"  - 动作维度: {config.action_dim}")
        # logger.info(f"  - 隐藏维度: {config.hidden_dim}")
        # logger.info(f"  - Transformer层数: {config.n_transformer_layers}")
        # logger.info(f"  - 注意力头数: {config.n_attention_heads}")
        # logger.info(f"  - 参数总数: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, nn.Linear):
            # Xavier初始化
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                latent_states: torch.Tensor, 
                actions: torch.Tensor, 
                return_confidence: bool = True,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播
        
        Args:
            latent_states: 潜空间状态序列 [batch_size, seq_len, latent_dim]
            actions: 动作序列 [batch_size, seq_len, action_dim]
            return_confidence: 是否返回置信度
            return_attention: 是否返回注意力权重
            
        Returns:
            如果return_confidence=True: (偏好分数, 置信度)
            如果return_confidence=False: 偏好分数
        """
        batch_size, seq_len = latent_states.shape[:2]
        
        # 1. 输入融合：将潜空间状态和动作拼接
        combined_input = torch.cat([latent_states, actions], dim=-1)  # [B, T, latent_dim + action_dim]
        
        # 2. 输入投影
        projected = self.input_projection(combined_input)  # [B, T, hidden_dim]
        
        # 3. 添加位置编码
        # Transformer期望 [seq_len, batch_size, hidden_dim] 格式
        projected_transposed = projected.transpose(0, 1)  # [T, B, hidden_dim]
        pos_encoded = self.pos_encoding(projected_transposed)  # [T, B, hidden_dim]
        pos_encoded = pos_encoded.transpose(0, 1)  # [B, T, hidden_dim]
        
        # 4. Transformer编码
        # 创建padding mask（如果需要）
        # 这里假设所有序列都是有效的，实际使用中可能需要mask
        encoded = self.transformer(pos_encoded)  # [B, T, hidden_dim]
        
        # 5. 全局池化：使用注意力加权平均而非简单的最后一步
        # 计算注意力权重
        attention_weights = F.softmax(
            torch.sum(encoded * projected, dim=-1), dim=-1
        )  # [B, T]
        
        # 加权平均
        final_representation = torch.sum(
            encoded * attention_weights.unsqueeze(-1), dim=1
        )  # [B, hidden_dim]
        
        # 6. 输出偏好分数
        preference_score = self.score_head(final_representation)  # [B, 1]
        
        if return_confidence and self.config.enable_uncertainty:
            # 计算不确定性（转换为置信度）
            uncertainty = self.uncertainty_head(final_representation)  # [B, 1]
            confidence = 1.0 - uncertainty  # 转换为置信度
            
            if return_attention:
                return preference_score, confidence, attention_weights
            else:
                return preference_score, confidence
        else:
            if return_attention:
                return preference_score, attention_weights
            else:
                return preference_score
    
    def compute_preference_loss(self, 
                              chosen_latents: torch.Tensor,
                              chosen_actions: torch.Tensor,
                              rejected_latents: torch.Tensor,
                              rejected_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算改进的偏好损失
        
        Args:
            chosen_latents: 偏好轨迹的潜空间状态
            chosen_actions: 偏好轨迹的动作
            rejected_latents: 非偏好轨迹的潜空间状态
            rejected_actions: 非偏好轨迹的动作
            
        Returns:
            损失字典
        """
        # 计算偏好分数和置信度
        if self.config.enable_uncertainty:
            chosen_result = self.forward(
                chosen_latents, chosen_actions, return_confidence=True
            )
            rejected_result = self.forward(
                rejected_latents, rejected_actions, return_confidence=True
            )
            
            # 正确解包元组返回值
            if isinstance(chosen_result, tuple):
                chosen_score, chosen_confidence = chosen_result
            else:
                chosen_score = chosen_result
                chosen_confidence = torch.ones_like(chosen_score, requires_grad=True)
                
            if isinstance(rejected_result, tuple):
                rejected_score, rejected_confidence = rejected_result
            else:
                rejected_score = rejected_result
                rejected_confidence = torch.ones_like(rejected_score, requires_grad=True)
        else:
            chosen_score = self.forward(
                chosen_latents, chosen_actions, return_confidence=False
            )
            rejected_score = self.forward(
                rejected_latents, rejected_actions, return_confidence=False
            )
            chosen_confidence = torch.ones_like(chosen_score, requires_grad=True)
            rejected_confidence = torch.ones_like(rejected_score, requires_grad=True)
        
        # Bradley-Terry损失
        score_diff = (chosen_score - rejected_score) / self.config.temperature
        
        # 标签平滑
        if self.config.label_smoothing > 0:
            smooth_label = 1.0 - self.config.label_smoothing + self.config.label_smoothing / 2
            preference_loss = -torch.log(
                torch.sigmoid(score_diff) * smooth_label + 
                torch.sigmoid(-score_diff) * (1 - smooth_label)
            )
        else:
            preference_loss = F.binary_cross_entropy_with_logits(
                score_diff, torch.ones_like(score_diff)
            )
        
        # 不确定性正则化损失
        uncertainty_loss = torch.tensor(0.0, device=chosen_score.device)
        if self.config.enable_uncertainty:
            # 鼓励模型在困难样本上表现出更高的不确定性
            uncertainty_loss = F.mse_loss(
                chosen_confidence, torch.sigmoid(chosen_score.detach())
            ) + F.mse_loss(
                rejected_confidence, torch.sigmoid(rejected_score.detach())
            )
        
        # 总损失
        total_loss = preference_loss.mean() + 0.1 * uncertainty_loss
        
        return {
            'total_loss': total_loss,
            'preference_loss': preference_loss.mean(),
            'uncertainty_loss': uncertainty_loss,
            'score_diff': score_diff.mean(),
            'chosen_confidence': chosen_confidence.mean(),
            'rejected_confidence': rejected_confidence.mean()
        }
    
    def update_training_state(self, episodes_completed: int = None):
        """更新训练状态
        
        Args:
            episodes_completed: 已完成的训练轮次，如果为None则增加1
        """
        if episodes_completed is not None:
            self.training_episodes_completed = episodes_completed
        else:
            self.training_episodes_completed += 1
        
        # 检查是否达到训练阈值
        if self.training_episodes_completed >= self.config.min_training_episodes_before_reward:
            self.is_trained = True
    
    def compute_preference_logits(self, 
                                latent_a: torch.Tensor, 
                                action_a: torch.Tensor,
                                latent_b: torch.Tensor, 
                                action_b: torch.Tensor) -> torch.Tensor:
        """计算两个轨迹之间的偏好logits
        
        Args:
            latent_a: 轨迹A的潜空间状态序列 [seq_len, latent_dim] 或 [latent_dim]
            action_a: 轨迹A的动作序列 [seq_len, action_dim] 或 [action_dim]
            latent_b: 轨迹B的潜空间状态序列 [seq_len, latent_dim] 或 [latent_dim]
            action_b: 轨迹B的动作序列 [seq_len, action_dim] 或 [action_dim]
            
        Returns:
            偏好logits: 正值表示偏好轨迹A，负值表示偏好轨迹B
        """
        # 移除eval()和no_grad()以保持梯度传播
        
        # 处理轨迹编码器输出的固定维度特征向量
        # 如果输入是轨迹编码器的输出（1D张量），需要特殊处理
        if latent_a.dim() == 1 and action_a.dim() == 1:
            # 轨迹编码器输出：创建单步序列
            latent_a = latent_a.unsqueeze(0).unsqueeze(0)  # [1, 1, latent_dim]
            action_a = action_a.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
        elif latent_a.dim() == 1:
            # 潜空间特征是1D，但动作是序列
            # 将潜空间特征扩展到与动作序列相同的长度
            seq_len = action_a.shape[0] if action_a.dim() == 2 else action_a.shape[1]
            latent_a = latent_a.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1)  # [1, seq_len, latent_dim]
            if action_a.dim() == 2:
                action_a = action_a.unsqueeze(0)  # [1, seq_len, action_dim]
        elif latent_a.dim() == 2:
            latent_a = latent_a.unsqueeze(0)  # [1, seq_len, latent_dim]
            if action_a.dim() == 1:
                action_a = action_a.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
            elif action_a.dim() == 2:
                action_a = action_a.unsqueeze(0)  # [1, seq_len, action_dim]
            
        # 同样处理轨迹B
        if latent_b.dim() == 1 and action_b.dim() == 1:
            # 轨迹编码器输出：创建单步序列
            latent_b = latent_b.unsqueeze(0).unsqueeze(0)  # [1, 1, latent_dim]
            action_b = action_b.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
        elif latent_b.dim() == 1:
            # 潜空间特征是1D，但动作是序列
            seq_len = action_b.shape[0] if action_b.dim() == 2 else action_b.shape[1]
            latent_b = latent_b.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1)  # [1, seq_len, latent_dim]
            if action_b.dim() == 2:
                action_b = action_b.unsqueeze(0)  # [1, seq_len, action_dim]
        elif latent_b.dim() == 2:
            latent_b = latent_b.unsqueeze(0)  # [1, seq_len, latent_dim]
            if action_b.dim() == 1:
                action_b = action_b.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
            elif action_b.dim() == 2:
                action_b = action_b.unsqueeze(0)  # [1, seq_len, action_dim]
        
        # 计算两个轨迹的偏好分数
        if self.config.enable_uncertainty:
            score_a, _ = self.forward(latent_a, action_a, return_confidence=True)
            score_b, _ = self.forward(latent_b, action_b, return_confidence=True)
        else:
            score_a = self.forward(latent_a, action_a, return_confidence=False)
            score_b = self.forward(latent_b, action_b, return_confidence=False)
        
        # 计算偏好logits (A相对于B的偏好)
        preference_logits = score_a - score_b
        
        return preference_logits.squeeze()  # 移除多余的维度
    
    def get_preference_reward(self, 
                            latent_state: torch.Tensor, 
                            action: torch.Tensor) -> Tuple[float, float]:
        """获取单步偏好奖励和置信度
        
        Args:
            latent_state: 单个潜空间状态 [latent_dim]
            action: 单个动作 [action_dim]
            
        Returns:
            (偏好奖励分数, 置信度)
        """
        # 直接禁用训练状态检查，允许模型在任何时候输出偏好奖励
        # 这样可以确保训练完成后立即生效
        # if self.config.enable_training_state_check and self.training_episodes_completed == 0:
        #     return 0.0, 0.0
        
        self.eval()
        with torch.no_grad():
            # 添加批次和序列维度
            latent_batch = latent_state.unsqueeze(0).unsqueeze(0)  # [1, 1, latent_dim]
            action_batch = action.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
            
            # 计算偏好分数和置信度
            if self.config.enable_uncertainty:
                score, confidence = self.forward(latent_batch, action_batch, return_confidence=True)
                raw_score = float(score.item())
                confidence_value = float(confidence.item())
                
                # 动态归一化：更新统计信息
                self._update_reward_stats(raw_score)
                
                # 标准化偏好奖励
                normalized_score = self._normalize_reward(raw_score)
                
                # 使用新的置信度映射方法
                final_reward = self._map_reward_with_confidence(normalized_score, confidence_value)
                
                return final_reward, confidence_value
            else:
                score = self.forward(latent_batch, action_batch, return_confidence=False)
                raw_score = float(score.item())
                
                # 动态归一化：更新统计信息
                self._update_reward_stats(raw_score)
                
                # 标准化偏好奖励
                normalized_score = self._normalize_reward(raw_score)
                
                # 没有置信度时，使用默认置信度1.0
                final_reward = self._map_reward_with_confidence(normalized_score, 1.0)
                
                return final_reward, 1.0
    
    def _update_reward_stats(self, reward: float):
        """动态更新奖励统计信息
        
        Args:
            reward: 当前原始奖励值
        """
        reward_tensor = torch.tensor(reward, device=self.reward_mean.device)
        
        if self.reward_count == 0:
            # 初始化
            self.reward_mean.copy_(reward_tensor)
            self.reward_std.copy_(torch.ones_like(reward_tensor))
            self.reward_count.copy_(torch.ones_like(self.reward_count))
        else:
            # 使用动量更新均值和标准差
            self.reward_count += 1
            
            # 更新均值
            delta = reward_tensor - self.reward_mean
            self.reward_mean += delta * (1 - self.momentum)
            
            # 更新标准差（使用Welford算法的简化版本）
            delta2 = reward_tensor - self.reward_mean
            variance_update = delta * delta2 * (1 - self.momentum)
            current_var = self.reward_std ** 2
            new_var = self.momentum * current_var + variance_update
            self.reward_std.copy_(torch.sqrt(torch.clamp(new_var, min=1e-8)))
    
    def _normalize_reward(self, reward: float) -> float:
        """标准化奖励到合理范围
        
        Args:
            reward: 原始奖励值
            
        Returns:
            标准化后的奖励值
        """
        if self.reward_count < 10:  # 前10个样本不进行标准化，避免统计不稳定
            return reward * 0.05  # 适度缩放，从0.01提升到0.05
        
        # Z-score标准化
        normalized = (reward - self.reward_mean.item()) / (self.reward_std.item() + 1e-8)
        
        # 将标准化结果映射到扩大的范围 [-1.0, 1.0]
        # 使用tanh函数进行软限制，避免极端值，但允许更大的信号强度
        scaled = torch.tanh(torch.tensor(normalized * 0.8)).item() * 1.0
        
        return scaled
    
    def _map_reward_with_confidence(self, raw_reward: float, confidence: float) -> float:
        """根据置信度映射偏好奖励到指定范围
        
        用户要求的映射规则：
        - 偏好奖励范围：(-0.4, -0.1) 和 (0.1, 0.4)
        - 置信度 >= 0.7 时：最大偏好(0.4)或最大惩罚(-0.4)
        - 置信度 <= 0.4 时：最低偏好(0.1)或最低惩罚(-0.1)
        - 置信度在 (0.4, 0.7) 之间：线性插值
        
        Args:
            raw_reward: 原始奖励值（已标准化）
            confidence: 置信度 [0, 1]
            
        Returns:
            映射后的偏好奖励
        """
        # 确保置信度在合理范围内
        confidence = max(0.0, min(1.0, confidence))
        
        # 判断奖励的正负性
        is_positive = raw_reward >= 0
        
        # 根据置信度计算奖励强度
        if confidence >= 0.7:
            # 高置信度：最大奖励/惩罚
            reward_magnitude = 0.4
        elif confidence <= 0.4:
            # 低置信度：最小奖励/惩罚
            reward_magnitude = 0.1
        else:
            # 中等置信度：线性插值 [0.1, 0.4]
            # 映射 (0.4, 0.7) -> [0.1, 0.4]
            alpha = (confidence - 0.4) / (0.7 - 0.4)  # [0, 1)
            reward_magnitude = 0.1 + alpha * (0.4 - 0.1)
        
        # 应用正负号
        if is_positive:
            return reward_magnitude
        else:
            return -reward_magnitude

# 工厂函数
def create_optimized_latent_preference_model(config: Optional[OptimizedLatentPreferenceConfig] = None) -> OptimizedLatentPreferenceModel:
    """创建优化的潜空间偏好模型
    
    Args:
        config: 配置对象
        
    Returns:
        优化的偏好模型
    """
    if config is None:
        config = OptimizedLatentPreferenceConfig()
    
    model = OptimizedLatentPreferenceModel(config)
    
    # logger.info("[优化潜空间偏好模型] 创建成功")
    return model

if __name__ == "__main__":
    # 测试代码
    print("优化潜空间偏好模型测试")
    
    # 创建配置
    config = OptimizedLatentPreferenceConfig(
        latent_dim=512,
        action_dim=61,
        hidden_dim=256
    )
    
    # 创建模型
    model = create_optimized_latent_preference_model(config)
    
    # 测试数据
    batch_size, seq_len = 4, 20
    latent_states = torch.randn(batch_size, seq_len, config.latent_dim)
    actions = torch.randn(batch_size, seq_len, config.action_dim)
    
    # 前向传播测试
    with torch.no_grad():
        scores, confidences = model(latent_states, actions)
        print(f"偏好分数形状: {scores.shape}")
        print(f"置信度形状: {confidences.shape}")
        print(f"偏好分数值: {scores.flatten()}")
        print(f"置信度值: {confidences.flatten()}")
    
    # 损失计算测试
    chosen_latents = torch.randn(batch_size, seq_len, config.latent_dim)
    chosen_actions = torch.randn(batch_size, seq_len, config.action_dim)
    rejected_latents = torch.randn(batch_size, seq_len, config.latent_dim)
    rejected_actions = torch.randn(batch_size, seq_len, config.action_dim)
    
    loss_dict = model.compute_preference_loss(
        chosen_latents, chosen_actions,
        rejected_latents, rejected_actions
    )
    
    print(f"\n损失信息:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n优化潜空间偏好模型测试完成！")