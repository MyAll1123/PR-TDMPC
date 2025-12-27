#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于轨迹编码的偏好模型 (Trajectory-based Preference Model)

改进点：
1. 使用轨迹编码器将变长轨迹编码为定长表示
2. 基于完整轨迹进行偏好比较，而非逐步比较
3. 更符合TD-MPC2的设计理念
4. 提供更稳定的训练和推理

作者：AI Assistant
日期：2025-01-11
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')
sys.path.append('/public/home/yaotianxiao2024/SPE/prm')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math

# 导入轨迹编码器
from trajectory_encoder import TrajectoryEncoder, TrajectoryEncoderConfig, create_trajectory_encoder

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryBasedPreferenceConfig:
    """基于轨迹编码的偏好模型配置"""
    # 轨迹编码器配置
    obs_dim: int = 151  # 观测维度（h1hand-walk-v0实际维度）
    action_dim: int = 61   # 动作维度
    latent_dim: int = 512  # TD-MPC2的潜空间维度
    
    # 偏好比较器配置
    hidden_dim: int = 256  # 隐藏层维度
    num_comparison_layers: int = 3  # 比较网络层数
    dropout: float = 0.1   # Dropout率
    
    # 轨迹编码器配置
    trajectory_encoder_config: Optional[TrajectoryEncoderConfig] = None
    pooling_method: str = "attention"  # 轨迹池化方法
    
    # 训练参数
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_seq_len: int = 1000  # 最大序列长度
    
    # 损失函数参数
    temperature: float = 1.0  # 温度参数
    label_smoothing: float = 0.05  # 标签平滑
    
    # 正则化参数
    weight_decay: float = 1e-4  # 权重衰减
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 不确定性估计
    enable_uncertainty: bool = True  # 启用不确定性估计
    
    # 训练状态控制参数
    min_training_episodes_before_reward: int = 100  # 开始输出偏好奖励前的最小训练轮次
    enable_training_state_check: bool = True  # 是否启用训练状态检查


class TrajectoryComparator(nn.Module):
    """轨迹比较器
    
    接收两个轨迹的编码表示，输出偏好分数和置信度
    """
    
    def __init__(self, config: TrajectoryBasedPreferenceConfig):
        super().__init__()
        self.config = config
        
        # 输入维度：两个轨迹编码的拼接
        input_dim = config.latent_dim * 2
        
        # 比较网络
        layers = []
        current_dim = input_dim
        
        for i in range(config.num_comparison_layers - 1):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout)
            ])
            current_dim = config.hidden_dim
        
        self.comparison_network = nn.Sequential(*layers)
        
        # 偏好分数头
        self.preference_head = nn.Sequential(
            nn.Linear(current_dim, config.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # 不确定性估计头
        if config.enable_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(current_dim, config.hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 4, 1),
                nn.Sigmoid()  # 输出0-1之间的不确定性
            )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                trajectory_a_encoding: torch.Tensor, 
                trajectory_b_encoding: torch.Tensor,
                return_confidence: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        比较两个轨迹编码
        
        Args:
            trajectory_a_encoding: 轨迹A的编码 [batch_size, latent_dim]
            trajectory_b_encoding: 轨迹B的编码 [batch_size, latent_dim]
            return_confidence: 是否返回置信度
            
        Returns:
            如果return_confidence=True: (偏好分数, 置信度)
            如果return_confidence=False: 偏好分数
        """
        # 拼接两个轨迹编码
        combined_encoding = torch.cat([trajectory_a_encoding, trajectory_b_encoding], dim=-1)
        
        # 通过比较网络
        features = self.comparison_network(combined_encoding)
        
        # 计算偏好分数
        preference_score = self.preference_head(features)  # [batch_size, 1]
        
        if return_confidence and self.config.enable_uncertainty:
            # 计算不确定性（转换为置信度）
            uncertainty = self.uncertainty_head(features)  # [batch_size, 1]
            confidence = 1.0 - uncertainty  # 转换为置信度
            return preference_score, confidence
        else:
            return preference_score


class TrajectoryBasedPreferenceModel(nn.Module):
    """基于轨迹编码的偏好模型
    
    使用轨迹编码器将变长轨迹编码为定长表示，然后进行偏好比较
    """
    
    def __init__(self, config: TrajectoryBasedPreferenceConfig):
        super().__init__()
        self.config = config
        
        # 训练状态跟踪
        self.training_episodes_completed = 0
        self.is_trained = False
        
        # 创建轨迹编码器
        if config.trajectory_encoder_config is None:
            encoder_config = TrajectoryEncoderConfig(
                obs_dim=config.obs_dim,
                action_dim=config.action_dim,
                latent_dim=config.latent_dim,
                pooling_method=config.pooling_method,
                max_seq_len=config.max_seq_len,
                device=config.device
            )
        else:
            encoder_config = config.trajectory_encoder_config
        
        self.trajectory_encoder = TrajectoryEncoder(encoder_config)
        
        # 创建轨迹比较器
        self.trajectory_comparator = TrajectoryComparator(config)
        
        logger.info(f"[轨迹偏好模型] 初始化完成")
        logger.info(f"  - 观测维度: {config.obs_dim}")
        logger.info(f"  - 动作维度: {config.action_dim}")
        logger.info(f"  - 潜空间维度: {config.latent_dim}")
        logger.info(f"  - 池化方法: {config.pooling_method}")
        logger.info(f"  - 参数总数: {sum(p.numel() for p in self.parameters()):,}")
    
    def encode_trajectory(self, 
                         observations: torch.Tensor, 
                         actions: torch.Tensor,
                         lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码轨迹为定长表示
        
        Args:
            observations: 观测序列 [batch_size, seq_len, obs_dim]
            actions: 动作序列 [batch_size, seq_len, action_dim]
            lengths: 序列长度 [batch_size] (可选)
            
        Returns:
            轨迹编码 [batch_size, latent_dim]
        """
        return self.trajectory_encoder(observations, actions, lengths)
    
    def forward(self, 
                observations_a: torch.Tensor, 
                actions_a: torch.Tensor,
                observations_b: torch.Tensor, 
                actions_b: torch.Tensor,
                lengths_a: Optional[torch.Tensor] = None,
                lengths_b: Optional[torch.Tensor] = None,
                return_confidence: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：比较两个轨迹的偏好
        
        Args:
            observations_a: 轨迹A的观测 [batch_size, seq_len_a, obs_dim]
            actions_a: 轨迹A的动作 [batch_size, seq_len_a, action_dim]
            observations_b: 轨迹B的观测 [batch_size, seq_len_b, obs_dim]
            actions_b: 轨迹B的动作 [batch_size, seq_len_b, action_dim]
            lengths_a: 轨迹A的长度 [batch_size] (可选)
            lengths_b: 轨迹B的长度 [batch_size] (可选)
            return_confidence: 是否返回置信度
            
        Returns:
            如果return_confidence=True: (偏好分数, 置信度)
            如果return_confidence=False: 偏好分数
        """
        # 编码两个轨迹
        encoding_a = self.encode_trajectory(observations_a, actions_a, lengths_a)
        encoding_b = self.encode_trajectory(observations_b, actions_b, lengths_b)
        
        # 比较轨迹
        return self.trajectory_comparator(encoding_a, encoding_b, return_confidence)
    
    def compute_preference_loss(self, 
                              chosen_obs: torch.Tensor,
                              chosen_actions: torch.Tensor,
                              rejected_obs: torch.Tensor,
                              rejected_actions: torch.Tensor,
                              chosen_lengths: Optional[torch.Tensor] = None,
                              rejected_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算偏好损失
        
        Args:
            chosen_obs: 偏好轨迹的观测
            chosen_actions: 偏好轨迹的动作
            rejected_obs: 非偏好轨迹的观测
            rejected_actions: 非偏好轨迹的动作
            chosen_lengths: 偏好轨迹的长度 (可选)
            rejected_lengths: 非偏好轨迹的长度 (可选)
            
        Returns:
            损失字典
        """
        # 计算偏好分数和置信度
        if self.config.enable_uncertainty:
            preference_score, confidence = self.forward(
                chosen_obs, chosen_actions,
                rejected_obs, rejected_actions,
                chosen_lengths, rejected_lengths,
                return_confidence=True
            )
        else:
            preference_score = self.forward(
                chosen_obs, chosen_actions,
                rejected_obs, rejected_actions,
                chosen_lengths, rejected_lengths,
                return_confidence=False
            )
            confidence = torch.ones_like(preference_score)
        
        # Bradley-Terry损失
        # 偏好分数 > 0 表示轨迹A更好，< 0 表示轨迹B更好
        score_logits = preference_score / self.config.temperature
        
        # 标签平滑
        if self.config.label_smoothing > 0:
            smooth_label = 1.0 - self.config.label_smoothing + self.config.label_smoothing / 2
            preference_loss = -torch.log(
                torch.sigmoid(score_logits) * smooth_label + 
                torch.sigmoid(-score_logits) * (1 - smooth_label)
            )
        else:
            preference_loss = F.binary_cross_entropy_with_logits(
                score_logits, torch.ones_like(score_logits)
            )
        
        # 不确定性正则化损失
        uncertainty_loss = torch.tensor(0.0, device=preference_score.device)
        if self.config.enable_uncertainty:
            # 鼓励模型在困难样本上表现出更高的不确定性
            target_confidence = torch.sigmoid(preference_score.detach().abs())
            uncertainty_loss = F.mse_loss(confidence, target_confidence)
        
        # 总损失
        total_loss = preference_loss.mean() + 0.1 * uncertainty_loss
        
        return {
            'total_loss': total_loss,
            'preference_loss': preference_loss.mean(),
            'uncertainty_loss': uncertainty_loss,
            'preference_score': preference_score.mean(),
            'confidence': confidence.mean()
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
    
    def get_preference_reward(self, 
                            observations: Union[torch.Tensor, np.ndarray, List], 
                            actions: Union[torch.Tensor, np.ndarray, List]) -> Tuple[float, float]:
        """
        获取轨迹的偏好奖励分数
        
        Args:
            observations: 轨迹观测序列
            actions: 轨迹动作序列
            
        Returns:
            (偏好奖励分数, 置信度)
        """
        # 检查训练状态
        if self.config.enable_training_state_check and not self.is_trained:
            return 0.0, 0.0
        
        self.eval()
        with torch.no_grad():
            # 转换为tensor
            if not isinstance(observations, torch.Tensor):
                observations = torch.tensor(observations, dtype=torch.float32, device=self.config.device)
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, dtype=torch.float32, device=self.config.device)
            
            # 确保有批次维度
            if observations.dim() == 2:
                observations = observations.unsqueeze(0)  # [1, seq_len, obs_dim]
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)  # [1, seq_len, action_dim]
            
            # 编码轨迹
            trajectory_encoding = self.encode_trajectory(observations, actions)
            
            # 计算偏好分数（相对于零轨迹的偏好）
            # 这里我们使用轨迹编码的L2范数作为偏好分数的基础
            preference_score = torch.norm(trajectory_encoding, dim=-1).mean()
            
            # 简单的置信度估计（基于编码的方差）
            if self.config.enable_uncertainty:
                encoding_std = torch.std(trajectory_encoding, dim=-1).mean()
                confidence = torch.sigmoid(-encoding_std + 1.0)  # 方差越小，置信度越高
            else:
                confidence = torch.tensor(1.0)
            
            return float(preference_score.item()), float(confidence.item())
    
    def get_preference_reward_single_step(self, 
                                        latent_state: torch.Tensor, 
                                        action: torch.Tensor) -> Tuple[float, float]:
        """
        获取单步偏好奖励（兼容性方法）
        
        注意：这个方法是为了兼容现有接口，但不是最优的使用方式。
        建议使用完整轨迹的get_preference_reward方法。
        
        Args:
            latent_state: 单个潜空间状态 [latent_dim]
            action: 单个动作 [action_dim]
            
        Returns:
            (偏好奖励分数, 置信度)
        """
        # 检查训练状态
        if self.config.enable_training_state_check and not self.is_trained:
            return 0.0, 0.0
        
        self.eval()
        with torch.no_grad():
            # 将单步状态和动作转换为简单的偏好分数
            # 这里使用一个简化的方法：基于状态和动作的组合
            combined = torch.cat([latent_state, action], dim=-1)
            
            # 简单的线性变换得到偏好分数
            preference_score = torch.mean(combined) * 0.1  # 缩放因子
            confidence = 0.5  # 单步预测的置信度较低
            
            return float(preference_score.item()), float(confidence)


# 工厂函数
def create_trajectory_based_preference_model(config: Optional[TrajectoryBasedPreferenceConfig] = None) -> TrajectoryBasedPreferenceModel:
    """创建基于轨迹编码的偏好模型
    
    Args:
        config: 配置对象
        
    Returns:
        轨迹偏好模型
    """
    if config is None:
        config = TrajectoryBasedPreferenceConfig()
    
    model = TrajectoryBasedPreferenceModel(config)
    
    logger.info("[轨迹偏好模型] 创建成功")
    return model


if __name__ == "__main__":
    # 测试代码
    print("基于轨迹编码的偏好模型测试")
    
    # 创建配置
    config = TrajectoryBasedPreferenceConfig(
        obs_dim=212,
        action_dim=61,
        latent_dim=512,
        pooling_method="attention"
    )
    
    # 创建模型
    model = create_trajectory_based_preference_model(config)
    
    # 测试数据
    batch_size = 2
    seq_len_a, seq_len_b = 30, 45
    
    observations_a = torch.randn(batch_size, seq_len_a, config.obs_dim)
    actions_a = torch.randn(batch_size, seq_len_a, config.action_dim)
    observations_b = torch.randn(batch_size, seq_len_b, config.obs_dim)
    actions_b = torch.randn(batch_size, seq_len_b, config.action_dim)
    
    lengths_a = torch.tensor([25, 30])
    lengths_b = torch.tensor([35, 40])
    
    # 前向传播测试
    with torch.no_grad():
        scores, confidences = model(
            observations_a, actions_a,
            observations_b, actions_b,
            lengths_a, lengths_b
        )
        print(f"偏好分数形状: {scores.shape}")
        print(f"置信度形状: {confidences.shape}")
        print(f"偏好分数值: {scores.flatten()}")
        print(f"置信度值: {confidences.flatten()}")
    
    # 损失计算测试
    loss_dict = model.compute_preference_loss(
        observations_a, actions_a,
        observations_b, actions_b,
        lengths_a, lengths_b
    )
    
    print(f"\n损失信息:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 单个轨迹奖励测试
    single_obs = torch.randn(20, config.obs_dim)
    single_actions = torch.randn(20, config.action_dim)
    
    reward, confidence = model.get_preference_reward(single_obs, single_actions)
    print(f"\n单个轨迹偏好奖励: {reward:.4f}, 置信度: {confidence:.4f}")
    
    print("\n基于轨迹编码的偏好模型测试完成！")