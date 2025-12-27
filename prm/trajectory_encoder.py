import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass


@dataclass
class TrajectoryEncoderConfig:
    """轨迹编码器配置"""
    # 输入维度
    obs_dim: int = 151  # 观测维度（h1hand-walk-v0实际维度）
    action_dim: int = 61  # 动作维度
    
    # 编码器架构
    latent_dim: int = 512  # TD-MPC2的潜空间维度
    hidden_dim: int = 256  # 隐藏层维度
    num_layers: int = 3  # MLP层数
    dropout: float = 0.1  # Dropout率
    
    # 序列处理参数
    max_seq_len: int = 1000  # 最大序列长度
    pooling_method: str = "attention"  # 池化方法: "mean", "max", "last", "attention"
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class AttentionPooling(nn.Module):
    """注意力池化模块"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] - True for valid positions
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # 计算注意力权重
        attn_weights = self.attention(x).squeeze(-1)  # [batch_size, seq_len]
        
        if mask is not None:
            # 将无效位置的权重设为负无穷
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1)  # [batch_size, seq_len]
        
        # 加权平均
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return pooled


class TrajectoryEncoder(nn.Module):
    """轨迹编码器
    
    将变长的(观测, 动作)序列编码为定长的潜空间表示，
    模仿TD-MPC2的编码方式但处理序列数据。
    """
    
    def __init__(self, config: TrajectoryEncoderConfig):
        super().__init__()
        self.config = config
        
        # 输入投影层 - 模仿TD-MPC2的MLP编码器结构
        input_dim = config.obs_dim + config.action_dim
        
        # 构建MLP编码器（类似TD-MPC2的state编码器）
        layers = []
        current_dim = input_dim
        
        for i in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.Mish(inplace=True),
                nn.Dropout(config.dropout)
            ])
            current_dim = config.hidden_dim
        
        # 最后一层输出到潜空间维度
        layers.append(nn.Linear(current_dim, config.latent_dim))
        
        self.step_encoder = nn.Sequential(*layers)
        
        # 位置编码（可选）
        self.pos_encoding = PositionalEncoding(config.latent_dim, config.max_seq_len)
        
        # 序列池化层
        if config.pooling_method == "attention":
            self.pooling = AttentionPooling(config.latent_dim)
        elif config.pooling_method in ["mean", "max", "last"]:
            self.pooling = None
        else:
            raise ValueError(f"Unsupported pooling method: {config.pooling_method}")
        
        # 最终输出投影（确保输出维度为latent_dim）
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
    
    def forward(self, 
                observations: torch.Tensor, 
                actions: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码轨迹序列为定长潜空间表示
        
        Args:
            observations: [batch_size, max_seq_len, obs_dim] 或 [batch_size, seq_len, obs_dim]
            actions: [batch_size, max_seq_len, action_dim] 或 [batch_size, seq_len, action_dim]
            lengths: [batch_size] - 每个序列的实际长度（可选）
            
        Returns:
            encoded: [batch_size, latent_dim] - 编码后的定长表示
        """
        batch_size, seq_len = observations.shape[:2]
        
        # 确保在推理时使用eval模式以保证一致性
        was_training = self.training
        if not was_training:
            self.eval()
        
        # 1. 拼接观测和动作
        combined_input = torch.cat([observations, actions], dim=-1)  # [B, T, obs_dim + action_dim]
        
        # 2. 逐步编码（类似TD-MPC2的方式）
        # 将序列展平进行批量编码
        flat_input = combined_input.view(-1, combined_input.size(-1))  # [B*T, input_dim]
        flat_encoded = self.step_encoder(flat_input)  # [B*T, latent_dim]
        step_encoded = flat_encoded.view(batch_size, seq_len, -1)  # [B, T, latent_dim]
        
        # 3. 添加位置编码
        # 转置以匹配位置编码的期望格式
        step_encoded_t = step_encoded.transpose(0, 1)  # [T, B, latent_dim]
        pos_encoded_t = self.pos_encoding(step_encoded_t)  # [T, B, latent_dim]
        pos_encoded = pos_encoded_t.transpose(0, 1)  # [B, T, latent_dim]
        
        # 4. 创建掩码（如果提供了长度信息）
        mask = None
        if lengths is not None:
            mask = torch.arange(seq_len, device=observations.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # 5. 序列池化
        if self.config.pooling_method == "attention":
            pooled = self.pooling(pos_encoded, mask)
        elif self.config.pooling_method == "mean":
            if mask is not None:
                # 掩码平均
                masked_encoded = pos_encoded * mask.unsqueeze(-1).float()
                pooled = masked_encoded.sum(dim=1) / lengths.unsqueeze(-1).float()
            else:
                pooled = pos_encoded.mean(dim=1)
        elif self.config.pooling_method == "max":
            if mask is not None:
                masked_encoded = pos_encoded.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                pooled = masked_encoded.max(dim=1)[0]
            else:
                pooled = pos_encoded.max(dim=1)[0]
        elif self.config.pooling_method == "last":
            if lengths is not None:
                # 获取每个序列的最后一个有效位置
                batch_indices = torch.arange(batch_size, device=observations.device)
                last_indices = (lengths - 1).clamp(min=0)
                pooled = pos_encoded[batch_indices, last_indices]
            else:
                pooled = pos_encoded[:, -1]  # 使用序列的最后一个位置
        
        # 6. 最终输出投影
        encoded = self.output_projection(pooled)  # [B, latent_dim]
        
        # 恢复原来的训练模式
        if was_training:
            self.train()
        
        return encoded
    
    def encode_single_trajectory(self, 
                               observations: Union[List, np.ndarray, torch.Tensor],
                               actions: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        编码单个轨迹（便捷方法）
        
        Args:
            observations: 观测序列
            actions: 动作序列
            
        Returns:
            encoded: [latent_dim] - 编码后的表示
        """
        # 转换为tensor
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32, device=self.config.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.config.device)
        
        # 处理单步输入：添加序列维度和批次维度
        # 输入: [obs_dim], [action_dim] -> [1, 1, obs_dim], [1, 1, action_dim]
        if observations.dim() == 1:
            observations = observations.unsqueeze(0).unsqueeze(0)  # [obs_dim] -> [1, 1, obs_dim]
        elif observations.dim() == 2:
            observations = observations.unsqueeze(0)  # [T, obs_dim] -> [1, T, obs_dim]
        
        if actions.dim() == 1:
            actions = actions.unsqueeze(0).unsqueeze(0)  # [action_dim] -> [1, 1, action_dim]
        elif actions.dim() == 2:
            actions = actions.unsqueeze(0)  # [T, action_dim] -> [1, T, action_dim]
        
        # 编码（确保使用eval模式）
        self.eval()
        with torch.no_grad():
            encoded = self.forward(observations, actions)  # [1, latent_dim]
        
        return encoded.squeeze(0)  # [latent_dim]
    
    def encode_single_trajectory_with_grad(self, 
                                         observations: Union[List, np.ndarray, torch.Tensor],
                                         actions: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        编码单个轨迹（保持梯度的版本）
        
        Args:
            observations: 观测序列
            actions: 动作序列
            
        Returns:
            encoded: [latent_dim] - 编码后的表示
        """
        # 转换为tensor
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32, device=self.config.device)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.config.device)
        
        # 处理单步输入：添加序列维度和批次维度
        # 输入: [obs_dim], [action_dim] -> [1, 1, obs_dim], [1, 1, action_dim]
        if observations.dim() == 1:
            observations = observations.unsqueeze(0).unsqueeze(0)  # [obs_dim] -> [1, 1, obs_dim]
        elif observations.dim() == 2:
            observations = observations.unsqueeze(0)  # [T, obs_dim] -> [1, T, obs_dim]
        
        if actions.dim() == 1:
            actions = actions.unsqueeze(0).unsqueeze(0)  # [action_dim] -> [1, 1, action_dim]
        elif actions.dim() == 2:
            actions = actions.unsqueeze(0)  # [T, action_dim] -> [1, T, action_dim]
        
        # 编码（保持梯度）
        encoded = self.forward(observations, actions)  # [1, latent_dim]
        
        return encoded.squeeze(0)  # [latent_dim]


def create_trajectory_encoder(obs_dim: int = 151, 
                            action_dim: int = 61,
                            latent_dim: int = 512,
                            **kwargs) -> TrajectoryEncoder:
    """创建轨迹编码器的便捷函数"""
    config = TrajectoryEncoderConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        **kwargs
    )
    return TrajectoryEncoder(config)


if __name__ == "__main__":
    # 测试代码
    print("测试轨迹编码器...")
    
    # 创建编码器
    encoder = create_trajectory_encoder()
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 测试数据
    batch_size = 4
    seq_len = 50
    obs_dim = 151
    action_dim = 61
    
    observations = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    lengths = torch.randint(10, seq_len, (batch_size,))
    
    # 前向传播
    encoded = encoder(observations, actions, lengths)
    print(f"输入形状: obs={observations.shape}, actions={actions.shape}")
    print(f"输出形状: {encoded.shape}")
    print(f"输出维度正确: {encoded.shape[-1] == 512}")
    
    # 测试单个轨迹编码
    single_obs = torch.randn(30, obs_dim)
    single_actions = torch.randn(30, action_dim)
    single_encoded = encoder.encode_single_trajectory(single_obs, single_actions)
    print(f"单个轨迹编码形状: {single_encoded.shape}")
    
    print("轨迹编码器测试完成！")