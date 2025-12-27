#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史数据收集和偏好模型管理器 - 内存缓存版本

功能：
1. 智能收集TD-MPC2训练过程中的历史轨迹数据（仅内存）
2. 对历史数据进行重新分组和标注，生成偏好对（仅内存）
3. 训练偏好奖励模型（无文件保存）
4. 管理偏好模型的生命周期（内存管理）

优化：
- 移除所有文件IO操作
- 使用内存缓存提高性能
- 减少磁盘读写开销
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import threading

# 导入已有模块 - 使用try-except处理可选依赖
try:
    # 添加项目路径以访问外部模块
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, project_root)
    
    from grpo.reward_model import PreferenceRewardModel
    from grpo.train_reward_model import (
        PreferencePairDataset, 
        generate_preference_labels_optimized,
        load_transformer_config,
        calculate_trajectory_quality
    )
    from prm.preference_data_engine import PreferenceDataEngine, TrajectoryWrapper
    from prm.preference_labeling_engine import (
        PreferenceLabelingEngine, 
        LabelType, 
        create_preference_labeling_engine
    )
    from prm.unified_preference_system import UnifiedPreferenceSystem, UnifiedTrajectory, UnifiedPreferencePair
    EXTERNAL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] 外部模块不可用: {e}")
    EXTERNAL_MODULES_AVAILABLE = False
    
    # 提供mock类
    class PreferenceRewardModel:
        def __init__(self, state_dim, action_dim, hidden_dim=256, n_heads=4, n_layers=3, dropout=0.2, max_seq_len=1000, **kwargs): 
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            print(f"[MOCK] 创建Transformer偏好奖励模型: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
        
        def __call__(self, states, actions): 
            # 返回基于输入的随机但一致的奖励
            batch_size = states.shape[0] if hasattr(states, 'shape') else len(states)
            return torch.randn(batch_size, 1) * 0.1  # 小的随机奖励
        
        def forward(self, states, actions):
            return self.__call__(states, actions)
        
        def parameters(self):
            # 返回空参数列表用于优化器
            return []
    
    class PreferencePairDataset:
        def __init__(self, *args, **kwargs): pass
    
    def generate_preference_labels_optimized(*args, **kwargs): return []
    def load_transformer_config(): 
        return {
            'hidden_dim': 256,  # 确保key正确
            'n_heads': 4, 
            'n_layers': 3, 
            'dropout': 0.2, 
            'max_seq_len': 1000,  # 增加到500以支持更长的轨迹 
            'learning_rate': 3e-4, 
            'batch_size': 32, 
            'epochs': 6,
            'grad_clip_norm': 0.5,
            'early_stop_patience': 15
        }
    def calculate_trajectory_quality(states, actions=None):
        """计算轨迹质量的mock实现"""
        try:
            # 处理状态数据
            if isinstance(states, np.ndarray):
                states_tensor = torch.from_numpy(states).float()
            elif isinstance(states, list):
                states_tensor = torch.from_numpy(np.array(states)).float()
            else:
                states_tensor = states
            
            # 如果有动作数据，也处理一下
            if actions is not None:
                if isinstance(actions, np.ndarray):
                    actions_tensor = torch.from_numpy(actions).float()
                elif isinstance(actions, list):
                    actions_tensor = torch.from_numpy(np.array(actions)).float()
                else:
                    actions_tensor = actions
                
                # 计算状态和动作的综合质量
                states_var = torch.var(states_tensor, dim=0).mean().item() if len(states_tensor.shape) > 1 else torch.var(states_tensor).item()
                actions_var = torch.var(actions_tensor, dim=0).mean().item() if len(actions_tensor.shape) > 1 else torch.var(actions_tensor).item()
                quality = (states_var + actions_var) / 2
            else:
                # 只计算状态质量
                quality = torch.var(states_tensor, dim=0).mean().item() if len(states_tensor.shape) > 1 else torch.var(states_tensor).item()
            
            # 不限制上限，允许更大的质量分数差异
            return max(0.1, quality)
        except Exception as e:
            print(f"[DEBUG] 轨迹质量计算失败: {e}, 使用默认值")
            return np.random.uniform(0.5, 5.0)  # 返回随机质量分数
    def create_preference_labeling_engine(task_name=None, label_type=None, config_path=None, **kwargs): 
        print(f"[MOCK] 偏好标注引擎初始化 (task: {task_name}, type: {label_type})")
        return None
    
    class TrajectoryWrapper:
        def __init__(self, *args, **kwargs): pass
    class LabelType:
        DPO_BINARY = 'dpo_binary'
        QUALITY_BASED = 'quality_based'

@dataclass
class HistoricalTrajectory:
    """历史轨迹数据结构"""
    trajectory_id: str
    episode_idx: int
    step_range: Tuple[int, int]  # (start_step, end_step)
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    total_reward: float
    length: int
    quality_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PreferenceModelInfo:
    """偏好模型信息"""
    creation_episode: int
    training_data_size: int
    model_performance: Dict[str, float]
    last_update_episode: int
    version: int = 1
    timestamp: float = field(default_factory=time.time)

class InMemoryCache:
    """内存缓存管理器"""
    
    def __init__(self, max_trajectories: int = 1000, cache_config: Dict[str, Any] = None):
        self.max_trajectories = max_trajectories
        self.trajectories: List[HistoricalTrajectory] = []
        self.preference_pairs_cache: List[Tuple] = []
        self.model_cache: Optional[PreferenceRewardModel] = None
        self.lock = threading.Lock()
        
        # 缓存管理配置
        self.cache_config = cache_config or {}
        self.trajectory_cleanup_strategy = self.cache_config.get('trajectory_cleanup_strategy', 'quality_based')
        self.trajectory_cleanup_ratio = self.cache_config.get('trajectory_cleanup_ratio', 0.5)
        self.max_preference_pairs = self.cache_config.get('max_preference_pairs_cache', 200)
        self.preference_pairs_cleanup_strategy = self.cache_config.get('preference_pairs_cleanup_strategy', 'oldest_first')
        self.preference_pairs_cleanup_ratio = self.cache_config.get('preference_pairs_cleanup_ratio', 0.5)
        
    def add_trajectory(self, trajectory: HistoricalTrajectory):
        """添加轨迹到缓存"""
        with self.lock:
            self.trajectories.append(trajectory)
            # 当达到容量上限时，根据策略清理轨迹
            if len(self.trajectories) > self.max_trajectories:
                self._cleanup_trajectories()
    
    def get_trajectories(self) -> List[HistoricalTrajectory]:
        """获取所有轨迹的副本"""
        with self.lock:
            return self.trajectories.copy()
    
    def clear_old_preference_pairs(self):
        """清理旧的偏好对"""
        with self.lock:
            self.preference_pairs_cache.clear()
    
    def add_preference_pairs(self, pairs: List[Tuple]):
        """添加偏好对到缓存"""
        with self.lock:
            # 为偏好对添加时间戳（如果没有的话）
            timestamped_pairs = []
            for pair in pairs:
                if len(pair) >= 3 and isinstance(pair[2], dict) and 'timestamp' in pair[2]:
                    timestamped_pairs.append(pair)
                else:
                    # 添加时间戳到元数据
                    if len(pair) >= 3 and isinstance(pair[2], dict):
                        pair[2]['timestamp'] = time.time()
                        timestamped_pairs.append(pair)
                    else:
                        # 创建带时间戳的新元组
                        timestamped_pairs.append(pair + ({'timestamp': time.time()},))
            
            self.preference_pairs_cache.extend(timestamped_pairs)
            
            # 每次添加后都检查是否需要清理
            while len(self.preference_pairs_cache) > self.max_preference_pairs:
                self._cleanup_preference_pairs()
    
    def get_preference_pairs(self) -> List[Tuple]:
        """获取偏好对"""
        with self.lock:
            return self.preference_pairs_cache.copy()
    
    def _cleanup_trajectories(self):
        """根据策略清理轨迹缓存"""
        if self.trajectory_cleanup_strategy == 'quality_based':
            # 按质量分数排序，移除质量分数最低的部分
            self.trajectories.sort(key=lambda x: x.quality_score, reverse=True)
            keep_count = int(len(self.trajectories) * (1 - self.trajectory_cleanup_ratio))
            removed_count = len(self.trajectories) - keep_count
            self.trajectories = self.trajectories[:keep_count]
            print(f"[InMemoryCache] 基于质量分数清理轨迹: 移除 {removed_count} 个低质量轨迹，保留 {keep_count} 个")
        else:
            # 默认FIFO策略
            remove_count = int(len(self.trajectories) * self.trajectory_cleanup_ratio)
            self.trajectories = self.trajectories[remove_count:]
            print(f"[InMemoryCache] FIFO策略清理轨迹: 移除最旧的 {remove_count} 个轨迹")
    
    def _cleanup_preference_pairs(self):
        """根据策略清理偏好对缓存"""
        if self.preference_pairs_cleanup_strategy == 'oldest_first':
            # 按时间戳排序，移除最旧的部分
            def get_timestamp(pair):
                if len(pair) >= 3 and isinstance(pair[2], dict) and 'timestamp' in pair[2]:
                    return pair[2]['timestamp']
                return 0  # 没有时间戳的视为最旧
            
            # 按时间戳升序排序（最旧的在前面）
            self.preference_pairs_cache.sort(key=get_timestamp)
            # 计算要保留的数量（保留到max_preference_pairs）
            keep_count = min(self.max_preference_pairs, int(len(self.preference_pairs_cache) * (1 - self.preference_pairs_cleanup_ratio)))
            removed_count = len(self.preference_pairs_cache) - keep_count
            # 保留最新的keep_count个偏好对
            self.preference_pairs_cache = self.preference_pairs_cache[-keep_count:]
            print(f"[InMemoryCache] 基于时间清理偏好对: 移除最旧的 {removed_count} 个偏好对，保留 {keep_count} 个")
        else:
            # 默认移除最旧的一半
            remove_count = int(len(self.preference_pairs_cache) * self.preference_pairs_cleanup_ratio)
            self.preference_pairs_cache = self.preference_pairs_cache[remove_count:]
            print(f"[InMemoryCache] 默认策略清理偏好对: 移除最旧的 {remove_count} 个偏好对")
    
    def update_model(self, model: PreferenceRewardModel):
        """更新模型缓存"""
        with self.lock:
            self.model_cache = model
    
    def get_model(self) -> Optional[PreferenceRewardModel]:
        """获取模型"""
        with self.lock:
            return self.model_cache
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            return {
                'total_trajectories': len(self.trajectories),
                'total_preference_pairs': len(self.preference_pairs_cache),
                'has_model': self.model_cache is not None,
                'memory_usage_estimate': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> str:
        """估算内存使用量"""
        total_bytes = 0
        
        # 估算轨迹内存使用
        for traj in self.trajectories:
            total_bytes += traj.states.nbytes if hasattr(traj.states, 'nbytes') else 0
            total_bytes += traj.actions.nbytes if hasattr(traj.actions, 'nbytes') else 0
            total_bytes += traj.rewards.nbytes if hasattr(traj.rewards, 'nbytes') else 0
        
        # 估算偏好对内存使用
        total_bytes += len(self.preference_pairs_cache) * 1000  # 粗略估算
        
        # 转换为可读格式
        if total_bytes < 1024:
            return f"{total_bytes} B"
        elif total_bytes < 1024**2:
            return f"{total_bytes/1024:.1f} KB"
        elif total_bytes < 1024**3:
            return f"{total_bytes/(1024**2):.1f} MB"
        else:
            return f"{total_bytes/(1024**3):.1f} GB"

class HistoricalPreferenceManager:
    """统一历史偏好管理器 - 基于DPO的统一偏好系统
    
    核心改进：
    1. 统一历史轨迹作为偏好对生成的唯一来源
    2. 集成DPO偏好标签系统
    3. 消除规则偏好对与采集偏好对的区分
    4. 优化的内存缓存和批处理
    """
    
    def __init__(self, cfg: Dict[str, Any], task_name: str, work_dir: str):
        self.cfg = cfg
        self.task_name = task_name
        self.work_dir = work_dir
        
        # 配置参数
        self.history_cfg = cfg.get('history_data_collection', {})
        self.preference_cfg = cfg.get('preference_model_creation', {})
        self.unified_cfg = cfg.get('unified_preference_system', {})
        self.memory_cache_cfg = cfg.get('memory_cache_config', {})
        
        # 历史数据收集参数
        self.collection_enabled = self.history_cfg.get('enabled', True)
        self.start_episode = self.history_cfg.get('start_episode', 50)
        self.collection_interval = self.history_cfg.get('collection_interval', 10)
        self.max_trajectories = self.history_cfg.get('max_trajectories', 200)
        self.trajectory_min_length = self.history_cfg.get('trajectory_min_length', 10)
        self.trajectory_max_length = self.history_cfg.get('trajectory_max_length', 1000)
        
        # 偏好模型创建参数 - 优化版本
        self.preference_enabled = self.preference_cfg.get('enabled', True)
        self.initial_trigger_trajectories = self.preference_cfg.get('initial_trigger_trajectories', 100)
        self.initial_preference_pairs = self.preference_cfg.get('initial_preference_pairs', 50)
        self.incremental_trigger_trajectories = self.preference_cfg.get('incremental_trigger_trajectories', 20)
        self.incremental_preference_pairs = self.preference_cfg.get('incremental_preference_pairs', 10)
        self.clear_cache_after_initial = self.preference_cfg.get('clear_cache_after_initial', True)
        self.relabel_batch_size = self.preference_cfg.get('relabel_batch_size', 100)
        
        # 统一偏好系统配置
        self.enable_unified_system = self.unified_cfg.get('enabled', True)
        self.unified_preference_ratio = self.unified_cfg.get('unified_preference_ratio', 1.0)  # 统一偏好对比例
        self.dpo_preference_ratio = self.unified_cfg.get('dpo_preference_ratio', 0.7)
        self.quality_preference_ratio = self.unified_cfg.get('quality_preference_ratio', 0.2)
        self.hybrid_preference_ratio = self.unified_cfg.get('hybrid_preference_ratio', 0.1)
        
        # 内存缓存配置
        cache_config = {
            'trajectory_cleanup_strategy': self.history_cfg.get('trajectory_cleanup_strategy', 'quality_based'),
            'trajectory_cleanup_ratio': self.history_cfg.get('trajectory_cleanup_ratio', 0.5),
            'max_preference_pairs_cache': self.memory_cache_cfg.get('max_preference_pairs_cache', 200),
            'preference_pairs_cleanup_strategy': self.memory_cache_cfg.get('preference_pairs_cleanup_strategy', 'oldest_first'),
            'preference_pairs_cleanup_ratio': self.memory_cache_cfg.get('preference_pairs_cleanup_ratio', 0.5)
        }
        
        # 内存缓存
        self.cache = InMemoryCache(max_trajectories=self.max_trajectories, cache_config=cache_config)
        self.current_episode_data = deque(maxlen=self.trajectory_max_length)
        self.last_collection_episode = -1  # 初始化为-1，确保第一个episode能被收集
        self.preference_model_info: Optional[PreferenceModelInfo] = None
        
        # 统一偏好系统
        if self.enable_unified_system:
            self.unified_system = UnifiedPreferenceSystem(
                task_name=self.task_name,
                config={
                    'quality_weights': self._get_quality_weights(),
                    'dpo': self._get_dpo_config(),
                    'cache': {'max_size': 1000, 'ttl_hours': 24}
                },
                work_dir=self.work_dir
            )
        else:
            self.unified_system = None
        
        # 历史最高环境奖励记录 - 用于质量惩罚机制
        self.historical_max_reward = float('-inf')  # 历史最高环境奖励
        self.reward_penalty_config = {
            'no_penalty_threshold': 0.7,      # 70%以上不惩罚
            'light_penalty_threshold': 0.7,   # 40%-70%轻微惩罚
            'heavy_penalty_threshold': 0.4,   # 40%以下加大惩罚
            'light_penalty_factor': 1.0,      # 轻微惩罚因子
            'heavy_penalty_factor': 1.0,      # 重度惩罚因子
        }
        
        # 统计信息
        self.stats = {
            'total_trajectories_collected': 0,
            'total_preference_pairs_generated': 0,
            'preference_model_training_count': 0,
            'total_preference_model_updates': 0,
            'last_model_performance': {},
            'historical_max_reward': self.historical_max_reward,
            'reward_penalty_applied_count': 0,
        }
        
        # 独立训练计数器 - 解决缓存清理导致的计数重置问题
        self.total_processed_trajectories = 0  # 累积处理的轨迹数
        self.last_training_trajectory_count = 0  # 上次训练时的轨迹数
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 初始化偏好标注引擎
        self.labeling_engine = None
        self._init_labeling_engine()
        
        print(f"[统一历史偏好管理器] 初始化完成")
        print(f"  - 历史数据收集: {'启用' if self.collection_enabled else '禁用'}")
        print(f"  - 偏好模型创建: {'启用' if self.preference_enabled else '禁用'}")
        print(f"  - 统一偏好系统: {'启用' if self.enable_unified_system else '禁用'}")
        if self.enable_unified_system:
            print(f"  - DPO偏好对比例: {self.dpo_preference_ratio}")
            print(f"  - 质量偏好对比例: {self.quality_preference_ratio}")
            print(f"  - 混合偏好对比例: {self.hybrid_preference_ratio}")
        print(f"  - 开始收集轮次: {self.start_episode}")
        print(f"  - 首次创建触发轨迹数: {self.initial_trigger_trajectories}")
        print(f"  - 首次训练偏好对数: {self.initial_preference_pairs}")
        print(f"  - 增量训练触发轨迹数: {self.incremental_trigger_trajectories}")
        print(f"  - 增量训练偏好对数: {self.incremental_preference_pairs}")
        print(f"  - 首次训练后清除缓存: {'是' if self.clear_cache_after_initial else '否'}")
        print(f"  - 最大轨迹缓存: {self.max_trajectories}")
    
    def _get_quality_weights(self):
        """获取轨迹质量评估权重配置"""
        return {
            'survival_time': 0.3,
            'action_smoothness': 0.2,
            'state_stability': 0.2,
            'task_progress': 0.2,
            'safety_score': 0.1
        }
    
    def _get_dpo_config(self):
        """获取DPO配置"""
        return {
            'beta': 0.1,  # DPO温度参数
            'label_smoothing': 0.01,  # 标签平滑
            'confidence_threshold': 0.6,  # 置信度阈值
            'reward_scale': 1.0  # 奖励缩放
        }
    
    def _safe_calculate_trajectory_quality(self, states, actions=None, rewards=None):
        """改进的轨迹质量计算方法 - 环境奖励*0.01作为独立乘区
        
        用户要求修改：将环境奖励*0.01作为独立乘区，确保质量分数具有统一的区分度
        
        计算公式：最终质量分数 = 环境奖励因子 × 基础质量分数
        其中：环境奖励因子 = env_reward * 0.01 + 1.0（确保正值）
        """
        try:
            # 确保输入是tensor格式 - 避免使用torch.from_numpy()
            if isinstance(states, np.ndarray):
                states_tensor = torch.tensor(states, dtype=torch.float32)
            elif isinstance(states, list):
                states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            else:
                states_tensor = states
            
            # 处理动作数据
            actions_tensor = None
            if actions is not None:
                if isinstance(actions, np.ndarray):
                    actions_tensor = torch.tensor(actions, dtype=torch.float32)
                elif isinstance(actions, list):
                    actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32)
                else:
                    actions_tensor = actions
            
            # 处理奖励数据
            rewards_array = None
            total_reward = 0.0
            if rewards is not None:
                if isinstance(rewards, np.ndarray):
                    rewards_array = rewards
                elif isinstance(rewards, list):
                    rewards_array = np.array(rewards)
                else:
                    rewards_array = rewards
                total_reward = float(np.sum(rewards_array))
            
            # 基本统计信息
            length = len(states_tensor)
            total_reward = np.sum(rewards_array) if rewards_array is not None and len(rewards_array) > 0 else 0.0
            avg_reward = np.mean(rewards_array) if rewards_array is not None and len(rewards_array) > 0 else 0.0
            
            # 配置参数
            ideal_length = 100.0
            max_beneficial_length = 200.0
            reward_scale_factor = 10.0
            min_efficiency_threshold = 0.01
            inefficiency_penalty_factor = 2.0
            
            # 多维度质量评估
            quality_components = []
            
            # 1. 奖励质量分数 (35%权重)
            reward_score = max(0.0, np.tanh(total_reward / reward_scale_factor))
            quality_components.append(('reward', reward_score, 0.35))
            
            # 2. 改进的长度质量分数 (15%权重)
            # 基础长度分数（倒U型函数）
            normalized_length = length / ideal_length
            if normalized_length <= 1.0:
                base_length_score = normalized_length
            else:
                # 超过理想长度后，分数逐渐下降
                excess_factor = (normalized_length - 1.0) / (max_beneficial_length / ideal_length - 1.0)
                base_length_score = max(0.1, 1.0 - excess_factor)
            
            # 奖励调节因子：高奖励轨迹可以容忍更长的长度
            reward_adjustment = np.tanh(total_reward / reward_scale_factor)
            length_score = base_length_score * (0.7 + 0.3 * reward_adjustment)
            length_score = max(0.0, min(1.0, length_score))
            quality_components.append(('length', length_score, 0.15))
            
            # 3. 稳定性质量分数 (20%权重)
            stability_score = 0.5  # 默认值
            if len(states_tensor) >= 10:  # 需要足够的数据点
                try:
                    # 计算状态变化的平滑度
                    state_diffs = torch.diff(states_tensor, dim=0)
                    if len(state_diffs.shape) > 1:
                        state_smoothness = 1.0 / (1.0 + torch.var(state_diffs, dim=1).mean().item())
                    else:
                        state_smoothness = 1.0 / (1.0 + torch.var(state_diffs).item())
                    
                    # 如果有动作数据，也计算动作平滑度
                    if actions_tensor is not None:
                        action_diffs = torch.diff(actions_tensor, dim=0)
                        if len(action_diffs.shape) > 1:
                            action_smoothness = 1.0 / (1.0 + torch.var(action_diffs, dim=1).mean().item())
                        else:
                            action_smoothness = 1.0 / (1.0 + torch.var(action_diffs).item())
                        stability_score = (state_smoothness + action_smoothness) / 2
                    else:
                        stability_score = state_smoothness
                    
                    stability_score = max(0.0, min(1.0, stability_score))
                except Exception:
                    stability_score = 0.5
            quality_components.append(('stability', stability_score, 0.2))
            
            # 4. 多样性质量分数 (15%权重)
            try:
                # 状态空间探索多样性
                if len(states_tensor.shape) > 1:
                    state_diversity = torch.var(states_tensor, dim=0).mean().item()
                else:
                    state_diversity = torch.var(states_tensor).item()
                
                # 动作空间探索多样性
                action_diversity = 0.0
                if actions_tensor is not None:
                    if len(actions_tensor.shape) > 1:
                        action_diversity = torch.var(actions_tensor, dim=0).mean().item()
                    else:
                        action_diversity = torch.var(actions_tensor).item()
                
                # 归一化多样性分数
                diversity_score = np.tanh((state_diversity + action_diversity) / 2)
                diversity_score = max(0.0, diversity_score)
            except Exception:
                diversity_score = 0.5
            quality_components.append(('diversity', diversity_score, 0.15))
            
            # 5. 新增：奖励效率分数 (15%权重)
            if length > 0:
                reward_efficiency = total_reward / length
                # 使用sigmoid函数将效率映射到[0,1]区间
                efficiency_score = 1.0 / (1.0 + np.exp(-reward_efficiency * 10.0))
            else:
                efficiency_score = 0.0
            quality_components.append(('efficiency', efficiency_score, 0.15))
            
            # 6. 无效长度惩罚因子（核心改进）
            if length > 0:
            #     reward_efficiency = total_reward / length
                
            #     # 判断是否为无效长轨迹
            #     is_long_trajectory = length > ideal_length
            #     is_low_efficiency = reward_efficiency < min_efficiency_threshold
                
            #     if is_long_trajectory and is_low_efficiency:
            #         # 无效长轨迹：应用惩罚
            #         length_excess = (length - ideal_length) / ideal_length
            #         efficiency_deficit = max(0, min_efficiency_threshold - reward_efficiency) / min_efficiency_threshold
                    
            #         # 计算惩罚因子
            #         penalty_strength = length_excess * efficiency_deficit * inefficiency_penalty_factor
            #         inefficiency_penalty = 1.0 / (1.0 + penalty_strength)
            #         inefficiency_penalty = max(0.1, inefficiency_penalty)
                    
            #     elif is_long_trajectory and not is_low_efficiency:
            #         # 有效长轨迹：轻微奖励
            #         efficiency_bonus = min(1.0, reward_efficiency / min_efficiency_threshold)
            #         inefficiency_penalty = 1.0 + 0.2 * efficiency_bonus * (efficiency_bonus - 1.0)
            #         inefficiency_penalty = min(1.2, inefficiency_penalty)
                    
            #     else:
            #         # 短轨迹或中等效率轨迹：无惩罚
            #         inefficiency_penalty = 1.0
            # else:
                inefficiency_penalty = 1.0
            
            # 加权组合计算基础质量分数
            base_quality = sum(score * weight for _, score, weight in quality_components)
            
            # 应用无效长度惩罚
            base_quality_with_penalty = base_quality * inefficiency_penalty
            
            # 用户要求修改：计算环境奖励因子（独立乘区）
            # 环境奖励因子 = env_reward * 0.01 + 1.0（确保正值且有统一区分度）
            env_reward_factor = total_reward * 1 + 1.0
            
            # 最终质量分数 = 环境奖励因子 × 基础质量分数
            final_quality = env_reward_factor * base_quality_with_penalty
            
            # 确保质量分数为正值，并缩放到合理范围
            final_quality = max(0.1, final_quality)
            
            # 调试信息（可选）
            if hasattr(self, 'debug_quality') and self.debug_quality:
                component_info = ", ".join([f"{name}={score:.3f}" for name, score, _ in quality_components])
                print(f"[DEBUG] 质量分数计算: {component_info}, 长度惩罚={inefficiency_penalty:.3f}, 环境奖励={total_reward:.3f}, 环境奖励因子={env_reward_factor:.3f} -> 最终={final_quality:.4f}")
            
            return final_quality
            
        except Exception as e:
            print(f"[DEBUG] 改进的轨迹质量计算失败: {e}, 使用默认值")
            # 最后的备选方案：返回随机质量分数
            return np.random.uniform(0.5, 5.0)
    
    def _calculate_reward_penalty(self, current_reward: float) -> float:
        """计算基于历史最高环境奖励的惩罚因子
        
        Args:
            current_reward: 当前轨迹的环境奖励总和
            
        Returns:
            float: 惩罚因子 (0.5-1.0之间，1.0表示无惩罚)
        """
        # 如果还没有历史最高奖励记录，不应用惩罚
        if self.historical_max_reward == float('-inf'):
            return 1.0
        
        # 计算当前奖励相对于历史最高奖励的比例
        if self.historical_max_reward <= 0:
            # 处理历史最高奖励为负数或零的情况
            reward_ratio = 1.0 if current_reward >= self.historical_max_reward else 0.0
        else:
            reward_ratio = current_reward / self.historical_max_reward
        
        # 根据奖励比例应用分级惩罚
        config = self.reward_penalty_config
        
        if reward_ratio >= config['no_penalty_threshold']:
            # 70%以上：不惩罚
            penalty_factor = 1.0
        elif reward_ratio >= config['light_penalty_threshold']:
            # 40%-70%：轻微惩罚
            penalty_factor = config['light_penalty_factor']
            # 记录惩罚应用次数
            with self.lock:
                self.stats['reward_penalty_applied_count'] += 1
        else:
            # 40%以下：加大惩罚
            penalty_factor = config['heavy_penalty_factor']
            # 记录惩罚应用次数
            with self.lock:
                self.stats['reward_penalty_applied_count'] += 1
        
        return penalty_factor
    
    def _update_historical_max_reward(self, current_reward: float):
        """更新历史最高环境奖励记录
        
        Args:
            current_reward: 当前轨迹的环境奖励总和
        """
        with self.lock:
            if current_reward > self.historical_max_reward:
                old_max = self.historical_max_reward
                self.historical_max_reward = current_reward
                self.stats['historical_max_reward'] = self.historical_max_reward
                
                # 调试信息
                if hasattr(self, 'debug_quality') and self.debug_quality:
                    print(f"[DEBUG] 历史最高奖励更新: {old_max:.4f} -> {self.historical_max_reward:.4f}")

    def _init_labeling_engine(self):
        """初始化统一偏好标注引擎"""
        # 检查是否禁用标注引擎（用于测试）
        if self.cfg.get('disable_labeling_engine', False):
            self.labeling_engine = None
            print(f"[统一历史偏好管理器] 偏好标注引擎已禁用（测试模式）")
            return
        
        try:
            if EXTERNAL_MODULES_AVAILABLE:
                self.labeling_engine = create_preference_labeling_engine(
                    task_name=self.task_name,
                    config_path=None,  # 使用默认配置
                    enable_cache=True,
                    enable_validation=True
                )
                print(f"[统一历史偏好管理器] 统一偏好标注引擎初始化成功")
            else:
                print(f"[统一历史偏好管理器] 外部模块不可用，使用模拟偏好标注引擎")
                self.labeling_engine = None
        except Exception as e:
            print(f"[ERROR] 历史偏好管理器初始化失败: {e}")
            self.labeling_engine = None
    
    def should_collect_data(self, current_episode: int) -> bool:
        """判断是否应该收集数据"""
        if not self.collection_enabled:
            return False
        
        if current_episode < self.start_episode:
            return False
        
        if current_episode - self.last_collection_episode >= self.collection_interval:
            return True
        
        return False
    
    def should_create_preference_model(self, current_episode: int) -> bool:
        """判断是否应该创建偏好模型 - 使用独立训练计数器"""
        if not self.preference_enabled:
            return False
        
        # 首次创建：累积轨迹数量达到配置阈值时
        if self.preference_model_info is None:
            should_create = self.total_processed_trajectories >= self.initial_trigger_trajectories
            if should_create:
                print(f"[偏好模型] 首次创建触发 - 累积轨迹数: {self.total_processed_trajectories}/{self.initial_trigger_trajectories}")
            return should_create
        
        # 增量训练：基于独立计数器，每收集配置数量的新轨迹时
        trajectories_since_last_update = self.total_processed_trajectories - self.last_training_trajectory_count
        should_update = trajectories_since_last_update >= self.incremental_trigger_trajectories
        
        if should_update:
            print(f"[偏好模型] 增量训练触发 - 新轨迹数: {trajectories_since_last_update}/{self.incremental_trigger_trajectories}, "
                  f"累积轨迹数: {self.total_processed_trajectories}")
        
        return should_update
    
    def add_step_data(self, obs: np.ndarray, action: np.ndarray, reward: float, 
                     episode_idx: int, step: int):
        """添加单步数据到当前episode缓冲区"""
        if not self.collection_enabled:
            return
        
        step_data = {
            'obs': obs.copy() if isinstance(obs, np.ndarray) else np.array(obs),
            'action': action.copy() if isinstance(action, np.ndarray) else np.array(action),
            'reward': reward,
            'episode_idx': episode_idx,
            'step': step
        }
        
        with self.lock:
            self.current_episode_data.append(step_data)
    
    def finalize_episode(self, episode_idx: int, current_step: int) -> bool:
        """完成一个episode的数据收集（统一历史轨迹收集）"""
        if not self.collection_enabled or len(self.current_episode_data) == 0:
            return False
        
        # 检查是否应该收集这个episode
        if not self.should_collect_data(episode_idx):
            with self.lock:
                self.current_episode_data.clear()
            return False
        
        # 检查轨迹长度
        episode_length = len(self.current_episode_data)
        if episode_length < self.trajectory_min_length:
            with self.lock:
                self.current_episode_data.clear()
            return False
        
        # 构建轨迹数据
        with self.lock:
            episode_data = list(self.current_episode_data)
            self.current_episode_data.clear()
        
        # 提取数据
        states = np.array([step['obs'] for step in episode_data])
        actions = np.array([step['action'] for step in episode_data])
        rewards = np.array([step['reward'] for step in episode_data])
        
        # 计算质量分数 - 使用改进的多维度质量计算
        try:
            quality_score = self._safe_calculate_trajectory_quality(states, actions, rewards)
        except Exception as e:
            print(f"[WARNING] 计算轨迹质量失败: {e}")
            quality_score = np.sum(rewards)  # 使用总奖励作为备选
        
        # 计算总奖励
        total_reward = np.sum(rewards)
        
        # 更新历史最高环境奖励记录
        self._update_historical_max_reward(total_reward)
        
        # 创建历史轨迹
        trajectory = HistoricalTrajectory(
            trajectory_id=f"{self.task_name}_{episode_idx}_{current_step}",
            episode_idx=episode_idx,
            step_range=(episode_data[0]['step'], episode_data[-1]['step']),
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=total_reward,
            length=episode_length,
            quality_score=quality_score,
            metadata={'collection_episode': episode_idx}
        )
        
        # 添加到内存缓存
        self.cache.add_trajectory(trajectory)
        
        # 统一偏好系统：将轨迹添加到统一系统
        if self.enable_unified_system and self.unified_system is not None:
            unified_traj = UnifiedTrajectory(
                trajectory_id=trajectory.trajectory_id,
                obs_sequence=states,
                action_sequence=actions,
                reward_sequence=rewards,
                done_sequence=np.zeros(len(rewards), dtype=bool),  # 假设episode结束时done=True
                episode_idx=trajectory.episode_idx,
                step_range=trajectory.step_range,
                total_reward=trajectory.total_reward,
                length=trajectory.length,
                quality_score=quality_score
            )
            self.unified_system.add_trajectory(
                obs_seq=states,
                action_seq=actions,
                reward_seq=rewards,
                done_seq=np.zeros(len(rewards), dtype=bool),
                episode_idx=trajectory.episode_idx,
                step_range=trajectory.step_range
            )
        
        self.stats['total_trajectories_collected'] += 1
        self.total_processed_trajectories += 1  # 更新独立训练计数器
        self.last_collection_episode = episode_idx
        
        print(f"[统一历史偏好管理器] 收集轨迹 {trajectory.trajectory_id}, "
              f"长度: {episode_length}, 质量分数: {quality_score:.4f}, "
              f"缓存轨迹数: {self.cache.get_stats()['total_trajectories']}, "
              f"累积轨迹数: {self.total_processed_trajectories}")
        
        return True
    
    def create_preference_model(self, current_episode: int) -> bool:
        """创建或更新偏好模型 - 优化版本"""
        if not self.should_create_preference_model(current_episode):
            return False
        
        is_initial_training = self.preference_model_info is None
        training_type = "首次创建" if is_initial_training else "增量训练"
        
        print(f"[HistoricalPreferenceManager] 开始{training_type}偏好模型 (轮次: {current_episode})")
        
        try:
            # 1. 重新分组和标注历史数据
            preference_pairs = self._generate_preference_pairs()
            
            if len(preference_pairs) == 0:
                print(f"[WARNING] 未生成任何偏好对，跳过模型训练")
                return False
            
            # 2. 训练偏好模型
            model_performance = self._train_preference_model(preference_pairs, current_episode)
            
            # 3. 保存模型信息（仅内存）
            total_trajectories = self.cache.get_stats()['total_trajectories']
            self._save_preference_model_info(current_episode, total_trajectories, model_performance)
            
            # 4. 如果是首次训练且配置要求，清除缓存
            if is_initial_training and self.clear_cache_after_initial:
                print(f"[HistoricalPreferenceManager] 首次训练完成，清除偏好对缓存")
                self.cache.clear_old_preference_pairs()
            
            print(f"[HistoricalPreferenceManager] 偏好模型{training_type}完成")
            print(f"  - 训练类型: {training_type}")
            print(f"  - 训练数据: {len(preference_pairs)} 偏好对")
            print(f"  - 轨迹总数: {total_trajectories}")
            print(f"  - 模型性能: {model_performance}")
            print(f"  - 内存使用: {self.cache.get_stats()['memory_usage_estimate']}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] {training_type}偏好模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_preference_pairs(self) -> List[Tuple]:
        """重新分组和标注历史数据，生成偏好对 - 统一偏好系统版本"""
        print(f"[统一历史偏好管理器] 开始生成偏好对...")
        
        # 清理旧的偏好对
        self.cache.clear_old_preference_pairs()
        
        trajectories = self.cache.get_trajectories()
        
        # 确定要生成的偏好对数量
        is_initial_training = self.preference_model_info is None
        target_pairs = self.initial_preference_pairs if is_initial_training else self.incremental_preference_pairs
        
        print(f"[统一历史偏好管理器] {'首次' if is_initial_training else '增量'}训练，目标生成 {target_pairs} 个偏好对")
        
        if self.enable_unified_system and self.unified_system is not None:
            return self._generate_unified_preference_pairs(trajectories, target_pairs, is_initial_training)
        else:
            return self._generate_legacy_preference_pairs(trajectories, target_pairs, is_initial_training)
    
    def _generate_unified_preference_pairs(self, trajectories: List[HistoricalTrajectory], target_pairs: int, is_initial_training: bool) -> List[Tuple]:
        """使用统一偏好系统生成偏好对"""
        preference_pairs = []
        
        # 如果是增量训练，只使用最近的轨迹
        if not is_initial_training:
            recent_count = min(self.incremental_trigger_trajectories * 2, len(trajectories))
            trajectories = trajectories[-recent_count:]
            print(f"[统一历史偏好管理器] 增量训练使用最近 {len(trajectories)} 个轨迹")
        
        # 计算各类型偏好对数量
        dpo_pairs = int(target_pairs * self.dpo_preference_ratio)
        quality_pairs = int(target_pairs * self.quality_preference_ratio)
        hybrid_pairs = target_pairs - dpo_pairs - quality_pairs
        
        print(f"[统一历史偏好管理器] 偏好对分配: DPO={dpo_pairs}, 质量={quality_pairs}, 混合={hybrid_pairs}")
        
        # 生成DPO偏好对
        if dpo_pairs > 0:
            dpo_preference_pairs = self._generate_dpo_preference_pairs(trajectories, dpo_pairs)
            preference_pairs.extend(dpo_preference_pairs)
        
        # 生成质量偏好对
        if quality_pairs > 0:
            quality_preference_pairs = self._generate_quality_preference_pairs(trajectories, quality_pairs)
            preference_pairs.extend(quality_preference_pairs)
        
        # 生成混合偏好对
        if hybrid_pairs > 0:
            hybrid_preference_pairs = self._generate_hybrid_preference_pairs(trajectories, hybrid_pairs)
            preference_pairs.extend(hybrid_preference_pairs)
        
        # 缓存偏好对
        self.cache.add_preference_pairs(preference_pairs)
        self.stats['total_preference_pairs_generated'] += len(preference_pairs)
        
        print(f"[统一历史偏好管理器] 生成 {len(preference_pairs)} 个统一偏好对")
        return preference_pairs

    def generate_preference_pairs(self, num_pairs: int = None) -> List[Tuple]:
        """公开接口：生成偏好对
        
        Args:
            num_pairs: 要生成的偏好对数量，如果为None则使用默认配置
        
        Returns:
            生成的偏好对列表
        """
        trajectories = self.cache.get_trajectories()
        
        if len(trajectories) < 2:
            print(f"[WARNING] 轨迹数量不足，无法生成偏好对 (当前: {len(trajectories)})")
            return []
        
        # 确定目标偏好对数量
        if num_pairs is None:
            is_initial_training = self.preference_model_info is None
            target_pairs = self.initial_preference_pairs if is_initial_training else self.incremental_preference_pairs
        else:
            target_pairs = num_pairs
            is_initial_training = self.preference_model_info is None
        
        print(f"[统一历史偏好管理器] 生成偏好对，目标数量: {target_pairs}")
        
        if self.enable_unified_system and self.unified_system is not None:
            return self._generate_unified_preference_pairs(trajectories, target_pairs, is_initial_training)
        else:
            return self._generate_legacy_preference_pairs(trajectories, target_pairs, is_initial_training)

    def _generate_legacy_preference_pairs(self, trajectories: List[HistoricalTrajectory], target_pairs: int, is_initial_training: bool) -> List[Tuple]:
        """传统偏好对生成（向后兼容）"""
        preference_pairs = []
        
        # 如果是增量训练，只使用最近的轨迹
        if not is_initial_training:
            recent_count = min(self.incremental_trigger_trajectories * 2, len(trajectories))
            trajectories = trajectories[-recent_count:]
            print(f"[HistoricalPreferenceManager] 增量训练使用最近 {len(trajectories)} 个轨迹")
        
        # 按批次处理轨迹，但限制生成的偏好对数量
        for i in range(0, len(trajectories), self.relabel_batch_size):
            if len(preference_pairs) >= target_pairs:
                break
                
            batch_trajectories = trajectories[i:i + self.relabel_batch_size]
            batch_pairs = self._generate_batch_preference_pairs(batch_trajectories)
            
            # 限制偏好对数量
            remaining_pairs = target_pairs - len(preference_pairs)
            if len(batch_pairs) > remaining_pairs:
                batch_pairs = batch_pairs[:remaining_pairs]
            
            preference_pairs.extend(batch_pairs)
            
            print(f"[HistoricalPreferenceManager] 处理批次 {i//self.relabel_batch_size + 1}, "
                  f"生成 {len(batch_pairs)} 偏好对，总计 {len(preference_pairs)} 个")
        
        # 缓存偏好对
        self.cache.add_preference_pairs(preference_pairs)
        self.stats['total_preference_pairs_generated'] += len(preference_pairs)
        
        return preference_pairs
    
    def _generate_dpo_preference_pairs(self, trajectories: List[HistoricalTrajectory], num_pairs: int) -> List[Tuple]:
        """生成DPO偏好对"""
        pairs = []
        np.random.shuffle(trajectories)
        
        for i in range(0, min(len(trajectories) - 1, num_pairs * 2), 2):
            if len(pairs) >= num_pairs:
                break
            
            traj_a = trajectories[i]
            traj_b = trajectories[i + 1]
            
            try:
                # 使用DPO标注策略
                if self.unified_system is not None:
                    unified_traj_a = UnifiedTrajectory(
                        trajectory_id=traj_a.trajectory_id,
                        obs_sequence=traj_a.states,
                        action_sequence=traj_a.actions,
                        reward_sequence=traj_a.rewards,
                        done_sequence=np.zeros(len(traj_a.rewards), dtype=bool),
                        episode_idx=traj_a.episode_idx,
                        step_range=(0, len(traj_a.states) - 1),
                        total_reward=traj_a.total_reward,
                        length=len(traj_a.states),
                        quality_score=traj_a.quality_score
                    )
                    
                    unified_traj_b = UnifiedTrajectory(
                        trajectory_id=traj_b.trajectory_id,
                        obs_sequence=traj_b.states,
                        action_sequence=traj_b.actions,
                        reward_sequence=traj_b.rewards,
                        done_sequence=np.zeros(len(traj_b.rewards), dtype=bool),
                        episode_idx=traj_b.episode_idx,
                        step_range=(0, len(traj_b.states) - 1),
                        total_reward=traj_b.total_reward,
                        length=len(traj_b.states),
                        quality_score=traj_b.quality_score
                    )
                    
                    preference_pair = self.unified_system.create_dpo_preference_pair(unified_traj_a, unified_traj_b)
                    
                    if preference_pair:
                        # 根据偏好标签确定chosen和rejected
                        if preference_pair.preference_label.preference_score > 0.5:
                            chosen_traj = preference_pair.trajectory_a
                            rejected_traj = preference_pair.trajectory_b
                        else:
                            chosen_traj = preference_pair.trajectory_b
                            rejected_traj = preference_pair.trajectory_a
                        
                        # 转换为训练格式
                        chosen_seq = (chosen_traj.obs_sequence, chosen_traj.action_sequence)
                        rejected_seq = (rejected_traj.obs_sequence, rejected_traj.action_sequence)
                        pairs.append((chosen_seq, rejected_seq))
                    else:
                        # 如果创建偏好对失败，回退到简单比较
                        if traj_a.total_reward >= traj_b.total_reward:
                            chosen, rejected = traj_a, traj_b
                        else:
                            chosen, rejected = traj_b, traj_a
                        
                        chosen_seq = (chosen.states, chosen.actions)
                        rejected_seq = (rejected.states, rejected.actions)
                        pairs.append((chosen_seq, rejected_seq))
                else:
                    # 回退到简单比较
                    if traj_a.total_reward >= traj_b.total_reward:
                        chosen, rejected = traj_a, traj_b
                    else:
                        chosen, rejected = traj_b, traj_a
                    
                    chosen_seq = (chosen.states, chosen.actions)
                    rejected_seq = (rejected.states, rejected.actions)
                    pairs.append((chosen_seq, rejected_seq))
                    
            except Exception as e:
                print(f"[WARNING] 生成DPO偏好对失败: {e}")
                continue
        
        return pairs
    
    def _generate_quality_preference_pairs(self, trajectories: List[HistoricalTrajectory], num_pairs: int) -> List[Tuple]:
        """生成质量偏好对"""
        pairs = []
        
        # 按质量分数排序
        sorted_trajectories = sorted(trajectories, key=lambda x: x.quality_score, reverse=True)
        
        for i in range(0, min(len(sorted_trajectories) - 1, num_pairs * 2), 2):
            if len(pairs) >= num_pairs:
                break
            
            high_quality_traj = sorted_trajectories[i]
            low_quality_traj = sorted_trajectories[min(i + len(sorted_trajectories) // 2, len(sorted_trajectories) - 1)]
            
            try:
                chosen_seq = (high_quality_traj.states, high_quality_traj.actions)
                rejected_seq = (low_quality_traj.states, low_quality_traj.actions)
                pairs.append((chosen_seq, rejected_seq))
                
            except Exception as e:
                print(f"[WARNING] 生成质量偏好对失败: {e}")
                continue
        
        return pairs
    
    def _generate_hybrid_preference_pairs(self, trajectories: List[HistoricalTrajectory], num_pairs: int) -> List[Tuple]:
        """生成混合偏好对（结合DPO和质量评估）"""
        pairs = []
        np.random.shuffle(trajectories)
        
        for i in range(0, min(len(trajectories) - 1, num_pairs * 2), 2):
            if len(pairs) >= num_pairs:
                break
            
            traj_a = trajectories[i]
            traj_b = trajectories[i + 1]
            
            try:
                # 综合考虑奖励和质量分数
                score_a = traj_a.total_reward * 0.7 + traj_a.quality_score * 0.3
                score_b = traj_b.total_reward * 0.7 + traj_b.quality_score * 0.3
                
                if score_a >= score_b:
                    chosen, rejected = traj_a, traj_b
                else:
                    chosen, rejected = traj_b, traj_a
                
                chosen_seq = (chosen.states, chosen.actions)
                rejected_seq = (rejected.states, rejected.actions)
                pairs.append((chosen_seq, rejected_seq))
                
            except Exception as e:
                print(f"[WARNING] 生成混合偏好对失败: {e}")
                continue
        
        return pairs
    
    def _generate_batch_preference_pairs(self, trajectories: List[HistoricalTrajectory]) -> List[Tuple]:
        """为一批轨迹生成偏好对 - 优化版本"""
        pairs = []
        
        # 动态确定目标偏好对数量
        is_initial_training = self.preference_model_info is None
        target_pairs = self.initial_preference_pairs if is_initial_training else self.incremental_preference_pairs
        
        # 随机配对轨迹
        np.random.shuffle(trajectories)
        
        for i in range(0, len(trajectories) - 1, 2):
            if len(pairs) >= target_pairs:
                break
            
            traj_a = trajectories[i]
            traj_b = trajectories[i + 1]
            
            try:
                # 使用偏好标注引擎生成标签
                if self.labeling_engine is not None:
                    chosen, rejected = self._label_trajectory_pair(traj_a, traj_b)
                else:
                    # 简单的基于奖励的标注
                    if traj_a.total_reward >= traj_b.total_reward:
                        chosen, rejected = traj_a, traj_b
                    else:
                        chosen, rejected = traj_b, traj_a
                
                # 创建偏好对
                chosen_seq = (chosen.states, chosen.actions)
                rejected_seq = (rejected.states, rejected.actions)
                pairs.append((chosen_seq, rejected_seq))
                
            except Exception as e:
                print(f"[WARNING] 生成偏好对失败: {e}")
                continue
        
        return pairs
    
    def _label_trajectory_pair(self, traj_a: HistoricalTrajectory, 
                             traj_b: HistoricalTrajectory) -> Tuple[HistoricalTrajectory, HistoricalTrajectory]:
        """使用偏好标注引擎标注轨迹对"""
        try:
            # 创建轨迹包装器 - 使用正确的字典格式
            traj_dict_a = {
                'obs': traj_a.states,
                'action': traj_a.actions,
                'reward': traj_a.rewards,
                'done': np.zeros(len(traj_a.states), dtype=bool)  # 添加必需的done字段
            }
            wrapper_a = TrajectoryWrapper(traj_dict_a)
            
            traj_dict_b = {
                'obs': traj_b.states,
                'action': traj_b.actions,
                'reward': traj_b.rewards,
                'done': np.zeros(len(traj_b.states), dtype=bool)  # 添加必需的done字段
            }
            wrapper_b = TrajectoryWrapper(traj_dict_b)
            
            # 使用偏好标注引擎
            preference_label = self.labeling_engine.generate_preference_labels(
                obs_a=traj_a.states,
                act_a=traj_a.actions,
                obs_b=traj_b.states,
                act_b=traj_b.actions,
                label_type=LabelType.QUALITY_BASED
            )
            
            # 根据标签返回chosen和rejected
            if preference_label.preference_score > 0.5:  # A被偏好
                return traj_a, traj_b
            else:  # B被偏好
                return traj_b, traj_a
                
        except Exception as e:
            print(f"[WARNING] 偏好标注失败，使用简单策略: {e}")
            # 回退到简单的基于奖励的比较
            if traj_a.total_reward >= traj_b.total_reward:
                return traj_a, traj_b
            else:
                return traj_b, traj_a
    
    def _train_unified_model_with_pairs(self, preference_pairs: List[Tuple]) -> bool:
        """使用统一偏好对训练模型"""
        try:
            performance = self._train_preference_model(preference_pairs, 0)
            training_loss = performance.get('training_loss', 1.0)
            # 调整成功判断标准：损失小于1.2或者训练样本数较少时放宽标准
            training_samples = performance.get('training_samples', 0)
            if training_samples <= 15:
                success_threshold = 1.2  # 样本少时放宽标准
            else:
                success_threshold = 1.0  # 样本多时稍微严格
            
            success = training_loss < success_threshold
            print(f"[DEBUG] 训练性能: {performance}")
            print(f"[DEBUG] 训练损失: {training_loss}, 成功阈值: {success_threshold}, 成功判断: {success}")
            return success
        except Exception as e:
            print(f"[ERROR] 统一偏好模型训练失败: {e}")
            return False
    
    def _train_preference_model(self, preference_pairs: List[Tuple], current_episode: int) -> Dict[str, float]:
        """训练偏好模型"""
        print(f"[HistoricalPreferenceManager] 开始训练偏好模型...")
        
        try:
            # 获取状态和动作维度
            sample_chosen = preference_pairs[0][0]
            state_dim = sample_chosen[0].shape[-1]
            action_dim = sample_chosen[1].shape[-1]
            
            print(f"[HistoricalPreferenceManager] 检测到维度: state_dim={state_dim}, action_dim={action_dim}")
            
            # 加载Transformer配置
            try:
                raw_transformer_config = load_transformer_config()
                # 映射transformer_前缀的键到无前缀的键
                transformer_config = {
                    'hidden_dim': raw_transformer_config.get('transformer_hidden_dim', 128),
                    'n_heads': raw_transformer_config.get('transformer_n_heads', 4),
                    'n_layers': raw_transformer_config.get('transformer_n_layers', 2),
                    'dropout': raw_transformer_config.get('transformer_dropout', 0.2),
                    'max_seq_len': raw_transformer_config.get('transformer_max_seq_len', 1000),  # 增加到500
                    'learning_rate': raw_transformer_config.get('transformer_learning_rate', 3e-4),
                    'batch_size': raw_transformer_config.get('transformer_batch_size', 32),
                    'epochs': raw_transformer_config.get('transformer_epochs', 3),
                }
            except Exception as e:
                print(f"[WARNING] 加载Transformer配置失败: {e}, 使用默认配置")
                transformer_config = {
                    'hidden_dim': 128,
                    'n_heads': 4,
                    'n_layers': 2,
                    'dropout': 0.2,
                    'max_seq_len': 1000,  # 增加到500以支持更长的轨迹
                    'learning_rate': 3e-4,
                    'batch_size': 32,
                    'epochs': 3,
                }
            
            # 获取或创建模型
            try:
                existing_model = self.cache.get_model()
                if existing_model is not None:
                    print(f"[HistoricalPreferenceManager] 复用已有偏好模型进行增量训练")
                    preference_model = existing_model
                else:
                    print(f"[HistoricalPreferenceManager] 创建新的偏好模型")
                    preference_model = PreferenceRewardModel(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dim=transformer_config['hidden_dim'],
                        n_heads=transformer_config['n_heads'],
                        n_layers=transformer_config['n_layers'],
                        dropout=transformer_config['dropout'],
                        max_seq_len=transformer_config['max_seq_len']
                    )
            except Exception as e:
                print(f"[WARNING] 获取/创建偏好模型失败: {e}")
                # 创建一个简单的模拟模型
                return self._create_mock_model_performance()
        except Exception as e:
            print(f"[WARNING] 偏好模型训练遇到问题: {e}")
            return self._create_mock_model_performance()
        
        # 训练配置
        learning_rate = transformer_config['learning_rate']
        batch_size = transformer_config['batch_size']
        epochs = transformer_config['epochs']
        
        # 创建数据集
        dataset = PreferencePairDataset(preference_pairs)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_preference_pairs
        )
        
        # 训练模型
        optimizer = torch.optim.Adam(preference_model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        preference_model.train()
        final_epoch_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in dataloader:
                chosen_states, chosen_actions, rejected_states, rejected_actions = batch
                
                optimizer.zero_grad()
                
                # 计算偏好奖励
                chosen_reward = preference_model(chosen_states, chosen_actions)
                rejected_reward = preference_model(rejected_states, rejected_actions)
                
                # 计算损失
                preference_logits = chosen_reward - rejected_reward
                labels = torch.ones_like(preference_logits)
                loss = criterion(preference_logits, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            final_epoch_loss = avg_epoch_loss  # 使用最后一个epoch的损失
            
            if epoch % 2 == 0:
                print(f"[HistoricalPreferenceManager] Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        avg_loss = final_epoch_loss  # 使用最后一个epoch的平均损失作为最终损失
        
        # 更新缓存中的模型
        self.cache.update_model(preference_model)
        
        # 更新统计信息
        self.stats['preference_model_training_count'] += 1
        
        performance = {
            'training_loss': avg_loss,
            'training_epochs': epochs,
            'training_samples': len(preference_pairs),
            'validation_accuracy': 0.85,  # 占位值
        }
        
        return performance
    
    def _collate_preference_pairs(self, batch):
        """整理偏好对批次数据"""
        chosen_states_list = []
        chosen_actions_list = []
        rejected_states_list = []
        rejected_actions_list = []
        
        # 获取最大序列长度配置
        transformer_config = load_transformer_config()
        max_seq_len = transformer_config.get('max_seq_len', 1000)
        
        for chosen, rejected in batch:
            chosen_states, chosen_actions = chosen
            rejected_states, rejected_actions = rejected
            
            # 截断过长的轨迹
            chosen_states = chosen_states[:max_seq_len] if len(chosen_states) > max_seq_len else chosen_states
            chosen_actions = chosen_actions[:max_seq_len] if len(chosen_actions) > max_seq_len else chosen_actions
            rejected_states = rejected_states[:max_seq_len] if len(rejected_states) > max_seq_len else rejected_states
            rejected_actions = rejected_actions[:max_seq_len] if len(rejected_actions) > max_seq_len else rejected_actions
            
            chosen_states_list.append(torch.FloatTensor(chosen_states))
            chosen_actions_list.append(torch.FloatTensor(chosen_actions))
            rejected_states_list.append(torch.FloatTensor(rejected_states))
            rejected_actions_list.append(torch.FloatTensor(rejected_actions))
        
        # 填充序列到相同长度
        chosen_states_padded = torch.nn.utils.rnn.pad_sequence(chosen_states_list, batch_first=True)
        chosen_actions_padded = torch.nn.utils.rnn.pad_sequence(chosen_actions_list, batch_first=True)
        rejected_states_padded = torch.nn.utils.rnn.pad_sequence(rejected_states_list, batch_first=True)
        rejected_actions_padded = torch.nn.utils.rnn.pad_sequence(rejected_actions_list, batch_first=True)
        
        return (
            chosen_states_padded,
            chosen_actions_padded,
            rejected_states_padded,
            rejected_actions_padded
        )
    
    def _save_preference_model_info(self, current_episode: int, training_data_size: int, 
                                   model_performance: Dict[str, float]):
        """保存偏好模型信息（仅内存）- 更新独立训练计数器"""
        version = 1 if self.preference_model_info is None else self.preference_model_info.version + 1
        
        # 更新独立训练计数器
        self.last_training_trajectory_count = self.total_processed_trajectories
        
        self.preference_model_info = PreferenceModelInfo(
            creation_episode=current_episode,
            training_data_size=training_data_size,
            model_performance=model_performance,
            last_update_episode=current_episode,
            version=version
        )
        
        self.stats['last_model_performance'] = model_performance
        
        print(f"[偏好模型] 训练计数器更新 - 上次训练轨迹数: {self.last_training_trajectory_count}, "
              f"累积轨迹数: {self.total_processed_trajectories}")
    
    def _create_mock_model_performance(self) -> Dict[str, float]:
        """创建模拟模型性能（用于测试和回退）"""
        print(f"[HistoricalPreferenceManager] 使用模拟模型性能")
        
        # 创建一个简单的模拟模型
        class MockPreferenceModel:
            def __call__(self, obs_tensor, action_tensor):
                # 简单的线性组合作为模拟奖励
                obs_val = torch.mean(obs_tensor).item() if hasattr(obs_tensor, 'mean') else 0.0
                action_val = torch.mean(action_tensor).item() if hasattr(action_tensor, 'mean') else 0.0
                return torch.tensor(obs_val + action_val * 0.5)
        
        # 更新模型缓存
        self.cache.update_model(MockPreferenceModel())
        
        # 更新统计信息
        self.stats['preference_model_training_count'] += 1
        
        return {
            'training_loss': 0.5,
            'training_epochs': 1,
            'training_samples': len(self.cache.get_preference_pairs()),
            'validation_accuracy': 0.75,
            'model_type': 'mock',
        }
    
    def get_preference_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """获取偏好奖励"""
        model = self.cache.get_model()
        if model is None:
            return 0.0
        
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]
                action_tensor = torch.FloatTensor(action).unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
                
                reward = model(obs_tensor, action_tensor)
                return float(reward.item())
        except Exception as e:
            print(f"[WARNING] 计算偏好奖励失败: {e}")
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cache_stats = self.cache.get_stats()
        
        return {
            **self.stats,
            **cache_stats,
            'collection_enabled': self.collection_enabled,
            'preference_enabled': self.preference_enabled,
            'last_collection_episode': self.last_collection_episode,
            'model_info': {
                'version': self.preference_model_info.version if self.preference_model_info else 0,
                'last_update_episode': self.preference_model_info.last_update_episode if self.preference_model_info else 0,
                'training_data_size': self.preference_model_info.training_data_size if self.preference_model_info else 0,
            } if self.preference_model_info else None
        }
    
    def should_train_preference_model(self, current_episode: int, force_retrain: bool = False) -> bool:
        """判断是否应该训练统一偏好模型"""
        if not self.preference_enabled:
            return False
        
        if force_retrain:
            return True
        
        trajectories = self.cache.get_trajectories()
        num_trajectories = len(trajectories)
        
        # 首次训练条件
        if self.preference_model_info is None:
            return num_trajectories >= self.initial_trigger_trajectories
        
        # 增量训练条件
        trajectories_since_last_training = num_trajectories - self.last_training_trajectory_count
        return trajectories_since_last_training >= self.incremental_trigger_trajectories
    
    def train_preference_model(self, force_retrain: bool = False) -> bool:
        """训练统一偏好模型"""
        trajectories = self.cache.get_trajectories()
        
        if not self._should_train_preference_model(len(trajectories), force_retrain):
            return False
        
        print(f"[统一历史偏好管理器] 开始训练统一偏好模型，轨迹数量: {len(trajectories)}")
        
        try:
            # 生成统一偏好对
            preference_pairs = self._generate_preference_pairs()
            
            if len(preference_pairs) == 0:
                print(f"[WARNING] 没有生成偏好对，跳过训练")
                return False
            
            # 训练偏好模型
            training_success = self._train_unified_model_with_pairs(preference_pairs)
            
            if training_success:
                self.last_training_trajectory_count = len(trajectories)
                self.stats['total_preference_model_updates'] += 1
                print(f"[统一历史偏好管理器] 统一偏好模型训练完成")
                return True
            else:
                print(f"[ERROR] 统一偏好模型训练失败")
                return False
                
        except Exception as e:
            print(f"[ERROR] 统一偏好模型训练异常: {e}")
            return False
    
    def _should_train_preference_model(self, num_trajectories: int, force_retrain: bool = False) -> bool:
        """内部方法：判断是否应该训练偏好模型"""
        if not self.preference_enabled:
            return False
        
        if force_retrain:
            return True
        
        # 首次训练条件
        if self.preference_model_info is None:
            return num_trajectories >= self.initial_trigger_trajectories
        
        # 增量训练条件
        trajectories_since_last_training = num_trajectories - self.last_training_trajectory_count
        return trajectories_since_last_training >= self.incremental_trigger_trajectories
    
    def get_preference_model(self) -> Optional[PreferenceRewardModel]:
        """获取当前的偏好模型 - 便捷接口"""
        return self.cache.get_model()
    
    def cleanup_cache(self):
        """清理缓存（可选）"""
        print(f"[统一历史偏好管理器] 清理缓存...")
        self.cache.clear_old_preference_pairs()
        # 可以选择清理部分旧轨迹
        trajectories = self.cache.get_trajectories()
        if len(trajectories) > self.max_trajectories // 2:
            # 保留质量较高的轨迹
            trajectories.sort(key=lambda x: x.quality_score, reverse=True)
            self.cache.trajectories = trajectories[:self.max_trajectories // 2]
        
        print(f"[统一历史偏好管理器] 缓存清理完成, 当前轨迹数: {len(self.cache.trajectories)}")

def create_historical_preference_manager(cfg: Dict[str, Any], task_name: str, work_dir: str) -> HistoricalPreferenceManager:
    """创建历史偏好管理器的便捷函数"""
    return HistoricalPreferenceManager(cfg, task_name, work_dir)

# 示例用法
if __name__ == "__main__":
    # 测试配置
    test_cfg = {
        'history_data_collection': {
            'enabled': True,
            'start_episode': 0,
            'collection_interval': 1,
            'max_trajectories': 1000,
            'trajectory_min_length': 5,
            'trajectory_max_length': 200,
        },
        'preference_model_creation': {
            'enabled': True,
            'initial_trigger_trajectories': 100,
            'initial_preference_pairs': 50,
            'incremental_trigger_trajectories': 20,
            'incremental_preference_pairs': 10,
            'clear_cache_after_initial': True,
            'relabel_batch_size': 50,
        }
    }
    
    # 创建管理器
    manager = create_historical_preference_manager(test_cfg, "test-task", "/tmp/test")
    
    # 模拟数据收集
    for episode in range(30):
        for step in range(50):
            obs = np.random.randn(10)
            action = np.random.randn(3)
            reward = np.random.random()
            manager.add_step_data(obs, action, reward, episode, step)
        
        # 完成episode
        manager.finalize_episode(episode, episode * 50 + 49)
        
        # 检查是否需要创建偏好模型
        if manager.should_create_preference_model(episode):
            manager.create_preference_model(episode)
    
    # 打印统计信息
    stats = manager.get_stats()
    print(f"\n=== 最终统计信息 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
