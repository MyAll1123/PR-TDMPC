#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的优先级计算器

解决前期低奖励偏好对干扰后期高奖励偏好对学习的问题
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPreferencePair:
    """增强的偏好对数据结构"""
    trajectory_a: Dict[str, np.ndarray]
    trajectory_b: Dict[str, np.ndarray]
    preference_label: float
    confidence_score: float
    rule_score_diff: float
    timestamp: float
    priority: float = 0.0
    sample_count: int = 0
    last_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 新增字段
    episode: Optional[int] = None
    training_phase: str = "unknown"  # early, middle, late, unknown
    _id: Optional[str] = None  # 用于唯一标识
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()
        if self._id is None:
            self._id = f"{id(self)}_{self.timestamp}"
    
    def __eq__(self, other):
        """重写相等比较方法"""
        if not isinstance(other, EnhancedPreferencePair):
            return False
        return self._id == other._id
    
    def __hash__(self):
        """重写哈希方法"""
        return hash(self._id)
    
    @property
    def env_reward_a(self) -> float:
        """获取轨迹A的环境奖励"""
        return self.metadata.get('env_reward_a', 0.0)
    
    @property
    def env_reward_b(self) -> float:
        """获取轨迹B的环境奖励"""
        return self.metadata.get('env_reward_b', 0.0)
    
    @property
    def max_env_reward(self) -> float:
        """获取最大环境奖励"""
        return max(self.env_reward_a, self.env_reward_b)
    
    @property
    def env_reward_diff_percentage(self) -> float:
        """计算环境奖励差异百分比"""
        max_reward = max(abs(self.env_reward_a), abs(self.env_reward_b), 1e-8)
        return abs(self.env_reward_a - self.env_reward_b) / max_reward * 100.0

class EnhancedPriorityCalculator:
    """增强的优先级计算器"""
    
    def __init__(self, 
                 confidence_weight: float = 0.5,  # 降低置信度权重
                 temporal_weight: float = 0.2,    # 降低时间权重
                 reward_weight: float = 0.3,      # 新增奖励权重
                 temporal_decay: float = 0.95,
                 min_priority: float = 1e-6,
                 max_priority: float = 1.0,
                 # 奖励感知参数
                 reward_scale_factor: float = 100.0,
                 phase_reward_weights: Optional[Dict[str, float]] = None,
                 # 时间窗口参数
                 similar_time_window_hours: float = 10.0,
                 reward_priority_boost: float = 2.0,
                 # 自适应参数
                 enable_adaptive_priority: bool = True,
                 high_reward_percentile_threshold: float = 90.0,
                 adaptive_boost_factor: float = 1.5):
        
        # 基础权重参数
        self.confidence_weight = confidence_weight
        self.temporal_weight = temporal_weight
        self.reward_weight = reward_weight
        self.temporal_decay = temporal_decay
        self.min_priority = min_priority
        self.max_priority = max_priority
        
        # 奖励感知参数
        self.reward_scale_factor = reward_scale_factor
        self.phase_reward_weights = phase_reward_weights or {
            "early": 0.1,
            "middle": 0.2,
            "late": 0.4,
            "unknown": 0.2
        }
        
        # 时间窗口参数
        self.similar_time_window = similar_time_window_hours * 3600  # 转换为秒
        self.reward_priority_boost = reward_priority_boost
        
        # 自适应参数
        self.enable_adaptive_priority = enable_adaptive_priority
        self.high_reward_percentile_threshold = high_reward_percentile_threshold
        self.adaptive_boost_factor = adaptive_boost_factor
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'reward_boosts_applied': 0,
            'adaptive_boosts_applied': 0,
            'time_window_boosts_applied': 0
        }
        
        logger.info(f"初始化增强优先级计算器: confidence_weight={confidence_weight}, "
                   f"temporal_weight={temporal_weight}, reward_weight={reward_weight}")
    
    def calculate_confidence_priority(self, preference_pair: EnhancedPreferencePair) -> float:
        """计算置信度优先级"""
        confidence_priority = preference_pair.confidence_score
        return np.clip(confidence_priority, 0.0, 1.0)
    
    def calculate_temporal_priority(self, preference_pair: EnhancedPreferencePair, 
                                  current_time: Optional[float] = None) -> float:
        """计算时间优先级"""
        if current_time is None:
            current_time = time.time()
        
        time_diff = current_time - preference_pair.timestamp
        temporal_priority = np.exp(-time_diff / 3600.0 * (1 - self.temporal_decay))
        return np.clip(temporal_priority, 0.1, 1.0)
    
    def calculate_reward_priority(self, preference_pair: EnhancedPreferencePair) -> float:
        """计算奖励优先级"""
        max_reward = preference_pair.max_env_reward
        
        # 使用tanh函数归一化奖励，避免极值问题
        normalized_reward = np.tanh(max_reward / self.reward_scale_factor)
        
        # 根据训练阶段调整权重
        phase_weight = self.phase_reward_weights.get(preference_pair.training_phase, 0.2)
        
        reward_priority = normalized_reward * phase_weight
        return np.clip(reward_priority, 0.0, 1.0)
    
    def calculate_base_priority(self, preference_pair: EnhancedPreferencePair, 
                              current_time: Optional[float] = None) -> float:
        """计算基础优先级"""
        confidence_priority = self.calculate_confidence_priority(preference_pair)
        temporal_priority = self.calculate_temporal_priority(preference_pair, current_time)
        reward_priority = self.calculate_reward_priority(preference_pair)
        
        # 加权组合
        base_priority = (
            self.confidence_weight * confidence_priority +
            self.temporal_weight * temporal_priority +
            self.reward_weight * reward_priority
        )
        
        return np.clip(base_priority, self.min_priority, self.max_priority)
    
    def apply_time_window_boost(self, preference_pair: EnhancedPreferencePair,
                               recent_pairs: List[EnhancedPreferencePair],
                               base_priority: float) -> float:
        """应用时间窗口内的奖励优先级提升"""
        if not recent_pairs:
            return base_priority
        
        # 找到相似时间窗口内的偏好对
        similar_time_pairs = [
            p for p in recent_pairs 
            if abs(p.timestamp - preference_pair.timestamp) <= self.similar_time_window
            and p != preference_pair
        ]
        
        if not similar_time_pairs:
            return base_priority
        
        # 在相似时间窗口内，奖励更高的偏好对获得额外提升
        all_rewards = [p.max_env_reward for p in similar_time_pairs + [preference_pair]]
        max_reward_in_window = max(all_rewards)
        
        if max_reward_in_window > 0 and preference_pair.max_env_reward >= max_reward_in_window * 0.8:
            reward_boost = (preference_pair.max_env_reward / max_reward_in_window) * self.reward_priority_boost
            boosted_priority = base_priority * (1 + reward_boost)
            
            self.stats['time_window_boosts_applied'] += 1
            logger.debug(f"应用时间窗口奖励提升: {base_priority:.4f} -> {boosted_priority:.4f}")
            
            return boosted_priority
        
        return base_priority
    
    def apply_adaptive_boost(self, preference_pair: EnhancedPreferencePair,
                           global_reward_stats: Dict[str, float],
                           base_priority: float) -> float:
        """应用自适应优先级提升"""
        if not self.enable_adaptive_priority or not global_reward_stats:
            return base_priority
        
        global_mean = global_reward_stats.get('mean', 100.0)
        global_std = global_reward_stats.get('std', 50.0)
        global_percentile = global_reward_stats.get(f'percentile_{self.high_reward_percentile_threshold}', 200.0)
        
        # 计算奖励相对位置
        reward_z_score = (preference_pair.max_env_reward - global_mean) / (global_std + 1e-8)
        
        # 高奖励偏好对获得额外提升
        if preference_pair.max_env_reward >= global_percentile:
            boost_factor = self.adaptive_boost_factor
            self.stats['adaptive_boosts_applied'] += 1
        elif reward_z_score > 1.0:
            boost_factor = 1.0 + 0.3 * reward_z_score
            self.stats['adaptive_boosts_applied'] += 1
        else:
            boost_factor = 1.0
        
        if boost_factor > 1.0:
            boosted_priority = base_priority * boost_factor
            logger.debug(f"应用自适应奖励提升: {base_priority:.4f} -> {boosted_priority:.4f} "
                        f"(z_score={reward_z_score:.2f})")
            return boosted_priority
        
        return base_priority
    
    def calculate_enhanced_priority(self, preference_pair: EnhancedPreferencePair,
                                  current_time: Optional[float] = None,
                                  recent_pairs: Optional[List[EnhancedPreferencePair]] = None,
                                  global_reward_stats: Optional[Dict[str, float]] = None) -> float:
        """计算增强的优先级"""
        self.stats['total_calculations'] += 1
        
        # 计算基础优先级
        base_priority = self.calculate_base_priority(preference_pair, current_time)
        
        # 应用时间窗口提升
        if recent_pairs:
            base_priority = self.apply_time_window_boost(preference_pair, recent_pairs, base_priority)
        
        # 应用自适应提升
        if global_reward_stats:
            base_priority = self.apply_adaptive_boost(preference_pair, global_reward_stats, base_priority)
        
        # 最终裁剪
        final_priority = np.clip(base_priority, self.min_priority, self.max_priority)
        
        return final_priority
    
    def update_priority_with_loss(self, preference_pair: EnhancedPreferencePair, 
                                loss: float, loss_weight: float = 0.5) -> float:
        """基于训练损失更新优先级"""
        if preference_pair.last_loss is None:
            # 首次训练，直接设置损失
            preference_pair.last_loss = loss
            loss_priority = loss
        else:
            # 使用损失变化来调整优先级
            loss_change = abs(loss - preference_pair.last_loss)
            loss_priority = loss + loss_weight * loss_change
            preference_pair.last_loss = loss
        
        # 将损失优先级与当前优先级结合
        current_priority = preference_pair.priority
        updated_priority = (1 - loss_weight) * current_priority + loss_weight * loss_priority
        
        preference_pair.priority = np.clip(updated_priority, self.min_priority, self.max_priority)
        return preference_pair.priority
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total_calculations'] > 0:
            stats['reward_boost_rate'] = stats['reward_boosts_applied'] / stats['total_calculations']
            stats['adaptive_boost_rate'] = stats['adaptive_boosts_applied'] / stats['total_calculations']
            stats['time_window_boost_rate'] = stats['time_window_boosts_applied'] / stats['total_calculations']
        else:
            stats['reward_boost_rate'] = 0.0
            stats['adaptive_boost_rate'] = 0.0
            stats['time_window_boost_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_calculations': 0,
            'reward_boosts_applied': 0,
            'adaptive_boosts_applied': 0,
            'time_window_boosts_applied': 0
        }

def test_enhanced_priority_calculator():
    """测试增强的优先级计算器"""
    print("=== 测试增强优先级计算器 ===")
    
    # 创建计算器
    calculator = EnhancedPriorityCalculator(
        confidence_weight=0.5,
        temporal_weight=0.2,
        reward_weight=0.3,
        reward_scale_factor=100.0
    )
    
    current_time = time.time()
    
    # 创建测试偏好对
    test_pairs = [
        # 前期低奖励偏好对
        EnhancedPreferencePair(
            trajectory_a={'obs': np.random.randn(50, 10)},
            trajectory_b={'obs': np.random.randn(50, 10)},
            preference_label=1.0,
            confidence_score=0.8,
            rule_score_diff=4.0,
            timestamp=current_time - 50 * 3600,  # 50小时前
            metadata={'env_reward_a': 30, 'env_reward_b': 40},
            episode=100,
            training_phase="early"
        ),
        # 后期高奖励偏好对
        EnhancedPreferencePair(
            trajectory_a={'obs': np.random.randn(50, 10)},
            trajectory_b={'obs': np.random.randn(50, 10)},
            preference_label=1.0,
            confidence_score=0.8,
            rule_score_diff=4.0,
            timestamp=current_time - 5 * 3600,   # 5小时前
            metadata={'env_reward_a': 250, 'env_reward_b': 300},
            episode=1500,
            training_phase="late"
        ),
        # 时间相近的低奖励偏好对
        EnhancedPreferencePair(
            trajectory_a={'obs': np.random.randn(50, 10)},
            trajectory_b={'obs': np.random.randn(50, 10)},
            preference_label=1.0,
            confidence_score=0.8,
            rule_score_diff=4.0,
            timestamp=current_time - 6 * 3600,   # 6小时前
            metadata={'env_reward_a': 60, 'env_reward_b': 80},
            episode=1480,
            training_phase="late"
        )
    ]
    
    # 计算全局奖励统计
    all_rewards = [pair.max_env_reward for pair in test_pairs]
    global_stats = {
        'mean': np.mean(all_rewards),
        'std': np.std(all_rewards),
        'percentile_90': np.percentile(all_rewards, 90)
    }
    
    print(f"\n全局奖励统计: mean={global_stats['mean']:.2f}, std={global_stats['std']:.2f}, "
          f"90%分位数={global_stats['percentile_90']:.2f}")
    
    # 计算优先级
    print("\n优先级计算结果:")
    for i, pair in enumerate(test_pairs):
        # 基础优先级
        base_priority = calculator.calculate_base_priority(pair, current_time)
        
        # 增强优先级
        enhanced_priority = calculator.calculate_enhanced_priority(
            pair, current_time, test_pairs, global_stats
        )
        
        print(f"\n偏好对 {i+1}:")
        print(f"  奖励: {pair.max_env_reward:.2f}")
        print(f"  阶段: {pair.training_phase}")
        print(f"  时间差: {(current_time - pair.timestamp) / 3600:.1f} 小时")
        print(f"  基础优先级: {base_priority:.4f}")
        print(f"  增强优先级: {enhanced_priority:.4f}")
        print(f"  提升倍数: {enhanced_priority / base_priority:.2f}x")
    
    # 显示统计信息
    print("\n计算器统计信息:")
    stats = calculator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== 测试完成 ===")
    
    return test_pairs, calculator

if __name__ == "__main__":
    test_enhanced_priority_calculator()