#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先级经验回放系统 (Prioritized Experience Replay System)

实现基于优先级的偏好对经验回放机制，包括：
1. 置信度优先级：启发式规则得分差异越大，置信度越高，优先级越高
2. 时间优先级：略微偏向新数据，但不是绝对优势
3. 综合优先级：将两者相乘得到最终优先级
4. 经验回放池：存储和采样偏好对
5. 动态优先级更新：根据训练损失更新优先级

作者：AI Assistant
日期：2025-01-19
版本：1.0.0
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import time
import threading
from abc import ABC, abstractmethod
import heapq
import random
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriorityType(Enum):
    """优先级类型枚举"""
    CONFIDENCE = "confidence"  # 置信度优先级
    TEMPORAL = "temporal"      # 时间优先级
    COMBINED = "combined"      # 综合优先级
    LOSS_BASED = "loss_based"  # 基于损失的优先级

@dataclass
class PreferencePair:
    """偏好对数据结构"""
    trajectory_a: Dict[str, np.ndarray]  # 轨迹A的数据
    trajectory_b: Dict[str, np.ndarray]  # 轨迹B的数据
    preference_label: float              # 偏好标签 [0, 1]
    confidence_score: float              # 置信度分数
    rule_score_diff: float              # 启发式规则得分差异
    timestamp: float                     # 时间戳
    priority: float = 0.0               # 当前优先级
    sample_count: int = 0               # 被采样次数
    last_loss: Optional[float] = None   # 最后一次训练损失
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp == 0:
            self.timestamp = time.time()

class PriorityCalculator:
    """优先级计算器"""
    
    def __init__(self, 
                 confidence_weight: float = 0.7,
                 temporal_weight: float = 0.3,
                 temporal_decay: float = 0.95,
                 min_priority: float = 1e-6,
                 max_priority: float = 1.0):
        """
        初始化优先级计算器
        
        Args:
            confidence_weight: 置信度权重
            temporal_weight: 时间权重
            temporal_decay: 时间衰减因子
            min_priority: 最小优先级
            max_priority: 最大优先级
        """
        self.confidence_weight = confidence_weight
        self.temporal_weight = temporal_weight
        self.temporal_decay = temporal_decay
        self.min_priority = min_priority
        self.max_priority = max_priority
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'avg_confidence_priority': 0.0,
            'avg_temporal_priority': 0.0,
            'avg_combined_priority': 0.0
        }
    
    def calculate_confidence_priority(self, preference_pair: PreferencePair) -> float:
        """
        计算置信度优先级
        
        基于启发式规则得分差异、置信度分数和环境奖励差异，综合评估偏好对质量
        优化后的版本提高了置信度的影响力，减少过度保守的计算
        
        Args:
            preference_pair: 偏好对
            
        Returns:
            置信度优先级 [0, 1]
        """
        # 1. 基础置信度分数 - 提高权重从0.4到0.6
        base_confidence = preference_pair.confidence_score
        
        # 2. 规则得分差异优先级 - 使用更激进的映射
        score_diff = abs(preference_pair.rule_score_diff)
        max_score_diff = 8.0  # 降低最大值，让差异更敏感
        normalized_score_diff = min(score_diff / max_score_diff, 1.0)
        score_diff_priority = np.power(normalized_score_diff, 0.3)  # 更激进的映射，从0.5改为0.3
        
        # 3. 环境奖励差异优先级（如果有的话）
        env_reward_diff_priority = 0.0
        if 'env_reward_diff' in preference_pair.metadata:
            env_reward_diff = preference_pair.metadata['env_reward_diff']
            max_env_reward_diff = 50.0  # 降低最大值，提高敏感度
            normalized_env_diff = min(env_reward_diff / max_env_reward_diff, 1.0)
            env_reward_diff_priority = np.power(normalized_env_diff, 0.2)  # 更激进的映射
        
        # 4. 偏好对类型加权 - 提高权重差异
        pair_type_weight = 1.0
        if 'pair_type' in preference_pair.metadata:
            pair_type = preference_pair.metadata['pair_type']
            if pair_type == 'high_quality':
                pair_type_weight = 2.0  # 从1.5提升到2.0
            elif pair_type == 'medium_quality':
                pair_type_weight = 1.5  # 从1.2提升到1.5
            # exploration类型保持默认权重1.0
        
        # 5. 综合计算置信度优先级 - 提高基础置信度权重
        confidence_priority = (
            0.6 * base_confidence +      # 从0.4提升到0.6
            0.25 * score_diff_priority + # 从0.3调整到0.25
            0.1 * env_reward_diff_priority + # 从0.2调整到0.1
            0.05  # 降低基础优先级从0.1到0.05
        ) * pair_type_weight
        
        # 应用非线性增强，让高置信度样本更突出
        confidence_priority = np.power(confidence_priority, 0.8)  # 轻微的非线性增强
        
        # 确保在合理范围内，但提高最小值
        confidence_priority = np.clip(confidence_priority, 0.05, 1.0)  # 最小值从0.0提升到0.05
        
        return float(confidence_priority)
    
    def calculate_temporal_priority(self, preference_pair: PreferencePair, current_time: Optional[float] = None) -> float:
        """
        计算时间优先级
        
        略微偏向新数据，但不是绝对优势
        
        Args:
            preference_pair: 偏好对
            current_time: 当前时间戳
            
        Returns:
            时间优先级 [0, 1]
        """
        if current_time is None:
            current_time = time.time()
        
        # 计算时间差（秒）
        time_diff = current_time - preference_pair.timestamp
        
        # 使用指数衰减，但保持较高的基础值
        # 这样新数据有优势，但旧数据也不会被完全忽略
        temporal_priority = np.exp(-time_diff / 3600.0 * (1 - self.temporal_decay))  # 1小时衰减周期
        
        # 确保在合理范围内
        temporal_priority = np.clip(temporal_priority, 0.1, 1.0)  # 最小保持0.1
        
        return float(temporal_priority)
    
    def calculate_combined_priority(self, preference_pair: PreferencePair, current_time: Optional[float] = None) -> float:
        """
        计算综合优先级
        
        将置信度优先级和时间优先级相乘得到最终优先级
        
        Args:
            preference_pair: 偏好对
            current_time: 当前时间戳
            
        Returns:
            综合优先级 [min_priority, max_priority]
        """
        confidence_priority = self.calculate_confidence_priority(preference_pair)
        temporal_priority = self.calculate_temporal_priority(preference_pair, current_time)
        
        # 加权组合
        combined_priority = (
            self.confidence_weight * confidence_priority + 
            self.temporal_weight * temporal_priority
        )
        
        # 应用最终的乘法组合（如用户要求）
        final_priority = confidence_priority * temporal_priority
        
        # 限制在合理范围内
        final_priority = np.clip(final_priority, self.min_priority, self.max_priority)
        
        # 更新统计信息
        self.stats['total_calculations'] += 1
        self.stats['avg_confidence_priority'] = (
            (self.stats['avg_confidence_priority'] * (self.stats['total_calculations'] - 1) + confidence_priority) /
            self.stats['total_calculations']
        )
        self.stats['avg_temporal_priority'] = (
            (self.stats['avg_temporal_priority'] * (self.stats['total_calculations'] - 1) + temporal_priority) /
            self.stats['total_calculations']
        )
        self.stats['avg_combined_priority'] = (
            (self.stats['avg_combined_priority'] * (self.stats['total_calculations'] - 1) + final_priority) /
            self.stats['total_calculations']
        )
        
        return float(final_priority)
    
    def update_priority_with_loss(self, preference_pair: PreferencePair, loss: float, loss_weight: float = 0.5) -> float:
        """
        基于训练损失更新优先级
        
        损失越大，说明模型越难拟合，优先级越高
        
        Args:
            preference_pair: 偏好对
            loss: 训练损失
            loss_weight: 损失权重
            
        Returns:
            更新后的优先级
        """
        # 记录损失
        preference_pair.last_loss = loss
        
        # 计算基础优先级
        base_priority = self.calculate_combined_priority(preference_pair)
        
        # 基于损失的优先级调整
        # 使用损失的平方根来避免极端值
        loss_factor = np.sqrt(max(loss, 1e-8))  # 避免零损失
        
        # 组合基础优先级和损失因子
        updated_priority = base_priority * (1 + loss_weight * loss_factor)
        
        # 限制在合理范围内
        updated_priority = np.clip(updated_priority, self.min_priority, self.max_priority)
        
        # 更新偏好对的优先级
        preference_pair.priority = updated_priority
        
        return float(updated_priority)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

class PrioritizedExperienceBuffer:
    """优先级经验回放缓冲池"""
    
    def __init__(self, 
                 capacity: int = 10000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 high_quality_boost: float = 3.0,
                 enable_quality_sampling: bool = True,
                 quality_thresholds: Optional[Dict[str, float]] = None):
        """
        初始化优先级经验回放缓冲池
        
        Args:
            capacity: 缓冲池容量
            alpha: 优先级指数，控制优先级的影响程度
            beta: 重要性采样指数
            beta_increment: beta的增长率
            epsilon: 数值稳定性常数
            high_quality_boost: 高质量经验的采样提升倍数
            enable_quality_sampling: 是否启用质量感知采样
            quality_thresholds: 质量识别阈值配置
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.high_quality_boost = high_quality_boost
        self.enable_quality_sampling = enable_quality_sampling
        
        # 设置质量识别阈值
        if quality_thresholds is None:
            quality_thresholds = {}
        
        self.confidence_threshold = quality_thresholds.get('confidence_threshold', 0.7)
        self.rule_score_threshold = quality_thresholds.get('rule_score_diff_threshold', 3.0)
        self.env_reward_threshold = quality_thresholds.get('env_reward_diff_threshold', 15.0)
        self.max_sample_count = quality_thresholds.get('max_sample_count', 5)
        self.min_quality_indicators = quality_thresholds.get('min_quality_indicators', 2)
        
        # 存储偏好对
        self.buffer: List[PreferencePair] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # 优先级计算器
        self.priority_calculator = PriorityCalculator()
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_added': 0,
            'total_sampled': 0,
            'avg_priority': 0.0,
            'max_priority': 0.0,
            'min_priority': float('inf'),
            'priority_updates': 0,
            'high_quality_samples': 0,
            'quality_boost_applied': 0
        }
    
    def add(self, preference_pair: PreferencePair) -> None:
        """
        添加偏好对到缓冲池
        
        Args:
            preference_pair: 偏好对
        """
        try:
            with self.lock:
                # 计算初始优先级
                # logger.info(f"[经验回放池] 开始计算偏好对优先级 - 置信度: {preference_pair.confidence_score:.3f}")
                priority = self.priority_calculator.calculate_combined_priority(preference_pair)
                
                # 验证优先级有效性
                if not isinstance(priority, (int, float)) or np.isnan(priority) or np.isinf(priority) or priority <= 0:
                    logger.error(f"[经验回放池] 计算出无效优先级: {priority}, 使用默认值")
                    priority = self.priority_calculator.min_priority
                
                preference_pair.priority = priority
                # logger.info(f"[经验回放池] 偏好对优先级计算完成: {priority:.6f}")
                
                # 添加到缓冲池
                size_before = self.size
                if self.size < self.capacity:
                    self.buffer.append(preference_pair)
                    self.size += 1
                else:
                    self.buffer[self.position] = preference_pair
                
                # logger.info(f"[经验回放池] 偏好对添加到缓冲池 - 大小: {size_before} -> {self.size}, 位置: {self.position}")
                
                # 设置优先级
                priority_alpha = priority ** self.alpha
                if np.isnan(priority_alpha) or np.isinf(priority_alpha):
                    logger.error(f"[经验回放池] 优先级的alpha次幂无效: {priority_alpha}, 使用原始优先级")
                    priority_alpha = priority
                
                self.priorities[self.position] = priority_alpha
                
                # 更新位置
                self.position = (self.position + 1) % self.capacity
                
                # 更新统计信息
                self.stats['total_added'] += 1
                self.stats['avg_priority'] = (
                    (self.stats['avg_priority'] * (self.stats['total_added'] - 1) + priority) /
                    self.stats['total_added']
                )
                self.stats['max_priority'] = max(self.stats['max_priority'], priority)
                self.stats['min_priority'] = min(self.stats['min_priority'], priority)
                
                # 静默处理添加完成，不输出详细日志
                
        except Exception as e:
            logger.error(f"[经验回放池] 添加偏好对时发生异常: {e}")
            logger.error(f"[经验回放池] 偏好对信息: 置信度={preference_pair.confidence_score}, 时间戳={preference_pair.timestamp}")
            raise e
    
    def sample(self, batch_size: int) -> Tuple[List[PreferencePair], np.ndarray, np.ndarray]:
        """
        从缓冲池中采样一个批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (采样的偏好对列表, 采样索引, 重要性权重)
        """
        with self.lock:
            if self.size == 0:
                return [], np.array([]), np.array([])
            
            if self.enable_quality_sampling:
                return self._quality_aware_sample(batch_size)
            else:
                return self._standard_sample(batch_size)
    
    def _standard_sample(self, batch_size: int) -> Tuple[List[PreferencePair], np.ndarray, np.ndarray]:
        """标准优先级采样"""
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probabilities = priorities / np.sum(priorities)
        
        # 采样索引
        indices = np.random.choice(self.size, size=min(batch_size, self.size), p=probabilities, replace=False)
        
        # 获取采样的偏好对
        sampled_pairs = [self.buffer[i] for i in indices]
        
        # 计算重要性权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # 归一化
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 更新采样统计
        for pair in sampled_pairs:
            pair.sample_count += 1
        
        self.stats['total_sampled'] += len(sampled_pairs)
        
        return sampled_pairs, indices, weights
    
    def _quality_aware_sample(self, batch_size: int) -> Tuple[List[PreferencePair], np.ndarray, np.ndarray]:
        """质量感知采样 - 增加高质量经验的采样权重和频率"""
        # 获取基础优先级
        base_priorities = self.priorities[:self.size].copy()
        
        # 应用质量提升
        enhanced_priorities = base_priorities.copy()
        high_quality_count = 0
        
        for i in range(self.size):
            pair = self.buffer[i]
            
            # 识别高质量经验的条件
            is_high_quality = self._is_high_quality_experience(pair)
            
            if is_high_quality:
                enhanced_priorities[i] *= self.high_quality_boost
                high_quality_count += 1
        
        # 计算增强后的采样概率
        probabilities = enhanced_priorities / np.sum(enhanced_priorities)
        
        # 采样索引
        indices = np.random.choice(self.size, size=min(batch_size, self.size), p=probabilities, replace=False)
        
        # 获取采样的偏好对
        sampled_pairs = [self.buffer[i] for i in indices]
        
        # 计算重要性权重（基于原始优先级）
        original_probabilities = base_priorities / np.sum(base_priorities)
        weights = (self.size * original_probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # 归一化
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 更新采样统计
        sampled_high_quality = 0
        for i, pair in enumerate(sampled_pairs):
            pair.sample_count += 1
            if self._is_high_quality_experience(pair):
                sampled_high_quality += 1
        
        self.stats['total_sampled'] += len(sampled_pairs)
        self.stats['high_quality_samples'] += sampled_high_quality
        if high_quality_count > 0:
            self.stats['quality_boost_applied'] += 1
        
        return sampled_pairs, indices, weights
    
    def _is_high_quality_experience(self, pair: PreferencePair) -> bool:
        """判断是否为高质量经验"""
        # 使用初始化时设置的阈值参数
        confidence_threshold = self.confidence_threshold
        rule_score_threshold = self.rule_score_threshold
        env_reward_threshold = self.env_reward_threshold  # 现在作为百分比阈值（如100.0表示100%）
        max_sample_count = self.max_sample_count
        min_quality_indicators = self.min_quality_indicators
        
        # 1. 高置信度
        high_confidence = pair.confidence_score >= confidence_threshold
        
        # 2. 显著的规则得分差异
        significant_rule_diff = abs(pair.rule_score_diff) >= rule_score_threshold
        
        # 3. 高质量标记（如果有的话）
        high_quality_type = False
        if 'pair_type' in pair.metadata:
            high_quality_type = pair.metadata['pair_type'] in ['high_quality', 'excellent']
        
        # 4. 显著的环境奖励差异（基于百分比）
        significant_env_diff = False
        if 'env_reward_a' in pair.metadata and 'env_reward_b' in pair.metadata:
            reward_a = pair.metadata['env_reward_a']
            reward_b = pair.metadata['env_reward_b']
            
            # 计算百分比差异
            # 使用较大的奖励值作为基准，避免除零错误
            max_reward = max(abs(reward_a), abs(reward_b), 1e-8)  # 避免除零
            reward_diff_percentage = abs(reward_a - reward_b) / max_reward * 100.0
            
            significant_env_diff = reward_diff_percentage >= env_reward_threshold
        elif 'env_reward_diff' in pair.metadata:
            # 兼容旧版本：如果直接提供了差异值，假设需要计算百分比
            # 这里需要额外的上下文信息来计算百分比，暂时保持原逻辑但添加警告
            logger.warning("使用旧版本的env_reward_diff，建议提供env_reward_a和env_reward_b以支持百分比计算")
            significant_env_diff = pair.metadata['env_reward_diff'] >= env_reward_threshold
        
        # 5. 较少被采样（保持多样性）
        not_over_sampled = pair.sample_count <= max_sample_count
        
        # 综合判断：满足多个条件的经验被认为是高质量的
        quality_indicators = [
            high_confidence,
            significant_rule_diff,
            high_quality_type,
            significant_env_diff,
            not_over_sampled
        ]
        
        # 根据配置的最小指标数量判断
        return sum(quality_indicators) >= min_quality_indicators
    
    def update_priorities(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """
        基于训练损失更新优先级
        
        Args:
            indices: 样本索引
            losses: 对应的训练损失
        """
        with self.lock:
            for idx, loss in zip(indices, losses):
                if idx < self.size:
                    preference_pair = self.buffer[idx]
                    
                    # 使用优先级计算器更新优先级
                    new_priority = self.priority_calculator.update_priority_with_loss(
                        preference_pair, float(loss)
                    )
                    
                    # 更新缓冲池中的优先级
                    self.priorities[idx] = (new_priority + self.epsilon) ** self.alpha
                    
                    self.stats['priority_updates'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            stats = self.stats.copy()
            stats['buffer_size'] = self.size
            stats['buffer_capacity'] = self.capacity
            stats['current_beta'] = self.beta
            stats['priority_calculator_stats'] = self.priority_calculator.get_statistics()
            
            if self.size > 0:
                current_priorities = self.priorities[:self.size]
                stats['current_avg_priority'] = float(np.mean(current_priorities))
                stats['current_max_priority'] = float(np.max(current_priorities))
                stats['current_min_priority'] = float(np.min(current_priorities))
            
            return stats
    
    def clear(self) -> None:
        """清空缓冲池"""
        with self.lock:
            self.buffer.clear()
            self.priorities.fill(0)
            self.position = 0
            self.size = 0
    
    def __len__(self) -> int:
        """返回缓冲池大小"""
        return self.size
    
    def is_ready_for_sampling(self, min_size: int = 200) -> bool:
        """检查是否准备好进行采样，从配置文件读取，不硬编码"""
        return self.size >= min_size

class PrioritizedPreferenceTrainer:
    """基于优先级经验回放的偏好训练器"""
    
    def __init__(self,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 min_buffer_size: int = 200,  # 从配置文件读取，不硬编码
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4,
                 priority_beta_increment: float = 0.001,
                 high_quality_boost: float = 3.0,
                 enable_quality_sampling: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化优先级偏好训练器
        
        Args:
            buffer_capacity: 缓冲池容量
            batch_size: 批次大小
            min_buffer_size: 最小缓冲池大小
            priority_alpha: 优先级指数
            priority_beta: 重要性采样指数
            priority_beta_increment: beta增长率
            high_quality_boost: 高质量经验的采样提升倍数
            enable_quality_sampling: 是否启用质量感知采样
            config: 配置字典（可选）
        """
        # 如果提供了config，优先使用config中的参数
        if config is not None:
            self.batch_size = config.get('batch_size', batch_size)
            self.min_buffer_size = config.get('min_buffer_size', min_buffer_size)
            self.high_quality_boost = config.get('high_quality_boost', high_quality_boost)
            self.enable_quality_sampling = config.get('enable_quality_sampling', enable_quality_sampling)
            buffer_capacity = config.get('buffer_capacity', buffer_capacity)
            priority_alpha = config.get('priority_alpha', priority_alpha)
            priority_beta = config.get('priority_beta', priority_beta)
            priority_beta_increment = config.get('priority_beta_increment', priority_beta_increment)
        else:
            self.batch_size = batch_size
            self.min_buffer_size = min_buffer_size
            self.high_quality_boost = high_quality_boost
            self.enable_quality_sampling = enable_quality_sampling
        
        self.config = config or {}
        
        # 从配置中获取质量阈值参数
        quality_thresholds = None
        if self.config and 'sampling_config' in self.config:
            sampling_config = self.config['sampling_config']
            if 'quality_thresholds' in sampling_config:
                quality_thresholds = sampling_config['quality_thresholds']
        
        # 创建优先级经验回放缓冲池
        self.experience_buffer = PrioritizedExperienceBuffer(
            capacity=buffer_capacity,
            alpha=priority_alpha,
            beta=priority_beta,
            beta_increment=priority_beta_increment,
            high_quality_boost=self.high_quality_boost,
            enable_quality_sampling=self.enable_quality_sampling,
            quality_thresholds=quality_thresholds
        )
        
        # 统计信息
        self.training_stats = {
            'total_training_steps': 0,
            'total_preference_pairs_processed': 0,
            'avg_training_loss': 0.0,
            'last_batch_size': 0,
            'high_quality_sample_ratio': 0.0,
            'quality_boost_usage': 0,
            'episodes_trained': 0,
            'buffer_utilization': 0.0,
            'quality_sampling_ratio': 0.0
        }
        
        logger.info(f"优先级偏好训练器初始化完成")
        logger.info(f"  - 缓冲池容量: {buffer_capacity}")
        logger.info(f"  - 批次大小: {batch_size}")
        logger.info(f"  - 最小缓冲池大小: {min_buffer_size}")
    
    def add_preference_pair(self, preference_pair: PreferencePair) -> None:
        """
        添加偏好对到经验回放池
        
        Args:
            preference_pair: 偏好对
        """
        self.experience_buffer.add(preference_pair)
        self.training_stats['total_preference_pairs_processed'] += 1
    
    def sample_batch(self) -> Tuple[List[PreferencePair], np.ndarray, np.ndarray]:
        """
        从经验回放池采样一个批次
        
        Returns:
            (偏好对列表, 采样索引, 重要性权重)
        """
        if not self.is_ready_for_training():
            return [], np.array([]), np.array([])
        
        sampled_pairs, indices, weights = self.experience_buffer.sample(self.batch_size)
        self.training_stats['last_batch_size'] = len(sampled_pairs)
        
        return sampled_pairs, indices, weights
    
    def update_priorities_with_losses(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """
        基于训练损失更新优先级
        
        Args:
            indices: 样本索引
            losses: 训练损失
        """
        self.experience_buffer.update_priorities(indices, losses)
        
        # 更新训练统计
        if len(losses) > 0:
            avg_loss = float(np.mean(losses))
            self.training_stats['total_training_steps'] += 1
            self.training_stats['avg_training_loss'] = (
                (self.training_stats['avg_training_loss'] * (self.training_stats['total_training_steps'] - 1) + avg_loss) /
                self.training_stats['total_training_steps']
            )
    
    def is_ready_for_training(self) -> bool:
        """
        检查是否准备好进行训练
        
        Returns:
            是否准备好训练
        """
        return self.experience_buffer.is_ready_for_sampling(self.min_buffer_size)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.training_stats.copy()
        buffer_stats = self.experience_buffer.get_statistics()
        stats['buffer_stats'] = buffer_stats
        
        # 计算高质量采样比例
        total_sampled = buffer_stats.get('total_sampled', 0)
        high_quality_sampled = buffer_stats.get('high_quality_samples', 0)
        quality_ratio = high_quality_sampled / total_sampled if total_sampled > 0 else 0.0
        
        # 更新训练统计信息
        self.training_stats['high_quality_samples'] = high_quality_sampled
        self.training_stats['quality_boost_applied'] = buffer_stats.get('quality_boost_applied', 0)
        self.training_stats['quality_sampling_ratio'] = quality_ratio
        self.training_stats['buffer_utilization'] = len(self.experience_buffer) / self.experience_buffer.capacity if self.experience_buffer.capacity > 0 else 0.0
        
        stats['high_quality_sample_ratio'] = quality_ratio
        stats['quality_boost_usage'] = buffer_stats.get('quality_boost_applied', 0)
        stats['enable_quality_sampling'] = self.enable_quality_sampling
        stats['high_quality_boost'] = self.high_quality_boost
        
        return stats
    
    def clear_buffer(self) -> None:
        """清空经验回放池"""
        self.experience_buffer.clear()
        logger.info("经验回放池已清空")

# 工厂函数
def create_prioritized_preference_trainer(**kwargs) -> PrioritizedPreferenceTrainer:
    """
    创建优先级偏好训练器的工厂函数
    
    Args:
        **kwargs: 训练器参数
        
    Returns:
        优先级偏好训练器实例
    """
    return PrioritizedPreferenceTrainer(**kwargs)

if __name__ == "__main__":
    # 测试代码
    print("测试优先级经验回放系统...")
    
    # 创建训练器
    trainer = create_prioritized_preference_trainer(
        buffer_capacity=1000,
        batch_size=32,
        min_buffer_size=200
    )
    
    # 创建测试偏好对
    for i in range(100):
        preference_pair = PreferencePair(
            trajectory_a={'obs': np.random.randn(50, 10), 'action': np.random.randn(50, 5)},
            trajectory_b={'obs': np.random.randn(45, 10), 'action': np.random.randn(45, 5)},
            preference_label=np.random.rand(),
            confidence_score=np.random.rand(),
            rule_score_diff=np.random.randn() * 5,
            timestamp=time.time() - np.random.rand() * 3600  # 随机时间戳
        )
        trainer.add_preference_pair(preference_pair)
    
    print(f"添加了100个偏好对到缓冲池")
    print(f"缓冲池大小: {len(trainer.experience_buffer)}")
    print(f"是否准备好训练: {trainer.is_ready_for_training()}")
    
    # 测试采样
    if trainer.is_ready_for_training():
        sampled_pairs, indices, weights = trainer.sample_batch()
        print(f"采样批次大小: {len(sampled_pairs)}")
        print(f"重要性权重范围: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
        
        # 模拟训练损失并更新优先级
        fake_losses = np.random.rand(len(indices)) * 2.0
        trainer.update_priorities_with_losses(indices, fake_losses)
        print(f"已更新 {len(indices)} 个样本的优先级")
    
    # 获取统计信息
    stats = trainer.get_statistics()
    print(f"\n训练器统计信息:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print("测试完成！")