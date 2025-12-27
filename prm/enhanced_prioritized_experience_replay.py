#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版优先级经验回放系统 (Enhanced Prioritized Experience Replay System)

在原有基础上增强高质量经验的采样权重和频率，包括：
1. 高质量经验识别机制：基于多维度指标识别高质量经验
2. 自适应采样策略：动态调整高质量经验的采样概率
3. 质量感知优先级计算：更精确地评估经验质量
4. 分层采样机制：对不同质量层级的经验采用不同采样策略
5. 经验质量追踪：持续监控和更新经验质量评估

作者：AI Assistant
日期：2025-01-26
版本：2.0.0
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import threading
from abc import ABC, abstractmethod
import heapq
import random
from enum import Enum
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperienceQuality(Enum):
    """经验质量等级"""
    EXCELLENT = "excellent"    # 优秀经验 (top 10%)
    HIGH = "high"             # 高质量经验 (10%-30%)
    MEDIUM = "medium"         # 中等质量经验 (30%-70%)
    LOW = "low"               # 低质量经验 (70%-90%)
    POOR = "poor"             # 差质量经验 (bottom 10%)

class SamplingStrategy(Enum):
    """采样策略类型"""
    QUALITY_AWARE = "quality_aware"        # 质量感知采样
    STRATIFIED = "stratified"              # 分层采样
    ADAPTIVE = "adaptive"                  # 自适应采样
    HYBRID = "hybrid"                      # 混合采样

@dataclass
class EnhancedPreferencePair:
    """增强版偏好对数据结构"""
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
    
    # 新增质量相关字段
    quality_score: float = 0.0          # 综合质量分数 [0, 1]
    quality_level: ExperienceQuality = ExperienceQuality.MEDIUM  # 质量等级
    learning_value: float = 0.0         # 学习价值评估
    diversity_score: float = 0.0        # 多样性分数
    stability_score: float = 0.0        # 稳定性分数
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp == 0:
            self.timestamp = time.time()

class QualityAssessment:
    """经验质量评估器"""
    
    def __init__(self, 
                 confidence_weight: float = 0.3,
                 rule_diff_weight: float = 0.25,
                 learning_value_weight: float = 0.2,
                 diversity_weight: float = 0.15,
                 stability_weight: float = 0.1):
        """
        初始化质量评估器
        
        Args:
            confidence_weight: 置信度权重
            rule_diff_weight: 规则差异权重
            learning_value_weight: 学习价值权重
            diversity_weight: 多样性权重
            stability_weight: 稳定性权重
        """
        self.confidence_weight = confidence_weight
        self.rule_diff_weight = rule_diff_weight
        self.learning_value_weight = learning_value_weight
        self.diversity_weight = diversity_weight
        self.stability_weight = stability_weight
        
        # 质量统计
        self.quality_stats = {
            'total_assessments': 0,
            'quality_distribution': defaultdict(int),
            'avg_quality_score': 0.0
        }
    
    def assess_quality(self, preference_pair: EnhancedPreferencePair) -> Tuple[float, ExperienceQuality]:
        """
        评估偏好对的质量
        
        Args:
            preference_pair: 偏好对
            
        Returns:
            (质量分数, 质量等级)
        """
        # 1. 置信度评估
        confidence_component = preference_pair.confidence_score
        
        # 2. 规则差异评估
        rule_diff = abs(preference_pair.rule_score_diff)
        max_rule_diff = 10.0
        rule_diff_component = min(rule_diff / max_rule_diff, 1.0)
        
        # 3. 学习价值评估（基于损失变化）
        learning_value_component = self._assess_learning_value(preference_pair)
        
        # 4. 多样性评估
        diversity_component = self._assess_diversity(preference_pair)
        
        # 5. 稳定性评估
        stability_component = self._assess_stability(preference_pair)
        
        # 综合质量分数
        quality_score = (
            self.confidence_weight * confidence_component +
            self.rule_diff_weight * rule_diff_component +
            self.learning_value_weight * learning_value_component +
            self.diversity_weight * diversity_component +
            self.stability_weight * stability_component
        )
        
        # 应用非线性增强
        quality_score = self._apply_quality_enhancement(quality_score)
        
        # 确定质量等级
        quality_level = self._determine_quality_level(quality_score)
        
        # 更新统计信息
        self._update_stats(quality_score, quality_level)
        
        return quality_score, quality_level
    
    def _assess_learning_value(self, preference_pair: EnhancedPreferencePair) -> float:
        """评估学习价值"""
        if preference_pair.last_loss is None:
            return 0.5  # 默认中等学习价值
        
        # 基于损失大小评估学习价值
        # 中等损失通常有更好的学习价值
        loss = preference_pair.last_loss
        if 0.1 <= loss <= 1.0:
            return 0.8  # 高学习价值
        elif 0.05 <= loss < 0.1 or 1.0 < loss <= 2.0:
            return 0.6  # 中等学习价值
        else:
            return 0.3  # 低学习价值
    
    def _assess_diversity(self, preference_pair: EnhancedPreferencePair) -> float:
        """评估多样性"""
        # 基于轨迹长度差异和动作多样性评估
        traj_a_len = len(preference_pair.trajectory_a.get('obs', []))
        traj_b_len = len(preference_pair.trajectory_b.get('obs', []))
        
        # 长度多样性
        length_diversity = abs(traj_a_len - traj_b_len) / max(traj_a_len, traj_b_len, 1)
        
        # 动作多样性（如果有动作数据）
        action_diversity = 0.5  # 默认值
        if 'action' in preference_pair.trajectory_a and 'action' in preference_pair.trajectory_b:
            actions_a = preference_pair.trajectory_a['action']
            actions_b = preference_pair.trajectory_b['action']
            if len(actions_a) > 0 and len(actions_b) > 0:
                action_std_a = np.std(actions_a)
                action_std_b = np.std(actions_b)
                action_diversity = abs(action_std_a - action_std_b) / (action_std_a + action_std_b + 1e-6)
        
        return (length_diversity + action_diversity) / 2.0
    
    def _assess_stability(self, preference_pair: EnhancedPreferencePair) -> float:
        """评估稳定性"""
        # 基于采样次数和时间稳定性评估
        sample_stability = min(preference_pair.sample_count / 10.0, 1.0)  # 被采样越多越稳定
        
        # 时间稳定性（较新的数据可能不够稳定）
        age = time.time() - preference_pair.timestamp
        time_stability = min(age / 3600.0, 1.0)  # 1小时后认为稳定
        
        return (sample_stability + time_stability) / 2.0
    
    def _apply_quality_enhancement(self, quality_score: float) -> float:
        """应用质量增强函数"""
        # 使用sigmoid函数增强质量区分度
        enhanced_score = 1.0 / (1.0 + np.exp(-10 * (quality_score - 0.5)))
        return float(enhanced_score)
    
    def _determine_quality_level(self, quality_score: float) -> ExperienceQuality:
        """确定质量等级"""
        if quality_score >= 0.8:
            return ExperienceQuality.EXCELLENT
        elif quality_score >= 0.6:
            return ExperienceQuality.HIGH
        elif quality_score >= 0.4:
            return ExperienceQuality.MEDIUM
        elif quality_score >= 0.2:
            return ExperienceQuality.LOW
        else:
            return ExperienceQuality.POOR
    
    def _update_stats(self, quality_score: float, quality_level: ExperienceQuality):
        """更新统计信息"""
        self.quality_stats['total_assessments'] += 1
        self.quality_stats['quality_distribution'][quality_level.value] += 1
        
        # 更新平均质量分数
        total = self.quality_stats['total_assessments']
        current_avg = self.quality_stats['avg_quality_score']
        self.quality_stats['avg_quality_score'] = (current_avg * (total - 1) + quality_score) / total

class EnhancedPriorityCalculator:
    """增强版优先级计算器"""
    
    def __init__(self, 
                 quality_weight: float = 0.5,
                 confidence_weight: float = 0.3,
                 temporal_weight: float = 0.2,
                 quality_boost_factor: float = 2.0,
                 min_priority: float = 1e-6,
                 max_priority: float = 10.0):
        """
        初始化增强版优先级计算器
        
        Args:
            quality_weight: 质量权重
            confidence_weight: 置信度权重
            temporal_weight: 时间权重
            quality_boost_factor: 高质量经验的提升因子
            min_priority: 最小优先级
            max_priority: 最大优先级
        """
        self.quality_weight = quality_weight
        self.confidence_weight = confidence_weight
        self.temporal_weight = temporal_weight
        self.quality_boost_factor = quality_boost_factor
        self.min_priority = min_priority
        self.max_priority = max_priority
        
        # 质量评估器
        self.quality_assessor = QualityAssessment()
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'quality_priority_distribution': defaultdict(list),
            'avg_priority_by_quality': defaultdict(float)
        }
    
    def calculate_enhanced_priority(self, preference_pair: EnhancedPreferencePair, 
                                  current_time: Optional[float] = None) -> float:
        """
        计算增强版优先级
        
        Args:
            preference_pair: 偏好对
            current_time: 当前时间
            
        Returns:
            增强版优先级
        """
        # 1. 评估质量
        quality_score, quality_level = self.quality_assessor.assess_quality(preference_pair)
        preference_pair.quality_score = quality_score
        preference_pair.quality_level = quality_level
        
        # 2. 计算基础优先级组件
        quality_priority = quality_score
        confidence_priority = preference_pair.confidence_score
        temporal_priority = self._calculate_temporal_priority(preference_pair, current_time)
        
        # 3. 综合优先级
        base_priority = (
            self.quality_weight * quality_priority +
            self.confidence_weight * confidence_priority +
            self.temporal_weight * temporal_priority
        )
        
        # 4. 应用质量提升
        enhanced_priority = self._apply_quality_boost(base_priority, quality_level)
        
        # 5. 确保在合理范围内
        enhanced_priority = np.clip(enhanced_priority, self.min_priority, self.max_priority)
        
        # 6. 更新统计信息
        self._update_priority_stats(enhanced_priority, quality_level)
        
        return float(enhanced_priority)
    
    def _calculate_temporal_priority(self, preference_pair: EnhancedPreferencePair, 
                                   current_time: Optional[float] = None) -> float:
        """计算时间优先级"""
        if current_time is None:
            current_time = time.time()
        
        age = current_time - preference_pair.timestamp
        # 使用指数衰减，但不要过于激进
        temporal_priority = np.exp(-age / 7200.0)  # 2小时半衰期
        return float(temporal_priority)
    
    def _apply_quality_boost(self, base_priority: float, quality_level: ExperienceQuality) -> float:
        """应用质量提升"""
        boost_factors = {
            ExperienceQuality.EXCELLENT: self.quality_boost_factor * 2.0,  # 4倍提升
            ExperienceQuality.HIGH: self.quality_boost_factor * 1.5,       # 3倍提升
            ExperienceQuality.MEDIUM: 1.0,                                 # 无提升
            ExperienceQuality.LOW: 0.7,                                    # 轻微降低
            ExperienceQuality.POOR: 0.5                                    # 显著降低
        }
        
        boost_factor = boost_factors.get(quality_level, 1.0)
        return base_priority * boost_factor
    
    def _update_priority_stats(self, priority: float, quality_level: ExperienceQuality):
        """更新优先级统计信息"""
        self.stats['total_calculations'] += 1
        self.stats['quality_priority_distribution'][quality_level.value].append(priority)
        
        # 更新平均优先级
        priorities = self.stats['quality_priority_distribution'][quality_level.value]
        self.stats['avg_priority_by_quality'][quality_level.value] = np.mean(priorities)

class EnhancedSampler:
    """增强版采样器"""
    
    def __init__(self, 
                 strategy: SamplingStrategy = SamplingStrategy.HYBRID,
                 high_quality_boost: float = 3.0,
                 stratified_ratios: Optional[Dict[str, float]] = None):
        """
        初始化增强版采样器
        
        Args:
            strategy: 采样策略
            high_quality_boost: 高质量经验的采样提升
            stratified_ratios: 分层采样比例
        """
        self.strategy = strategy
        self.high_quality_boost = high_quality_boost
        
        # 默认分层采样比例
        self.stratified_ratios = stratified_ratios or {
            ExperienceQuality.EXCELLENT.value: 0.3,  # 30%来自优秀经验
            ExperienceQuality.HIGH.value: 0.3,       # 30%来自高质量经验
            ExperienceQuality.MEDIUM.value: 0.25,    # 25%来自中等质量经验
            ExperienceQuality.LOW.value: 0.1,        # 10%来自低质量经验
            ExperienceQuality.POOR.value: 0.05       # 5%来自差质量经验
        }
        
        # 统计信息
        self.sampling_stats = {
            'total_samples': 0,
            'samples_by_quality': defaultdict(int),
            'samples_by_strategy': defaultdict(int)
        }
    
    def enhanced_sample(self, 
                       buffer: List[EnhancedPreferencePair], 
                       priorities: np.ndarray,
                       batch_size: int,
                       beta: float = 0.4) -> Tuple[List[EnhancedPreferencePair], np.ndarray, np.ndarray]:
        """
        增强版采样方法
        
        Args:
            buffer: 经验缓冲池
            priorities: 优先级数组
            batch_size: 批次大小
            beta: 重要性采样参数
            
        Returns:
            (采样的偏好对列表, 采样索引, 重要性权重)
        """
        if len(buffer) == 0:
            return [], np.array([]), np.array([])
        
        if self.strategy == SamplingStrategy.QUALITY_AWARE:
            return self._quality_aware_sample(buffer, priorities, batch_size, beta)
        elif self.strategy == SamplingStrategy.STRATIFIED:
            return self._stratified_sample(buffer, priorities, batch_size, beta)
        elif self.strategy == SamplingStrategy.ADAPTIVE:
            return self._adaptive_sample(buffer, priorities, batch_size, beta)
        elif self.strategy == SamplingStrategy.HYBRID:
            return self._hybrid_sample(buffer, priorities, batch_size, beta)
        else:
            # 默认使用质量感知采样
            return self._quality_aware_sample(buffer, priorities, batch_size, beta)
    
    def _quality_aware_sample(self, buffer: List[EnhancedPreferencePair], 
                            priorities: np.ndarray, batch_size: int, beta: float):
        """质量感知采样"""
        # 根据质量等级调整采样概率
        adjusted_priorities = priorities.copy()
        
        for i, pair in enumerate(buffer):
            if pair.quality_level in [ExperienceQuality.EXCELLENT, ExperienceQuality.HIGH]:
                adjusted_priorities[i] *= self.high_quality_boost
        
        # 标准优先级采样
        probabilities = adjusted_priorities / np.sum(adjusted_priorities)
        indices = np.random.choice(len(buffer), size=min(batch_size, len(buffer)), 
                                 p=probabilities, replace=False)
        
        sampled_pairs = [buffer[i] for i in indices]
        weights = (len(buffer) * probabilities[indices]) ** (-beta)
        weights = weights / np.max(weights)
        
        self._update_sampling_stats(sampled_pairs, 'quality_aware')
        return sampled_pairs, indices, weights
    
    def _stratified_sample(self, buffer: List[EnhancedPreferencePair], 
                          priorities: np.ndarray, batch_size: int, beta: float):
        """分层采样"""
        # 按质量等级分组
        quality_groups = defaultdict(list)
        for i, pair in enumerate(buffer):
            quality_groups[pair.quality_level.value].append((i, pair, priorities[i]))
        
        sampled_pairs = []
        indices = []
        weights = []
        
        # 按比例从各层采样
        for quality_level, ratio in self.stratified_ratios.items():
            if quality_level in quality_groups:
                group_size = max(1, int(batch_size * ratio))
                group_data = quality_groups[quality_level]
                
                if len(group_data) > 0:
                    # 从该质量层采样
                    group_indices, group_pairs, group_priorities = zip(*group_data)
                    group_priorities = np.array(group_priorities)
                    
                    if len(group_data) <= group_size:
                        # 全部采样
                        selected_indices = list(range(len(group_data)))
                    else:
                        # 按优先级采样
                        probs = group_priorities / np.sum(group_priorities)
                        selected_indices = np.random.choice(len(group_data), size=group_size, 
                                                          p=probs, replace=False)
                    
                    for idx in selected_indices:
                        original_idx = group_indices[idx]
                        pair = group_pairs[idx]
                        priority = group_priorities[idx]
                        
                        sampled_pairs.append(pair)
                        indices.append(original_idx)
                        
                        # 计算重要性权重
                        prob = priority / np.sum(priorities)
                        weight = (len(buffer) * prob) ** (-beta)
                        weights.append(weight)
        
        indices = np.array(indices)
        weights = np.array(weights)
        if len(weights) > 0:
            weights = weights / np.max(weights)
        
        self._update_sampling_stats(sampled_pairs, 'stratified')
        return sampled_pairs, indices, weights
    
    def _adaptive_sample(self, buffer: List[EnhancedPreferencePair], 
                        priorities: np.ndarray, batch_size: int, beta: float):
        """自适应采样"""
        # 根据当前质量分布自适应调整采样策略
        quality_counts = defaultdict(int)
        for pair in buffer:
            quality_counts[pair.quality_level.value] += 1
        
        total_count = len(buffer)
        high_quality_ratio = (quality_counts[ExperienceQuality.EXCELLENT.value] + 
                            quality_counts[ExperienceQuality.HIGH.value]) / total_count
        
        # 如果高质量经验较少，增加其采样权重
        if high_quality_ratio < 0.2:
            boost_factor = self.high_quality_boost * 2.0
        elif high_quality_ratio < 0.4:
            boost_factor = self.high_quality_boost * 1.5
        else:
            boost_factor = self.high_quality_boost
        
        # 应用自适应权重
        adjusted_priorities = priorities.copy()
        for i, pair in enumerate(buffer):
            if pair.quality_level in [ExperienceQuality.EXCELLENT, ExperienceQuality.HIGH]:
                adjusted_priorities[i] *= boost_factor
        
        # 采样
        probabilities = adjusted_priorities / np.sum(adjusted_priorities)
        indices = np.random.choice(len(buffer), size=min(batch_size, len(buffer)), 
                                 p=probabilities, replace=False)
        
        sampled_pairs = [buffer[i] for i in indices]
        weights = (len(buffer) * probabilities[indices]) ** (-beta)
        weights = weights / np.max(weights)
        
        self._update_sampling_stats(sampled_pairs, 'adaptive')
        return sampled_pairs, indices, weights
    
    def _hybrid_sample(self, buffer: List[EnhancedPreferencePair], 
                      priorities: np.ndarray, batch_size: int, beta: float):
        """混合采样策略"""
        # 50%使用质量感知采样，50%使用分层采样
        half_batch = batch_size // 2
        
        # 质量感知采样
        qa_pairs, qa_indices, qa_weights = self._quality_aware_sample(
            buffer, priorities, half_batch, beta)
        
        # 分层采样（排除已采样的）
        remaining_buffer = [pair for i, pair in enumerate(buffer) if i not in qa_indices]
        remaining_priorities = np.array([priorities[i] for i in range(len(buffer)) 
                                       if i not in qa_indices])
        
        if len(remaining_buffer) > 0:
            st_pairs, st_indices_rel, st_weights = self._stratified_sample(
                remaining_buffer, remaining_priorities, batch_size - len(qa_pairs), beta)
            
            # 转换相对索引为绝对索引
            remaining_indices = [i for i in range(len(buffer)) if i not in qa_indices]
            st_indices = np.array([remaining_indices[i] for i in st_indices_rel])
        else:
            st_pairs, st_indices, st_weights = [], np.array([]), np.array([])
        
        # 合并结果
        sampled_pairs = qa_pairs + st_pairs
        indices = np.concatenate([qa_indices, st_indices]) if len(st_indices) > 0 else qa_indices
        weights = np.concatenate([qa_weights, st_weights]) if len(st_weights) > 0 else qa_weights
        
        self._update_sampling_stats(sampled_pairs, 'hybrid')
        return sampled_pairs, indices, weights
    
    def _update_sampling_stats(self, sampled_pairs: List[EnhancedPreferencePair], strategy: str):
        """更新采样统计信息"""
        self.sampling_stats['total_samples'] += len(sampled_pairs)
        self.sampling_stats['samples_by_strategy'][strategy] += len(sampled_pairs)
        
        for pair in sampled_pairs:
            self.sampling_stats['samples_by_quality'][pair.quality_level.value] += 1

class EnhancedPrioritizedExperienceBuffer:
    """增强版优先级经验回放缓冲池"""
    
    def __init__(self, 
                 capacity: int = 10000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6,
                 sampling_strategy: SamplingStrategy = SamplingStrategy.HYBRID):
        """
        初始化增强版缓冲池
        
        Args:
            capacity: 缓冲池容量
            alpha: 优先级指数
            beta: 重要性采样指数
            beta_increment: beta增长率
            epsilon: 小常数
            sampling_strategy: 采样策略
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 缓冲池
        self.buffer: List[EnhancedPreferencePair] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        
        # 增强组件
        self.priority_calculator = EnhancedPriorityCalculator()
        self.sampler = EnhancedSampler(strategy=sampling_strategy)
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_added': 0,
            'total_sampled': 0,
            'priority_updates': 0,
            'quality_distribution': defaultdict(int)
        }
    
    def add(self, preference_pair: EnhancedPreferencePair) -> None:
        """添加偏好对到缓冲池"""
        with self.lock:
            try:
                # 计算增强优先级
                priority = self.priority_calculator.calculate_enhanced_priority(preference_pair)
                preference_pair.priority = priority
                
                # 添加到缓冲池
                if self.size < self.capacity:
                    self.buffer.append(preference_pair)
                    self.priorities[self.size] = (priority + self.epsilon) ** self.alpha
                    self.size += 1
                else:
                    # 替换最旧的经验
                    self.buffer[self.position] = preference_pair
                    self.priorities[self.position] = (priority + self.epsilon) ** self.alpha
                    self.position = (self.position + 1) % self.capacity
                
                # 更新统计信息
                self.stats['total_added'] += 1
                self.stats['quality_distribution'][preference_pair.quality_level.value] += 1
                
            except Exception as e:
                logger.error(f"[增强经验回放池] 添加偏好对时发生异常: {e}")
                raise e
    
    def sample(self, batch_size: int) -> Tuple[List[EnhancedPreferencePair], np.ndarray, np.ndarray]:
        """从缓冲池中采样一个批次"""
        with self.lock:
            if self.size == 0:
                return [], np.array([]), np.array([])
            
            # 使用增强采样器
            current_buffer = self.buffer[:self.size]
            current_priorities = self.priorities[:self.size]
            
            sampled_pairs, indices, weights = self.sampler.enhanced_sample(
                current_buffer, current_priorities, batch_size, self.beta)
            
            # 更新beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # 更新采样统计
            for pair in sampled_pairs:
                pair.sample_count += 1
            
            self.stats['total_sampled'] += len(sampled_pairs)
            
            return sampled_pairs, indices, weights
    
    def update_priorities(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """基于训练损失更新优先级"""
        with self.lock:
            for idx, loss in zip(indices, losses):
                if idx < self.size:
                    preference_pair = self.buffer[idx]
                    
                    # 重新计算增强优先级
                    preference_pair.last_loss = float(loss)
                    new_priority = self.priority_calculator.calculate_enhanced_priority(preference_pair)
                    
                    # 更新缓冲池中的优先级
                    self.priorities[idx] = (new_priority + self.epsilon) ** self.alpha
                    preference_pair.priority = new_priority
                    
                    self.stats['priority_updates'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            stats = self.stats.copy()
            stats['buffer_size'] = self.size
            stats['buffer_capacity'] = self.capacity
            stats['current_beta'] = self.beta
            
            # 添加质量分布统计
            if self.size > 0:
                quality_counts = defaultdict(int)
                for pair in self.buffer[:self.size]:
                    quality_counts[pair.quality_level.value] += 1
                stats['current_quality_distribution'] = dict(quality_counts)
            
            # 添加优先级计算器和采样器统计
            stats['priority_calculator_stats'] = self.priority_calculator.stats
            stats['sampler_stats'] = self.sampler.sampling_stats
            
            return stats
    
    def clear(self) -> None:
        """清空缓冲池"""
        with self.lock:
            self.buffer.clear()
            self.priorities.fill(0)
            self.size = 0
            self.position = 0
            
            # 重置统计信息
            self.stats = {
                'total_added': 0,
                'total_sampled': 0,
                'priority_updates': 0,
                'quality_distribution': defaultdict(int)
            }
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready_for_sampling(self, min_size: int = 200) -> bool:
        return self.size >= min_size

def create_enhanced_prioritized_trainer(**kwargs) -> 'EnhancedPrioritizedPreferenceTrainer':
    """
    创建增强版优先级偏好训练器的工厂函数
    
    Args:
        **kwargs: 训练器参数
        
    Returns:
        增强版优先级偏好训练器实例
    """
    return EnhancedPrioritizedPreferenceTrainer(**kwargs)

class EnhancedPrioritizedPreferenceTrainer:
    """增强版优先级偏好训练器"""
    
    def __init__(self,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 min_buffer_size: int = 200,
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4,
                 priority_beta_increment: float = 0.001,
                 sampling_strategy: str = "hybrid"):
        """
        初始化增强版训练器
        
        Args:
            buffer_capacity: 缓冲池容量
            batch_size: 批次大小
            min_buffer_size: 最小缓冲池大小
            priority_alpha: 优先级指数
            priority_beta: 重要性采样指数
            priority_beta_increment: beta增长率
            sampling_strategy: 采样策略
        """
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        
        # 转换采样策略
        strategy_map = {
            "quality_aware": SamplingStrategy.QUALITY_AWARE,
            "stratified": SamplingStrategy.STRATIFIED,
            "adaptive": SamplingStrategy.ADAPTIVE,
            "hybrid": SamplingStrategy.HYBRID
        }
        strategy = strategy_map.get(sampling_strategy, SamplingStrategy.HYBRID)
        
        # 创建增强版经验回放缓冲池
        self.experience_buffer = EnhancedPrioritizedExperienceBuffer(
            capacity=buffer_capacity,
            alpha=priority_alpha,
            beta=priority_beta,
            beta_increment=priority_beta_increment,
            sampling_strategy=strategy
        )
        
        # 统计信息
        self.training_stats = {
            'total_training_steps': 0,
            'total_preference_pairs': 0,
            'avg_batch_quality': 0.0
        }
    
    def add_preference_pair(self, preference_pair_data: Dict[str, Any]) -> None:
        """添加偏好对"""
        # 转换为增强版偏好对
        enhanced_pair = EnhancedPreferencePair(
            trajectory_a=preference_pair_data['trajectory_a'],
            trajectory_b=preference_pair_data['trajectory_b'],
            preference_label=preference_pair_data['preference_label'],
            confidence_score=preference_pair_data['confidence_score'],
            rule_score_diff=preference_pair_data['rule_score_diff'],
            timestamp=preference_pair_data.get('timestamp', time.time()),
            metadata=preference_pair_data.get('metadata', {})
        )
        
        self.experience_buffer.add(enhanced_pair)
        self.training_stats['total_preference_pairs'] += 1
    
    def sample_batch(self) -> Tuple[List[EnhancedPreferencePair], np.ndarray, np.ndarray]:
        """采样训练批次"""
        if not self.is_ready_for_training():
            return [], np.array([]), np.array([])
        
        sampled_pairs, indices, weights = self.experience_buffer.sample(self.batch_size)
        
        # 计算批次质量
        if sampled_pairs:
            batch_quality = np.mean([pair.quality_score for pair in sampled_pairs])
            self.training_stats['avg_batch_quality'] = batch_quality
        
        return sampled_pairs, indices, weights
    
    def update_priorities_with_losses(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """基于损失更新优先级"""
        self.experience_buffer.update_priorities(indices, losses)
        self.training_stats['total_training_steps'] += 1
    
    def is_ready_for_training(self) -> bool:
        """检查是否准备好训练"""
        return len(self.experience_buffer) >= self.min_buffer_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.training_stats.copy()
        stats['buffer_stats'] = self.experience_buffer.get_statistics()
        return stats
    
    def clear_buffer(self) -> None:
        """清空缓冲池"""
        self.experience_buffer.clear()
        self.training_stats = {
            'total_training_steps': 0,
            'total_preference_pairs': 0,
            'avg_batch_quality': 0.0
        }

if __name__ == "__main__":
    # 测试增强版优先级经验回放系统
    print("测试增强版优先级经验回放系统...")
    
    # 创建训练器
    trainer = create_enhanced_prioritized_trainer(
        buffer_capacity=1000,
        batch_size=32,
        min_buffer_size=100,
        sampling_strategy="hybrid"
    )
    
    # 添加测试数据
    for i in range(200):
        preference_pair_data = {
            'trajectory_a': {'obs': np.random.randn(50, 10), 'action': np.random.randn(50, 5)},
            'trajectory_b': {'obs': np.random.randn(45, 10), 'action': np.random.randn(45, 5)},
            'preference_label': np.random.rand(),
            'confidence_score': np.random.rand(),
            'rule_score_diff': np.random.randn() * 5,
            'timestamp': time.time() - np.random.rand() * 3600,
            'metadata': {
                'env_reward_diff': np.random.rand() * 20,
                'pair_type': np.random.choice(['high_quality', 'medium_quality', 'exploration'])
            }
        }
        trainer.add_preference_pair(preference_pair_data)
    
    print(f"添加了200个偏好对到缓冲池")
    print(f"缓冲池大小: {len(trainer.experience_buffer)}")
    print(f"是否准备好训练: {trainer.is_ready_for_training()}")
    
    # 测试采样
    if trainer.is_ready_for_training():
        sampled_pairs, indices, weights = trainer.sample_batch()
        print(f"采样批次大小: {len(sampled_pairs)}")
        print(f"重要性权重范围: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
        
        # 分析采样质量分布
        quality_counts = defaultdict(int)
        for pair in sampled_pairs:
            quality_counts[pair.quality_level.value] += 1
        print(f"采样质量分布: {dict(quality_counts)}")
        
        # 模拟训练损失更新
        fake_losses = np.random.rand(len(indices)) * 2.0
        trainer.update_priorities_with_losses(indices, fake_losses)
        print(f"已更新 {len(indices)} 个样本的优先级")
    
    # 显示统计信息
    stats = trainer.get_statistics()
    print(f"\n增强版训练器统计信息:")
    print(f"  总训练步数: {stats['total_training_steps']}")
    print(f"  总偏好对数: {stats['total_preference_pairs']}")
    print(f"  平均批次质量: {stats['avg_batch_quality']:.3f}")
    
    buffer_stats = stats['buffer_stats']
    print(f"\n缓冲池统计信息:")
    print(f"  当前大小: {buffer_stats['buffer_size']}")
    print(f"  总添加数: {buffer_stats['total_added']}")
    print(f"  总采样数: {buffer_stats['total_sampled']}")
    print(f"  优先级更新数: {buffer_stats['priority_updates']}")
    
    if 'current_quality_distribution' in buffer_stats:
        print(f"  当前质量分布: {buffer_stats['current_quality_distribution']}")
    
    print("测试完成！")