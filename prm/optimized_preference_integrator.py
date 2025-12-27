#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的偏好集成器 (Optimized Preference Integrator)

改进点：
1. 更智能的奖励融合策略
2. 动态权重调整机制
3. 性能驱动的权重优化
4. 改进的缓存机制
5. 更好的数值稳定性

作者：AI Assistant
日期：2025-01-11
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import hashlib
import time
from collections import deque, defaultdict
import threading

# 延迟导入以避免循环依赖
# from .optimized_preference_trainer import OptimizedPreferenceTrainer
# from .optimized_latent_preference_model import OptimizedLatentPreferenceModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedIntegrationConfig:
    """优化集成配置"""
    # 权重配置
    initial_preference_weight: float = 0.4  # 初始偏好权重（调整到中间值）
    initial_environment_weight: float = 0.6  # 初始环境权重
    
    # 权重调整参数
    min_preference_weight: float = 0.2  # 最小偏好权重
    max_preference_weight: float = 0.7  # 最大偏好权重
    
    # 置信度和不确定性
    confidence_threshold_low: float = 0.2   # 低置信度阈值
    confidence_threshold_high: float = 0.6  # 高置信度阈值
    confidence_sensitivity: float = 0.8     # 置信度敏感性
    uncertainty_penalty: float = 0.08       # 不确定性惩罚
    weight_update_rate: float = 0.05        # 权重更新率
    enable_confidence_weighting: bool = True # 启用基于置信度的权重调整
    
    # 性能驱动的权重调整
    enable_performance_weighting: bool = True
    performance_window_size: int = 10
    reward_improvement_threshold: float = 0.01
    performance_weight_factor: float = 0.5
    
    # 奖励归一化
    reward_normalization_method: str = "adaptive_sigmoid"  # "tanh", "sigmoid", "adaptive_sigmoid"
    reward_clip_range: Tuple[float, float] = (-3.000, 3000.0)  # 扩大裁剪范围以支持大范围环境奖励
    adaptive_scaling_factor: float = 2.0  # 减小缩放因子以提高稳定性
    
    # 缓存配置
    enable_caching: bool = True
    max_cache_size: int = 10000
    cache_batch_size: int = 128
    
    # 在线学习
    enable_online_learning: bool = True
    online_buffer_size: int = 1000
    
    # 数值稳定性
    epsilon: float = 1e-8
    gradient_clip_value: float = 1.0
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.baseline_performance = 0.0
        self.performance_trend = 0.0
        self.lock = threading.Lock()
    
    def update(self, reward: float) -> float:
        """更新性能并返回改善情况
        
        Args:
            reward: 当前奖励
            
        Returns:
            性能改善值（正值表示改善，负值表示恶化）
        """
        with self.lock:
            self.reward_history.append(reward)
            
            if len(self.reward_history) < 6:  # 需要至少6个数据点才能有效比较
                return 0.0
            
            # 计算当前性能（最近3个的平均）
            current_performance = np.mean(list(self.reward_history)[-3:])
            
            # 计算基线性能（前面数据的平均，确保不与当前性能重叠）
            if len(self.reward_history) >= 10:
                # 有足够数据时，使用前面一半作为基线
                baseline_data = list(self.reward_history)[:-3]
                self.baseline_performance = np.mean(baseline_data[-5:])  # 基线也用最近5个
            else:
                # 数据较少时，使用除最近3个外的所有数据
                baseline_data = list(self.reward_history)[:-3]
                if len(baseline_data) > 0:
                    self.baseline_performance = np.mean(baseline_data)
                else:
                    self.baseline_performance = current_performance
            
            # 计算改善（添加最小阈值避免微小波动）
            improvement = current_performance - self.baseline_performance
            if abs(improvement) < 0.1:  # 小于0.1的变化认为是噪声
                improvement = 0.0
            
            # 更新趋势
            if len(self.reward_history) >= 3:
                recent_rewards = list(self.reward_history)[-3:]
                self.performance_trend = (recent_rewards[-1] - recent_rewards[0]) / 2.0
            
            return improvement
    
    def get_trend(self) -> float:
        """获取性能趋势"""
        return self.performance_trend
    
    def get_baseline(self) -> float:
        """获取基线性能"""
        return self.baseline_performance

class OptimizedPreferenceIntegrator:
    """优化的偏好集成器"""
    
    def __init__(self, 
                 trainer: 'OptimizedPreferenceTrainer',
                 config: OptimizedIntegrationConfig):
        self.trainer = trainer
        self.config = config
        
        # 当前权重
        self.current_preference_weight = config.initial_preference_weight
        self.current_environment_weight = config.initial_environment_weight
        
        # 性能跟踪
        if config.enable_performance_weighting:
            self.performance_tracker = PerformanceTracker(config.performance_window_size)
        else:
            self.performance_tracker = None
        
        # 缓存系统
        if config.enable_caching:
            self.reward_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
        
        # 在线学习缓冲区
        if config.enable_online_learning:
            self.online_buffer = deque(maxlen=config.online_buffer_size)
        
        # 统计信息
        self.stats = {
            'total_calls': 0,
            'cache_hit_rate': 0.0,
            'avg_preference_weight': config.initial_preference_weight,
            'avg_environment_weight': config.initial_environment_weight,
            'avg_confidence': 0.0,
            'avg_preference_reward': 0.0,
            'avg_environment_reward': 0.0,
            'weight_updates': 0,
            'performance_improvements': 0
        }
        
        # 权重历史（用于分析）
        self.weight_history = deque(maxlen=50)
        
        # 轨迹级别的偏好统计
        self.trajectory_stats = {
            'positive_preference_count': 0,
            'negative_preference_count': 0,
            'total_steps': 0,
            'avg_confidence': 0.0
        }
        
        # logger.info(f"[优化集成器] 初始化完成")
        # logger.info(f"  - 初始偏好权重: {config.initial_preference_weight}")
        # logger.info(f"  - 初始环境权重: {config.initial_environment_weight}")
        # logger.info(f"  - 性能驱动权重: {config.enable_performance_weighting}")
        # logger.info(f"  - 缓存启用: {config.enable_caching}")
    
    def reset_trajectory_stats(self):
        """重置轨迹级别的统计信息"""
        self.trajectory_stats = {
            'positive_preference_count': 0,
            'negative_preference_count': 0,
            'total_steps': 0,
            'avg_confidence': 0.0
        }
    
    def _build_cache_key(self, latent_state: torch.Tensor, action: torch.Tensor) -> str:
        """构建缓存键
        
        Args:
            latent_state: 潜空间状态
            action: 动作
            
        Returns:
            缓存键
        """
        try:
            # 安全地将tensor转换为CPU，避免CUDA初始化问题
            if latent_state.is_cuda:
                state_data = latent_state.detach().cpu().flatten()[:10].tolist()
            else:
                state_data = latent_state.detach().flatten()[:10].tolist()
                
            if action.is_cuda:
                action_data = action.detach().cpu().flatten()[:10].tolist()
            else:
                action_data = action.detach().flatten()[:10].tolist()
                
            state_str = str(state_data)
            action_str = str(action_data)
        except Exception as e:
            # 如果tensor操作失败，使用tensor的形状和设备信息作为备用键
            state_str = f"shape_{latent_state.shape}_device_{latent_state.device}"
            action_str = f"shape_{action.shape}_device_{action.device}"
        
        state_hash = hashlib.md5(state_str.encode()).hexdigest()[:16]
        action_hash = hashlib.md5(action_str.encode()).hexdigest()[:16]
        return f"{state_hash}_{action_hash}"
    
    def _compute_preference_reward(self, 
                                 latent_state: torch.Tensor, 
                                 action: torch.Tensor) -> Tuple[float, float]:
        """计算偏好奖励和置信度（改进版）
        
        重要：单步偏好奖励保持原值不变，不受轨迹统计影响
        修正因子只在轨迹级别的融合中使用
        
        Args:
            latent_state: 潜空间状态
            action: 动作
            
        Returns:
            (原始偏好奖励, 置信度)
        """
        # 检查缓存
        if self.config.enable_caching:
            cache_key = self._build_cache_key(latent_state, action)
            if cache_key in self.reward_cache:
                self.cache_hits += 1
                return self.reward_cache[cache_key]
            else:
                self.cache_misses += 1
        
        # 计算偏好奖励
        raw_preference_reward, raw_confidence = self.trainer.get_preference_reward(latent_state, action)
        
        # 改进置信度计算
        improved_confidence = self._improve_confidence_calculation(raw_confidence, raw_preference_reward)
        
        # 重要：保持原始偏好奖励不变，不进行稳定化处理
        # 单步偏好奖励应该保持原值，修正因子只在轨迹级别应用
        original_reward = raw_preference_reward
        
        # 缓存结果
        if self.config.enable_caching:
            # 限制缓存大小
            if len(self.reward_cache) >= self.config.max_cache_size:
                # 随机删除一些旧的缓存项
                keys_to_remove = list(self.reward_cache.keys())[:self.config.max_cache_size // 4]
                for key in keys_to_remove:
                    del self.reward_cache[key]
            
            self.reward_cache[cache_key] = (original_reward, improved_confidence)
        
        return original_reward, improved_confidence
    
    def _improve_confidence_calculation(self, raw_confidence: float, preference_reward: float) -> float:
        """改进置信度计算
        
        Args:
            raw_confidence: 原始置信度
            preference_reward: 偏好奖励值
            
        Returns:
            改进后的置信度
        """
        # 基于偏好奖励的绝对值调整置信度
        reward_magnitude = abs(preference_reward)
        
        # 如果偏好奖励绝对值较大，提高置信度
        if reward_magnitude > 0.2:
            confidence_boost = min(0.15, reward_magnitude * 0.3)
            improved_confidence = min(0.95, raw_confidence + confidence_boost)
        elif reward_magnitude < 0.05:
            # 如果偏好奖励绝对值很小，降低置信度
            confidence_penalty = min(0.1, (0.05 - reward_magnitude) * 2.0)
            improved_confidence = max(0.4, raw_confidence - confidence_penalty)
        else:
            improved_confidence = raw_confidence
        
        # 应用平滑处理
        if hasattr(self, '_last_confidence'):
            smoothing_factor = 0.7
            improved_confidence = (smoothing_factor * improved_confidence + 
                                 (1 - smoothing_factor) * self._last_confidence)
        
        self._last_confidence = improved_confidence
        return improved_confidence
    
    def _stabilize_preference_reward(self, raw_reward: float, confidence: float) -> float:
        """稳定化偏好奖励
        
        Args:
            raw_reward: 原始偏好奖励
            confidence: 置信度
            
        Returns:
            稳定化后的偏好奖励
        """
        # 基于置信度调整奖励幅度
        if confidence < 0.4:
            # 低置信度时大幅衰减奖励
            stabilized_reward = raw_reward * 0.3
        elif confidence < 0.6:
            # 中等置信度时适度衰减
            stabilized_reward = raw_reward * (0.5 + confidence * 0.5)
        else:
            # 高置信度时保持或轻微增强
            stabilized_reward = raw_reward * min(1.2, 0.8 + confidence * 0.4)
        
        # 应用移动平均平滑
        if hasattr(self, '_reward_history'):
            self._reward_history.append(stabilized_reward)
            if len(self._reward_history) > 10:
                self._reward_history.pop(0)
            
            # 计算移动平均
            avg_reward = sum(self._reward_history) / len(self._reward_history)
            smoothing_factor = 0.8
            stabilized_reward = (smoothing_factor * stabilized_reward + 
                               (1 - smoothing_factor) * avg_reward)
        else:
            self._reward_history = [stabilized_reward]
        
        return stabilized_reward
    
    def _normalize_rewards(self, 
                         environment_reward: float, 
                         preference_reward: float) -> Tuple[float, float]:
        """改进的奖励归一化
        
        Args:
            environment_reward: 环境奖励
            preference_reward: 偏好奖励
            
        Returns:
            (归一化的环境奖励, 归一化的偏好奖励)
        """
        # 裁剪奖励以确保数值稳定性
        env_reward_clipped = np.clip(
            environment_reward, 
            self.config.reward_clip_range[0], 
            self.config.reward_clip_range[1]
        )
        pref_reward_clipped = np.clip(
            preference_reward,
            self.config.reward_clip_range[0], 
            self.config.reward_clip_range[1]
        )
        
        # 环境奖励保持不变（已经是合理的尺度）
        normalized_env_reward = env_reward_clipped
        
        # 偏好奖励归一化 - 修改为与推理时一致的范围(-0.3, 0.3)
        if self.config.reward_normalization_method == "tanh":
            # 使用tanh并缩放到(-0.3, 0.3)范围，与推理时保持一致
            normalized_pref_reward = np.tanh(pref_reward_clipped) * 0.3
        elif self.config.reward_normalization_method == "sigmoid":
            # sigmoid映射到(-0.3, 0.3)范围
            sigmoid_val = 1.0 / (1.0 + np.exp(-pref_reward_clipped))
            normalized_pref_reward = 0.3 * (2.0 * sigmoid_val - 1.0)
        elif self.config.reward_normalization_method == "adaptive_sigmoid":
            # 修改adaptive_sigmoid也使用(-0.3, 0.3)范围，与推理时保持一致
            sigmoid_output = 1.0 / (1.0 + np.exp(-pref_reward_clipped))
            normalized_pref_reward = 0.3 * (2.0 * sigmoid_output - 1.0)
        else:
            # 默认使用tanh并缩放到(-0.3, 0.3)
            normalized_pref_reward = np.tanh(pref_reward_clipped) * 0.3
        
        return normalized_env_reward, normalized_pref_reward
    
    def _compute_confidence_based_weight(self, confidence: float, preference_reward: float = None) -> float:
        """基于置信度和偏好奖励计算偏好权重
        
        Args:
            confidence: 置信度值 (0.0 - 1.0)
            preference_reward: 偏好奖励值，用于判断是否应该降低权重
            
        Returns:
            偏好权重 (min_preference_weight - max_preference_weight)
        """
        # 基础权重计算：使用分段线性函数映射置信度到权重
        if confidence <= self.config.confidence_threshold_low:
            # 低置信度：使用最小权重
            base_weight = self.config.min_preference_weight
        elif confidence >= self.config.confidence_threshold_high:
            # 高置信度：使用较高权重，但不是最大值
            base_weight = min(0.5, self.config.max_preference_weight)  # 限制最大权重为0.5
        else:
            # 中等置信度：线性插值
            confidence_range = self.config.confidence_threshold_high - self.config.confidence_threshold_low
            weight_range = min(0.5, self.config.max_preference_weight) - self.config.min_preference_weight
            
            normalized_confidence = (confidence - self.config.confidence_threshold_low) / confidence_range
            base_weight = self.config.min_preference_weight + normalized_confidence * weight_range
        
        # 关键修复：如果偏好奖励为负，大幅降低偏好权重
        if preference_reward is not None and preference_reward < -0.1:
            # 偏好奖励为负时，降低权重以减少负面影响
            negative_penalty = min(0.3, abs(preference_reward) * 0.5)  # 根据负奖励程度调整
            base_weight = max(self.config.min_preference_weight, base_weight - negative_penalty)
        
        # 应用置信度敏感性调整（减小调整幅度）
        if hasattr(self, 'weight_history') and len(self.weight_history) > 10:
            # 获取最近的置信度历史
            recent_confidences = [entry['confidence'] for entry in list(self.weight_history)[-10:]]
            avg_confidence = np.mean(recent_confidences)
            confidence_deviation = confidence - avg_confidence
            sensitivity_adjustment = self.config.confidence_sensitivity * confidence_deviation * 0.05  # 减小调整幅度
            base_weight = base_weight + sensitivity_adjustment
        
        # 确保权重在合理范围内，并进一步限制最大权重
        final_weight = max(self.config.min_preference_weight, 
                          min(0.5, base_weight))  # 硬限制最大权重为0.5
        
        return final_weight
    
    def _update_adaptive_weights(self, 
                               confidence: float, 
                               preference_reward: Optional[float] = None,
                               performance_improvement: Optional[float] = None) -> bool:
        """更新自适应权重
        
        Args:
            confidence: 当前置信度
            preference_reward: 偏好奖励值
            performance_improvement: 性能改善情况
            
        Returns:
            是否更新了权重
        """
        old_pref_weight = self.current_preference_weight
        
        # 初始化权重稳定性跟踪器
        if not hasattr(self, '_weight_stability_counter'):
            self._weight_stability_counter = 0
            self._last_significant_change = 0
        
        # 1. 基于置信度的动态权重调整（优化为更灵敏的调整机制）
        if self.config.enable_confidence_weighting:
            target_weight = self._compute_confidence_based_weight(confidence, preference_reward)
            weight_diff = target_weight - self.current_preference_weight
            
            # 优化权重调整：降低阈值，减少稳定性要求，增加调整幅度
            if abs(weight_diff) > 0.02:  # 降低阈值到2%
                self._weight_stability_counter += 1
                # 只需要连续2次都需要调整就执行，提高响应性
                if self._weight_stability_counter >= 2:
                    # 增加调整幅度，使权重能更快适应变化
                    adjustment_factor = 0.8 if abs(weight_diff) > 0.1 else 0.6  # 大变化时更激进
                    self.current_preference_weight += weight_diff * self.config.weight_update_rate * adjustment_factor
                    self.current_environment_weight = 1.0 - self.current_preference_weight
                    self._weight_stability_counter = 0
                    self._last_significant_change = time.time()
            else:
                self._weight_stability_counter = max(0, self._weight_stability_counter - 1)  # 逐渐减少计数器
        else:
            # 保持初始权重配置
            self.current_preference_weight = self.config.initial_preference_weight
            self.current_environment_weight = self.config.initial_environment_weight
        
        # 2. 基于性能的微调（优化为更积极的调整）
        if (self.config.enable_performance_weighting and 
            performance_improvement is not None and 
            abs(performance_improvement) > 0.1 and  # 降低性能改善阈值，提高敏感性
            self.performance_tracker is not None and
            time.time() - self._last_significant_change > 5):  # 缩短间隔时间到5秒
            
            if performance_improvement > self.config.reward_improvement_threshold:  # 降低阈值
                # 性能改善：增加偏好权重
                perf_factor = min(performance_improvement / self.config.reward_improvement_threshold, 2.0)
                pref_weight_increase = self.config.performance_weight_factor * perf_factor * self.config.weight_update_rate * 0.5  # 增加调整幅度
                self.current_preference_weight += pref_weight_increase
                self.stats['performance_improvements'] += 1
                self._last_significant_change = time.time()
            elif performance_improvement < -self.config.reward_improvement_threshold:
                # 性能恶化：减少偏好权重
                perf_factor = min(abs(performance_improvement) / self.config.reward_improvement_threshold, 2.0)
                pref_weight_decrease = self.config.performance_weight_factor * perf_factor * self.config.weight_update_rate * 0.5  # 增加调整幅度
                self.current_preference_weight -= pref_weight_decrease
                self._last_significant_change = time.time()
        
        # 3. 确保权重在有效范围内
        self.current_preference_weight = max(
            self.config.min_preference_weight,
            min(self.config.max_preference_weight, self.current_preference_weight)
        )
        self.current_environment_weight = 1.0 - self.current_preference_weight
        
        # 4. 记录权重变化（降低阈值以捕获更多变化）
        weight_changed = abs(self.current_preference_weight - old_pref_weight) > 0.01  # 降低阈值到1%

        # 添加详细的权重调整日志（适度减少频率但保持可观察性）
        if weight_changed:
            # 增加日志计数器，每20次权重变化才输出一次日志
            if not hasattr(self, '_log_counter'):
                self._log_counter = 0
            self._log_counter += 1
            
            if self._log_counter % 20 == 0:  # 每20次输出一次日志，提高可观察性
                logger.info(f"[动态权重调整] 偏好权重: {old_pref_weight:.3f} -> {self.current_preference_weight:.3f}, "
                           f"置信度: {confidence:.3f}, 偏好奖励: {preference_reward or 0.0:.3f}, 性能改善: {performance_improvement or 0.0:.3f}")

        if weight_changed:
            self.stats['weight_updates'] += 1
            self.weight_history.append({
                'preference_weight': self.current_preference_weight,
                'environment_weight': self.current_environment_weight,
                'confidence': confidence,
                'performance_improvement': performance_improvement or 0.0,
                'timestamp': time.time()
            })
        
        return weight_changed
    
    def fuse_rewards(self, 
                    environment_reward: float, 
                    preference_reward: float, 
                    confidence: float,
                    performance_improvement: Optional[float] = None) -> Dict[str, float]:
        """融合环境奖励和偏好奖励
        
        使用新的融合公式：融合奖励 = 环境奖励 × (1 + 修正因子)
        修正因子 = (正向偏好个数-负向偏好个数)/(总轨迹长度)*置信度
        修正因子上下限：(-0.4, -0.1)和(0.1, 0.4)
        
        Args:
            environment_reward: 环境奖励（原版tdmpc2得出的奖励值，不做任何处理）
            preference_reward: 偏好奖励
            confidence: 置信度
            performance_improvement: 性能改善情况
            
        Returns:
            融合结果字典
        """
        # 更新轨迹统计
        self.trajectory_stats['total_steps'] += 1
        if preference_reward > 0:
            self.trajectory_stats['positive_preference_count'] += 1
        elif preference_reward < 0:
            self.trajectory_stats['negative_preference_count'] += 1
        
        # 更新平均置信度
        total_steps = self.trajectory_stats['total_steps']
        self.trajectory_stats['avg_confidence'] = (
            (self.trajectory_stats['avg_confidence'] * (total_steps - 1) + confidence) / total_steps
        )
        
        # 计算修正因子
        if total_steps > 0:
            positive_count = self.trajectory_stats['positive_preference_count']
            negative_count = self.trajectory_stats['negative_preference_count']
            avg_confidence = self.trajectory_stats['avg_confidence']
            
            # 修正因子 = (正向偏好个数-负向偏好个数)/(总轨迹长度)*置信度
            correction_factor = ((positive_count - negative_count) / total_steps) * avg_confidence
            
            # 应用上下限：(-0.4, -0.1)和(0.1, 0.4)
            if correction_factor > 0:
                # 正向修正因子限制在(0.1, 0.4)
                correction_factor = max(0.1, min(0.4, correction_factor))
            elif correction_factor < 0:
                # 负向修正因子限制在(-0.4, -0.1)
                correction_factor = max(-0.4, min(-0.1, correction_factor))
            else:
                # 修正因子为0时，设为最小正值或负值
                correction_factor = 0.1 if positive_count >= negative_count else -0.1
        else:
            correction_factor = 0.1  # 默认小正值
        
        # 融合奖励 = 环境奖励 × (1 + 修正因子)
        integrated_reward = environment_reward * (1.0 + correction_factor)
        
        return {
            'integrated_reward': integrated_reward,
            'environment_reward': environment_reward,
            'preference_reward': preference_reward,
            'normalized_environment_reward': environment_reward,  # 简化处理，直接使用原始值
            'normalized_preference_reward': preference_reward,    # 简化处理，直接使用原始值
            'correction_factor': correction_factor,
            'positive_preference_count': self.trajectory_stats['positive_preference_count'],
            'negative_preference_count': self.trajectory_stats['negative_preference_count'],
            'total_steps': self.trajectory_stats['total_steps'],
            'avg_confidence': self.trajectory_stats['avg_confidence'],
            'confidence': confidence
        }
    
    def compute_integrated_reward(self, 
                                latent_state: torch.Tensor, 
                                action: torch.Tensor, 
                                environment_reward: float) -> Dict[str, Any]:
        """计算集成奖励
        
        Args:
            latent_state: 潜空间状态
            action: 动作
            environment_reward: 环境奖励
            
        Returns:
            详细的奖励信息
        """
        start_time = time.time()
        
        # 计算偏好奖励和置信度
        preference_reward, confidence = self._compute_preference_reward(latent_state, action)
        
        # 更新性能跟踪
        performance_improvement = 0.0
        if self.performance_tracker is not None:
            performance_improvement = self.performance_tracker.update(environment_reward)
        
        # 融合奖励
        fusion_result = self.fuse_rewards(
            environment_reward, preference_reward, confidence, performance_improvement
        )
        
        # 收集在线学习数据
        if self.config.enable_online_learning:
            try:
                # 安全地将tensor转换为CPU，避免CUDA初始化问题
                if latent_state.is_cuda:
                    latent_state_data = latent_state.detach().cpu().tolist()
                else:
                    latent_state_data = latent_state.detach().tolist()
                    
                if action.is_cuda:
                    action_data = action.detach().cpu().tolist()
                else:
                    action_data = action.detach().tolist()
            except Exception as e:
                # 如果tensor操作失败，使用备用数据
                latent_state_data = f"tensor_shape_{latent_state.shape}"
                action_data = f"tensor_shape_{action.shape}"
                
            online_data = {
                'latent_state': latent_state_data,
                'action': action_data,
                'environment_reward': environment_reward,
                'preference_reward': preference_reward,
                'confidence': confidence,
                'integrated_reward': fusion_result['integrated_reward'],
                'timestamp': time.time()
            }
            self.online_buffer.append(online_data)
        
        # 更新统计信息
        self.stats['total_calls'] += 1
        self.stats['avg_confidence'] = (
            self.stats['avg_confidence'] * (self.stats['total_calls'] - 1) + confidence
        ) / self.stats['total_calls']
        self.stats['avg_preference_reward'] = (
            self.stats['avg_preference_reward'] * (self.stats['total_calls'] - 1) + preference_reward
        ) / self.stats['total_calls']
        self.stats['avg_environment_reward'] = (
            self.stats['avg_environment_reward'] * (self.stats['total_calls'] - 1) + environment_reward
        ) / self.stats['total_calls']
        self.stats['avg_preference_weight'] = (
            self.stats['avg_preference_weight'] * (self.stats['total_calls'] - 1) + self.current_preference_weight
        ) / self.stats['total_calls']
        self.stats['avg_environment_weight'] = (
            self.stats['avg_environment_weight'] * (self.stats['total_calls'] - 1) + self.current_environment_weight
        ) / self.stats['total_calls']
        
        # 更新缓存命中率
        if self.config.enable_caching:
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests > 0:
                self.stats['cache_hit_rate'] = self.cache_hits / total_cache_requests
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 返回详细信息
        result = {
            'integrated_reward': fusion_result['integrated_reward'],
            'environment_reward': environment_reward,
            'preference_reward': preference_reward,
            'normalized_environment_reward': fusion_result['normalized_environment_reward'],
            'normalized_preference_reward': fusion_result['normalized_preference_reward'],
            'confidence': confidence,
            'preference_weight': self.current_preference_weight,
            'environment_weight': self.current_environment_weight,
            'performance_improvement': performance_improvement,
            'baseline_performance': self.performance_tracker.get_baseline() if self.performance_tracker else 0.0,
            'processing_time': processing_time
        }
        
        return result
    
    def get_online_learning_data(self) -> List[Dict[str, Any]]:
        """获取在线学习数据
        
        Returns:
            在线学习数据列表
        """
        if not self.config.enable_online_learning:
            return []
        
        return list(self.online_buffer)
    
    def clear_online_buffer(self):
        """清空在线学习缓冲区"""
        if self.config.enable_online_learning:
            self.online_buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = dict(self.stats)
        
        # 添加权重历史统计
        if len(self.weight_history) > 0:
            recent_weights = list(self.weight_history)[-10:]
            stats['recent_preference_weights'] = [w['preference_weight'] for w in recent_weights]
            stats['recent_environment_weights'] = [w['environment_weight'] for w in recent_weights]
            stats['weight_variance'] = np.var([w['preference_weight'] for w in recent_weights])
        
        # 添加性能跟踪统计
        if self.performance_tracker is not None:
            stats['performance_trend'] = self.performance_tracker.get_trend()
            stats['baseline_performance'] = self.performance_tracker.get_baseline()
        
        # 添加缓存统计
        if self.config.enable_caching:
            stats['cache_size'] = len(self.reward_cache)
            stats['cache_hits'] = self.cache_hits
            stats['cache_misses'] = self.cache_misses
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_calls': 0,
            'cache_hit_rate': 0.0,
            'avg_preference_weight': self.config.initial_preference_weight,
            'avg_environment_weight': self.config.initial_environment_weight,
            'avg_confidence': 0.0,
            'avg_preference_reward': 0.0,
            'avg_environment_reward': 0.0,
            'weight_updates': 0,
            'performance_improvements': 0
        }
        
        if self.config.enable_caching:
            self.cache_hits = 0
            self.cache_misses = 0
        
        self.weight_history.clear()
        logger.info("[优化集成器] 统计信息已重置")

# 工厂函数
def create_optimized_preference_system(
    model_config: Optional[Dict[str, Any]] = None,
    integration_config: Optional[Dict[str, Any]] = None,
    tdmpc2_cfg=None
) -> Tuple['OptimizedPreferenceTrainer', 'OptimizedPreferenceIntegrator']:
    """
    创建完整的优化偏好系统
    
    Args:
        model_config: 模型配置
        integration_config: 集成配置
        tdmpc2_cfg: TD-MPC2配置对象，用于读取置信度阈值等参数
        
    Returns:
        (训练器, 集成器)
    """
    try:
        # 延迟导入以避免循环依赖
        from .optimized_preference_trainer import OptimizedTrainingConfig, create_optimized_preference_trainer
        from .optimized_latent_preference_model import OptimizedLatentPreferenceConfig
        
        # 创建默认配置
        if model_config is None:
            model_config = OptimizedLatentPreferenceConfig()
        elif isinstance(model_config, dict):
            # 如果是字典，创建配置对象
            config_obj = OptimizedLatentPreferenceConfig()
            for key, value in model_config.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
            model_config = config_obj
        
        if integration_config is None:
            integration_config = OptimizedIntegrationConfig()
        elif isinstance(integration_config, dict):
            # 如果是字典，创建配置对象
            config_obj = OptimizedIntegrationConfig()
            for key, value in integration_config.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
            integration_config = config_obj
        
        # 从TD-MPC2配置中读取置信度阈值
        if tdmpc2_cfg is not None:
            try:
                # 尝试从多个可能的路径读取置信度阈值
                confidence_threshold = None
                
                # 路径1: prioritized_experience_replay.sampling_config.quality_thresholds.confidence_threshold
                if hasattr(tdmpc2_cfg, 'prioritized_experience_replay'):
                    per_config = tdmpc2_cfg.prioritized_experience_replay
                    if hasattr(per_config, 'sampling_config'):
                        sampling_config = per_config.sampling_config
                        if hasattr(sampling_config, 'quality_thresholds'):
                            quality_thresholds = sampling_config.quality_thresholds
                            if hasattr(quality_thresholds, 'confidence_threshold'):
                                confidence_threshold = quality_thresholds.confidence_threshold
                                logger.info(f"[优化偏好系统] 从TD-MPC2配置读取置信度阈值: {confidence_threshold}")
                
                # 如果找到了置信度阈值，更新集成配置
                if confidence_threshold is not None:
                    integration_config.confidence_threshold_low = confidence_threshold * 0.8  # 低阈值为80%
                    integration_config.confidence_threshold_high = confidence_threshold  # 高阈值为配置值
                    logger.info(f"[优化偏好系统] 更新置信度阈值 - 低: {integration_config.confidence_threshold_low}, 高: {integration_config.confidence_threshold_high}")
                else:
                    logger.warning(f"[优化偏好系统] 未在TD-MPC2配置中找到置信度阈值，使用默认值")
                    
            except Exception as e:
                logger.warning(f"[优化偏好系统] 读取TD-MPC2配置失败: {e}，使用默认配置")
        
        # 创建训练配置
        training_config = OptimizedTrainingConfig(
            device=integration_config.device
        )
        
        # 创建训练器
        trainer = create_optimized_preference_trainer(model_config, training_config)
        
        # 创建集成器
        integrator = OptimizedPreferenceIntegrator(trainer, integration_config)
        
        logger.info("[优化偏好系统] 创建成功")
        return trainer, integrator
        
    except Exception as e:
        logger.error(f"创建优化偏好系统失败: {e}")
        raise

if __name__ == "__main__":
    # 测试代码
    print("优化偏好集成器测试")
    
    # 创建系统
    trainer, integrator = create_optimized_preference_system()
    
    # 模拟测试数据
    latent_state = torch.randn(512)
    action = torch.randn(61)
    environment_reward = 1.5
    
    # 测试集成奖励计算
    result = integrator.compute_integrated_reward(latent_state, action, environment_reward)
    
    print(f"集成结果:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试多次调用
    print("\n多次调用测试:")
    for i in range(5):
        env_reward = np.random.uniform(0.5, 2.0)
        result = integrator.compute_integrated_reward(latent_state, action, env_reward)
        print(f"  调用 {i+1}: 集成奖励={result['integrated_reward']:.4f}, "
              f"偏好权重={result['preference_weight']:.3f}, "
              f"置信度={result['confidence']:.3f}")
    
    # 获取统计信息
    stats = integrator.get_stats()
    print(f"\n统计信息:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list) and len(value) <= 5:
            print(f"  {key}: {value}")
    
    print("\n优化偏好集成器测试完成！")