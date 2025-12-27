#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于DPO (Direct Preference Optimization) 的统一偏好标签系统

核心设计理念：
1. 基于最新DPO研究 (Rafailov et al., 2024) 的直接偏好优化方法
2. 统一轨迹生成与标签系统，确保评估意义一致性
3. 采用二元分类损失函数进行偏好学习
4. 支持多层次偏好标签：规则基础、质量评估、混合标签
5. 集成轨迹质量评估与偏好对生成的统一框架

参考文献：
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2024)
- Deep Reinforcement Learning for Robotics: A Survey of Real-World Applications (2024)
"""

import os
import torch
from typing import List, Tuple, Dict, Optional, Union, Any, Callable

# NumPy兼容性处理
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # 创建numpy的替代实现
    class NumpyCompat:
        @staticmethod
        def array(data):
            if isinstance(data, torch.Tensor):
                return data
            return torch.tensor(data)
        
        @staticmethod
        def ndarray(*args, **kwargs):
            return torch.Tensor
    
    np = NumpyCompat()
from dataclasses import dataclass, field
from enum import Enum
import yaml
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from functools import lru_cache
import hashlib
import importlib.util
import sys
from pathlib import Path

# 全局缓存变量
_global_engine_cache = {}
_global_cache_timestamps = {}
_global_cache_max_age = 600  # 10分钟缓存

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelType(Enum):
    """基于DPO的标签类型枚举
    
    参考DPO论文的分类方法，支持多层次偏好标签生成
    """
    # 核心标签类型（基于DPO二元分类）
    DPO_BINARY = "dpo_binary"  # DPO二元偏好标签（核心）
    DPO_CONFIDENCE = "dpo_confidence"  # 带置信度的DPO标签
    
    # 传统标签类型（向后兼容）
    RULE_BASED = "rule_based"  # 基于规则的标签
    QUALITY_BASED = "quality_based"  # 基于质量评估的标签
    HEURISTIC_BASED = "heuristic_based"  # 基于启发式的标签
    
    # 混合标签类型
    HYBRID_DPO_RULE = "hybrid_dpo_rule"  # DPO+规则混合
    HYBRID_DPO_QUALITY = "hybrid_dpo_quality"  # DPO+质量混合
    
    # 特殊标签类型
    TRAJECTORY_ALIGNED = "trajectory_aligned"  # 轨迹生成对齐标签
    UNCERTAIN = "uncertain"  # 不确定标签
    INVALID = "invalid"  # 无效标签

@dataclass
class LabelingStatistics:
    """标签生成统计信息"""
    total_generated: int = 0
    rule_based_count: int = 0
    heuristic_based_count: int = 0
    valid_labels: int = 0
    invalid_labels: int = 0
    total_generation_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_confidence: float = 0.0
    label_type_distribution: Dict[str, int] = None
    cache_hits: int = 0
    cache_misses: int = 0
    
    def __post_init__(self):
        if self.label_type_distribution is None:
            self.label_type_distribution = defaultdict(int)
    
    def update_generation_stats(self, label_type: LabelType, generation_time: float, is_valid: bool):
        """更新生成统计信息"""
        self.total_generated += 1
        self.total_generation_time += generation_time
        self.avg_generation_time = self.total_generation_time / self.total_generated
        
        if label_type == LabelType.RULE_BASED:
            self.rule_based_count += 1
        elif label_type == LabelType.HEURISTIC_BASED:
            self.heuristic_based_count += 1
        
        if is_valid:
            self.valid_labels += 1
        else:
            self.invalid_labels += 1
        
        self.label_type_distribution[label_type.value] += 1
    
    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests

@dataclass
class LabelMetadata:
    """标签元数据"""
    label_type: LabelType
    confidence: float  # 置信度 [0, 1]
    quality_score_a: float  # 轨迹A的质量分数
    quality_score_b: float  # 轨迹B的质量分数
    score_difference: float  # 分数差异
    generation_time: float  # 生成时间
    features_used: List[str]  # 使用的特征列表
    additional_info: Dict[str, Any]  # 额外信息

@dataclass
class PreferenceLabel:
    """基于DPO的偏好标签数据结构
    
    参考DPO论文的偏好建模方法
    """
    preference_score: float  # 偏好分数 [0, 1]，0.5表示无偏好
    logit_preference: Optional[float] = None  # DPO logit空间的偏好值
    binary_preference: Optional[int] = None  # 二元偏好 {-1, 0, 1}
    metadata: LabelMetadata = None
    is_valid: bool = True
    validation_errors: Optional[List[str]] = None
    
    def to_dpo_format(self) -> Dict[str, Any]:
        """转换为DPO训练格式"""
        return {
            'preference_logit': self.logit_preference or self._score_to_logit(self.preference_score),
            'binary_label': self.binary_preference or self._score_to_binary(self.preference_score),
            'confidence': self.metadata.confidence if self.metadata else 0.5,
            'is_valid': self.is_valid
        }
    
    def _score_to_logit(self, score: float) -> float:
        """将偏好分数转换为logit空间"""
        # 避免极值，使用epsilon平滑
        epsilon = 1e-8
        score = np.clip(score, epsilon, 1.0 - epsilon)
        return np.log(score / (1.0 - score))
    
    def _score_to_binary(self, score: float) -> int:
        """将偏好分数转换为二元标签"""
        if score > 0.6:
            return 1  # A > B
        elif score < 0.4:
            return -1  # B > A
        else:
            return 0  # A ≈ B

class DPOPreferenceEvaluator:
    """基于DPO的偏好评估器
    
    实现Direct Preference Optimization论文中的核心算法
    """
    
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0, 
                 task_name: str = None, config: Dict[str, Any] = None):
        """
        Args:
            beta: DPO温度参数，控制偏好强度
            label_smoothing: 标签平滑参数，处理噪声偏好数据
            task_name: 任务名称，用于任务特定配置
            config: 配置字典，包含任务特定设置
        """
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.task_name = task_name or 'default'
        self.config = config or {}
        self.preference_history = deque(maxlen=1000)  # 偏好历史记录
        
        # 初始化质量评估器（用于启发式奖励估计）
        self.quality_evaluator = None
    
    def evaluate_dpo_preference(self, 
                               trajectory_a: Dict[str, np.ndarray],
                               trajectory_b: Dict[str, np.ndarray],
                               reward_model: Optional[Callable] = None) -> Tuple[float, float]:
        """使用DPO方法评估轨迹偏好
        
        Args:
            trajectory_a: 轨迹A数据
            trajectory_b: 轨迹B数据 
            reward_model: 可选的奖励模型
            
        Returns:
            (preference_logit, confidence)
        """
        # 计算轨迹奖励（如果有奖励模型）
        if reward_model is not None:
            reward_a = self._compute_trajectory_reward(trajectory_a, reward_model)
            reward_b = self._compute_trajectory_reward(trajectory_b, reward_model)
        else:
            # 使用启发式奖励估计
            reward_a = self._heuristic_reward_estimate(trajectory_a)
            reward_b = self._heuristic_reward_estimate(trajectory_b)
        
        # DPO偏好概率计算
        # P(τ1 ≻ τ2) = σ(β * (R(τ1) - R(τ2)))
        reward_diff = float(reward_a) - float(reward_b)
        preference_logit = float(self.beta * reward_diff)
        
        # 应用标签平滑
        if self.label_smoothing > 0:
            preference_logit = self._apply_label_smoothing(preference_logit)
        
        # 计算置信度
        confidence = self._compute_confidence(reward_a, reward_b, preference_logit)
        
        # 记录偏好历史
        self.preference_history.append({
            'reward_a': reward_a,
            'reward_b': reward_b,
            'preference_logit': preference_logit,
            'confidence': confidence
        })
        
        return preference_logit, confidence
    
    def _compute_trajectory_reward(self, trajectory: Dict[str, np.ndarray], reward_model: Callable) -> float:
        """计算轨迹总奖励"""
        obs_seq = trajectory.get('obs', [])
        act_seq = trajectory.get('action', [])
        
        if len(obs_seq) == 0:
            return 0.0
        
        # 使用奖励模型计算每步奖励
        total_reward = 0.0
        for i in range(len(obs_seq)):
            obs = obs_seq[i] if i < len(obs_seq) else obs_seq[-1]
            act = act_seq[i] if i < len(act_seq) else np.zeros_like(act_seq[0] if act_seq else [])
            step_reward = reward_model(obs, act)
            total_reward += step_reward
        
        return float(total_reward / len(obs_seq))  # 平均奖励
    
    def _heuristic_reward_estimate(self, trajectory: Dict[str, np.ndarray]) -> float:
        """启发式奖励估计（当没有奖励模型时）- 基于配置的增强版"""
        obs_seq = trajectory.get('obs', [])
        act_seq = trajectory.get('action', [])
        
        if len(obs_seq) == 0:
            return 0.0
        
        # 使用TrajectoryQualityEvaluator进行质量评估
        try:
            obs_array = np.array(obs_seq) if not isinstance(obs_seq, np.ndarray) else obs_seq
            act_array = np.array(act_seq) if not isinstance(act_seq, np.ndarray) else act_seq
            
            # 创建质量评估器（如果还没有）
            if not hasattr(self, 'quality_evaluator') or self.quality_evaluator is None:
                # 从配置中获取任务名称
                task_name = getattr(self, 'task_name', 'default')
                config = getattr(self, 'config', {})
                self.quality_evaluator = TrajectoryQualityEvaluator(task_name, config)
            
            # 使用质量评估器计算综合质量分数
            quality_score, feature_scores = self.quality_evaluator.evaluate_trajectory_quality(
                obs_array, act_array
            )
            
            # 应用任务特定的启发式规则
            heuristic_bonus = self._apply_task_specific_heuristics(obs_array, act_array, feature_scores)
            
            # 综合启发式奖励
            heuristic_reward = quality_score + heuristic_bonus
            
            return float(np.clip(heuristic_reward, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"启发式奖励计算失败，使用基础方法: {e}")
            # 回退到基础方法
            return self._basic_heuristic_reward_estimate(obs_seq, act_seq)
    
    def _basic_heuristic_reward_estimate(self, obs_seq, act_seq) -> float:
        """基础启发式奖励估计（回退方法）"""
        # 基于轨迹长度、动作平滑性等启发式特征
        survival_score = min(len(obs_seq) / 200.0, 1.0)  # 生存时间
        
        # 动作平滑性
        if len(act_seq) > 1:
            act_array = np.array(act_seq)
            action_variance = np.mean(np.var(act_array, axis=0))
            smoothness_score = np.exp(-action_variance)
        else:
            smoothness_score = 0.5
        
        # 状态稳定性（基于观测方差）
        if len(obs_seq) > 1:
            obs_array = np.array(obs_seq)
            obs_variance = np.mean(np.var(obs_array, axis=0))
            stability_score = np.exp(-obs_variance * 0.1)
        else:
            stability_score = 0.5
        
        # 综合评分
        heuristic_reward = 0.4 * survival_score + 0.3 * smoothness_score + 0.3 * stability_score
        return float(heuristic_reward)
    
    def _apply_task_specific_heuristics(self, obs_seq: np.ndarray, act_seq: np.ndarray, 
                                       feature_scores: Dict[str, float]) -> float:
        """应用任务特定的启发式规则"""
        if not hasattr(self, 'config') or not self.config:
            return 0.0
        
        task_name = getattr(self, 'task_name', 'default')
        tasks_config = self.config.get('tasks', {})
        task_config = tasks_config.get(task_name, tasks_config.get('default', {}))
        
        heuristic_rules = task_config.get('heuristic_rules', {})
        bonus = 0.0
        
        try:
            # 应用内置启发式规则
            if heuristic_rules.get('use_standing_stability', False):
                bonus += self._calculate_standing_stability_bonus(obs_seq) * 0.1
            
            if heuristic_rules.get('use_forward_motion', False):
                bonus += self._calculate_forward_motion_bonus(obs_seq) * 0.1
            
            if heuristic_rules.get('use_minimal_movement', False):
                bonus += self._calculate_minimal_movement_bonus(act_seq) * 0.05
            
            if heuristic_rules.get('use_action_efficiency', False):
                bonus += self._calculate_action_efficiency_bonus(act_seq) * 0.05
            
            if heuristic_rules.get('use_balance_control', False):
                bonus += self._calculate_balance_control_bonus(obs_seq, act_seq) * 0.1
            
            if heuristic_rules.get('use_energy_efficiency', False):
                bonus += self._calculate_energy_efficiency_bonus(act_seq) * 0.05
            
            # 应用API规则（如果有的话）
            if hasattr(self, 'api_rules') and self.api_rules:
                api_bonus = self._apply_api_rules(obs_seq, act_seq, feature_scores, task_config)
                bonus += api_bonus
            
            return float(np.clip(bonus, -0.2, 0.2))  # 限制奖励范围
            
        except Exception as e:
            logger.warning(f"任务特定启发式规则应用失败: {e}")
            return 0.0
    
    # _load_api_rules方法已移动到PreferenceLabelingEngine类中







    def _apply_api_rules(self, obs_seq: np.ndarray, act_seq: np.ndarray, 
                        feature_scores: Dict[str, float], task_config: Dict[str, Any]) -> float:
        """应用API规则计算额外奖励"""
        if not self.api_rules:
            return 0.0
        
        try:
            # 构建轨迹数据
            trajectory = {
                'obs': obs_seq,
                'action': act_seq,
                'features': feature_scores,
                'states': obs_seq,  # 兼容性别名
                'actions': act_seq,  # 兼容性别名
                'observations': obs_seq  # 兼容性别名
            }
            
            total_bonus = 0.0
            rule_count = 0
            
            # 1. 查找并应用compute_*_reward_components函数
            reward_component_funcs = [name for name in self.api_rules.keys() 
                                    if name.startswith('compute_') and name.endswith('_reward_components')]
            
            for func_name in reward_component_funcs:
                try:
                    # 尝试不同的参数格式
                    func = self.api_rules[func_name]
                    components = None
                    
                    # 尝试多种调用方式
                    try:
                        # 方式1: 传入obs_seq, act_seq, reward_seq (最常见格式)
                        reward_seq = np.zeros(len(obs_seq))  # 默认奖励序列
                        components = func(obs_seq, act_seq, reward_seq)
                    except (TypeError, ValueError):
                        try:
                            # 方式2: 传入轨迹字典
                            components = func(trajectory)
                        except (TypeError, ValueError):
                            try:
                                # 方式3: 传入states, actions
                                components = func(obs_seq, act_seq)
                            except (TypeError, ValueError):
                                try:
                                    # 方式4: 传入轨迹和配置
                                    components = func(trajectory, task_config)
                                except (TypeError, ValueError):
                                    try:
                                        # 方式5: 传入轨迹包装器
                                        from .preference_data_engine import TrajectoryWrapper
                                        trajectory_wrapper = TrajectoryWrapper(trajectory)
                                        components = func(trajectory_wrapper)
                                    except (TypeError, ValueError):
                                        logger.warning(f"无法调用API奖励组件函数 {func_name}")
                                        continue
                    
                    if components and isinstance(components, dict):
                        # 计算组件的加权平均值
                        component_values = [v for v in components.values() if isinstance(v, (int, float))]
                        if component_values:
                            avg_component_score = np.mean(component_values)
                            # 标准化到合理范围
                            normalized_score = np.tanh(float(avg_component_score))  # 标准化到[-1, 1]
                            total_bonus += normalized_score * 0.15  # 转换为[-0.15, 0.15]的奖励
                            rule_count += 1
                            logger.debug(f"应用API奖励组件 {func_name}: {avg_component_score:.4f}")
                        
                except Exception as e:
                    logger.warning(f"API奖励组件规则 {func_name} 应用失败: {e}")
            
            # 2. 查找并应用DPO偏好评估函数
            dpo_func_name = f"evaluate_dpo_preference"
            if dpo_func_name in self.api_rules:
                try:
                    # 构建两个轨迹进行比较（与基准轨迹比较）
                    baseline_trajectory = self._create_baseline_trajectory(obs_seq, act_seq)
                    
                    # 调用API规则的DPO评估函数
                    preference_result = self.api_rules[dpo_func_name](
                        trajectory, baseline_trajectory
                    )
                    
                    if isinstance(preference_result, (tuple, list)) and len(preference_result) >= 2:
                        preference_score, confidence = preference_result[:2]
                        # 将偏好分数转换为奖励加成
                        bonus = (preference_score - 0.5) * 0.2  # 偏好分数转换为[-0.1, 0.1]的奖励
                        total_bonus += bonus * confidence  # 按置信度加权
                        rule_count += 1
                    elif isinstance(preference_result, (int, float)):
                        bonus = (float(preference_result) - 0.5) * 0.2
                        total_bonus += bonus
                        rule_count += 1
                        
                except Exception as e:
                    logger.warning(f"API DPO规则应用失败: {e}")
            
            # 3. 查找并应用轨迹评分函数
            score_func_names = [name for name in self.api_rules.keys() 
                              if name in ['_compute_trajectory_score', 'compute_trajectory_score']]
            
            for score_func_name in score_func_names:
                try:
                    func = self.api_rules[score_func_name]
                    score = None
                    
                    # 尝试不同的调用方式
                    try:
                        score = func(trajectory)
                    except (TypeError, ValueError):
                        try:
                            score = func(obs_seq, act_seq)
                        except (TypeError, ValueError):
                            try:
                                score = func(obs_seq, act_seq, None)  # goal参数
                            except (TypeError, ValueError):
                                logger.warning(f"无法调用API评分函数 {score_func_name}")
                                continue
                    
                    if isinstance(score, (int, float)):
                        # 将分数标准化为奖励加成
                        normalized_score = np.tanh(float(score))  # 标准化到[-1, 1]
                        total_bonus += normalized_score * 0.1  # 转换为[-0.1, 0.1]的奖励
                        rule_count += 1
                        logger.debug(f"应用API评分函数 {score_func_name}: {score:.4f}")
                        
                except Exception as e:
                    logger.warning(f"API评分规则 {score_func_name} 应用失败: {e}")
            
            # 4. 查找并应用比较函数
            compare_func_names = [name for name in self.api_rules.keys() 
                                if name.startswith('compare_') and name.endswith('_trajectories')]
            
            for func_name in compare_func_names:
                try:
                    # 导入TrajectoryWrapper
                    from .preference_data_engine import TrajectoryWrapper
                    
                    # 创建轨迹包装器对象
                    trajectory_wrapper = TrajectoryWrapper(trajectory)
                    baseline_trajectory_dict = self._create_baseline_trajectory(obs_seq, act_seq)
                    baseline_wrapper = TrajectoryWrapper(baseline_trajectory_dict)
                    
                    # 调用比较函数，尝试不同的参数格式
                    try:
                        # 尝试带goal参数的调用
                        goal = task_config.get('goal', {})
                        if not goal:  # 如果goal为空，提供默认目标
                            goal = {'target_position': [0.5, 0.0, 1.0]}  # 默认目标位置
                        comparison_result = self.api_rules[func_name](trajectory_wrapper, baseline_wrapper, goal)
                    except TypeError as te:
                        logger.debug(f"函数 {func_name} 不支持goal参数: {te}")
                        try:
                            # 如果不支持goal参数，尝试只传两个轨迹
                            comparison_result = self.api_rules[func_name](trajectory_wrapper, baseline_wrapper)
                        except Exception as e2:
                            logger.debug(f"函数 {func_name} 两参数调用也失败: {e2}")
                            raise e2
                    except Exception as e:
                        logger.debug(f"函数 {func_name} 调用失败: {e}")
                        raise e
                    
                    # 处理比较函数的返回结果
                    if comparison_result is not None:
                        if isinstance(comparison_result, tuple) and len(comparison_result) == 2:
                            better_traj, worse_traj = comparison_result
                            if better_traj is not None and worse_traj is not None:
                                # 如果当前轨迹是更好的，给予正奖励
                                if better_traj == trajectory_wrapper or (
                                    hasattr(better_traj, 'to_dict') and 
                                    hasattr(trajectory_wrapper, 'to_dict') and
                                    better_traj.to_dict() == trajectory_wrapper.to_dict()
                                ):
                                    bonus = 0.1  # 正奖励
                                else:
                                    bonus = -0.05  # 负奖励
                                total_bonus += bonus
                                rule_count += 1
                                logger.debug(f"应用API比较规则 {func_name}: 奖励={bonus}")
                        elif isinstance(comparison_result, (int, float)):
                            # 兼容数值返回格式
                            bonus = float(comparison_result) * 0.05
                            total_bonus += bonus
                            rule_count += 1
                            logger.debug(f"应用API比较规则 {func_name}: {comparison_result}")
                        
                except Exception as e:
                    logger.warning(f"API比较规则 {func_name} 应用失败: {e}")
            
            # 5. 查找并应用其他评估函数模式
            eval_func_names = [name for name in self.api_rules.keys() 
                             if name.startswith('_evaluate_') or name.startswith('evaluate_')]
            
            for func_name in eval_func_names:
                if func_name in ['evaluate_dpo_preference']:  # 跳过已处理的函数
                    continue
                    
                try:
                    func = self.api_rules[func_name]
                    result = None
                    
                    # 尝试不同的调用方式
                    try:
                        result = func(trajectory)
                    except (TypeError, ValueError):
                        try:
                            result = func(obs_seq)
                        except (TypeError, ValueError):
                            try:
                                result = func(obs_seq, act_seq)
                            except (TypeError, ValueError):
                                continue
                    
                    if isinstance(result, (int, float)):
                        # 标准化评估结果
                        normalized_result = np.tanh(float(result))  # 标准化到[-1, 1]
                        total_bonus += normalized_result * 0.05  # 转换为[-0.05, 0.05]的奖励
                        rule_count += 1
                        logger.debug(f"应用API评估函数 {func_name}: {result:.4f}")
                        
                except Exception as e:
                    logger.warning(f"API评估规则 {func_name} 应用失败: {e}")
            
            # 平均奖励加成并限制范围
            if rule_count > 0:
                final_bonus = total_bonus / rule_count
                # 限制API规则贡献在 (-0.3, 0.3) 范围内
                final_bonus = np.clip(final_bonus, -0.3, 0.3)
                logger.debug(f"API规则总加成: {final_bonus:.4f} (来自 {rule_count} 个规则)")
                return final_bonus
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"API规则应用失败: {e}")
            return 0.0
    
    def _create_baseline_trajectory(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> Dict[str, np.ndarray]:
        """创建基准轨迹用于比较"""
        # 创建一个中性的基准轨迹（零动作或平均动作）
        baseline_obs = np.zeros_like(obs_seq)
        baseline_act = np.zeros_like(act_seq)
        
        # 如果有历史数据，可以使用平均值作为基准
        if len(obs_seq) > 0:
            baseline_obs = np.full_like(obs_seq, np.mean(obs_seq, axis=0))
        if len(act_seq) > 0:
            baseline_act = np.full_like(act_seq, np.mean(act_seq, axis=0))
        
        return {
            'obs': baseline_obs,
            'action': baseline_act
        }
    
    def _calculate_standing_stability_bonus(self, obs_seq: np.ndarray) -> float:
        """计算站立稳定性奖励加成"""
        if len(obs_seq) < 2:
            return 0.0
        
        try:
            # 假设观测中包含身体姿态信息（前几个维度通常是位置和方向）
            # 计算重心高度稳定性（假设z坐标在第3个位置）
            if obs_seq.shape[1] > 2:
                height_variance = np.var(obs_seq[:, 2])  # z坐标方差
                stability_bonus = np.exp(-height_variance * 10)  # 高度越稳定奖励越高
                return float(stability_bonus)
        except Exception:
            pass
        return 0.0
    
    def _calculate_forward_motion_bonus(self, obs_seq: np.ndarray) -> float:
        """计算前进运动奖励加成"""
        if len(obs_seq) < 2:
            return 0.0
        
        try:
            # 计算x方向的位移（假设x坐标在第1个位置）
            if obs_seq.shape[1] > 0:
                x_displacement = obs_seq[-1, 0] - obs_seq[0, 0]
                forward_bonus = np.tanh(x_displacement * 0.1)  # 前进距离奖励
                return float(max(0, forward_bonus))
        except Exception:
            pass
        return 0.0
    
    def _calculate_minimal_movement_bonus(self, act_seq: np.ndarray) -> float:
        """计算最小运动奖励加成（适用于平衡任务）"""
        if len(act_seq) < 2:
            return 0.0
        
        try:
            # 计算动作变化的平均幅度
            action_changes = np.diff(act_seq, axis=0)
            avg_change = np.mean(np.abs(action_changes))
            minimal_bonus = np.exp(-avg_change * 2)  # 动作变化越小奖励越高
            return float(minimal_bonus)
        except Exception:
            pass
        return 0.0
    
    def _calculate_action_efficiency_bonus(self, act_seq: np.ndarray) -> float:
        """计算动作效率奖励加成"""
        if len(act_seq) == 0:
            return 0.0
        
        try:
            # 计算动作的能量消耗（L2范数）
            action_energy = np.mean(np.sum(act_seq**2, axis=1))
            efficiency_bonus = np.exp(-action_energy * 0.1)  # 能量消耗越低奖励越高
            return float(efficiency_bonus)
        except Exception:
            pass
        return 0.0
    
    def _calculate_balance_control_bonus(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> float:
        """计算平衡控制奖励加成"""
        if len(obs_seq) < 2 or len(act_seq) == 0:
            return 0.0
        
        try:
            # 结合姿态稳定性和控制平滑性
            stability_bonus = self._calculate_standing_stability_bonus(obs_seq)
            smoothness_bonus = self._calculate_minimal_movement_bonus(act_seq)
            balance_bonus = (stability_bonus + smoothness_bonus) / 2
            return float(balance_bonus)
        except Exception:
            pass
        return 0.0
    
    def _calculate_energy_efficiency_bonus(self, act_seq: np.ndarray) -> float:
        """计算能量效率奖励加成"""
        if len(act_seq) == 0:
            return 0.0
        
        try:
            # 计算动作序列的总能量和效率
            total_energy = np.sum(np.sum(act_seq**2, axis=1))
            trajectory_length = len(act_seq)
            
            # 能量效率 = 轨迹长度 / 总能量消耗
            if total_energy > 0:
                efficiency = trajectory_length / (total_energy + 1e-8)
                efficiency_bonus = np.tanh(efficiency * 0.01)  # 归一化到[0,1]
                return float(efficiency_bonus)
        except Exception:
            pass
        return 0.0
    
    def _apply_label_smoothing(self, preference_logit: float) -> float:
        """应用标签平滑"""
        # Conservative DPO: 考虑标签噪声
        noise_factor = np.random.normal(0, self.label_smoothing)
        return preference_logit + noise_factor
    
    def _compute_confidence(self, reward_a: float, reward_b: float, preference_logit: float) -> float:
        """计算偏好置信度"""
        # 基于奖励差异和历史偏好分布计算置信度
        reward_diff = abs(float(reward_a) - float(reward_b))
        
        # 基于sigmoid函数的置信度
        logit_value = float(abs(preference_logit))
        base_confidence = torch.sigmoid(torch.tensor(logit_value, dtype=torch.float32)).item()
        
        # 考虑历史偏好的一致性
        if len(self.preference_history) > 10:
            recent_logits = [h['preference_logit'] for h in list(self.preference_history)[-10:]]
            consistency = 1.0 - np.std(recent_logits) / (np.mean(np.abs(recent_logits)) + 1e-8)
            consistency_bonus = max(0, consistency - 0.5) * 0.2
        else:
            consistency_bonus = 0.0
        
        final_confidence = min(base_confidence + consistency_bonus, 1.0)
        return final_confidence
    
    def evaluate_preference(self, trajectory_a: Dict[str, np.ndarray], 
                          trajectory_b: Dict[str, np.ndarray], 
                          preference_label: Optional[Any] = None) -> float:
        """评估轨迹偏好分数（兼容性方法）
        
        Args:
            trajectory_a: 轨迹A数据
            trajectory_b: 轨迹B数据
            preference_label: 偏好标签（可选）
            
        Returns:
            preference_score: 偏好分数 [0, 1]
        """
        preference_logit, _ = self.evaluate_dpo_preference(trajectory_a, trajectory_b)
        return torch.sigmoid(torch.tensor(preference_logit)).item()

class TrajectoryQualityEvaluator:
    """轨迹质量评估器 - 增强版，支持配置文件和任务特定指标"""
    
    def __init__(self, task_name: str = None, config: Dict[str, Any] = None):
        self.task_name = task_name.lower() if task_name else "default"
        self.config = config or {}
        self.feature_weights = self._load_feature_weights()
        self.task_specific_weights = self._load_task_specific_weights()
        self.thresholds = self._load_thresholds()
        
        # 导入轨迹指标计算器
        try:
            from trajectory_metrics import TrajectoryMetrics
            self.trajectory_metrics = TrajectoryMetrics()
        except ImportError:
            logger.warning("无法导入TrajectoryMetrics，将使用基础评估")
            self.trajectory_metrics = None
        
    def _load_feature_weights(self) -> Dict[str, float]:
        """从配置文件加载特征权重 - 改进版"""
        # 改进的默认权重配置 - 增加活跃度权重，减少生存权重
        default_weights = {
            'survival_time': 0.15,      # 降低从0.25
            'action_smoothness': 0.15,  # 保持
            'state_stability': 0.15,    # 保持
            'activity_score': 0.25,     # 增加从0.15
            'task_progress': 0.20,      # 增加从0.25
            'safety_score': 0.10        # 增加从0.05
        }
        
        # 任务特定权重调整
        task_weight_adjustments = {
            'walk': {'task_progress': 0.30, 'activity_score': 0.25},
            'run': {'task_progress': 0.35, 'activity_score': 0.25},
            'stand': {'state_stability': 0.30, 'activity_score': 0.10},
            'sit': {'state_stability': 0.25, 'activity_score': 0.05},
            'balance': {'state_stability': 0.30, 'activity_score': 0.15}
        }
        
        # 应用任务特定调整
        for task_key, adjustments in task_weight_adjustments.items():
            if task_key in self.task_name:
                for key, value in adjustments.items():
                    default_weights[key] = value
                break
        
        # 从配置中获取任务特定权重（覆盖默认调整）
        if self.config and 'tasks' in self.config:
            task_config = self.config['tasks'].get(self.task_name, {})
            if not task_config:
                # 尝试匹配任务名称模式
                task_config = self._match_task_config()
            
            quality_weights = task_config.get('quality_weights', {})
            default_weights.update(quality_weights)
        
        # 确保权重和为1
        total_weight = sum(default_weights.values())
        if total_weight > 0:
            default_weights = {k: v/total_weight for k, v in default_weights.items()}
            
        return default_weights
    
    def _load_task_specific_weights(self) -> Dict[str, float]:
        """加载任务特定指标权重"""
        if self.config and 'tasks' in self.config:
            task_config = self.config['tasks'].get(self.task_name, {})
            if not task_config:
                task_config = self._match_task_config()
            
            task_weights = task_config.get('task_specific_weights', {})
            
            # 确保权重和为1
            if task_weights:
                total_weight = sum(task_weights.values())
                if total_weight > 0:
                    task_weights = {k: v/total_weight for k, v in task_weights.items()}
            
            return task_weights
        
        return {}
    
    def _load_thresholds(self) -> Dict[str, float]:
        """加载评估阈值"""
        default_thresholds = {
            'min_survival_time': 10,
            'max_action_variance': 2.0,
            'min_task_progress': 0.1,
            'safety_threshold': 0.8
        }
        
        if self.config and 'tasks' in self.config:
            task_config = self.config['tasks'].get(self.task_name, {})
            if not task_config:
                task_config = self._match_task_config()
            
            thresholds = task_config.get('thresholds', {})
            default_thresholds.update(thresholds)
        
        return default_thresholds
    
    def _match_task_config(self) -> Dict[str, Any]:
        """匹配任务配置（支持模糊匹配）"""
        if not self.config or 'tasks' not in self.config:
            return {}
        
        tasks_config = self.config['tasks']
        
        # 精确匹配
        if self.task_name in tasks_config:
            return tasks_config[self.task_name]
        
        # 模糊匹配
        for task_key in tasks_config.keys():
            if task_key == 'default':
                continue
            
            # 检查任务名称中是否包含关键词
            if any(keyword in self.task_name for keyword in task_key.split('_')):
                return tasks_config[task_key]
            
            # 检查关键词匹配
            if 'walk' in self.task_name and 'walk' in task_key:
                return tasks_config[task_key]
            elif 'run' in self.task_name and 'run' in task_key:
                return tasks_config[task_key]
            elif 'stand' in self.task_name and 'stand' in task_key:
                return tasks_config[task_key]
            elif 'reach' in self.task_name and 'reach' in task_key:
                return tasks_config[task_key]
            elif 'push' in self.task_name and 'push' in task_key:
                return tasks_config[task_key]
            elif 'balance' in self.task_name and 'balance' in task_key:
                return tasks_config[task_key]
        
        # 返回默认配置
        return tasks_config.get('default', {})
    
    def evaluate_trajectory_quality(self, obs_seq: np.ndarray, act_seq: np.ndarray, 
                                  rewards: np.ndarray = None) -> Tuple[float, Dict[str, float]]:
        """评估轨迹质量 - 增强版
        
        Args:
            obs_seq: 观测序列 [T, obs_dim]
            act_seq: 动作序列 [T, act_dim]
            rewards: 奖励序列 [T] (可选)
            
        Returns:
            quality_score: 质量分数 [0, 1]
            feature_scores: 各特征分数字典
        """
        if len(obs_seq) == 0 or len(act_seq) == 0:
            return 0.0, {}
        
        # 转换为tensor以便计算，改进NumPy兼容性处理
        try:
            # 检查是否为numpy数组（避免版本兼容性问题）
            if hasattr(obs_seq, 'dtype') and hasattr(obs_seq, 'shape'):
                # 可能是numpy数组，尝试转换
                try:
                    obs_seq = torch.tensor(obs_seq).float()
                except:
                    obs_seq = torch.tensor(list(obs_seq)).float()
            elif not isinstance(obs_seq, torch.Tensor):
                obs_seq = torch.tensor(obs_seq).float()
                
            if hasattr(act_seq, 'dtype') and hasattr(act_seq, 'shape'):
                # 可能是numpy数组，尝试转换
                try:
                    act_seq = torch.tensor(act_seq).float()
                except:
                    act_seq = torch.tensor(list(act_seq)).float()
            elif not isinstance(act_seq, torch.Tensor):
                act_seq = torch.tensor(act_seq).float()
        except Exception as e:
            logger.warning(f"数据转换问题，使用默认质量分数: {e}")
            return 0.5, {'default_score': 0.5, 'error': str(e)}
            
        scores = {}
        
        # 基础特征分数
        scores['survival_time'] = self._calculate_survival_score(obs_seq)
        scores['action_smoothness'] = self._calculate_smoothness_score(act_seq)
        scores['state_stability'] = self._calculate_stability_score(obs_seq)
        scores['activity_score'] = self._calculate_activity_score(obs_seq)
        scores['task_progress'] = self._calculate_progress_score(obs_seq)
        scores['safety_score'] = self._calculate_safety_score(obs_seq, act_seq)
        
        # 计算环境奖励得分（如果提供了rewards）
        if rewards is not None and len(rewards) > 0:
            # 直接使用环境奖励总和，不进行归一化处理
            # 保持实际奖励权重，用于轨迹质量比较
            if isinstance(rewards, torch.Tensor):
                env_reward_sum = float(rewards.sum().item())
            else:
                env_reward_sum = float(np.sum(rewards))
            scores['env_reward_score'] = env_reward_sum
        else:
            # 如果没有提供环境奖励，使用默认值0.0
            scores['env_reward_score'] = 0.0
        
        # 计算任务特定指标
        if self.trajectory_metrics and self.task_specific_weights:
            task_metrics = self._compute_task_specific_metrics(obs_seq, act_seq, rewards)
            scores.update(task_metrics)
        
        # 应用阈值过滤
        scores = self._apply_thresholds(scores)
        
        # 计算API规则贡献
        api_contribution = 0.0
        if hasattr(self, 'api_rules') and self.api_rules:
            try:
                # 转换为numpy格式用于API规则计算
                obs_np = obs_seq.cpu().numpy() if isinstance(obs_seq, torch.Tensor) else obs_seq
                act_np = act_seq.cpu().numpy() if isinstance(act_seq, torch.Tensor) else act_seq
                api_contribution = self._apply_api_rules(obs_np, act_np, scores, {})
            except Exception as e:
                logger.warning(f"API规则计算失败: {e}")
                api_contribution = 0.0
        
        # 计算质量分数（新公式：环境奖励 × 基础质量因子 × (1 + API规则贡献)）
        total_score = self._compute_weighted_score(scores, api_contribution)
        
        scores['overall'] = total_score
        scores['api_contribution'] = api_contribution
        
        # 应用改进的质量过滤器
        scores = self._apply_quality_filters(scores)
        final_score = scores.get('overall', total_score)
        
        return final_score, scores
    
    def _calculate_safety_score(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> float:
        """计算安全性得分"""
        try:
            # 检查动作是否在合理范围内
            action_magnitude = torch.norm(act_seq, dim=-1).mean().item()
            action_safety = 1.0 / (1.0 + action_magnitude * 0.5)
            
            # 检查状态是否稳定（避免异常值）
            obs_magnitude = torch.norm(obs_seq, dim=-1).mean().item()
            state_safety = 1.0 / (1.0 + obs_magnitude * 0.1)
            
            # 综合安全性得分
            safety_score = (action_safety + state_safety) / 2.0
            return float(np.clip(safety_score, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _apply_thresholds(self, scores: Dict[str, float]) -> Dict[str, float]:
        """应用阈值过滤"""
        filtered_scores = {}
        
        for key, score in scores.items():
            # 应用最小阈值
            if key in self.thresholds:
                threshold_key = f"min_{key}"
                if threshold_key in self.thresholds:
                    min_threshold = self.thresholds[threshold_key]
                    if score < min_threshold:
                        score *= 0.5  # 惩罚低于阈值的分数
            
            filtered_scores[key] = score
        
        return filtered_scores
    
    def _apply_quality_filters(self, scores: Dict[str, float]) -> Dict[str, float]:
        """应用改进的质量过滤器"""
        filtered_scores = scores.copy()
        
        # 质量阈值
        quality_thresholds = {
            'min_activity': 0.15,    # 提高最小活跃度要求
            'min_progress': 0.08,    # 提高最小进展要求
            'max_static_penalty': 0.5,  # 增加静止轨迹惩罚
            'very_low_activity': 0.05,  # 极低活跃度阈值
            'severe_penalty': 0.7       # 严重惩罚
        }
        
        # 分级惩罚系统
        activity_score = scores.get('activity_score', 0.5)
        progress_score = scores.get('task_progress', 0.5)
        
        # 极低活跃度的严重惩罚
        if activity_score < quality_thresholds['very_low_activity']:
            penalty = quality_thresholds['severe_penalty']
            if 'overall' in filtered_scores:
                filtered_scores['overall'] = filtered_scores['overall'] * (1 - penalty)
        # 低活跃度的一般惩罚
        elif activity_score < quality_thresholds['min_activity']:
            penalty = quality_thresholds['max_static_penalty']
            if 'overall' in filtered_scores:
                filtered_scores['overall'] = filtered_scores['overall'] * (1 - penalty)
        
        # 无进展轨迹惩罚
        if progress_score < quality_thresholds['min_progress']:
            if 'overall' in filtered_scores:
                filtered_scores['overall'] *= 0.6  # 增加惩罚力度
        
        # 组合惩罚：既无活跃度又无进展的轨迹
        if (activity_score < quality_thresholds['min_activity'] and 
            progress_score < quality_thresholds['min_progress']):
            if 'overall' in filtered_scores:
                filtered_scores['overall'] *= 0.3  # 严重惩罚
        
        return filtered_scores
    
    def _compute_weighted_score(self, scores: Dict[str, float], api_contribution: float = 0.0) -> float:
        """计算质量分数 - 新公式：环境奖励 × 基础质量因子 × (1 + API规则贡献)"""
        # 提取核心分数
        survival_score = scores.get('survival_time', 0.5)
        env_reward_score = scores.get('env_reward_score', 0.0)  # 环境奖励得分（原始数值）
        stability_score = scores.get('state_stability', 0.5)
        smoothness_score = scores.get('action_smoothness', 0.5)
        
        # 确保其他分数在合理范围内
        survival_score = max(min(survival_score, 1.0), 0.1)
        stability_score = max(min(stability_score, 1.0), 0.1)
        smoothness_score = max(min(smoothness_score, 1.0), 0.1)
        
        # 计算基础质量因子（其他指标的乘积）
        base_quality_factor = survival_score * stability_score * smoothness_score
        
        # 限制API规则贡献范围到(-0.3, 0.3)
        api_contribution = max(min(api_contribution, 0.3), -0.3)
        
        # 新公式：最终分数 = 环境奖励 × 基础质量因子 × (1 + API规则贡献)
        quality_score = env_reward_score * base_quality_factor * (1 + api_contribution)
        
        return float(quality_score)  # 保持原始数值，不进行范围限制
    
    def _compute_task_specific_metrics(self, obs_seq: torch.Tensor, act_seq: torch.Tensor, 
                                       rewards = None) -> Dict[str, float]:
        """计算任务特定指标"""
        task_metrics = {}
        
        if not self.trajectory_metrics:
            return task_metrics
        
        try:
            # 转换为numpy格式（trajectory_metrics期望numpy输入），添加兼容性处理
            if NUMPY_AVAILABLE:
                obs_np = obs_seq.cpu().numpy() if isinstance(obs_seq, torch.Tensor) else obs_seq
                act_np = act_seq.cpu().numpy() if isinstance(act_seq, torch.Tensor) else act_seq
                default_rewards = np.zeros(len(obs_np))
            else:
                obs_np = obs_seq.cpu().tolist() if isinstance(obs_seq, torch.Tensor) else obs_seq
                act_np = act_seq.cpu().tolist() if isinstance(act_seq, torch.Tensor) else act_seq
                default_rewards = [0.0] * len(obs_np)
            
            # 准备轨迹数据
            trajectory_data = {
                'observations': obs_np,
                'actions': act_np,
                'rewards': rewards if rewards is not None else default_rewards
            }
            
            # 根据任务类型计算相应指标
            if 'balance' in self.task_name:
                task_metrics.update(self._compute_balance_metrics(trajectory_data))
            if 'basketball' in self.task_name:
                task_metrics.update(self._compute_basketball_metrics(trajectory_data))
            if 'bookshelf' in self.task_name:
                task_metrics.update(self._compute_bookshelf_metrics(trajectory_data))
            if 'cabinet' in self.task_name:
                task_metrics.update(self._compute_cabinet_metrics(trajectory_data))
            if 'climb' in self.task_name:
                task_metrics.update(self._compute_climb_metrics(trajectory_data))
            if 'crawl' in self.task_name:
                task_metrics.update(self._compute_crawl_metrics(trajectory_data))
            if 'cube' in self.task_name:
                task_metrics.update(self._compute_cube_metrics(trajectory_data))
            if 'door' in self.task_name:
                task_metrics.update(self._compute_door_metrics(trajectory_data))
            if 'hurdle' in self.task_name:
                task_metrics.update(self._compute_hurdle_metrics(trajectory_data))
            if 'insert' in self.task_name:
                task_metrics.update(self._compute_insert_metrics(trajectory_data))
            if 'kitchen' in self.task_name:
                task_metrics.update(self._compute_kitchen_metrics(trajectory_data))
            if 'maze' in self.task_name:
                task_metrics.update(self._compute_maze_metrics(trajectory_data))
            if 'package' in self.task_name:
                task_metrics.update(self._compute_package_metrics(trajectory_data))
            if 'pole' in self.task_name:
                task_metrics.update(self._compute_pole_metrics(trajectory_data))
            if 'powerlift' in self.task_name:
                task_metrics.update(self._compute_powerlift_metrics(trajectory_data))
            if 'push' in self.task_name:
                task_metrics.update(self._compute_push_metrics(trajectory_data))
            if 'reach' in self.task_name:
                task_metrics.update(self._compute_reach_metrics(trajectory_data))
            if 'room' in self.task_name:
                task_metrics.update(self._compute_room_metrics(trajectory_data))
            if 'run' in self.task_name:
                task_metrics.update(self._compute_run_metrics(trajectory_data))
            if 'sit' in self.task_name:
                task_metrics.update(self._compute_sit_metrics(trajectory_data))
            if 'slide' in self.task_name:
                task_metrics.update(self._compute_slide_metrics(trajectory_data))
            if 'spoon' in self.task_name:
                task_metrics.update(self._compute_spoon_metrics(trajectory_data))
            if 'stand' in self.task_name:
                task_metrics.update(self._compute_stand_metrics(trajectory_data))
            if 'truck' in self.task_name:
                task_metrics.update(self._compute_truck_metrics(trajectory_data))
            if 'walk' in self.task_name:
                task_metrics.update(self._compute_walk_metrics(trajectory_data))
            if 'window' in self.task_name:
                task_metrics.update(self._compute_window_metrics(trajectory_data))
            
        except Exception as e:
            logger.warning(f"任务特定指标计算失败: {e}")
        
        return task_metrics

    def _compute_walk_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算行走任务指标 - 基于新的humanoid_bench真实奖励函数"""
        metrics = {}
        try:
            # 新的指标计算，基于真实奖励函数组件
            if hasattr(self.trajectory_metrics, 'survival_stability'):
                metrics['survival_stability'] = self.trajectory_metrics.survival_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'forward_motion'):
                metrics['forward_motion'] = self.trajectory_metrics.forward_motion(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'action_efficiency'):
                metrics['action_efficiency'] = self.trajectory_metrics.action_efficiency(
                    trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'forward_progress'):
                metrics['forward_progress'] = self.trajectory_metrics.forward_progress(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'gait_quality'):
                metrics['gait_quality'] = self.trajectory_metrics.gait_quality(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'balance_stability'):
                metrics['balance_stability'] = self.trajectory_metrics.balance_stability(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"行走指标计算失败: {e}")
        return metrics
    
    def _compute_run_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算跑步任务指标 - 基于新的humanoid_bench真实奖励函数"""
        metrics = {}
        try:
            # 新的指标计算，基于真实奖励函数组件
            if hasattr(self.trajectory_metrics, 'standing_stability'):
                metrics['standing_stability'] = self.trajectory_metrics.standing_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'high_speed_motion'):
                metrics['high_speed_motion'] = self.trajectory_metrics.high_speed_motion(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'forward_progress'):
                metrics['forward_progress'] = self.trajectory_metrics.forward_progress(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'running_efficiency'):
                metrics['running_efficiency'] = self.trajectory_metrics.running_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'speed_consistency'):
                metrics['speed_consistency'] = self.trajectory_metrics.speed_consistency(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"跑步指标计算失败: {e}")
        return metrics
    
    def _compute_stand_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算站立任务指标 - 基于新的humanoid_bench真实奖励函数"""
        metrics = {}
        try:
            # 新的指标计算，基于真实奖励函数组件
            if hasattr(self.trajectory_metrics, 'standing_stability'):
                metrics['standing_stability'] = self.trajectory_metrics.standing_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'minimal_movement'):
                metrics['minimal_movement'] = self.trajectory_metrics.minimal_movement(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'height_stability'):
                metrics['height_stability'] = self.trajectory_metrics.height_stability(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'minimal_motion'):
                metrics['minimal_motion'] = self.trajectory_metrics.minimal_motion(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"站立指标计算失败: {e}")
        return metrics
    
    def _compute_reach_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算到达任务指标 - 基于新的humanoid_bench真实奖励函数"""
        metrics = {}
        try:
            # 新的指标计算，基于真实奖励函数组件
            if hasattr(self.trajectory_metrics, 'reaching_accuracy'):
                metrics['reaching_accuracy'] = self.trajectory_metrics.reaching_accuracy(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'torso_stability'):
                metrics['torso_stability'] = self.trajectory_metrics.torso_stability(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'motion_efficiency'):
                metrics['motion_efficiency'] = self.trajectory_metrics.motion_efficiency(
                    trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'hand_to_goal_distance'):
                metrics['hand_to_goal_distance'] = self.trajectory_metrics.hand_to_goal_distance(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'reach_efficiency'):
                metrics['reach_efficiency'] = self.trajectory_metrics.reach_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'reach_smoothness'):
                metrics['reach_smoothness'] = self.trajectory_metrics.reach_smoothness(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"到达指标计算失败: {e}")
        return metrics
    
    def _compute_push_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算推箱子任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'push_success_rate'):
                metrics['push_success_rate'] = self.trajectory_metrics.push_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'contact_efficiency'):
                metrics['contact_efficiency'] = self.trajectory_metrics.contact_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_stability'):
                metrics['control_stability'] = self.trajectory_metrics.control_stability(
                    trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'box_to_goal_distance'):
                metrics['box_to_goal_distance'] = self.trajectory_metrics.box_to_goal_distance(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'hand_to_box_distance'):
                metrics['hand_to_box_distance'] = self.trajectory_metrics.hand_to_box_distance(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'push_efficiency'):
                metrics['push_efficiency'] = self.trajectory_metrics.push_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"推箱子指标计算失败: {e}")
        return metrics
    
    def _compute_balance_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算平衡任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'standing_stability'):
                metrics['standing_stability'] = self.trajectory_metrics.standing_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'static_stability'):
                metrics['static_stability'] = self.trajectory_metrics.static_stability(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'stillness_stability'):
                metrics['stillness_stability'] = self.trajectory_metrics.stillness_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_smoothness'):
                metrics['control_smoothness'] = self.trajectory_metrics.control_smoothness(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"平衡指标计算失败: {e}")
        return metrics
    
    def _compute_hurdle_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算跨栏任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'forward_progress'):
                metrics['forward_progress'] = self.trajectory_metrics.forward_progress(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'obstacle_clearance'):
                metrics['obstacle_clearance'] = self.trajectory_metrics.obstacle_clearance(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'running_stability'):
                metrics['running_stability'] = self.trajectory_metrics.running_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'hurdle_obstacle_clearance'):
                metrics['hurdle_obstacle_clearance'] = self.trajectory_metrics.hurdle_obstacle_clearance(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'hurdle_speed_consistency'):
                metrics['hurdle_speed_consistency'] = self.trajectory_metrics.hurdle_speed_consistency(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'hurdle_balance_stability'):
                metrics['hurdle_balance_stability'] = self.trajectory_metrics.hurdle_balance_stability(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"跨栏指标计算失败: {e}")
        return metrics
    
    def _compute_maze_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算迷宫任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'navigation_success_rate'):
                metrics['navigation_success_rate'] = self.trajectory_metrics.navigation_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'standing_stability'):
                metrics['standing_stability'] = self.trajectory_metrics.standing_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'checkpoint_progress'):
                metrics['checkpoint_progress'] = self.trajectory_metrics.checkpoint_progress(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'navigation_efficiency'):
                metrics['navigation_efficiency'] = self.trajectory_metrics.navigation_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'maze_standing_stability'):
                metrics['maze_standing_stability'] = self.trajectory_metrics.maze_standing_stability(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"迷宫指标计算失败: {e}")
        return metrics
    
    def _compute_climb_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算爬楼梯任务指标 - 基于新的humanoid_bench真实奖励函数"""
        metrics = {}
        try:
            # 新的指标计算，基于真实奖励函数组件
            if hasattr(self.trajectory_metrics, 'climbing_stability'):
                metrics['climbing_stability'] = self.trajectory_metrics.climbing_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'forward_motion'):
                metrics['forward_motion'] = self.trajectory_metrics.forward_motion(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'height_gain'):
                metrics['height_gain'] = self.trajectory_metrics.height_gain(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'climbing_efficiency'):
                metrics['climbing_efficiency'] = self.trajectory_metrics.climbing_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'step_coordination'):
                metrics['step_coordination'] = self.trajectory_metrics.step_coordination(
                    trajectory_data['observations'], trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"爬楼梯指标计算失败: {e}")
        return metrics
    
    def _compute_pole_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算杆子平衡任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'standing_stability'):
                metrics['standing_stability'] = self.trajectory_metrics.standing_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"杆子平衡指标计算失败: {e}")
        return metrics
    
    def _compute_sit_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算坐姿任务指标 - 基于新的humanoid_bench真实奖励函数"""
        metrics = {}
        try:
            # 新的指标计算，基于真实奖励函数组件
            if hasattr(self.trajectory_metrics, 'sitting_quality'):
                metrics['sitting_quality'] = self.trajectory_metrics.sitting_quality(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'position_accuracy'):
                metrics['position_accuracy'] = self.trajectory_metrics.position_accuracy(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'control_stability'):
                metrics['control_stability'] = self.trajectory_metrics.control_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            
            # 保持向后兼容性，支持旧的指标名称
            if hasattr(self.trajectory_metrics, 'sitting_position_accuracy'):
                metrics['sitting_position_accuracy'] = self.trajectory_metrics.sitting_position_accuracy(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'sitting_stability'):
                metrics['sitting_stability'] = self.trajectory_metrics.sitting_stability(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"坐姿指标计算失败: {e}")
        return metrics
    

    def _compute_basketball_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算basketball任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'hand_coordination'):
                metrics['hand_coordination'] = self.trajectory_metrics.hand_coordination(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'precision_control'):
                metrics['precision_control'] = self.trajectory_metrics.precision_control(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"basketball指标计算失败: {e}")
        return metrics

    def _compute_bookshelf_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算bookshelf任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'reaching_accuracy'):
                metrics['reaching_accuracy'] = self.trajectory_metrics.reaching_accuracy(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'manipulation_success'):
                metrics['manipulation_success'] = self.trajectory_metrics.manipulation_success(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"bookshelf指标计算失败: {e}")
        return metrics

    def _compute_cabinet_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算cabinet任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'reaching_accuracy'):
                metrics['reaching_accuracy'] = self.trajectory_metrics.reaching_accuracy(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'manipulation_success'):
                metrics['manipulation_success'] = self.trajectory_metrics.manipulation_success(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"cabinet指标计算失败: {e}")
        return metrics

    def _compute_crawl_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算crawl任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'crawling_stability'):
                metrics['crawling_stability'] = self.trajectory_metrics.crawling_stability(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'low_profile_motion'):
                metrics['low_profile_motion'] = self.trajectory_metrics.low_profile_motion(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"crawl指标计算失败: {e}")
        return metrics

    def _compute_cube_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算cube任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'lifting_efficiency'):
                metrics['lifting_efficiency'] = self.trajectory_metrics.lifting_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'load_stability'):
                metrics['load_stability'] = self.trajectory_metrics.load_stability(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"cube指标计算失败: {e}")
        return metrics

    def _compute_door_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算door任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'reaching_accuracy'):
                metrics['reaching_accuracy'] = self.trajectory_metrics.reaching_accuracy(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'manipulation_success'):
                metrics['manipulation_success'] = self.trajectory_metrics.manipulation_success(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"door指标计算失败: {e}")
        return metrics

    def _compute_insert_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算insert任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'insertion_accuracy'):
                metrics['insertion_accuracy'] = self.trajectory_metrics.insertion_accuracy(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'fine_motor_control'):
                metrics['fine_motor_control'] = self.trajectory_metrics.fine_motor_control(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"insert指标计算失败: {e}")
        return metrics

    def _compute_kitchen_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算kitchen任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'navigation_success_rate'):
                metrics['navigation_success_rate'] = self.trajectory_metrics.navigation_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'object_interaction_success'):
                metrics['object_interaction_success'] = self.trajectory_metrics.object_interaction_success(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"kitchen指标计算失败: {e}")
        return metrics

    def _compute_package_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算package任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'lifting_efficiency'):
                metrics['lifting_efficiency'] = self.trajectory_metrics.lifting_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'load_stability'):
                metrics['load_stability'] = self.trajectory_metrics.load_stability(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"package指标计算失败: {e}")
        return metrics

    def _compute_powerlift_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算powerlift任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'lifting_power'):
                metrics['lifting_power'] = self.trajectory_metrics.lifting_power(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'strength_efficiency'):
                metrics['strength_efficiency'] = self.trajectory_metrics.strength_efficiency(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"powerlift指标计算失败: {e}")
        return metrics

    def _compute_room_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算room任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'navigation_success_rate'):
                metrics['navigation_success_rate'] = self.trajectory_metrics.navigation_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'object_interaction_success'):
                metrics['object_interaction_success'] = self.trajectory_metrics.object_interaction_success(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"room指标计算失败: {e}")
        return metrics

    def _compute_slide_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算slide任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'insertion_accuracy'):
                metrics['insertion_accuracy'] = self.trajectory_metrics.insertion_accuracy(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'fine_motor_control'):
                metrics['fine_motor_control'] = self.trajectory_metrics.fine_motor_control(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"slide指标计算失败: {e}")
        return metrics

    def _compute_spoon_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算spoon任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'hand_coordination'):
                metrics['hand_coordination'] = self.trajectory_metrics.hand_coordination(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'precision_control'):
                metrics['precision_control'] = self.trajectory_metrics.precision_control(
                    trajectory_data['actions']
                )
        except Exception as e:
            logger.warning(f"spoon指标计算失败: {e}")
        return metrics

    def _compute_truck_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算truck任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'lifting_efficiency'):
                metrics['lifting_efficiency'] = self.trajectory_metrics.lifting_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'load_stability'):
                metrics['load_stability'] = self.trajectory_metrics.load_stability(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"truck指标计算失败: {e}")
        return metrics

    def _compute_window_metrics(self, trajectory_data: Dict) -> Dict[str, float]:
        """计算window任务指标 - 基于真实奖励函数的DPO偏好评估"""
        metrics = {}
        try:
            # 基于真实奖励函数组件的新指标计算
            if hasattr(self.trajectory_metrics, 'task_success_rate'):
                metrics['task_success_rate'] = self.trajectory_metrics.task_success_rate(
                    trajectory_data['observations']
                )
            if hasattr(self.trajectory_metrics, 'movement_efficiency'):
                metrics['movement_efficiency'] = self.trajectory_metrics.movement_efficiency(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'control_efficiency'):
                metrics['control_efficiency'] = self.trajectory_metrics.control_efficiency(
                    trajectory_data['actions']
                )
            
            # 任务特定指标
            if hasattr(self.trajectory_metrics, 'reaching_accuracy'):
                metrics['reaching_accuracy'] = self.trajectory_metrics.reaching_accuracy(
                    trajectory_data['observations'], trajectory_data['actions']
                )
            if hasattr(self.trajectory_metrics, 'manipulation_success'):
                metrics['manipulation_success'] = self.trajectory_metrics.manipulation_success(
                    trajectory_data['observations']
                )
        except Exception as e:
            logger.warning(f"window指标计算失败: {e}")
        return metrics
    def _calculate_survival_score(self, obs_seq: torch.Tensor) -> float:
        """改进的生存得分计算 - 减少对长轨迹的过度奖励"""
        seq_len = len(obs_seq)
        
        # 使用对数缩放而不是线性缩放，避免过度奖励长轨迹
        length_factor = np.log(seq_len + 1) / np.log(1000 + 1)
        
        # 添加轨迹完整性检查
        try:
            # 检查是否有NaN或无穷大值
            obs_valid = not (torch.isnan(obs_seq).any() or torch.isinf(obs_seq).any())
            # 检查数据范围是否合理
            obs_range_valid = torch.all(torch.abs(obs_seq) < 1000)
            # 检查轨迹长度是否合理
            length_valid = 10 <= seq_len <= 1000
            
            completeness_factors = [obs_valid.item(), obs_range_valid.item(), length_valid]
            completeness = sum(completeness_factors) / len(completeness_factors)
        except Exception:
            completeness = 0.8  # 默认完整性
        
        # 结合长度和完整性，但限制长轨迹的过度奖励
        survival_score = 0.6 * length_factor + 0.4 * completeness
        
        return float(np.clip(survival_score, 0.1, 1.0))
    
    def _calculate_smoothness_score(self, act_seq: torch.Tensor) -> float:
        """计算动作平滑性得分"""
        if len(act_seq) <= 1:
            return 0.7  # 提高单步动作的默认分数
        
        # 计算动作变化的方差
        act_diff = torch.diff(act_seq, dim=0)
        action_variance = torch.mean(torch.norm(act_diff, dim=-1)).item()
        
        # 使用更温和的指数衰减函数，降低惩罚系数
        smoothness_score = np.exp(-action_variance * 0.5)  # 从2.0降低到0.5
        return float(max(min(smoothness_score, 1.0), 0.3))  # 提高最小值从0.1到0.3
    
    def _calculate_stability_score(self, obs_seq: torch.Tensor) -> float:
        """计算状态稳定性得分"""
        if len(obs_seq) <= 1:
            return 0.7  # 提高单步状态的默认分数
        
        # 计算状态的标准差
        obs_std = torch.std(obs_seq, dim=0).mean().item()
        
        # 稳定性与标准差成反比，使用更温和的惩罚系数
        stability_score = 1.0 / (1.0 + obs_std * 1.0)  # 从3.0降低到1.0
        return float(max(min(stability_score, 1.0), 0.3))  # 提高最小值从0.1到0.3
    
    def _calculate_activity_score(self, obs_seq: torch.Tensor) -> float:
        """改进的活跃度得分计算 - 更严格地惩罚静止轨迹"""
        if len(obs_seq) <= 1:
            return 0.1  # 避免返回0导致质量分数为0
        
        try:
            # 计算状态变化
            obs_change = torch.diff(obs_seq, dim=0)
            movement_magnitude = torch.mean(torch.norm(obs_change, dim=-1)).item()
            
            # 使用更陡峭的sigmoid函数，更严格地惩罚低活跃度
            activity_score = 2 / (1 + np.exp(-10 * movement_magnitude)) - 1
            
            # 额外的静止检测
            static_threshold = 0.01
            if movement_magnitude < static_threshold:
                # 对静止轨迹施加严厉惩罚，但保持最小值0.1
                activity_score = max(activity_score * 0.1, 0.1)
            
            return float(np.clip(activity_score, 0.1, 1.0))  # 最小值改为0.1
        except Exception:
            return 0.1  # 异常时返回0.1而不是0.0
    
    def _calculate_progress_score(self, obs_seq: torch.Tensor) -> float:
        """改进的进展得分计算 - 基于任务特定的进展指标"""
        if len(obs_seq) < 2:
            return 0.1  # 避免返回0导致质量分数为0
        
        try:
            # 基于位置变化计算进展（假设前3维是位置信息）
            pos_dim = min(3, obs_seq.shape[1])
            start_pos = obs_seq[0][:pos_dim]
            end_pos = obs_seq[-1][:pos_dim]
            
            displacement = torch.norm(end_pos - start_pos).item()
            
            # 任务特定的进展计算
            task_name = self.task_name.lower()
            if 'walk' in task_name or 'run' in task_name:
                # 对于移动任务，主要看前进距离
                forward_progress = (end_pos[0] - start_pos[0]).item() if pos_dim > 0 else displacement
                progress_score = np.tanh(forward_progress / 5.0)  # 归一化到[-1,1]
                progress_score = (progress_score + 1) / 2  # 映射到[0,1]
            elif 'stand' in task_name or 'balance' in task_name:
                # 对于站立任务，进展应该很小
                progress_score = 1.0 - np.tanh(displacement / 2.0)
            else:
                # 通用进展计算
                progress_score = np.tanh(displacement / 3.0)
            
            return float(np.clip(progress_score, 0.1, 1.0))  # 最小值改为0.1
        except Exception:
            return 0.1  # 异常时返回0.1而不是0.0

class PreferenceLabelingEngine:
    """基于DPO的统一偏好标签引擎 - 重构优化版
    
    核心改进：
    1. 统一DPO和传统方法的偏好标签生成
    2. 优化缓存机制和性能
    3. 增强轨迹质量评估
    4. 支持多种标签生成策略
    5. 集成统一偏好系统接口
    """
    
    def __init__(self, task_name_or_config: Union[str, Dict[str, Any]] = None, config_path: str = None, 
                 enable_cache: bool = True, enable_validation: bool = True,
                 dpo_beta: float = 0.1, dpo_label_smoothing: float = 0.0,
                 reward_model: Optional[Callable] = None,
                 unified_system: Optional[Any] = None):
        
        # 处理不同的初始化参数格式
        if isinstance(task_name_or_config, dict):
            # 如果第一个参数是字典，则作为配置使用
            self.config = task_name_or_config
            self.task_name = self.config.get('task_name', 'default')
        else:
            # 传统方式：第一个参数是任务名称
            self.task_name = task_name_or_config or 'default'
            self.config = self._load_config(config_path)
        
        self.reward_model = reward_model
        self.unified_system = unified_system  # 统一偏好系统引用
        self.enable_cache = enable_cache
        self.enable_validation = enable_validation
        
        # 初始化质量评估器（传递配置）
        self.quality_evaluator = TrajectoryQualityEvaluator(self.task_name, self.config)
        
        # 初始化DPO评估器（增强版，传递任务配置）
        self.dpo_evaluator = DPOPreferenceEvaluator(dpo_beta, dpo_label_smoothing, self.task_name, self.config)
        
        # 注册任务规则（确保规则可用）
        self.api_rules = {}
        self.api_rules_cache = {}
        if self.task_name:
            try:
                # 动态导入API规则
                self._load_api_rules()
                logger.info(f"成功加载任务 {self.task_name} 的API规则")
            except Exception as e:
                logger.warning(f"无法加载API规则: {e}，规则标签功能可能不可用")
        
        # 初始化统计信息
        self.statistics = LabelingStatistics()
        self.label_stats = defaultdict(int)  # 保持向后兼容
        
        # 初始化缓存管理器
        self.cache_manager = None
        self.batch_processor = None
        if self.enable_cache:
            self._initialize_cache_manager()
        
        # 初始化质量验证器
        self.quality_validator = None
        if self.enable_validation:
            self._initialize_quality_validator()
        
        # 缓存相关（优化版）
        self.quality_cache = {}
        self.preference_cache = {}  # 新增偏好缓存
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 性能优化参数
        self.batch_size_threshold = self.config.get('batch_size_threshold', 10)
        self.parallel_threshold = self.config.get('parallel_threshold', 4)
    
    def _load_api_rules(self):
        """动态加载API目录下的所有任务规则"""
        try:
            # 构建API规则文件路径
            api_dir = Path(__file__).parent / "api"
            if not api_dir.exists():
                logger.warning(f"API规则目录不存在: {api_dir}")
                return
            
            # 初始化API规则字典
            self.api_rules = {}
            self.api_base_path = "api"
            
            # 获取所有规则文件
            rule_files = list(api_dir.glob("rules_h1hand_*_v0.py"))
            
            if not rule_files:
                logger.info("未找到任何API规则文件")
                return
            
            logger.info(f"发现 {len(rule_files)} 个API规则文件")
            
            # 加载所有规则文件
            loaded_count = 0
            
            # 定义所有场景规则 - 包含完整的33个任务
            scenarios = [
                # H1Hand任务 (30个)
                'h1hand_balance_simple_v0', 'h1hand_balance_hard_v0', 'h1hand_basketball_v0',
                'h1hand_bookshelf_hard_v0', 'h1hand_bookshelf_simple_v0', 'h1hand_cabinet_v0',
                'h1hand_crawl_v0', 'h1hand_cube_v0', 'h1hand_door_v0', 'h1hand_hurdle_v0',
                'h1hand_insert_normal_v0', 'h1hand_insert_small_v0', 'h1hand_kitchen_v0',
                'h1hand_maze_v0', 'h1hand_package_v0', 'h1hand_pole_v0', 'h1hand_powerlift_v0',
                'h1hand_push_v0', 'h1hand_reach_v0', 'h1hand_room_v0', 'h1hand_run_v0',
                'h1hand_sit_hard_v0', 'h1hand_sit_simple_v0', 'h1hand_slide_v0', 'h1hand_spoon_v0',
                'h1hand_stair_v0', 'h1hand_stand_v0', 'h1hand_truck_v0', 'h1hand_walk_v0', 'h1hand_window_v0',
                # 经典控制任务 (3个)
                'cartpole_balance', 'cheetah_run', 'walker_walk'
            ]
            
            # 动态加载每个场景规则
            for scenario in scenarios:
                try:
                    # 经典控制任务没有_v0后缀
                    if scenario in ['cartpole_balance', 'cheetah_run', 'walker_walk']:
                        module_name = f'{self.api_base_path}.rules_{scenario}'
                        compare_func_name = f'compare_{scenario}_trajectories'
                    else:
                        # H1Hand任务已经包含_v0后缀，直接使用
                        module_name = f'{self.api_base_path}.rules_{scenario}'
                        compare_func_name = f'compare_{scenario}_trajectories'
                    
                    scenario_module = importlib.import_module(module_name)
                    
                    if hasattr(scenario_module, compare_func_name):
                        self.api_rules[scenario] = getattr(scenario_module, compare_func_name)
                        logger.info(f'成功加载 {scenario} 场景规则: {compare_func_name}')
                        loaded_count += 1
                    else:
                        logger.warning(f'{scenario} 模块中未找到函数: {compare_func_name}')
                        
                except Exception as e:
                    logger.warning(f'加载 {scenario} 场景规则失败: {e}')
            
            logger.info(f"总计成功加载 {loaded_count} 个API规则场景")
            
            # 如果当前任务有对应的规则，优先使用
            if self.task_name and self.task_name in self.api_rules:
                logger.info(f"当前任务 {self.task_name} 有对应的API规则")
            
        except Exception as e:
            logger.error(f"加载API规则失败: {e}")
            self.api_rules = {}
    
    def _apply_api_rules(self, obs_seq: np.ndarray, act_seq: np.ndarray, 
                        feature_scores: Dict[str, float], task_config: Dict[str, Any]) -> float:
        """应用API规则计算额外奖励"""
        if not self.api_rules:
            return 0.0
        
        try:
            # 构建轨迹数据
            trajectory = {
                'obs': obs_seq,
                'action': act_seq,
                'features': feature_scores,
                'states': obs_seq,  # 兼容性别名
                'actions': act_seq,  # 兼容性别名
                'observations': obs_seq  # 兼容性别名
            }
            
            total_bonus = 0.0
            rule_count = 0
            
            # 1. 查找并应用compute_*_reward_components函数
            reward_component_funcs = [name for name in self.api_rules.keys() 
                                    if name.startswith('compute_') and name.endswith('_reward_components')]
            
            for func_name in reward_component_funcs:
                try:
                    # 尝试不同的参数格式
                    func = self.api_rules[func_name]
                    components = None
                    
                    # 尝试多种调用方式
                    try:
                        # 方式1: 传入轨迹字典
                        components = func(trajectory)
                    except (TypeError, ValueError):
                        try:
                            # 方式2: 传入obs_seq, act_seq, reward_seq
                            reward_seq = np.zeros(len(obs_seq))  # 默认奖励序列
                            components = func(obs_seq, act_seq, reward_seq)
                        except (TypeError, ValueError):
                            try:
                                # 方式3: 传入states, actions
                                components = func(obs_seq, act_seq)
                            except (TypeError, ValueError):
                                try:
                                    # 方式4: 传入轨迹和配置
                                    components = func(trajectory, task_config)
                                except (TypeError, ValueError):
                                    logger.warning(f"无法调用API奖励组件函数 {func_name}")
                                    continue
                    
                    if components and isinstance(components, dict):
                        # 计算组件的加权平均值
                        component_values = [v for v in components.values() if isinstance(v, (int, float))]
                        if component_values:
                            avg_component_score = np.mean(component_values)
                            # 标准化到合理范围
                            normalized_score = np.tanh(float(avg_component_score))  # 标准化到[-1, 1]
                            total_bonus += normalized_score * 0.15  # 转换为[-0.15, 0.15]的奖励
                            rule_count += 1
                            logger.debug(f"应用API奖励组件 {func_name}: {avg_component_score:.4f}")
                        
                except Exception as e:
                    logger.warning(f"API奖励组件规则 {func_name} 应用失败: {e}")
            
            # 2. 查找并应用DPO偏好评估函数
            dpo_func_name = f"evaluate_dpo_preference"
            if dpo_func_name in self.api_rules:
                try:
                    # 构建两个轨迹进行比较（与基准轨迹比较）
                    baseline_trajectory = self._create_baseline_trajectory(obs_seq, act_seq)
                    
                    # 调用API规则的DPO评估函数
                    preference_result = self.api_rules[dpo_func_name](
                        trajectory, baseline_trajectory
                    )
                    
                    if isinstance(preference_result, (tuple, list)) and len(preference_result) >= 2:
                        preference_score, confidence = preference_result[:2]
                        # 将偏好分数转换为奖励加成
                        bonus = (preference_score - 0.5) * 0.2  # 偏好分数转换为[-0.1, 0.1]的奖励
                        total_bonus += bonus * confidence  # 按置信度加权
                        rule_count += 1
                    elif isinstance(preference_result, (int, float)):
                        bonus = (float(preference_result) - 0.5) * 0.2
                        total_bonus += bonus
                        rule_count += 1
                        
                except Exception as e:
                    logger.warning(f"API DPO规则应用失败: {e}")
            
            # 3. 查找并应用轨迹评分函数
            score_func_names = [name for name in self.api_rules.keys() 
                              if name in ['_compute_trajectory_score', 'compute_trajectory_score']]
            
            for score_func_name in score_func_names:
                try:
                    func = self.api_rules[score_func_name]
                    score = None
                    
                    # 尝试不同的调用方式
                    try:
                        score = func(trajectory)
                    except (TypeError, ValueError):
                        try:
                            score = func(obs_seq, act_seq)
                        except (TypeError, ValueError):
                            try:
                                score = func(obs_seq, act_seq, None)  # goal参数
                            except (TypeError, ValueError):
                                logger.warning(f"无法调用API评分函数 {score_func_name}")
                                continue
                    
                    if isinstance(score, (int, float)):
                        # 将分数标准化为奖励加成
                        normalized_score = np.tanh(float(score))  # 标准化到[-1, 1]
                        total_bonus += normalized_score * 0.1  # 转换为[-0.1, 0.1]的奖励
                        rule_count += 1
                        logger.debug(f"应用API评分函数 {score_func_name}: {score:.4f}")
                        
                except Exception as e:
                    logger.warning(f"API评分规则 {score_func_name} 应用失败: {e}")
            
            # 4. 查找并应用比较函数
            compare_func_names = [name for name in self.api_rules.keys() 
                                if name.startswith('compare_') and name.endswith('_trajectories')]
            
            for func_name in compare_func_names:
                try:
                    baseline_trajectory = self._create_baseline_trajectory(obs_seq, act_seq)
                    comparison_result = self.api_rules[func_name](trajectory, baseline_trajectory)
                    
                    if isinstance(comparison_result, (int, float)):
                        # 比较结果转换为奖励加成
                        bonus = float(comparison_result) * 0.05
                        total_bonus += bonus
                        rule_count += 1
                        
                except Exception as e:
                    logger.warning(f"API比较规则 {func_name} 应用失败: {e}")
            
            # 5. 查找并应用其他评估函数模式
            eval_func_names = [name for name in self.api_rules.keys() 
                             if name.startswith('_evaluate_') or name.startswith('evaluate_')]
            
            for func_name in eval_func_names:
                if func_name in ['evaluate_dpo_preference']:  # 跳过已处理的函数
                    continue
                    
                try:
                    func = self.api_rules[func_name]
                    result = None
                    
                    # 尝试不同的调用方式
                    try:
                        result = func(trajectory)
                    except (TypeError, ValueError):
                        try:
                            result = func(obs_seq)
                        except (TypeError, ValueError):
                            try:
                                result = func(obs_seq, act_seq)
                            except (TypeError, ValueError):
                                continue
                    
                    if isinstance(result, (int, float)):
                        # 标准化评估结果
                        normalized_result = np.tanh(float(result))  # 标准化到[-1, 1]
                        total_bonus += normalized_result * 0.05  # 转换为[-0.05, 0.05]的奖励
                        rule_count += 1
                        logger.debug(f"应用API评估函数 {func_name}: {result:.4f}")
                        
                except Exception as e:
                    logger.warning(f"API评估规则 {func_name} 应用失败: {e}")
            
            # 平均奖励加成
            if rule_count > 0:
                final_bonus = total_bonus / rule_count
                logger.debug(f"API规则总加成: {final_bonus:.4f} (来自 {rule_count} 个规则)")
                return final_bonus
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"API规则应用失败: {e}")
            return 0.0
    
    def _create_baseline_trajectory(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> Dict[str, np.ndarray]:
        """创建基准轨迹用于比较"""
        # 创建一个简单的基准轨迹（零动作或平均动作）
        baseline_actions = np.zeros_like(act_seq)
        return {
            'obs': obs_seq,
            'action': baseline_actions,
            'features': {}
        }
        self.batch_size_threshold = self.config.get('batch_size_threshold', 10)
        self.parallel_threshold = self.config.get('parallel_threshold', 4)
        
        logger.info(f"初始化统一偏好标签引擎，任务: {task_name}，缓存: {enable_cache}，验证: {enable_validation}")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """加载配置文件"""
        # 默认配置
        default_config = {
            'quality_threshold': 0.1,      # 质量差异阈值
            'confidence_threshold': 0.6,   # 置信度阈值
            'uncertainty_range': 0.05,     # 不确定性范围
            'label_smoothing': 0.1,        # 标签平滑参数
            'enable_caching': True,        # 启用缓存
            'parallel_processing': True,   # 并行处理
            'max_workers': 4,              # 最大工作线程数
            'batch_size': 32,              # 批处理大小
            'cache_size': 10000,           # 缓存大小
            'enable_disk_cache': True,     # 启用磁盘缓存
            'validation_level': 'standard' # 验证级别
        }
        
        # 尝试加载标签引擎专用配置
        labeling_config_path = os.path.join(os.path.dirname(__file__), "labeling_config.yaml")
        if os.path.exists(labeling_config_path):
            try:
                with open(labeling_config_path, 'r', encoding='utf-8') as f:
                    labeling_config = yaml.safe_load(f)
                
                # 合并引擎配置
                if 'engine' in labeling_config:
                    default_config.update(labeling_config['engine'])
                
                # 合并缓存配置
                if 'cache' in labeling_config:
                    default_config.update(labeling_config['cache'])
                
                # 合并验证配置
                if 'validation' in labeling_config:
                    default_config.update(labeling_config['validation'])
                
                # 合并任务特定配置
                if 'tasks' in labeling_config and self.task_name:
                    task_config = labeling_config['tasks'].get(self.task_name, 
                                                             labeling_config['tasks'].get('default', {}))
                    default_config.update(task_config)
                
                logger.info(f"已加载标签引擎配置文件: {labeling_config_path}")
            except Exception as e:
                logger.warning(f"加载标签引擎配置文件失败: {e}")
        
        # 加载用户指定的配置文件
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config.get('preference_labeling', {}))
                logger.info(f"已加载用户配置文件: {config_path}")
            except Exception as e:
                logger.warning(f"配置文件加载失败: {e}，使用默认配置")
        
        return default_config
    
    def _initialize_cache_manager(self):
        """初始化缓存管理器"""
        if self.enable_cache and self.cache_manager is None:
            try:
                from prm.label_cache_manager import create_label_cache_manager, create_batch_processor
                
                cache_dir = f"./cache/labels/{self.task_name}"
                self.cache_manager = create_label_cache_manager(
                    cache_dir=cache_dir,
                    max_cache_size=self.config.get('cache_size', 10000),
                    enable_disk_cache=self.config.get('enable_disk_cache', True)
                )
                
                self.batch_processor = create_batch_processor(
                    cache_manager=self.cache_manager,
                    labeling_engine=self,
                    batch_size=self.config.get('batch_size', 32),
                    max_workers=self.config.get('max_workers', 4)
                )
                
                logger.info("缓存管理器初始化完成")
            except ImportError:
                logger.warning("缓存管理器模块不可用，禁用缓存功能")
                self.enable_cache = False
    
    def _initialize_quality_validator(self):
        """初始化质量验证器"""
        # 检查配置中是否明确禁用质量验证器
        if self.config.get('quality_validator') is None:
            logger.info("配置中明确禁用质量验证器")
            self.enable_validation = False
            return
            
        if self.enable_validation and self.quality_validator is None:
            try:
                from prm.label_quality_validator import LabelQualityValidator, ValidationLevel
                
                validation_level_str = self.config.get('validation_level', 'standard')
                validation_level = ValidationLevel(validation_level_str)
                
                self.quality_validator = LabelQualityValidator(validation_level)
                logger.info(f"质量验证器初始化完成，验证级别: {validation_level_str}")
            except ImportError:
                logger.warning("质量验证器模块不可用，禁用验证功能")
                self.enable_validation = False
    
    def generate_unified_preference_labels(self, 
                                          obs_a: Union[np.ndarray, torch.Tensor], 
                                          act_a: Union[np.ndarray, torch.Tensor],
                                          obs_b: Union[np.ndarray, torch.Tensor], 
                                          act_b: Union[np.ndarray, torch.Tensor],
                                          label_type: LabelType = LabelType.DPO_BINARY,
                                          use_dpo: bool = True,
                                          batch_mode: bool = False) -> Union[PreferenceLabel, List[PreferenceLabel]]:
        """生成统一偏好标签（DPO优先）
        
        Args:
            obs_a, act_a: 轨迹A的观测和动作序列
            obs_b, act_b: 轨迹B的观测和动作序列
            label_type: 标签生成类型
            use_dpo: 是否优先使用DPO方法
            batch_mode: 是否为批处理模式
            
        Returns:
            偏好标签或标签列表
        """
        start_time = time.time()
        
        # 初始化缓存管理器
        if self.enable_cache:
            self._initialize_cache_manager()
        
        if batch_mode:
            return self._generate_unified_batch_labels(obs_a, act_a, obs_b, act_b, label_type, use_dpo)
        else:
            # 尝试从统一缓存获取
            cache_key = self._generate_cache_key(obs_a, act_a, obs_b, act_b, label_type, use_dpo)
            if self.enable_cache and cache_key in self.preference_cache:
                self.cache_hit_count += 1
                self.statistics.cache_hits += 1
                return self.preference_cache[cache_key]
            else:
                self.cache_miss_count += 1
                self.statistics.cache_misses += 1
            
            label = self._generate_unified_single_label(obs_a, act_a, obs_b, act_b, label_type, use_dpo, start_time)
            
            # 缓存新生成的标签
            if self.enable_cache and label.is_valid:
                self.preference_cache[cache_key] = label
                # 限制缓存大小
                if len(self.preference_cache) > self.config.get('cache_size', 10000):
                    # 移除最旧的缓存项
                    oldest_key = next(iter(self.preference_cache))
                    del self.preference_cache[oldest_key]
            
            return label
    
    def generate_preference_labels(self, 
                                 obs_a: Union[np.ndarray, torch.Tensor], 
                                 act_a: Union[np.ndarray, torch.Tensor],
                                 obs_b: Union[np.ndarray, torch.Tensor], 
                                 act_b: Union[np.ndarray, torch.Tensor],
                                 label_type: LabelType = LabelType.QUALITY_BASED,
                                 batch_mode: bool = False) -> Union[PreferenceLabel, List[PreferenceLabel]]:
        """生成偏好标签（向后兼容方法）
        
        Args:
            obs_a, act_a: 轨迹A的观测和动作序列
            obs_b, act_b: 轨迹B的观测和动作序列
            label_type: 标签生成类型
            batch_mode: 是否为批处理模式
            
        Returns:
            偏好标签或标签列表
        """
        # 根据标签类型决定是否使用DPO
        use_dpo = label_type in [LabelType.DPO_BINARY, LabelType.DPO_CONFIDENCE, 
                                LabelType.HYBRID_DPO_RULE, LabelType.HYBRID_DPO_QUALITY]
        
        return self.generate_unified_preference_labels(
            obs_a, act_a, obs_b, act_b, label_type, use_dpo, batch_mode
        )
    
    def _generate_cache_key(self, obs_a, act_a, obs_b, act_b, label_type, use_dpo=False) -> str:
        """生成缓存键"""
        try:
            # 处理label_type类型
            if isinstance(label_type, str):
                label_type_str = label_type
            elif hasattr(label_type, 'value'):
                label_type_str = label_type.value
            else:
                label_type_str = str(label_type)
            
            # 使用轨迹数据的哈希值作为缓存键
            obs_a_hash = hashlib.md5(np.array(obs_a).tobytes()).hexdigest()[:8]
            act_a_hash = hashlib.md5(np.array(act_a).tobytes()).hexdigest()[:8]
            obs_b_hash = hashlib.md5(np.array(obs_b).tobytes()).hexdigest()[:8]
            act_b_hash = hashlib.md5(np.array(act_b).tobytes()).hexdigest()[:8]
            
            cache_key = f"{obs_a_hash}_{act_a_hash}_{obs_b_hash}_{act_b_hash}_{label_type_str}_{use_dpo}"
            return cache_key
        except Exception:
            # 回退到简单的字符串键
            if isinstance(label_type, str):
                label_type_str = label_type
            elif hasattr(label_type, 'value'):
                label_type_str = label_type.value
            else:
                label_type_str = str(label_type)
            return f"traj_{len(obs_a)}_{len(act_a)}_{len(obs_b)}_{len(act_b)}_{label_type_str}_{use_dpo}"
    
    def _generate_unified_single_label(self, obs_a, act_a, obs_b, act_b, label_type, use_dpo, start_time) -> PreferenceLabel:
        """生成统一单个偏好标签（DPO优先）"""
        try:
            # 确保在子线程中numpy可用
            import numpy as np
            
            # 转换数据格式
            obs_a = self._ensure_numpy(obs_a)
            act_a = self._ensure_numpy(act_a)
            obs_b = self._ensure_numpy(obs_b)
            act_b = self._ensure_numpy(act_b)
            
            # 验证输入数据
            validation_errors = self._validate_trajectory_data(obs_a, act_a, obs_b, act_b)
            if validation_errors:
                return self._create_invalid_label(validation_errors, start_time)
            
            # 根据use_dpo标志选择生成方法
            if use_dpo:
                return self._generate_dpo_label(obs_a, act_a, obs_b, act_b, label_type, start_time)
            else:
                return self._generate_traditional_label(obs_a, act_a, obs_b, act_b, label_type, start_time)
                
        except Exception as e:
            logger.error(f"统一标签生成失败: {e}")
            return self._create_error_label(str(e), start_time)
    
    def _generate_dpo_label(self, obs_a, act_a, obs_b, act_b, label_type, start_time) -> PreferenceLabel:
        """使用DPO方法生成偏好标签"""
        try:
            # 构建轨迹字典
            trajectory_a = {'obs': obs_a, 'action': act_a}
            trajectory_b = {'obs': obs_b, 'action': act_b}
            
            # 使用DPO评估器计算偏好
            dpo_logit, confidence = self.dpo_evaluator.evaluate_dpo_preference(
                trajectory_a, trajectory_b, self.reward_model
            )
            
            # 转换为偏好分数
            preference_score = torch.sigmoid(torch.tensor(dpo_logit)).item()
            
            # 计算轨迹质量（用于元数据）
            quality_a, features_a = self._get_trajectory_quality(obs_a, act_a)
            quality_b, features_b = self._get_trajectory_quality(obs_b, act_b)
            
            # 创建元数据
            metadata = LabelMetadata(
                label_type=label_type,
                confidence=confidence,
                quality_score_a=quality_a,
                quality_score_b=quality_b,
                score_difference=abs(quality_a - quality_b),
                generation_time=time.time() - start_time,
                features_used=list(features_a.keys()),
                additional_info={
                    'dpo_logit': dpo_logit,
                    'method': 'dpo_unified',
                    'features_a': features_a,
                    'features_b': features_b
                }
            )
            
            # 更新统计信息
            generation_time = time.time() - start_time
            self.statistics.update_generation_stats(label_type, generation_time, True)
            self.label_stats[label_type.value] += 1
            
            return PreferenceLabel(
                preference_score=preference_score,
                logit_preference=dpo_logit,
                binary_preference=1 if preference_score > 0.6 else (-1 if preference_score < 0.4 else 0),
                metadata=metadata,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"DPO标签生成失败: {e}")
            return self._create_error_label(str(e), start_time)
    
    def _generate_traditional_label(self, obs_a, act_a, obs_b, act_b, label_type, start_time) -> PreferenceLabel:
        """使用传统方法生成偏好标签"""
        try:
            # 计算轨迹质量
            quality_a, features_a = self._get_trajectory_quality(obs_a, act_a)
            quality_b, features_b = self._get_trajectory_quality(obs_b, act_b)
            
            # 生成偏好分数和置信度
            preference_score, confidence = self._calculate_preference_score(
                quality_a, quality_b, label_type, obs_a, act_a, obs_b, act_b
            )
            
            # 创建元数据
            metadata = LabelMetadata(
                label_type=label_type,
                confidence=confidence,
                quality_score_a=quality_a,
                quality_score_b=quality_b,
                score_difference=abs(quality_a - quality_b),
                generation_time=time.time() - start_time,
                features_used=list(features_a.keys()),
                additional_info={
                    'method': 'traditional',
                    'features_a': features_a,
                    'features_b': features_b
                }
            )
            
            # 更新统计信息
            generation_time = time.time() - start_time
            self.statistics.update_generation_stats(label_type, generation_time, True)
            self.label_stats[label_type.value] += 1
            
            return PreferenceLabel(
                preference_score=preference_score,
                metadata=metadata,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"传统标签生成失败: {e}")
            return self._create_error_label(str(e), start_time)
    
    def _create_invalid_label(self, validation_errors, start_time) -> PreferenceLabel:
        """创建无效标签"""
        return PreferenceLabel(
            preference_score=0.5,
            metadata=LabelMetadata(
                label_type=LabelType.UNCERTAIN,
                confidence=0.0,
                quality_score_a=0.0,
                quality_score_b=0.0,
                score_difference=0.0,
                generation_time=time.time() - start_time,
                features_used=[],
                additional_info={'errors': validation_errors}
            ),
            is_valid=False,
            validation_errors=validation_errors
        )
    
    def _create_error_label(self, error_msg, start_time) -> PreferenceLabel:
        """创建错误标签"""
        return PreferenceLabel(
            preference_score=0.5,
            metadata=LabelMetadata(
                label_type=LabelType.UNCERTAIN,
                confidence=0.0,
                quality_score_a=0.0,
                quality_score_b=0.0,
                score_difference=0.0,
                generation_time=time.time() - start_time,
                features_used=[],
                additional_info={'error': error_msg}
            ),
            is_valid=False,
            validation_errors=[error_msg]
        )
    
    def _generate_single_label(self, obs_a, act_a, obs_b, act_b, label_type, start_time) -> PreferenceLabel:
        """生成单个偏好标签"""
        try:
            # 确保在子线程中numpy可用
            import numpy as np
            
            # 转换数据格式
            obs_a = self._ensure_numpy(obs_a)
            act_a = self._ensure_numpy(act_a)
            obs_b = self._ensure_numpy(obs_b)
            act_b = self._ensure_numpy(act_b)
            
            # 验证输入数据
            validation_errors = self._validate_trajectory_data(obs_a, act_a, obs_b, act_b)
            if validation_errors:
                return PreferenceLabel(
                    preference_score=0.5,
                    metadata=LabelMetadata(
                        label_type=LabelType.UNCERTAIN,
                        confidence=0.0,
                        quality_score_a=0.0,
                        quality_score_b=0.0,
                        score_difference=0.0,
                        generation_time=time.time() - start_time,
                        features_used=[],
                        additional_info={'errors': validation_errors}
                    ),
                    is_valid=False,
                    validation_errors=validation_errors
                )
            
            # 计算轨迹质量
            quality_a, features_a = self._get_trajectory_quality(obs_a, act_a)
            quality_b, features_b = self._get_trajectory_quality(obs_b, act_b)
            
            # 生成偏好分数和置信度
            preference_score, confidence = self._calculate_preference_score(
                quality_a, quality_b, label_type, obs_a, act_a, obs_b, act_b
            )
            
            # 创建元数据
            metadata = LabelMetadata(
                label_type=label_type,
                confidence=confidence,
                quality_score_a=quality_a,
                quality_score_b=quality_b,
                score_difference=abs(quality_a - quality_b),
                generation_time=time.time() - start_time,
                features_used=list(features_a.keys()),
                additional_info={
                    'features_a': features_a,
                    'features_b': features_b
                }
            )
            
            # 更新统计信息
            generation_time = time.time() - start_time
            self.statistics.update_generation_stats(label_type, generation_time, True)
            self.label_stats[label_type.value] += 1  # 保持向后兼容
            
            return PreferenceLabel(
                preference_score=preference_score,
                metadata=metadata,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"标签生成失败: {e}")
            return PreferenceLabel(
                preference_score=0.5,
                metadata=LabelMetadata(
                    label_type=LabelType.UNCERTAIN,
                    confidence=0.0,
                    quality_score_a=0.0,
                    quality_score_b=0.0,
                    score_difference=0.0,
                    generation_time=time.time() - start_time,
                    features_used=[],
                    additional_info={'error': str(e)}
                ),
                is_valid=False,
                validation_errors=[str(e)]
            )
    
    def _generate_batch_labels(self, obs_a, act_a, obs_b, act_b, label_type) -> List[PreferenceLabel]:
        """批量生成偏好标签（支持缓存和并行处理）"""
        try:
            # 确保输入是numpy数组或列表
            if isinstance(obs_a, np.ndarray) and obs_a.ndim == 2:
                # 单个轨迹，转换为批量格式
                obs_a = [obs_a]
                act_a = [act_a]
                obs_b = [obs_b]
                act_b = [act_b]
            
            batch_size = len(obs_a) if isinstance(obs_a, (list, tuple)) else obs_a.shape[0]
            labels = []
            
            # 初始化批处理器
            if self.enable_cache:
                self._initialize_cache_manager()
                
                if self.batch_processor is not None:
                    batch_data = [(obs_a[i], act_a[i], obs_b[i], act_b[i], label_type) for i in range(batch_size)]
                    return self.batch_processor.process_batch(batch_data)
            
            # 回退到原有处理方式
            if self.config['parallel_processing'] and batch_size > 4:
                # 并行处理
                with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                    futures = []
                    for i in range(batch_size):
                        future = executor.submit(
                            self._generate_single_label,
                            obs_a[i], act_a[i], obs_b[i], act_b[i],
                            label_type, time.time()
                        )
                        futures.append(future)
                    
                    labels = [future.result() for future in futures]
            else:
                # 串行处理
                for i in range(batch_size):
                    label = self._generate_single_label(
                        obs_a[i], act_a[i], obs_b[i], act_b[i],
                        label_type, time.time()
                    )
                    labels.append(label)
            
            return labels
            
        except Exception as e:
            logger.error(f"批量标签生成失败: {e}")
            # 回退到单个标签生成
            return [self._generate_single_label(obs_a, act_a, obs_b, act_b, label_type, time.time())]
    
    def _ensure_numpy(self, data: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        """确保数据为numpy格式"""
        import numpy as np
        
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def _validate_trajectory_data(self, obs_a, act_a, obs_b, act_b) -> List[str]:
        """验证轨迹数据的有效性"""
        import numpy as np
        
        errors = []
        
        # 检查数据形状
        if len(obs_a) == 0 or len(act_a) == 0:
            errors.append("轨迹A数据为空")
        if len(obs_b) == 0 or len(act_b) == 0:
            errors.append("轨迹B数据为空")
        
        # 检查序列长度匹配
        if len(obs_a) != len(act_a):
            errors.append(f"轨迹A观测和动作长度不匹配: {len(obs_a)} vs {len(act_a)}")
        if len(obs_b) != len(act_b):
            errors.append(f"轨迹B观测和动作长度不匹配: {len(obs_b)} vs {len(act_b)}")
        
        # 检查数据类型和范围
        try:
            if np.any(np.isnan(obs_a)) or np.any(np.isnan(act_a)):
                errors.append("轨迹A包含NaN值")
            if np.any(np.isnan(obs_b)) or np.any(np.isnan(act_b)):
                errors.append("轨迹B包含NaN值")
        except Exception as e:
            errors.append(f"数据验证异常: {e}")
        
        return errors
    
    def _get_trajectory_quality(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """获取轨迹质量（带缓存）"""
        import numpy as np
        
        if not self.config['enable_caching']:
            return self.quality_evaluator.evaluate_trajectory_quality(obs_seq, act_seq)
        
        # 生成缓存键
        cache_key = hash((obs_seq.tobytes(), act_seq.tobytes()))
        
        if cache_key in self.quality_cache:
            self.statistics.cache_hits += 1
            return self.quality_cache[cache_key]
        
        self.statistics.cache_misses += 1
        quality, features = self.quality_evaluator.evaluate_trajectory_quality(obs_seq, act_seq)
        
        # 限制缓存大小
        if len(self.quality_cache) > 1000:
            # 清除最旧的一半缓存
            keys_to_remove = list(self.quality_cache.keys())[:500]
            for key in keys_to_remove:
                del self.quality_cache[key]
        
        self.quality_cache[cache_key] = (quality, features)
        return quality, features
    
    def _calculate_preference_score(self, quality_a: float, quality_b: float, label_type: LabelType, 
                                  obs_a: np.ndarray = None, act_a: np.ndarray = None,
                                  obs_b: np.ndarray = None, act_b: np.ndarray = None) -> Tuple[float, float]:
        """计算偏好分数和置信度
        
        Args:
            quality_a, quality_b: 轨迹A和B的质量分数
            label_type: 标签类型
            obs_a, act_a, obs_b, act_b: 轨迹数据（用于规则比较）
        """
        # 使用DPO方法计算偏好
        if label_type in [LabelType.DPO_BINARY, LabelType.DPO_CONFIDENCE, 
                         LabelType.HYBRID_DPO_RULE, LabelType.HYBRID_DPO_QUALITY]:
            # 构建轨迹数据字典
            trajectory_a = {'obs': obs_a, 'action': act_a}
            trajectory_b = {'obs': obs_b, 'action': act_b}
            
            # 使用DPO评估器计算偏好
            preference_logit, confidence = self.dpo_evaluator.evaluate_dpo_preference(
                trajectory_a, trajectory_b, self.reward_model
            )
            
            # 将logit转换为概率分数
            preference_score = torch.sigmoid(torch.tensor(preference_logit)).item()
            
            return preference_score, confidence
        elif label_type == LabelType.RULE_BASED:
            # 规则偏好对：使用启发式规则进行比较
            return self._calculate_rule_based_score(obs_a, act_a, obs_b, act_b)
        elif label_type == LabelType.HYBRID_DPO_RULE:
            # DPO与规则混合：DPO为主，规则为辅
            trajectory_a = {'obs': obs_a, 'action': act_a}
            trajectory_b = {'obs': obs_b, 'action': act_b}
            
            dpo_logit, dpo_conf = self.dpo_evaluator.evaluate_dpo_preference(
                trajectory_a, trajectory_b, self.reward_model
            )
            rule_score, rule_conf = self._calculate_rule_based_score(obs_a, act_a, obs_b, act_b)
            
            # DPO主导，规则调节
            dpo_score = torch.sigmoid(torch.tensor(dpo_logit)).item()
            combined_score = 0.7 * dpo_score + 0.3 * rule_score
            combined_conf = max(dpo_conf, rule_conf)  # 取最高置信度
            
            return combined_score, combined_conf
        elif label_type == LabelType.HYBRID_DPO_QUALITY:
            # DPO与质量混合：DPO为主，质量为辅
            trajectory_a = {'obs': obs_a, 'action': act_a}
            trajectory_b = {'obs': obs_b, 'action': act_b}
            
            dpo_logit, dpo_conf = self.dpo_evaluator.evaluate_dpo_preference(
                trajectory_a, trajectory_b, self.reward_model
            )
            quality_score, quality_conf = self._calculate_quality_based_score(quality_a, quality_b, LabelType.QUALITY_BASED)
            
            # DPO主导，质量调节
            dpo_score = torch.sigmoid(torch.tensor(dpo_logit)).item()
            combined_score = 0.8 * dpo_score + 0.2 * quality_score
            combined_conf = (dpo_conf + quality_conf) / 2
            
            return combined_score, combined_conf
        elif label_type == LabelType.TRAJECTORY_ALIGNED:
            # 轨迹对齐标签：确保生成与评估一致性
            trajectory_a = {'obs': obs_a, 'action': act_a}
            trajectory_b = {'obs': obs_b, 'action': act_b}
            
            # 使用DPO评估器，但加入轨迹一致性检查
            preference_logit, confidence = self.dpo_evaluator.evaluate_dpo_preference(
                trajectory_a, trajectory_b, self.reward_model
            )
            
            # 轨迹一致性检查
            consistency_score = self._check_trajectory_consistency(trajectory_a, trajectory_b)
            adjusted_confidence = confidence * consistency_score
            
            preference_score = torch.sigmoid(torch.tensor(preference_logit)).item()
            return preference_score, adjusted_confidence
        else:
            # 采集偏好对：使用质量评估进行比较
            return self._calculate_quality_based_score(quality_a, quality_b, label_type)
    
    def _calculate_rule_based_score(self, obs_a: np.ndarray, act_a: np.ndarray, 
                                  obs_b: np.ndarray, act_b: np.ndarray) -> Tuple[float, float]:
        """基于启发式规则计算偏好分数（修复：统一使用质量评估避免训练偏差）"""
        try:
            # 修复：规则偏好对也使用质量评估，确保训练一致性
            quality_a, _ = self.quality_evaluator.evaluate_trajectory_quality(obs_a, act_a)
            quality_b, _ = self.quality_evaluator.evaluate_trajectory_quality(obs_b, act_b)
            
            # 使用统一的质量评估方法，但为规则偏好对增加轻微的置信度提升
            preference_score, confidence = self._calculate_quality_based_score(quality_a, quality_b, LabelType.RULE_BASED)
            
            # 规则偏好对的置信度略微提升，但不改变偏好分数的计算逻辑
            confidence = min(confidence * 1.1, 1.0)
            
            return preference_score, confidence
            
        except Exception as e:
            logger.error(f"规则比较失败: {e}，使用默认中性分数")
            return 0.5, 0.1
    
    def _calculate_quality_based_score(self, quality_a: float, quality_b: float, label_type: LabelType) -> Tuple[float, float]:
        """基于质量评估计算偏好分数（修复训练问题：生成更多有意义的偏好）"""
        import numpy as np
        
        quality_diff = quality_a - quality_b
        abs_diff = abs(quality_diff)
        
        # 修复：极大降低不确定性阈值，让几乎所有样本都参与训练
        uncertainty_range = self.config.get('uncertainty_range', 0.01)  # 从0.05降到0.01
        min_uncertainty_threshold = uncertainty_range * 0.1  # 从0.5降到0.1，进一步降低阈值
        
        if abs_diff < min_uncertainty_threshold:
            # 只有极极小的差异才标记为不确定
            preference_score = 0.5
            confidence = 0.3
        else:
            # 修复：使用极敏感的sigmoid缩放，让微小差异也能产生明显偏好
            sigmoid_input = quality_diff * 10.0  # 从3.0增加到10.0，大幅增加敏感度
            raw_score = torch.sigmoid(torch.tensor(sigmoid_input)).item()
            
            # 修复：几乎不使用标签平滑，最大程度保留偏好信息
            smoothing = self.config.get('label_smoothing', 0.01) * 0.1  # 从0.1*0.2降到0.01*0.1
            preference_score = (1 - smoothing) * raw_score + smoothing * 0.5
            
            # 修复：大幅提高置信度，让模型更积极学习
            confidence = min(abs_diff * 10.0 + 0.5, 0.95)  # 从3.0+0.4提高到10.0+0.5
        
        # 修复：统一所有标签类型的置信度处理
        if label_type == LabelType.RULE_BASED:
            # 规则标签置信度轻微提升
            confidence = min(confidence * 1.1, 0.9)
        elif label_type == LabelType.HEURISTIC_BASED:
            # 启发式标签置信度轻微降低
            confidence = confidence * 0.9
        
        return float(np.clip(preference_score, 0.0, 1.0)), float(np.clip(confidence, 0.0, 1.0))
    
    def _check_trajectory_consistency(self, trajectory_a: Dict[str, np.ndarray], 
                                    trajectory_b: Dict[str, np.ndarray]) -> float:
        """检查轨迹一致性，确保生成与评估对齐
        
        Args:
            trajectory_a, trajectory_b: 轨迹数据字典
            
        Returns:
            consistency_score: 一致性分数 [0, 1]
        """
        try:
            obs_a = trajectory_a.get('obs', [])
            obs_b = trajectory_b.get('obs', [])
            act_a = trajectory_a.get('action', [])
            act_b = trajectory_b.get('action', [])
            
            if len(obs_a) == 0 or len(obs_b) == 0:
                return 0.5  # 数据不足，返回中性分数
            
            # 1. 轨迹长度一致性
            len_ratio = min(len(obs_a), len(obs_b)) / max(len(obs_a), len(obs_b))
            length_consistency = len_ratio
            
            # 2. 动作空间一致性
            if len(act_a) > 0 and len(act_b) > 0:
                act_a_array = np.array(act_a)
                act_b_array = np.array(act_b)
                
                # 检查动作维度一致性
                if act_a_array.shape[-1] == act_b_array.shape[-1]:
                    action_consistency = 1.0
                else:
                    action_consistency = 0.0
            else:
                action_consistency = 0.5
            
            # 3. 观测空间一致性
            obs_a_array = np.array(obs_a)
            obs_b_array = np.array(obs_b)
            
            if obs_a_array.shape[-1] == obs_b_array.shape[-1]:
                obs_consistency = 1.0
            else:
                obs_consistency = 0.0
            
            # 4. 任务特定一致性检查
            task_consistency = self._check_task_specific_consistency(trajectory_a, trajectory_b)
            
            # 综合一致性分数
            overall_consistency = (
                0.3 * length_consistency +
                0.3 * action_consistency +
                0.2 * obs_consistency +
                0.2 * task_consistency
            )
            
            return float(np.clip(overall_consistency, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"轨迹一致性检查失败: {e}")
            return 0.5  # 出错时返回中性分数
    
    def _check_task_specific_consistency(self, trajectory_a: Dict[str, np.ndarray],
                                       trajectory_b: Dict[str, np.ndarray]) -> float:
        """任务特定的一致性检查
        
        Args:
            trajectory_a, trajectory_b: 轨迹数据字典
            
        Returns:
            task_consistency: 任务一致性分数 [0, 1]
        """
        try:
            # 基于任务名称进行特定检查
            if self.task_name is None:
                return 1.0
            
            task_lower = self.task_name.lower()
            
            # 不同任务的特定一致性检查
            if 'reach' in task_lower:
                return self._check_reach_consistency(trajectory_a, trajectory_b)
            elif 'walk' in task_lower or 'run' in task_lower:
                return self._check_locomotion_consistency(trajectory_a, trajectory_b)
            elif 'stand' in task_lower:
                return self._check_stand_consistency(trajectory_a, trajectory_b)
            elif 'push' in task_lower:
                return self._check_manipulation_consistency(trajectory_a, trajectory_b)
            else:
                # 通用一致性检查
                return self._check_general_consistency(trajectory_a, trajectory_b)
                
        except Exception as e:
            logger.warning(f"任务特定一致性检查失败: {e}")
            return 1.0
    
    def _check_reach_consistency(self, trajectory_a: Dict[str, np.ndarray],
                               trajectory_b: Dict[str, np.ndarray]) -> float:
        """到达任务一致性检查"""
        # 检查手部位置变化的合理性
        obs_a = trajectory_a.get('obs', [])
        obs_b = trajectory_b.get('obs', [])
        
        if len(obs_a) < 2 or len(obs_b) < 2:
            return 1.0
        
        # 假设手部位置在观测的前3维
        try:
            hand_pos_a_start = np.array(obs_a[0][:3])
            hand_pos_a_end = np.array(obs_a[-1][:3])
            hand_pos_b_start = np.array(obs_b[0][:3])
            hand_pos_b_end = np.array(obs_b[-1][:3])
            
            # 检查移动距离的合理性
            move_dist_a = np.linalg.norm(hand_pos_a_end - hand_pos_a_start)
            move_dist_b = np.linalg.norm(hand_pos_b_end - hand_pos_b_start)
            
            # 移动距离应该在合理范围内
            reasonable_range = (0.01, 2.0)  # 1cm到2m
            consistency_a = 1.0 if reasonable_range[0] <= move_dist_a <= reasonable_range[1] else 0.5
            consistency_b = 1.0 if reasonable_range[0] <= move_dist_b <= reasonable_range[1] else 0.5
            
            return (consistency_a + consistency_b) / 2
            
        except Exception:
            return 1.0
    
    def _check_locomotion_consistency(self, trajectory_a: Dict[str, np.ndarray],
                                    trajectory_b: Dict[str, np.ndarray]) -> float:
        """运动任务一致性检查"""
        # 检查位置变化和速度的合理性
        obs_a = trajectory_a.get('obs', [])
        obs_b = trajectory_b.get('obs', [])
        
        if len(obs_a) < 3 or len(obs_b) < 3:
            return 1.0
        
        try:
            # 假设根部位置在观测的某个位置
            pos_changes_a = []
            pos_changes_b = []
            
            for i in range(1, min(len(obs_a), 10)):  # 检查前10步
                if len(obs_a[i]) >= 3 and len(obs_a[i-1]) >= 3:
                    pos_change = np.linalg.norm(np.array(obs_a[i][:3]) - np.array(obs_a[i-1][:3]))
                    pos_changes_a.append(pos_change)
            
            for i in range(1, min(len(obs_b), 10)):
                if len(obs_b[i]) >= 3 and len(obs_b[i-1]) >= 3:
                    pos_change = np.linalg.norm(np.array(obs_b[i][:3]) - np.array(obs_b[i-1][:3]))
                    pos_changes_b.append(pos_change)
            
            if not pos_changes_a or not pos_changes_b:
                return 1.0
            
            # 检查移动速度的合理性
            avg_speed_a = np.mean(pos_changes_a)
            avg_speed_b = np.mean(pos_changes_b)
            
            # 合理的移动速度范围
            reasonable_speed_range = (0.001, 0.5)  # 每步0.1cm到50cm
            consistency_a = 1.0 if reasonable_speed_range[0] <= avg_speed_a <= reasonable_speed_range[1] else 0.5
            consistency_b = 1.0 if reasonable_speed_range[0] <= avg_speed_b <= reasonable_speed_range[1] else 0.5
            
            return (consistency_a + consistency_b) / 2
            
        except Exception:
            return 1.0
    
    def _check_stand_consistency(self, trajectory_a: Dict[str, np.ndarray],
                               trajectory_b: Dict[str, np.ndarray]) -> float:
        """站立任务一致性检查"""
        # 检查姿态稳定性
        obs_a = trajectory_a.get('obs', [])
        obs_b = trajectory_b.get('obs', [])
        
        if len(obs_a) < 5 or len(obs_b) < 5:
            return 1.0
        
        try:
            # 检查观测的方差（稳定性指标）
            obs_a_array = np.array(obs_a)
            obs_b_array = np.array(obs_b)
            
            variance_a = np.mean(np.var(obs_a_array, axis=0))
            variance_b = np.mean(np.var(obs_b_array, axis=0))
            
            # 站立任务应该有较低的方差
            max_reasonable_variance = 1.0
            consistency_a = 1.0 if variance_a <= max_reasonable_variance else 0.5
            consistency_b = 1.0 if variance_b <= max_reasonable_variance else 0.5
            
            return (consistency_a + consistency_b) / 2
            
        except Exception:
            return 1.0
    
    def _check_manipulation_consistency(self, trajectory_a: Dict[str, np.ndarray],
                                      trajectory_b: Dict[str, np.ndarray]) -> float:
        """操作任务一致性检查"""
        # 检查操作动作的合理性
        act_a = trajectory_a.get('action', [])
        act_b = trajectory_b.get('action', [])
        
        if len(act_a) < 2 or len(act_b) < 2:
            return 1.0
        
        try:
            # 检查动作的变化幅度
            act_a_array = np.array(act_a)
            act_b_array = np.array(act_b)
            
            action_changes_a = np.mean(np.abs(np.diff(act_a_array, axis=0)))
            action_changes_b = np.mean(np.abs(np.diff(act_b_array, axis=0)))
            
            # 操作任务应该有适度的动作变化
            reasonable_change_range = (0.01, 2.0)
            consistency_a = 1.0 if reasonable_change_range[0] <= action_changes_a <= reasonable_change_range[1] else 0.5
            consistency_b = 1.0 if reasonable_change_range[0] <= action_changes_b <= reasonable_change_range[1] else 0.5
            
            return (consistency_a + consistency_b) / 2
            
        except Exception:
            return 1.0
    
    def _check_general_consistency(self, trajectory_a: Dict[str, np.ndarray],
                                 trajectory_b: Dict[str, np.ndarray]) -> float:
        """通用一致性检查"""
        # 基本的数据完整性检查
        obs_a = trajectory_a.get('obs', [])
        obs_b = trajectory_b.get('obs', [])
        act_a = trajectory_a.get('action', [])
        act_b = trajectory_b.get('action', [])
        
        # 检查数据完整性
        data_completeness_a = 1.0 if len(obs_a) > 0 and len(act_a) > 0 else 0.5
        data_completeness_b = 1.0 if len(obs_b) > 0 and len(act_b) > 0 else 0.5
        
        return (data_completeness_a + data_completeness_b) / 2
    
    def validate_label_quality(self, labels: List[PreferenceLabel]) -> Optional[Any]:
        """验证标签质量"""
        if not self.enable_validation or not labels:
            return None
        
        self._initialize_quality_validator()
        
        if self.quality_validator is not None:
            return self.quality_validator.validate_labels(labels)
        
        return None
    
    def validate_labels(self, labels: List[PreferenceLabel]) -> Dict[str, Any]:
        """验证标签质量（保持向后兼容）"""
        import numpy as np
        
        if not labels:
            return {'valid': False, 'error': '标签列表为空'}
        
        validation_result = {
            'total_labels': len(labels),
            'valid_labels': sum(1 for label in labels if label.is_valid),
            'invalid_labels': sum(1 for label in labels if not label.is_valid),
            'average_confidence': np.mean([label.metadata.confidence for label in labels if label.is_valid]),
            'label_distribution': defaultdict(int),
            'quality_stats': {
                'mean_score_diff': np.mean([label.metadata.score_difference for label in labels if label.is_valid]),
                'std_score_diff': np.std([label.metadata.score_difference for label in labels if label.is_valid])
            }
        }
        
        # 统计标签分布
        for label in labels:
            if label.preference_score < 0.4:
                validation_result['label_distribution']['prefer_b'] += 1
            elif label.preference_score > 0.6:
                validation_result['label_distribution']['prefer_a'] += 1
            else:
                validation_result['label_distribution']['uncertain'] += 1
        
        validation_result['valid'] = validation_result['valid_labels'] > 0
        
        return validation_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取标签引擎统计信息"""
        total_labels = sum(self.label_stats.values())
        
        stats = {
            'total_labels_generated': total_labels,
            'label_type_distribution': dict(self.label_stats),
            'cache_statistics': {
                'hits': self.statistics.cache_hits,
                'misses': self.statistics.cache_misses,
                'hit_rate': self.statistics.cache_hit_rate
            },
            'cache_size': len(self.quality_cache),
            # 新的统计信息
            'total_generated': self.statistics.total_generated,
            'avg_generation_time': self.statistics.avg_generation_time,
            'rule_based_count': self.statistics.rule_based_count,
            'heuristic_based_count': self.statistics.heuristic_based_count,
            'valid_labels': self.statistics.valid_labels,
            'invalid_labels': self.statistics.invalid_labels
        }
        
        # 添加缓存管理器统计信息
        if self.cache_manager is not None:
            cache_stats = self.cache_manager.get_cache_stats()
            stats.update({
                'advanced_cache_hit_rate': cache_stats.hit_rate,
                'advanced_cache_size': cache_stats.cache_size,
                'avg_cache_time': cache_stats.avg_cache_time
            })
        
        return stats
    
    def cleanup_cache(self):
        """清理缓存"""
        if self.cache_manager is not None:
            self.cache_manager.cleanup_expired_cache()
    
    def clear_cache(self):
        """清除缓存"""
        self.quality_cache.clear()
        self.statistics.cache_hits = 0
        self.statistics.cache_misses = 0
        
        if self.cache_manager is not None:
            self.cache_manager.clear_cache()
        
        logger.info("缓存已清除")
    
    def get_cache_stats(self) -> Optional[Any]:
        """获取缓存统计信息"""
        if self.cache_manager is not None:
            return self.cache_manager.get_cache_stats()
        return None
    
    def print_cache_stats(self):
        """打印缓存统计信息"""
        if self.cache_manager is not None:
            self.cache_manager.print_cache_stats()
        else:
            print("缓存功能未启用")
    
    def save_config(self, config_path: str):
        """保存当前配置"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump({'preference_labeling': self.config}, f, default_flow_style=False)
            logger.info(f"配置已保存到: {config_path}")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")

# 工具函数
def create_preference_labeling_engine(task_name: str = None, config_path: str = None, 
                                    enable_cache: bool = True, 
                                    enable_validation: bool = True) -> PreferenceLabelingEngine:
    """创建偏好标签引擎的工厂函数（带全局缓存优化）"""
    global _global_engine_cache, _global_cache_timestamps, _global_cache_max_age
    
    # 清理过期缓存
    current_time = time.time()
    expired_keys = []
    for key, timestamp in _global_cache_timestamps.items():
        if current_time - timestamp > _global_cache_max_age:
            expired_keys.append(key)
    
    for key in expired_keys:
        _global_engine_cache.pop(key, None)
        _global_cache_timestamps.pop(key, None)
    
    # 生成缓存键
    cache_key = f"{task_name}_{config_path}_{enable_cache}_{enable_validation}"
    
    # 检查缓存
    if (cache_key in _global_engine_cache and 
        cache_key in _global_cache_timestamps and
        current_time - _global_cache_timestamps[cache_key] < _global_cache_max_age):
        logger.info(f"使用缓存的偏好标签引擎: {task_name}")
        return _global_engine_cache[cache_key]
    
    # 创建新引擎
    engine = PreferenceLabelingEngine(task_name_or_config=task_name, config_path=config_path, enable_cache=enable_cache, enable_validation=enable_validation)
    
    # 缓存引擎
    _global_engine_cache[cache_key] = engine
    _global_cache_timestamps[cache_key] = current_time
    
    logger.info(f"创建并缓存新的偏好标签引擎: {task_name}")
    return engine

def batch_label_preference_pairs(engine: PreferenceLabelingEngine,
                                trajectory_pairs: List[Tuple],
                                label_type: LabelType = LabelType.QUALITY_BASED) -> List[PreferenceLabel]:
    """批量处理偏好对标签"""
    labels = []
    for (obs_a, act_a), (obs_b, act_b) in trajectory_pairs:
        label = engine.generate_preference_labels(obs_a, act_a, obs_b, act_b, label_type)
        labels.append(label)
    return labels

if __name__ == "__main__":
    # 测试代码
    print("测试偏好标签引擎...")
    
    # 创建引擎
    engine = create_preference_labeling_engine("humanoid_walk")
    
    # 生成测试数据
    obs_a = np.random.randn(50, 151)
    act_a = np.random.randn(50, 61)
    obs_b = np.random.randn(45, 151)
    act_b = np.random.randn(45, 61)
    
    # 生成标签
    label = engine.generate_preference_labels(obs_a, act_a, obs_b, act_b)
    
    print(f"生成的标签:")
    print(f"  偏好分数: {label.preference_score:.3f}")
    print(f"  置信度: {label.metadata.confidence:.3f}")
    print(f"  标签类型: {label.metadata.label_type.value}")
    print(f"  质量分数A: {label.metadata.quality_score_a:.3f}")
    print(f"  质量分数B: {label.metadata.quality_score_b:.3f}")
    print(f"  是否有效: {label.is_valid}")
    
    # 获取统计信息
    stats = engine.get_statistics()
    print(f"\n引擎统计: {stats}")
    
    print("测试完成！")