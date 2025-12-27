#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一偏好标签系统 - 基于DPO的优化实现

核心设计理念：
1. 统一历史轨迹作为偏好对生成的唯一来源
2. 基于DPO (Direct Preference Optimization) 的统一标签生成
3. 消除规则偏好对和采集偏好对的区分
4. 优化标签引擎，最大化DPO系统效果
5. 集成轨迹质量评估与偏好学习的统一框架

主要优化：
- 统一偏好对生成流程
- 增强DPO标签质量
- 优化缓存和性能
- 改进轨迹质量评估
"""

import os
import sys
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import hashlib
import pickle
import yaml

# 导入现有模块
try:
    from .preference_labeling_engine import (
        PreferenceLabelingEngine, 
        LabelType, 
        DPOPreferenceEvaluator,
        PreferenceLabel,
        LabelMetadata,
        TrajectoryQualityEvaluator
    )
    from .trajectory_metrics import TrajectoryMetrics
    from .label_cache_manager import LabelCacheManager
except ImportError:
    # 处理相对导入问题
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preference_labeling_engine import (
        PreferenceLabelingEngine, 
        LabelType, 
        DPOPreferenceEvaluator,
        PreferenceLabel,
        LabelMetadata,
        TrajectoryQualityEvaluator
    )
    from trajectory_metrics import TrajectoryMetrics
    from label_cache_manager import LabelCacheManager

logger = logging.getLogger(__name__)

@dataclass
class UnifiedTrajectory:
    """统一轨迹数据结构"""
    trajectory_id: str
    obs_sequence: np.ndarray  # 观测序列 [T, obs_dim]
    action_sequence: np.ndarray  # 动作序列 [T, act_dim]
    reward_sequence: np.ndarray  # 奖励序列 [T]
    done_sequence: np.ndarray  # 完成标志序列 [T]
    
    # 元数据
    episode_idx: int
    step_range: Tuple[int, int]
    total_reward: float
    length: int
    quality_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    # DPO相关属性
    dpo_reward_estimate: Optional[float] = None
    preference_features: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """转换为字典格式，兼容现有接口"""
        return {
            'obs': self.obs_sequence,
            'action': self.action_sequence,
            'reward': self.reward_sequence,
            'done': self.done_sequence
        }
    
    def get_trajectory_hash(self) -> str:
        """计算轨迹哈希值用于缓存"""
        content = f"{self.episode_idx}_{self.step_range}_{self.obs_sequence.shape}_{self.action_sequence.shape}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

@dataclass
class UnifiedPreferencePair:
    """统一偏好对数据结构"""
    trajectory_a: UnifiedTrajectory
    trajectory_b: UnifiedTrajectory
    preference_label: PreferenceLabel
    
    # 生成信息
    generation_method: str  # "dpo_unified", "quality_based", "hybrid"
    generation_timestamp: float = field(default_factory=time.time)
    
    # DPO特定信息
    dpo_logit: Optional[float] = None
    confidence_score: float = 0.5
    
    def to_training_format(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """转换为训练格式"""
        chosen_seq = (self.trajectory_a.obs_sequence, self.trajectory_a.action_sequence)
        rejected_seq = (self.trajectory_b.obs_sequence, self.trajectory_b.action_sequence)
        
        # 根据偏好分数决定chosen/rejected
        if self.preference_label.preference_score > 0.5:
            return chosen_seq, rejected_seq
        else:
            return rejected_seq, chosen_seq

class UnifiedPreferenceSystem:
    """统一偏好标签系统 - 核心类"""
    
    def __init__(self, 
                 task_name: str,
                 config: Dict[str, Any] = None,
                 work_dir: str = None):
        self.task_name = task_name
        self.config = config or {}
        self.work_dir = work_dir or "/tmp/unified_preference"
        
        # 初始化组件
        self._init_components()
        
        # 统一轨迹存储
        self.unified_trajectories: List[UnifiedTrajectory] = []
        self.trajectory_index: Dict[str, UnifiedTrajectory] = {}
        
        # 统一偏好对存储
        self.unified_preference_pairs: List[UnifiedPreferencePair] = []
        
        # 性能统计
        self.stats = {
            'total_trajectories': 0,
            'total_preference_pairs': 0,
            'dpo_pairs_generated': 0,
            'quality_pairs_generated': 0,
            'hybrid_pairs_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'generation_time_total': 0.0,
            'avg_generation_time': 0.0
        }
        
        # 线程安全
        self.lock = threading.RLock()
        
        logger.info(f"统一偏好系统初始化完成 - 任务: {task_name}")
    
    def _init_components(self):
        """初始化系统组件"""
        # DPO评估器配置
        dpo_config = self.config.get('dpo', {})
        self.dpo_evaluator = DPOPreferenceEvaluator(
            beta=dpo_config.get('beta', 0.1),
            label_smoothing=dpo_config.get('label_smoothing', 0.0)
        )
        
        # 轨迹质量评估器
        self.quality_evaluator = TrajectoryQualityEvaluator(
            task_name=self.task_name,
            config=self.config
        )
        
        # 偏好标签引擎
        labeling_config = self.config.get('labeling', {})
        self.labeling_engine = PreferenceLabelingEngine(
            task_name_or_config=self.task_name,
            config_path=labeling_config.get('config_path'),
            **labeling_config
        )
        
        # 轨迹指标计算器
        try:
            self.trajectory_metrics = TrajectoryMetrics()
        except Exception as e:
            logger.warning(f"轨迹指标计算器初始化失败: {e}")
            self.trajectory_metrics = None
        
        # 缓存管理器
        cache_config = self.config.get('cache', {})
        # 修正参数名称映射
        cache_params = {
            'cache_dir': os.path.join(self.work_dir, 'cache'),
            'max_cache_size': cache_config.get('max_size', 1000),
            'enable_disk_cache': cache_config.get('enable_disk_cache', True),
            'enable_memory_cache': cache_config.get('enable_memory_cache', True),
            'cache_ttl': cache_config.get('ttl_hours', 24) * 3600
        }
        self.cache_manager = LabelCacheManager(**cache_params)
        
        # 配置参数
        self.max_trajectories = self.config.get('max_trajectories', 1000)
        self.preference_pairs_per_generation = self.config.get('preference_pairs_per_generation', 50)
        self.quality_threshold = self.config.get('quality_threshold', 0.1)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
    
    def add_trajectory(self, 
                      obs_seq: np.ndarray,
                      action_seq: np.ndarray,
                      reward_seq: np.ndarray,
                      done_seq: np.ndarray,
                      episode_idx: int,
                      step_range: Tuple[int, int]) -> str:
        """添加轨迹到统一系统
        
        Args:
            obs_seq: 观测序列
            action_seq: 动作序列  
            reward_seq: 奖励序列
            done_seq: 完成标志序列
            episode_idx: 回合索引
            step_range: 步数范围
            
        Returns:
            trajectory_id: 轨迹ID
        """
        with self.lock:
            # 创建统一轨迹
            trajectory_id = f"traj_{episode_idx}_{step_range[0]}_{step_range[1]}_{int(time.time())}"
            
            # 计算轨迹质量
            quality_score, quality_features = self.quality_evaluator.evaluate_trajectory_quality(
                obs_seq, action_seq, reward_seq
            )
            
            # 计算DPO奖励估计
            dpo_reward = self._estimate_dpo_reward(obs_seq, action_seq, reward_seq)
            
            unified_traj = UnifiedTrajectory(
                trajectory_id=trajectory_id,
                obs_sequence=obs_seq,
                action_sequence=action_seq,
                reward_sequence=reward_seq,
                done_sequence=done_seq,
                episode_idx=episode_idx,
                step_range=step_range,
                total_reward=np.sum(reward_seq),
                length=len(obs_seq),
                quality_score=quality_score,
                dpo_reward_estimate=dpo_reward,
                preference_features=quality_features
            )
            
            # 添加到存储
            self.unified_trajectories.append(unified_traj)
            self.trajectory_index[trajectory_id] = unified_traj
            
            # 维护最大轨迹数量限制
            if len(self.unified_trajectories) > self.max_trajectories:
                removed_traj = self.unified_trajectories.pop(0)
                del self.trajectory_index[removed_traj.trajectory_id]
            
            self.stats['total_trajectories'] += 1
            
            logger.debug(f"添加轨迹: {trajectory_id}, 质量分数: {quality_score:.4f}")
            return trajectory_id
    
    def _estimate_dpo_reward(self, obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> float:
        """估计DPO奖励"""
        try:
            # 构建轨迹字典
            trajectory_dict = {
                'obs': obs_seq,
                'action': action_seq,
                'reward': reward_seq
            }
            
            # 使用DPO评估器的启发式奖励估计
            dpo_reward = self.dpo_evaluator._heuristic_reward_estimate(trajectory_dict)
            return dpo_reward
        except Exception as e:
            logger.warning(f"DPO奖励估计失败: {e}")
            return np.mean(reward_seq) if len(reward_seq) > 0 else 0.0
    
    def generate_unified_preference_pairs(self, 
                                        num_pairs: Optional[int] = None,
                                        generation_method: str = "dpo_unified") -> List[UnifiedPreferencePair]:
        """生成统一偏好对
        
        Args:
            num_pairs: 要生成的偏好对数量，None表示使用默认配置
            generation_method: 生成方法 ("dpo_unified", "quality_based", "hybrid")
            
        Returns:
            生成的统一偏好对列表
        """
        start_time = time.time()
        
        with self.lock:
            if len(self.unified_trajectories) < 2:
                logger.warning("轨迹数量不足，无法生成偏好对")
                return []
            
            num_pairs = num_pairs or self.preference_pairs_per_generation
            generated_pairs = []
            
            # 根据生成方法选择策略
            if generation_method == "dpo_unified":
                generated_pairs = self._generate_dpo_unified_pairs(num_pairs)
            elif generation_method == "quality_based":
                generated_pairs = self._generate_quality_based_pairs(num_pairs)
            elif generation_method == "hybrid":
                generated_pairs = self._generate_hybrid_pairs(num_pairs)
            else:
                raise ValueError(f"未知的生成方法: {generation_method}")
            
            # 更新统计信息
            generation_time = time.time() - start_time
            self.stats['total_preference_pairs'] += len(generated_pairs)
            self.stats[f'{generation_method.split("_")[0]}_pairs_generated'] += len(generated_pairs)
            self.stats['generation_time_total'] += generation_time
            self.stats['avg_generation_time'] = self.stats['generation_time_total'] / max(1, self.stats['total_preference_pairs'])
            
            # 添加到统一存储
            self.unified_preference_pairs.extend(generated_pairs)
            
            logger.info(f"生成 {len(generated_pairs)} 个统一偏好对，方法: {generation_method}, 耗时: {generation_time:.4f}s")
            return generated_pairs
    
    def _generate_dpo_unified_pairs(self, num_pairs: int) -> List[UnifiedPreferencePair]:
        """使用DPO统一方法生成偏好对"""
        pairs = []
        trajectories = self.unified_trajectories.copy()
        
        # 基于DPO奖励估计排序
        trajectories.sort(key=lambda t: t.dpo_reward_estimate or 0.0, reverse=True)
        
        # 智能配对策略：混合随机和质量差异配对
        for _ in range(num_pairs):
            if len(trajectories) < 2:
                break
            
            # 70%概率使用质量差异配对，30%概率随机配对
            if np.random.random() < 0.7:
                # 质量差异配对：选择质量差异较大的轨迹对
                traj_a, traj_b = self._select_quality_diverse_pair(trajectories)
            else:
                # 随机配对：增加多样性
                traj_a, traj_b = np.random.choice(trajectories, 2, replace=False)
            
            # 生成DPO偏好标签
            preference_pair = self._create_dpo_preference_pair(traj_a, traj_b, "dpo_unified")
            if preference_pair:
                pairs.append(preference_pair)
        
        return pairs
    
    def _generate_quality_based_pairs(self, num_pairs: int) -> List[UnifiedPreferencePair]:
        """基于质量评估生成偏好对"""
        pairs = []
        trajectories = self.unified_trajectories.copy()
        
        # 基于质量分数排序
        trajectories.sort(key=lambda t: t.quality_score, reverse=True)
        
        for _ in range(num_pairs):
            if len(trajectories) < 2:
                break
            
            # 选择质量差异明显的轨迹对
            traj_a, traj_b = self._select_quality_diverse_pair(trajectories)
            
            # 生成质量基础偏好标签
            preference_pair = self._create_quality_preference_pair(traj_a, traj_b, "quality_based")
            if preference_pair:
                pairs.append(preference_pair)
        
        return pairs
    
    def _generate_hybrid_pairs(self, num_pairs: int) -> List[UnifiedPreferencePair]:
        """混合方法生成偏好对"""
        pairs = []
        
        # 50% DPO方法，50% 质量方法
        dpo_pairs = self._generate_dpo_unified_pairs(num_pairs // 2)
        quality_pairs = self._generate_quality_based_pairs(num_pairs - len(dpo_pairs))
        
        # 合并并标记为混合方法
        for pair in dpo_pairs + quality_pairs:
            pair.generation_method = "hybrid"
            pairs.append(pair)
        
        return pairs
    
    def _select_quality_diverse_pair(self, trajectories: List[UnifiedTrajectory]) -> Tuple[UnifiedTrajectory, UnifiedTrajectory]:
        """选择质量差异较大的轨迹对"""
        if len(trajectories) < 2:
            return trajectories[0], trajectories[0]
        
        # 计算所有轨迹对的质量差异
        max_diff = 0.0
        best_pair = (trajectories[0], trajectories[1])
        
        # 采样策略：不遍历所有组合，而是采样一定数量
        sample_size = min(50, len(trajectories) * (len(trajectories) - 1) // 2)
        
        for _ in range(sample_size):
            i, j = np.random.choice(len(trajectories), 2, replace=False)
            traj_a, traj_b = trajectories[i], trajectories[j]
            
            # 计算综合差异（质量分数 + DPO奖励估计）
            quality_diff = abs(traj_a.quality_score - traj_b.quality_score)
            dpo_diff = abs((traj_a.dpo_reward_estimate or 0.0) - (traj_b.dpo_reward_estimate or 0.0))
            total_diff = quality_diff + 0.5 * dpo_diff
            
            if total_diff > max_diff:
                max_diff = total_diff
                best_pair = (traj_a, traj_b)
        
        return best_pair
    
    def _create_dpo_preference_pair(self, 
                                   traj_a: UnifiedTrajectory, 
                                   traj_b: UnifiedTrajectory,
                                   method: str) -> Optional[UnifiedPreferencePair]:
        """创建DPO偏好对"""
        try:
            # 使用DPO评估器计算偏好
            dpo_logit, confidence = self.dpo_evaluator.evaluate_dpo_preference(
                traj_a.to_dict(), traj_b.to_dict()
            )
            
            # 转换为偏好分数
            preference_score = torch.sigmoid(torch.tensor(dpo_logit)).item()
            
            # 创建偏好标签
            preference_label = PreferenceLabel(
                preference_score=preference_score,
                logit_preference=dpo_logit,
                binary_preference=1 if preference_score > 0.6 else (-1 if preference_score < 0.4 else 0),
                metadata=LabelMetadata(
                    label_type=LabelType.DPO_BINARY,
                    confidence=confidence,
                    quality_score_a=traj_a.quality_score,
                    quality_score_b=traj_b.quality_score,
                    score_difference=abs(traj_a.quality_score - traj_b.quality_score),
                    generation_time=0.0,  # 会在外部更新
                    features_used=list(traj_a.preference_features.keys()) if traj_a.preference_features else [],
                    additional_info={
                        'dpo_logit': dpo_logit,
                        'method': method
                    }
                ),
                is_valid=confidence >= self.confidence_threshold
            )
            
            # 创建统一偏好对
            unified_pair = UnifiedPreferencePair(
                trajectory_a=traj_a,
                trajectory_b=traj_b,
                preference_label=preference_label,
                generation_method=method,
                dpo_logit=dpo_logit,
                confidence_score=confidence
            )
            
            return unified_pair
            
        except Exception as e:
            logger.error(f"创建DPO偏好对失败: {e}")
            return None
    
    def create_dpo_preference_pair(self, 
                                  traj_a: UnifiedTrajectory, 
                                  traj_b: UnifiedTrajectory,
                                  method: str = "dpo_unified") -> Optional[UnifiedPreferencePair]:
        """创建DPO偏好对的公共接口"""
        return self._create_dpo_preference_pair(traj_a, traj_b, method)
    
    def _create_quality_preference_pair(self, 
                                       traj_a: UnifiedTrajectory, 
                                       traj_b: UnifiedTrajectory,
                                       method: str) -> Optional[UnifiedPreferencePair]:
        """创建基于质量的偏好对"""
        try:
            # 使用现有的偏好标签引擎
            preference_label = self.labeling_engine.generate_preference_labels(
                obs_a=traj_a.obs_sequence,
                act_a=traj_a.action_sequence,
                obs_b=traj_b.obs_sequence,
                act_b=traj_b.action_sequence,
                label_type=LabelType.QUALITY_BASED
            )
            
            # 计算置信度
            confidence = preference_label.metadata.confidence if preference_label.metadata else 0.5
            
            # 创建统一偏好对
            unified_pair = UnifiedPreferencePair(
                trajectory_a=traj_a,
                trajectory_b=traj_b,
                preference_label=preference_label,
                generation_method=method,
                confidence_score=confidence
            )
            
            return unified_pair
            
        except Exception as e:
            logger.error(f"创建质量偏好对失败: {e}")
            return None
    
    def get_training_data(self, 
                         num_pairs: Optional[int] = None,
                         min_confidence: Optional[float] = None) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """获取训练数据
        
        Args:
            num_pairs: 返回的偏好对数量
            min_confidence: 最小置信度阈值
            
        Returns:
            训练格式的偏好对列表
        """
        with self.lock:
            # 过滤有效的偏好对
            valid_pairs = [
                pair for pair in self.unified_preference_pairs
                if pair.preference_label.is_valid and 
                   pair.confidence_score >= (min_confidence or self.confidence_threshold)
            ]
            
            # 限制数量
            if num_pairs and len(valid_pairs) > num_pairs:
                # 按置信度排序，选择最好的
                valid_pairs.sort(key=lambda p: p.confidence_score, reverse=True)
                valid_pairs = valid_pairs[:num_pairs]
            
            # 转换为训练格式
            training_data = [pair.to_training_format() for pair in valid_pairs]
            
            logger.info(f"返回 {len(training_data)} 个训练偏好对")
            return training_data
    
    def clear_cache(self):
        """清除缓存"""
        with self.lock:
            self.unified_preference_pairs.clear()
            logger.info("统一偏好对缓存已清除")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'current_trajectories': len(self.unified_trajectories),
                'current_preference_pairs': len(self.unified_preference_pairs),
                'valid_preference_pairs': len([
                    p for p in self.unified_preference_pairs 
                    if p.preference_label.is_valid
                ]),
                'avg_trajectory_quality': np.mean([
                    t.quality_score for t in self.unified_trajectories
                ]) if self.unified_trajectories else 0.0,
                'avg_confidence_score': np.mean([
                    p.confidence_score for p in self.unified_preference_pairs
                ]) if self.unified_preference_pairs else 0.0
            })
            return stats
    
    def save_state(self, filepath: str):
        """保存系统状态"""
        with self.lock:
            state = {
                'trajectories': self.unified_trajectories,
                'preference_pairs': self.unified_preference_pairs,
                'stats': self.stats,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"系统状态已保存到: {filepath}")
    
    def load_state(self, filepath: str):
        """加载系统状态"""
        with self.lock:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.unified_trajectories = state.get('trajectories', [])
            self.unified_preference_pairs = state.get('preference_pairs', [])
            self.stats = state.get('stats', {})
            
            # 重建索引
            self.trajectory_index = {
                traj.trajectory_id: traj for traj in self.unified_trajectories
            }
            
            logger.info(f"系统状态已从 {filepath} 加载")

def create_unified_preference_system(task_name: str, 
                                   config: Dict[str, Any] = None,
                                   work_dir: str = None) -> UnifiedPreferenceSystem:
    """创建统一偏好系统实例"""
    return UnifiedPreferenceSystem(task_name, config, work_dir)

# 兼容性函数
def migrate_from_legacy_system(legacy_trajectories: List[Dict[str, Any]], 
                             unified_system: UnifiedPreferenceSystem) -> int:
    """从旧系统迁移轨迹数据
    
    Args:
        legacy_trajectories: 旧格式的轨迹列表
        unified_system: 统一偏好系统实例
        
    Returns:
        迁移的轨迹数量
    """
    migrated_count = 0
    
    for i, traj in enumerate(legacy_trajectories):
        try:
            # 提取轨迹数据
            obs_seq = np.array(traj.get('obs', []))
            action_seq = np.array(traj.get('action', []))
            reward_seq = np.array(traj.get('reward', []))
            done_seq = np.array(traj.get('done', [False] * len(obs_seq)))
            
            # 添加到统一系统
            unified_system.add_trajectory(
                obs_seq=obs_seq,
                action_seq=action_seq,
                reward_seq=reward_seq,
                done_seq=done_seq,
                episode_idx=traj.get('episode_idx', i),
                step_range=traj.get('step_range', (0, len(obs_seq)))
            )
            
            migrated_count += 1
            
        except Exception as e:
            logger.warning(f"迁移轨迹 {i} 失败: {e}")
    
    logger.info(f"成功迁移 {migrated_count} 个轨迹到统一系统")
    return migrated_count