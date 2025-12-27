#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先级偏好系统 (Prioritized Preference System)

集成优先级经验回放与现有偏好训练流程的完整系统，包括：
1. 轨迹收集和偏好对生成
2. 优先级计算和经验回放池管理
3. 批次采样和模型训练
4. 基于损失的优先级更新
5. 与TD-MPC2训练流程的集成

作者：AI Assistant
日期：2025-01-19
版本：1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import time
import threading
from pathlib import Path

# 导入现有模块
from .prioritized_experience_replay import (
    PrioritizedPreferenceTrainer,
    PreferencePair,
    PriorityCalculator,
    create_prioritized_preference_trainer
)
from .preference_labeling_engine import (
    PreferenceLabelingEngine,
    create_preference_labeling_engine,
    PreferenceLabel
)
from .optimized_preference_trainer import OptimizedPreferenceTrainer
from .optimized_latent_preference_model import OptimizedLatentPreferenceModel
from .trajectory_encoder import TrajectoryEncoder, create_trajectory_encoder
from .adaptive_threshold_manager import AdaptiveThresholdManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PrioritizedSystemConfig:
    """优先级偏好系统配置"""
    # 缓冲池配置
    buffer_capacity: int = 10000
    batch_size: int = 64
    min_buffer_size: int = 100
    
    # 优先级配置
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 0.001
    confidence_weight: float = 0.7
    temporal_weight: float = 0.3
    temporal_decay: float = 0.95
    
    # 训练配置
    train_every_n_episodes: int = 5
    preference_model_lr: float = 1e-3
    preference_model_weight_decay: float = 1e-4
    preference_model_grad_clip: float = 1.0
    max_training_epochs: int = 10
    
    # 质量检测阈值 - 现在使用自适应阈值管理器
    # 这些参数将传递给AdaptiveThresholdManager
    # 注意：所有参数都将从配置文件动态读取，不设置默认值
    adaptive_thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # 模型配置
    action_dim: int = 61
    latent_dim: int = 512
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 保存配置
    save_dir: str = "./prioritized_preference_checkpoints"
    save_interval: int = 100
    
    # 偏好对生成配置 - 置信度阈值将从配置文件动态读取
    preference_pair_generation: Dict[str, Any] = field(default_factory=lambda: {
        'pairs_per_generation': 20,
        'generation_methods': {
            'dpo': {
                'enabled': True,
                'ratio': 0.4
                # confidence_threshold将从配置文件动态设置
            },
            'quality': {
                'enabled': True,
                'ratio': 0.4
                # confidence_threshold将从配置文件动态设置
            },
            'hybrid': {
                'enabled': True,
                'ratio': 0.2
                # confidence_threshold将从配置文件动态设置
            }
        }
    })
    
    # 兼容性属性 - 用于向后兼容，将从配置文件动态设置
    min_confidence_threshold: float = None
    min_score_diff_threshold: float = 8.0
    
    @classmethod
    def from_tdmpc2_config(cls, tdmpc2_cfg=None):
        """从TD-MPC2配置文件创建配置对象"""
        config = cls()
        
        if tdmpc2_cfg is not None:
            # 从TD-MPC2配置中读取优化参数
            # 初始化threshold_found标志
            threshold_found = False
            
            try:
                # 读取偏好对生成配置
                if hasattr(tdmpc2_cfg, 'preference_pair_generation'):
                    pref_gen_cfg = tdmpc2_cfg.preference_pair_generation
                    
                    # 更新置信度阈值 - 优先从generation_methods.dpo中读取
                    threshold_found = False
                    if hasattr(pref_gen_cfg, 'generation_methods'):
                        gen_methods = pref_gen_cfg.generation_methods
                        # 从dpo方法读取置信度阈值
                        if hasattr(gen_methods, 'dpo') and hasattr(gen_methods.dpo, 'confidence_threshold'):
                            threshold = float(gen_methods.dpo.confidence_threshold)
                            config.min_confidence_threshold = threshold
                            # 更新adaptive_thresholds中的置信度阈值
                            config.adaptive_thresholds['confidence_threshold'] = threshold
                            # 更新所有生成方法的置信度阈值
                            for method in config.preference_pair_generation['generation_methods'].values():
                                method['confidence_threshold'] = threshold
                            logger.info(f"[配置] 从preference_pair_generation.generation_methods.dpo读取置信度阈值: {threshold}")
                            threshold_found = True
                    
                    # 兼容旧的配置方式
                    if not threshold_found and hasattr(pref_gen_cfg, 'confidence_threshold'):
                        threshold = float(pref_gen_cfg.confidence_threshold)
                        config.min_confidence_threshold = threshold
                        # 更新adaptive_thresholds中的置信度阈值
                        config.adaptive_thresholds['confidence_threshold'] = threshold
                        # 更新所有生成方法的置信度阈值
                        for method in config.preference_pair_generation['generation_methods'].values():
                            method['confidence_threshold'] = threshold
                        logger.info(f"[配置] 从preference_pair_generation.confidence_threshold读取置信度阈值: {threshold}")
                        threshold_found = True
                    
                    # 更新得分差阈值
                    if hasattr(pref_gen_cfg, 'score_diff_threshold'):
                        config.min_score_diff_threshold = float(pref_gen_cfg.score_diff_threshold)
                
                # 读取优先级经验回放配置
                if hasattr(tdmpc2_cfg, 'prioritized_experience_replay'):
                    per_cfg = tdmpc2_cfg.prioritized_experience_replay
                    
                    # 读取缓冲池配置
                    if hasattr(per_cfg, 'min_buffer_size'):
                        config.min_buffer_size = int(per_cfg.min_buffer_size)
                        logger.info(f"[配置] 从prioritized_experience_replay读取min_buffer_size: {config.min_buffer_size}")
                    
                    if hasattr(per_cfg, 'buffer_capacity'):
                        config.buffer_capacity = int(per_cfg.buffer_capacity)
                        logger.info(f"[配置] 从prioritized_experience_replay读取buffer_capacity: {config.buffer_capacity}")
                    
                    if hasattr(per_cfg, 'batch_size'):
                        config.batch_size = int(per_cfg.batch_size)
                        logger.info(f"[配置] 从prioritized_experience_replay读取batch_size: {config.batch_size}")
                    
                    if hasattr(per_cfg, 'sampling_config'):
                        sampling_cfg = per_cfg.sampling_config
                        if hasattr(sampling_cfg, 'quality_thresholds'):
                            quality_cfg = sampling_cfg.quality_thresholds
                            
                            # 如果之前没有找到置信度阈值，从这里读取
                            if not threshold_found and hasattr(quality_cfg, 'confidence_threshold'):
                                threshold = float(quality_cfg.confidence_threshold)
                                config.min_confidence_threshold = threshold
                                # 更新adaptive_thresholds中的置信度阈值
                                config.adaptive_thresholds['confidence_threshold'] = threshold
                                # 更新所有生成方法的置信度阈值
                                for method in config.preference_pair_generation['generation_methods'].values():
                                    method['confidence_threshold'] = threshold
                                logger.info(f"[配置] 从prioritized_experience_replay.sampling_config.quality_thresholds读取置信度阈值: {threshold}")
                                threshold_found = True
                            
                            # 读取其他质量阈值参数
                            if hasattr(quality_cfg, 'rule_score_diff_multiplier'):
                                config.adaptive_thresholds['rule_score_diff_multiplier'] = float(quality_cfg.rule_score_diff_multiplier)
                            if hasattr(quality_cfg, 'rule_score_diff_min_threshold'):
                                config.adaptive_thresholds['rule_score_diff_min_threshold'] = float(quality_cfg.rule_score_diff_min_threshold)
                            if hasattr(quality_cfg, 'env_reward_diff_std_multiplier'):
                                config.adaptive_thresholds['env_reward_diff_std_multiplier'] = float(quality_cfg.env_reward_diff_std_multiplier)
                            if hasattr(quality_cfg, 'env_reward_diff_percentile_threshold'):
                                config.adaptive_thresholds['env_reward_diff_percentile_threshold'] = float(quality_cfg.env_reward_diff_percentile_threshold)
                            if hasattr(quality_cfg, 'min_quality_indicators'):
                                config.adaptive_thresholds['min_quality_indicators'] = int(quality_cfg.min_quality_indicators)
                            # 读取信号强度相关参数
                            if hasattr(quality_cfg, 'weak_signal_std_multiplier'):
                                config.adaptive_thresholds['weak_signal_std_multiplier'] = float(quality_cfg.weak_signal_std_multiplier)
                            if hasattr(quality_cfg, 'strong_signal_std_multiplier'):
                                config.adaptive_thresholds['strong_signal_std_multiplier'] = float(quality_cfg.strong_signal_std_multiplier)
                            if hasattr(quality_cfg, 'stability_std_multiplier'):
                                config.adaptive_thresholds['stability_std_multiplier'] = float(quality_cfg.stability_std_multiplier)
                            if hasattr(quality_cfg, 'significance_mean_multiplier'):
                                config.adaptive_thresholds['significance_mean_multiplier'] = float(quality_cfg.significance_mean_multiplier)
                
                # 兼容旧的质量阈值配置
                if hasattr(tdmpc2_cfg, 'quality_thresholds'):
                    quality_cfg = tdmpc2_cfg.quality_thresholds
                    
                    if not threshold_found and hasattr(quality_cfg, 'confidence_threshold'):
                        threshold = float(quality_cfg.confidence_threshold)
                        config.min_confidence_threshold = threshold
                        # 更新adaptive_thresholds中的置信度阈值
                        config.adaptive_thresholds['confidence_threshold'] = threshold
                        # 更新所有生成方法的置信度阈值
                        for method in config.preference_pair_generation['generation_methods'].values():
                            method['confidence_threshold'] = threshold
                        logger.info(f"[配置] 从quality_thresholds读取置信度阈值: {threshold}")
                    
                    if hasattr(quality_cfg, 'env_reward_diff_threshold'):
                        config.min_score_diff_threshold = float(quality_cfg.env_reward_diff_threshold)
                
                logger.info(f"[配置] 从TD-MPC2配置读取参数: confidence_threshold={config.min_confidence_threshold}, score_diff_threshold={config.min_score_diff_threshold}")
                
            except Exception as e:
                logger.warning(f"[配置] 从TD-MPC2配置读取参数时出错: {e}，使用默认优化值")
            
            # 如果仍然没有找到置信度阈值，使用默认值并记录警告
            if not threshold_found:
                default_threshold = 0.75
                config.min_confidence_threshold = default_threshold
                config.adaptive_thresholds['confidence_threshold'] = default_threshold
                # 更新所有生成方法的置信度阈值
                for method in config.preference_pair_generation['generation_methods'].values():
                    method['confidence_threshold'] = default_threshold
                logger.warning(
                    f"[配置] 未在配置文件中找到置信度阈值设置，使用默认值: {default_threshold}。\n"
                    "建议在以下位置之一设置confidence_threshold:\n"
                    "1. preference_pair_generation.generation_methods.dpo.confidence_threshold\n"
                    "2. preference_pair_generation.confidence_threshold\n"
                    "3. prioritized_experience_replay.sampling_config.quality_thresholds.confidence_threshold\n"
                    "4. quality_thresholds.confidence_threshold"
                )
            
            # 检查必需的adaptive_thresholds参数是否都已设置，如果缺失则使用默认值
            default_params = {
                'rule_score_diff_multiplier': 2.0,
                'rule_score_diff_min_threshold': 0.1,
                'env_reward_diff_std_multiplier': 1.5,
                'env_reward_diff_percentile_threshold': 0.7,
                'min_quality_indicators': 2,
                'weak_signal_std_multiplier': 0.5,
                'strong_signal_std_multiplier': 2.0,
                'stability_std_multiplier': 1.0,
                'significance_mean_multiplier': 1.2
            }
            
            missing_params = []
            for param, default_value in default_params.items():
                if param not in config.adaptive_thresholds:
                    config.adaptive_thresholds[param] = default_value
                    missing_params.append(param)
            
            if missing_params:
                logger.warning(
                    f"[配置] 以下参数未在配置文件中找到，使用默认值: {missing_params}\n"
                    "建议在 prioritized_experience_replay.sampling_config.quality_thresholds 中设置这些参数"
                )
        
        return config

class TrajectoryCollector:
    """轨迹收集器"""
    
    def __init__(self, max_trajectory_length: int = 1000, cache_size: int = 100):
        self.max_trajectory_length = max_trajectory_length
        self.cache_size = cache_size  # 缓存区大小限制
        self.current_trajectories = []
        self.completed_trajectories = []
        self.lock = threading.RLock()
    
    def start_trajectory(self) -> int:
        """开始新轨迹收集"""
        with self.lock:
            trajectory_id = len(self.current_trajectories)
            self.current_trajectories.append({
                'obs': [],
                'action': [],
                'reward': [],
                'done': [],
                'info': [],
                'start_time': time.time()
            })
            return trajectory_id
    
    def add_step(self, trajectory_id: int, obs: np.ndarray, action: np.ndarray, 
                 reward: float, done: bool, info: Dict = None) -> None:
        """添加轨迹步骤"""
        with self.lock:
            if trajectory_id < len(self.current_trajectories):
                traj = self.current_trajectories[trajectory_id]
                traj['obs'].append(obs.copy())
                traj['action'].append(action.copy())
                traj['reward'].append(reward)
                traj['done'].append(done)
                traj['info'].append(info or {})
    
    def complete_trajectory(self, trajectory_id: int) -> Dict[str, np.ndarray]:
        """完成轨迹收集"""
        with self.lock:
            if trajectory_id < len(self.current_trajectories):
                traj = self.current_trajectories[trajectory_id]
                
                # 转换为numpy数组
                completed_traj = {
                    'obs': np.array(traj['obs']),
                    'action': np.array(traj['action']),
                    'reward': np.array(traj['reward']),
                    'done': np.array(traj['done']),
                    'info': traj['info'],
                    'start_time': traj['start_time'],
                    'end_time': time.time(),
                    'length': len(traj['obs'])
                }
                
                self.completed_trajectories.append(completed_traj)
                
                # 检查缓存大小，达到上限时清空
                if len(self.completed_trajectories) >= self.cache_size:
                    logger.info(f"轨迹缓存达到上限({self.cache_size})，清空缓存")
                    self.completed_trajectories = []
                
                return completed_traj
        logger.warning(f"[优先级偏好系统] 训练失败 - 无有效批次数据")
        return None
    
    def get_recent_trajectories(self, n: int = 10) -> List[Dict[str, np.ndarray]]:
        """获取最近的n条轨迹"""
        with self.lock:
            return self.completed_trajectories[-n:] if self.completed_trajectories else []
    
    def clear_old_trajectories(self, keep_last_n: int = 100) -> None:
        """清理旧轨迹，保留最近的n条"""
        with self.lock:
            if keep_last_n == 0:
                # 完全清空
                self.completed_trajectories = []
            elif len(self.completed_trajectories) > keep_last_n:
                self.completed_trajectories = self.completed_trajectories[-keep_last_n:]

class PrioritizedPreferenceSystem:
    """优先级偏好系统主类"""
    
    def __init__(self, 
                 task_name: str,
                 config: Optional[PrioritizedSystemConfig] = None,
                 existing_preference_model: Optional[nn.Module] = None,
                 tdmpc2_agent=None,
                 tdmpc2_cfg=None):
        """
        初始化优先级偏好系统
        
        Args:
            task_name: 任务名称
            config: 系统配置
            existing_preference_model: 现有的偏好模型
            tdmpc2_agent: TD-MPC2 agent引用
            tdmpc2_cfg: TD-MPC2配置对象
        """
        self.task_name = task_name
        # 优先使用从TD-MPC2配置创建的配置对象
        if config is None:
            self.config = PrioritizedSystemConfig.from_tdmpc2_config(tdmpc2_cfg)
        else:
            self.config = config
        self.tdmpc2_agent = tdmpc2_agent  # TD-MPC2 agent引用，用于编码器
        self.tdmpc2_cfg = tdmpc2_cfg  # TD-MPC2配置对象
        
        # 创建组件
        self.trajectory_collector = TrajectoryCollector(cache_size=100)  # 使用固定的缓存大小
        
        # 创建自适应阈值管理器
        self.adaptive_threshold_manager = AdaptiveThresholdManager(
            config=self.config.adaptive_thresholds,
            window_size=50
        )
        
        self.preference_labeling_engine = create_preference_labeling_engine(task_name)
        self.prioritized_trainer = create_prioritized_preference_trainer(
            buffer_capacity=self.config.buffer_capacity,
            batch_size=self.config.batch_size,
            min_buffer_size=self.config.min_buffer_size,
            priority_alpha=self.config.priority_alpha,
            priority_beta=self.config.priority_beta,
            priority_beta_increment=self.config.priority_beta_increment
        )
        
        # 动态获取观测维度
        obs_dim = 151  # 默认值
        if self.tdmpc2_cfg and hasattr(self.tdmpc2_cfg, 'obs_shape') and self.tdmpc2_cfg.obs_shape:
            # 导入OmegaConf类型以处理配置对象
            try:
                from omegaconf import DictConfig, ListConfig
            except ImportError:
                DictConfig = dict
                ListConfig = list
            
            if isinstance(self.tdmpc2_cfg.obs_shape, (dict, DictConfig)):
                # 处理字典格式的obs_shape，如{'state': (213,)} 或 {'state': [213]}
                for obs_type, shape in self.tdmpc2_cfg.obs_shape.items():
                    if isinstance(shape, (tuple, list, ListConfig)) and len(shape) > 0:
                        try:
                            obs_dim = int(shape[0])
                            break
                        except (TypeError, ValueError):
                            continue
            elif isinstance(self.tdmpc2_cfg.obs_shape, (tuple, list, ListConfig)) and len(self.tdmpc2_cfg.obs_shape) > 0:
                # 处理元组/列表格式的obs_shape
                try:
                    obs_dim = int(self.tdmpc2_cfg.obs_shape[0])
                except (TypeError, ValueError):
                    pass
        
        # 创建轨迹编码器
        self.trajectory_encoder = create_trajectory_encoder(
            obs_dim=obs_dim,
            action_dim=self.config.action_dim,
            latent_dim=self.config.latent_dim
        ).to(self.config.device)
        
        logger.info(f"[优先级偏好系统] 轨迹编码器创建成功 (obs_dim={obs_dim}, action_dim={self.config.action_dim}, latent_dim={self.config.latent_dim})")
        
        # 偏好模型和优化器 - 确保同时创建
        self.preference_model = None
        self.preference_optimizer = None
        
        # 设置偏好模型和优化器
        self._setup_preference_model_and_optimizer(existing_preference_model)
        
        # 统计信息
        self.stats = {
            'total_trajectories_collected': 0,
            'total_preference_pairs_generated': 0,
            'total_training_steps': 0,
            'avg_training_loss': 0.0,
            'last_training_time': 0.0,
            'system_start_time': time.time()
        }
        
        # 创建保存目录
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # 简化系统初始化信息
        logger.info(f"[优先级偏好系统] 系统初始化完成 - 任务: {task_name}")
    
    def _setup_preference_model_and_optimizer(self, existing_preference_model: Optional[nn.Module] = None) -> None:
        """设置偏好模型和优化器，确保同时创建"""
        if existing_preference_model is not None:
            # 使用现有模型
            self.preference_model = existing_preference_model
            logger.info(f"[优先级偏好系统] 使用现有偏好模型")
        else:
            # 创建默认的优化潜空间偏好模型
            from .optimized_latent_preference_model import OptimizedLatentPreferenceModel, OptimizedLatentPreferenceConfig
            model_config = OptimizedLatentPreferenceConfig(
                latent_dim=getattr(self.config, 'latent_dim', 512),
                action_dim=getattr(self.config, 'action_dim', 61),
                device=self.config.device
            )
            self.preference_model = OptimizedLatentPreferenceModel(model_config).to(self.config.device)
            logger.info(f"[优先级偏好系统] 创建了默认偏好模型")
        
        # 无论使用现有模型还是创建新模型，都立即创建优化器
        if self.preference_model is not None:
            self.preference_optimizer = torch.optim.Adam(
                self.preference_model.parameters(),
                lr=self.config.preference_model_lr,
                weight_decay=self.config.preference_model_weight_decay
            )
            logger.info(f"[优先级偏好系统] 偏好模型优化器已创建")
        else:
            logger.error(f"[优先级偏好系统] ❌ 偏好模型创建失败，无法创建优化器")
    
    def collect_trajectory_step(self, trajectory_id: int, obs: np.ndarray, 
                               action: np.ndarray, reward: float, done: bool, 
                               info: Dict = None) -> None:
        """收集轨迹步骤"""
        self.trajectory_collector.add_step(trajectory_id, obs, action, reward, done, info)
    
    def start_new_trajectory(self) -> int:
        """开始新轨迹"""
        trajectory_id = self.trajectory_collector.start_trajectory()
        current_cache_size = len(self.trajectory_collector.completed_trajectories)
        logger.info(f"[优先级偏好系统] 开始新轨迹 - ID: {trajectory_id}, 当前缓存轨迹数: {current_cache_size}/{self.config.train_every_n_episodes}")
        return trajectory_id
    
    def complete_trajectory(self, trajectory_id: int) -> None:
        """完成轨迹收集"""
        completed_traj = self.trajectory_collector.complete_trajectory(trajectory_id)
        if completed_traj is not None:
            self.stats['total_trajectories_collected'] += 1
            
            # 新的缓存逻辑：只有当缓存的轨迹数量达到train_every_n_episodes时才开始处理
            current_cache_size = len(self.trajectory_collector.completed_trajectories)
            buffer_stats = self.prioritized_trainer.get_statistics()
            buffer_size_before = buffer_stats.get('buffer_stats', {}).get('buffer_size', 0)
            # 计算轨迹质量分数
            try:
                from trajectory_metrics import TrajectoryMetrics
                trajectory_metrics = TrajectoryMetrics()
                
                # 准备轨迹数据用于质量评估
                trajectory_data = {
                    'obs': completed_traj['obs'],
                    'action': completed_traj['action'],
                    'reward': completed_traj['reward'],
                    'done': completed_traj['done']
                }
                
                # 计算质量指标
                quality_metrics = trajectory_metrics.compute_metrics(trajectories=trajectory_data, task_name=self.task_name)
                
                # 计算综合质量分数
                total_reward = np.sum(completed_traj['reward'])
                avg_reward = np.mean(completed_traj['reward'])
                survival_time = len(completed_traj['obs'])
                
                # 使用偏好标签引擎计算质量分数
                try:
                    quality_score, quality_features = self.preference_labeling_engine.quality_evaluator.evaluate_trajectory_quality(
                        completed_traj['obs'], completed_traj['action'], completed_traj['reward']
                    )
                except Exception as e:
                    logger.warning(f"[轨迹质量评估] 质量分数计算失败: {e}")
                    quality_score = 0.0
                    quality_features = {}
                
                # 将轨迹奖励数据添加到自适应阈值管理器
                self.adaptive_threshold_manager.add_reward_sample(total_reward, quality_score)
                
                # 输出轨迹质量日志
                logger.info(f"[优先级偏好系统] 完成轨迹收集 - 轨迹ID: {trajectory_id}, 轨迹长度: {len(completed_traj['obs'])}")
                logger.info(f"[轨迹质量评估] 总奖励: {total_reward:.4f}, 平均奖励: {avg_reward:.4f}, 生存时间: {survival_time}, 质量分数: {quality_score:.4f}")
                
                # 输出任务特定指标（如果有的话）
                if quality_metrics:
                    key_metrics = ['survival_time', 'distance_to_goal', 'is_task_successful', 'total_reward']
                    metric_values = []
                    for metric in key_metrics:
                        if metric in quality_metrics:
                            metric_values.append(f"{metric}: {quality_metrics[metric]:.4f}")
                    
                    if metric_values:
                        logger.info(f"[轨迹质量评估] 关键指标 - {', '.join(metric_values)}")
                        
            except Exception as e:
                # 质量评估失败，但不输出详细日志
                pass
            
            # 检查是否达到训练阈值
            if current_cache_size >= self.config.train_every_n_episodes:
                # 生成偏好对（必须在清空缓存之前完成）
                if current_cache_size >= 2:
                    logger.info(f"[优先级偏好系统] 开始生成偏好对，当前缓存轨迹数: {current_cache_size}")
                    self._generate_preference_pairs_from_cached_trajectories()
                    logger.info(f"[优先级偏好系统] 偏好对生成完成")
                
                # 检查是否应该训练偏好模型
                if self.should_train_preference_model():
                    training_loss = self.train_preference_model()
                    if training_loss is not None:
                        logger.info(f"[优先级偏好系统] 偏好模型训练完成 - 平均损失: {training_loss:.6f}")
                
                # 清理缓存（训练后清空所有轨迹，重新开始计数以确保严格按频率训练）
                logger.info(f"[优先级偏好系统] 清理轨迹缓存，保留0条轨迹")
                self.trajectory_collector.clear_old_trajectories(keep_last_n=0)
    
    def _generate_preference_pairs_from_cached_trajectories(self) -> None:
        """从缓存的轨迹生成偏好对，处理所有缓存中的轨迹"""
        # 从配置文件获取参数
        generation_config = getattr(self.config, 'preference_pair_generation', {})
        n_pairs = generation_config.get('pairs_per_generation', 20)
        
        # 获取所有缓存的轨迹
        cached_trajectories = self.trajectory_collector.completed_trajectories
        
        if len(cached_trajectories) < 2:
            logger.warning(f"缓存轨迹数量不足({len(cached_trajectories)})，无法生成偏好对")
            return
        
        logger.info(f"[优先级偏好系统] 开始从{len(cached_trajectories)}条缓存轨迹生成{n_pairs}个偏好对")
        
        # 获取生成方法配置
        methods_config = generation_config.get('generation_methods', {})
        dpo_config = methods_config.get('dpo', {})
        quality_config = methods_config.get('quality', {})
        hybrid_config = methods_config.get('hybrid', {})
        
        # 计算每种方法应生成的偏好对数量
        dpo_pairs = int(n_pairs * dpo_config.get('ratio', 0.4)) if dpo_config.get('enabled', True) else 0
        quality_pairs = int(n_pairs * quality_config.get('ratio', 0.4)) if quality_config.get('enabled', True) else 0
        hybrid_pairs = n_pairs - dpo_pairs - quality_pairs if hybrid_config.get('enabled', True) else 0
        
        # 确保总数不超过n_pairs
        if hybrid_pairs < 0:
            hybrid_pairs = 0
            quality_pairs = n_pairs - dpo_pairs
        
        logger.info(f"[优先级偏好系统] 偏好对生成计划: DPO={dpo_pairs}, 质量={quality_pairs}, 混合={hybrid_pairs}")
        
        # 获取置信度阈值，确保有正确的回退逻辑
        confidence_threshold = None
        if 'confidence_threshold' in self.config.adaptive_thresholds:
            confidence_threshold = self.config.adaptive_thresholds['confidence_threshold']
        elif self.config.min_confidence_threshold is not None:
            confidence_threshold = self.config.min_confidence_threshold
        else:
            # 如果都没有设置，使用默认值
            confidence_threshold = 0.75
            logger.warning(f"[优先级偏好系统] 未找到置信度阈值配置，使用默认值: {confidence_threshold}")
        
        logger.info(f"[优先级偏好系统] 生成配置: 置信度阈值={confidence_threshold}")
        
        # 生成不同类型的偏好对
        generated_pairs = 0
        
        # 1. 生成DPO偏好对
        if dpo_pairs > 0:
            generated_pairs += self._generate_pairs_by_method(
                cached_trajectories, dpo_pairs, 'DPO_BINARY', dpo_config
            )
        
        # 2. 生成质量偏好对
        if quality_pairs > 0:
            generated_pairs += self._generate_pairs_by_method(
                cached_trajectories, quality_pairs, 'QUALITY_BASED', quality_config
            )
        
        # 3. 生成混合偏好对
        if hybrid_pairs > 0:
            generated_pairs += self._generate_pairs_by_method(
                cached_trajectories, hybrid_pairs, 'HYBRID_DPO_QUALITY', hybrid_config
            )
        
        # 输出偏好对生成结果
        logger.info(f"[优先级偏好系统] 偏好对生成完成，总共生成: {generated_pairs}个偏好对")
        buffer_stats = self.prioritized_trainer.get_statistics()
        buffer_size = buffer_stats.get('buffer_stats', {}).get('buffer_size', 0)
        total_processed = buffer_stats.get('total_preference_pairs_processed', 0)
        logger.info(f"[优先级偏好系统] 当前缓冲区状态: 总偏好对数={buffer_size}，已处理偏好对数={total_processed}")
    
    # 废弃的旧方法已删除：_generate_preference_pairs_from_recent_trajectories
    # 现在统一使用 _generate_preference_pairs_from_cached_trajectories 方法
    # 该方法使用优先级经验回放池进行偏好对管理和采样
    
    def _generate_pairs_by_method(self, trajectories: List[Dict], n_pairs: int, 
                                 method_type: str, method_config: Dict) -> int:
        """使用指定方法生成偏好对，优先选择高质量轨迹对"""
        from .preference_labeling_engine import LabelType
        
        # 更新自适应阈值管理器的奖励统计
        for traj in trajectories:
            total_reward = np.sum(traj.get('reward', [0]))
            self.adaptive_threshold_manager.add_reward_sample(total_reward)
        
        # 映射方法类型到LabelType
        label_type_map = {
            'DPO_BINARY': LabelType.DPO_BINARY,
            'QUALITY_BASED': LabelType.QUALITY_BASED,
            'HYBRID_DPO_QUALITY': LabelType.HYBRID_DPO_QUALITY,
            'HYBRID_DPO_RULE': LabelType.HYBRID_DPO_RULE
        }
        
        label_type = label_type_map.get(method_type, LabelType.QUALITY_BASED)
        # 使用自适应阈值管理器获取动态置信度阈值
        confidence_threshold = self.adaptive_threshold_manager.get_confidence_threshold()
        # 获取动态环境奖励差异阈值
        env_reward_diff_threshold = self.adaptive_threshold_manager.get_env_reward_diff_threshold()
        
        generated_count = 0
        max_attempts = n_pairs * 3  # 最多尝试3倍数量
        attempts = 0
        
        # 统计信息
        quality_rejected = 0
        labeling_failed = 0
        confidence_scores = []
        
        # 获取自适应阈值管理器的统计信息
        threshold_stats = self.adaptive_threshold_manager.get_stats()
        logger.info(f"[优先级偏好系统] 开始使用{method_type}方法生成{n_pairs}个偏好对") 
        # 计算每条轨迹的环境奖励并排序
        traj_rewards = []
        for i, traj in enumerate(trajectories):
            env_reward = np.sum(traj.get('reward', [0]))
            traj_rewards.append((i, traj, env_reward))
        
        # 按奖励排序（降序）
        traj_rewards.sort(key=lambda x: x[2], reverse=True)
        
        # 分层轨迹选择策略
        high_quality_ratio = 0.6  # 高质量对比例
        medium_quality_ratio = 0.3  # 中等质量对比例
        exploration_ratio = 0.1  # 探索性对比例
        
        high_quality_pairs = int(n_pairs * high_quality_ratio)
        medium_quality_pairs = int(n_pairs * medium_quality_ratio)
        exploration_pairs = n_pairs - high_quality_pairs - medium_quality_pairs
        
        # 生成不同类型的偏好对
        generated_count += self._generate_stratified_pairs(
            traj_rewards, high_quality_pairs, "high_quality", method_type, label_type, confidence_threshold
        )
        generated_count += self._generate_stratified_pairs(
            traj_rewards, medium_quality_pairs, "medium_quality", method_type, label_type, confidence_threshold
        )
        generated_count += self._generate_stratified_pairs(
            traj_rewards, exploration_pairs, "exploration", method_type, label_type, confidence_threshold
        )
        
        logger.info(f"[偏好对生成] {method_type}方法统计: 高质量对={sum(1 for _ in range(high_quality_pairs))}, 中等质量对={sum(1 for _ in range(medium_quality_pairs))}, 探索性对={sum(1 for _ in range(exploration_pairs))}")
        
        return generated_count
    
    def _generate_stratified_pairs(self, traj_rewards: List, n_pairs: int, pair_type: str, 
                                  method_type: str, label_type, confidence_threshold: float) -> int:
        """生成分层偏好对"""
        generated_count = 0
        max_attempts = n_pairs * 3
        attempts = 0
        
        while generated_count < n_pairs and attempts < max_attempts:
            attempts += 1
            
            # 根据偏好对类型选择轨迹
            if pair_type == "high_quality":
                # 高质量对：选择高奖励轨迹 vs 低奖励轨迹
                high_idx = np.random.choice(len(traj_rewards) // 3)  # 前1/3
                low_idx = np.random.choice(range(2 * len(traj_rewards) // 3, len(traj_rewards)))  # 后1/3
                traj_a = traj_rewards[high_idx][1]
                traj_b = traj_rewards[low_idx][1]
            elif pair_type == "medium_quality":
                # 中等质量对：选择相近奖励的轨迹
                mid_start = len(traj_rewards) // 4
                mid_end = 3 * len(traj_rewards) // 4
                indices = np.random.choice(range(mid_start, mid_end), size=2, replace=False)
                traj_a = traj_rewards[indices[0]][1]
                traj_b = traj_rewards[indices[1]][1]
            else:  # exploration
                # 探索性对：随机选择
                indices = np.random.choice(len(traj_rewards), size=2, replace=False)
                traj_a = traj_rewards[indices[0]][1]
                traj_b = traj_rewards[indices[1]][1]
            
            # 计算轨迹的环境奖励用于诊断
            reward_a = np.sum(traj_a.get('reward', [0]))
            reward_b = np.sum(traj_b.get('reward', [0]))
            
            # 获取环境平均奖励和统计信息
            stats = self.adaptive_threshold_manager.get_statistics_summary()
            env_mean_reward = stats.get('mean', 0.0)
            reward_samples_count = len(self.adaptive_threshold_manager.reward_history)
            
            # 改进的过滤逻辑：单一判断机制
            # 要求其中一条轨迹必须高于历史最高45%阈值，避免垃圾数据学习
            if reward_samples_count >= 30:  # 使用新的窗口大小30
                # 获取历史最高环境平均值的45%阈值
                historical_max_threshold = self.adaptive_threshold_manager.get_historical_max_threshold(0.45)
                
                # 单一判断：至少有一条轨迹高于历史最高环境平均值的45%
                at_least_one_above_historical_threshold = (reward_a >= historical_max_threshold or reward_b >= historical_max_threshold)
                
                if not at_least_one_above_historical_threshold:
                    historical_max = self.adaptive_threshold_manager.get_historical_max_env_avg()
                    logger.debug(f"抛弃偏好对 - 两条轨迹都未达到历史最高45%阈值: 轨迹A={reward_a:.3f}, 轨迹B={reward_b:.3f}, 历史最高45%阈值={historical_max_threshold:.3f} (历史最高={historical_max:.3f})")
                    continue
                    
                # 通过单一判断，保留有意义的对比对
                historical_max = self.adaptive_threshold_manager.get_historical_max_env_avg()
                logger.debug(f"保留偏好对 - 轨迹A奖励={reward_a:.3f}, 轨迹B奖励={reward_b:.3f}, 历史最高45%阈值={historical_max_threshold:.3f} (历史最高={historical_max:.3f})")
            
            # 早期训练阶段检测：轨迹数量少或平均奖励很低时放宽过滤条件
            is_early_training = (reward_samples_count < 30 or env_mean_reward < 5.0)
            reward_diff = abs(reward_a - reward_b)
            
            if is_early_training:
                # 早期训练阶段：只要两条轨迹有明显差异就保留
                min_reward_diff = max(0.05, env_mean_reward * 0.05)  # 降低最小差异阈值
                if reward_diff < min_reward_diff:
                    logger.debug(f"[早期训练] 跳过偏好对 - 奖励差异太小: {reward_diff:.3f} < {min_reward_diff:.3f}")
                    continue
                else:
                    logger.debug(f"[早期训练] 保留偏好对 - 奖励A: {reward_a:.3f}, 奖励B: {reward_b:.3f}, 差异: {reward_diff:.3f}")
            else:
                # 训练后期：特别保留低质量vs高质量的对比对
                # 这些对比对对于偏好模型学习"什么是不好的行为"非常重要
                high_quality_threshold = env_mean_reward * 0.8  # 高质量阈值
                low_quality_threshold = env_mean_reward * 0.4   # 低质量阈值
                
                # 如果一个轨迹是高质量，另一个是低质量，这是非常有价值的对比
                is_valuable_contrast = (
                    (reward_a > high_quality_threshold and reward_b < low_quality_threshold) or
                    (reward_b > high_quality_threshold and reward_a < low_quality_threshold)
                )
                
                if is_valuable_contrast:
                    logger.debug(f"[训练后期] 保留有价值对比对 - 奖励A: {reward_a:.3f}, 奖励B: {reward_b:.3f}, 高质量阈值: {high_quality_threshold:.3f}, 低质量阈值: {low_quality_threshold:.3f}")
                else:
                    # 非早期训练阶段：检查奖励差异是否足够显著
                    reward_diff_threshold = max(1.0, env_mean_reward * 0.1)  # 至少1.0或平均奖励的10%
                    if reward_diff < reward_diff_threshold:
                        logger.debug(f"[正常训练] 跳过偏好对 - 奖励差异不够显著: {reward_diff:.3f} < {reward_diff_threshold:.3f}")
                        continue
                    else:
                        logger.debug(f"[正常训练] 保留偏好对 - 奖励A: {reward_a:.3f}, 奖励B: {reward_b:.3f}, 差异: {reward_diff:.3f}")
            
            # 生成偏好标签
            try:
                preference_label = self.preference_labeling_engine.generate_preference_labels(
                    traj_a['obs'], traj_a['action'],
                    traj_b['obs'], traj_b['action'],
                    label_type=label_type
                )
                
                if not preference_label.is_valid:
                    continue
                
                # 使用自适应阈值管理器检查数据质量
                quality_indicators = {
                    'confidence': preference_label.metadata.confidence,
                    'rule_score_diff': abs(preference_label.metadata.quality_score_a - 
                                         preference_label.metadata.quality_score_b),
                    'env_reward_diff': abs(reward_a - reward_b)
                }
                
                # 质量检查日志
                thresholds = self.adaptive_threshold_manager.get_quality_thresholds()
                    
                # 创建偏好对
                preference_pair = PreferencePair(
                    trajectory_a={
                        'obs': traj_a['obs'],
                        'action': traj_a['action'],
                        'reward': traj_a['reward'],
                        'done': traj_a['done']
                    },
                    trajectory_b={
                        'obs': traj_b['obs'],
                        'action': traj_b['action'],
                        'reward': traj_b['reward'],
                        'done': traj_b['done']
                    },
                    preference_label=preference_label.preference_score,
                    confidence_score=preference_label.metadata.confidence,
                    rule_score_diff=abs(preference_label.metadata.quality_score_a - 
                                       preference_label.metadata.quality_score_b),
                    timestamp=time.time(),
                    metadata={
                        'label_type': preference_label.metadata.label_type.value,
                        'quality_score_a': preference_label.metadata.quality_score_a,
                        'quality_score_b': preference_label.metadata.quality_score_b,
                        'trajectory_length_a': len(traj_a['obs']),
                        'trajectory_length_b': len(traj_b['obs']),
                        'generation_method': method_type,
                        'pair_type': pair_type,
                        'env_reward_a': reward_a,
                        'env_reward_b': reward_b,
                        'env_reward_diff': abs(reward_a - reward_b)
                    }
                )
                
                # 添加到经验回放池
                try:
                    self.prioritized_trainer.add_preference_pair(preference_pair)
                    self.stats['total_preference_pairs_generated'] += 1
                    generated_count += 1
                    logger.debug(f"[偏好对添加] 成功添加偏好对到缓冲区，当前总数: {self.stats['total_preference_pairs_generated']}")
                except Exception as add_error:
                    logger.error(f"[偏好对添加] 添加偏好对到缓冲区失败: {add_error}")
                    import traceback
                    logger.error(f"[偏好对添加] 详细错误信息: {traceback.format_exc()}")
                    continue
                    
            except Exception as e:
                logger.warning(f"使用{method_type}方法生成{pair_type}偏好对时出错: {e}")
                import traceback
                logger.warning(f"详细错误信息: {traceback.format_exc()}")
                continue
        
        # 输出自适应阈值系统汇总信息
        final_stats = self.adaptive_threshold_manager.get_stats()      
        return generated_count
    
    def train_preference_model(self) -> Optional[float]:
        """训练偏好模型"""
        if self.preference_model is None or not self.prioritized_trainer.is_ready_for_training():
            return None
        
        start_time = time.time()
        total_loss = 0.0
        n_batches = 0
        
        # 获取缓冲池统计信息
        buffer_stats = self.prioritized_trainer.get_statistics()
        buffer_size = buffer_stats.get('buffer_stats', {}).get('buffer_size', 0)
        
        # 静默开始训练，不输出详细配置信息
        
        # 使用轨迹编码器进行潜空间训练
        encoder = self.trajectory_encoder
        
        # 多个epoch的训练
        for epoch in range(self.config.max_training_epochs):
            # 采样批次
            sampled_pairs, indices, weights = self.prioritized_trainer.sample_batch()
            
            if len(sampled_pairs) == 0:
                break
            
            # 准备训练数据
            batch_losses = []
            pair_details = []
            
            for i, pair in enumerate(sampled_pairs):
                try:
                    # 准备输入数据
                    obs_a = torch.FloatTensor(pair.trajectory_a['obs']).to(self.config.device)
                    action_a = torch.FloatTensor(pair.trajectory_a['action']).to(self.config.device)
                    obs_b = torch.FloatTensor(pair.trajectory_b['obs']).to(self.config.device)
                    action_b = torch.FloatTensor(pair.trajectory_b['action']).to(self.config.device)
                    
                    # 计算偏好预测
                    if hasattr(self.preference_model, 'compute_preference_logits'):
                        # 使用轨迹编码器进行特征编码（保持梯度）
                        if hasattr(self, 'trajectory_encoder') and self.trajectory_encoder is not None:
                            # 使用轨迹编码器的单轨迹编码方法（保持梯度）
                            encoded_a = self.trajectory_encoder.encode_single_trajectory_with_grad(obs_a, action_a)
                            encoded_b = self.trajectory_encoder.encode_single_trajectory_with_grad(obs_b, action_b)
                                
                            logits = self.preference_model.compute_preference_logits(
                                encoded_a, action_a, encoded_b, action_b
                            )
                        else:
                            logits = self.preference_model.compute_preference_logits(
                                obs_a, action_a, obs_b, action_b
                            )
                    else:
                        # 假设模型有标准的前向传播方法
                        if hasattr(self, 'trajectory_encoder') and self.trajectory_encoder is not None:
                            # 使用轨迹编码器编码轨迹数据（保持梯度）
                            encoded_a = self.trajectory_encoder.encode_single_trajectory_with_grad(obs_a, action_a)
                            encoded_b = self.trajectory_encoder.encode_single_trajectory_with_grad(obs_b, action_b)
                                
                            reward_a = self.preference_model(encoded_a, action_a).mean()
                            reward_b = self.preference_model(encoded_b, action_b).mean()
                        else:
                            reward_a = self.preference_model(obs_a, action_a).mean()
                            reward_b = self.preference_model(obs_b, action_b).mean()
                        logits = reward_a - reward_b
                    
                    # 计算目标
                    target = torch.FloatTensor([pair.preference_label]).to(self.config.device)
                    
                    # 计算损失（使用二元交叉熵）
                    loss = nn.BCEWithLogitsLoss()(logits.unsqueeze(0), target)
                    
                    # 应用重要性权重
                    # 将numpy权重转换为PyTorch张量，但不需要梯度
                    weight_tensor = torch.tensor(weights[i], dtype=torch.float32, device=self.config.device, requires_grad=False)
                    weighted_loss = loss * weight_tensor
                    
                    # 损失稳定化处理
                    stabilized_loss = self._stabilize_training_loss(weighted_loss, pair.confidence_score)
                    batch_losses.append(stabilized_loss.item())
                    
                    # 记录偏好对详细信息
                    pair_details.append({
                        'pair_idx': indices[i],
                        'preference_label': pair.preference_label,
                        'confidence_score': pair.confidence_score,
                        'priority': pair.priority,
                        'sample_count': pair.sample_count,
                        'loss': stabilized_loss.item(),
                        'weight': weights[i],
                        'rule_score_diff': pair.rule_score_diff
                    })
                    
                    # 反向传播
                    self.preference_optimizer.zero_grad()
                    stabilized_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.preference_model.parameters(), 
                                                  self.config.preference_model_grad_clip)
                    
                    self.preference_optimizer.step()
                    
                except Exception as e:
                    logger.warning(f"训练偏好对 {i} 时出错: {e}")
                    batch_losses.append(0.0)
                    continue
            
            # 输出核心统计信息（用户要求保留的信息）
            if pair_details:
                avg_confidence = np.mean([d['confidence_score'] for d in pair_details])
                avg_priority = np.mean([d['priority'] for d in pair_details])
                # logger.info(f"[优先级偏好系统] Epoch {epoch+1} 批次统计 - 平均置信度: {avg_confidence:.3f}, 平均优先级: {avg_priority:.4f}")
            
            # 更新优先级
            if batch_losses:
                self.prioritized_trainer.update_priorities_with_losses(
                    indices, np.array(batch_losses)
                )
                
                batch_avg_loss = np.mean(batch_losses)
                total_loss += batch_avg_loss
                n_batches += 1
                
                # 输出核心损失和成功率统计（用户要求保留的信息）
                # logger.info(f"[优先级偏好系统] Epoch {epoch+1} 完成 - 平均损失: {batch_avg_loss:.6f}, 损失范围: [{np.min(batch_losses):.6f}, {np.max(batch_losses):.6f}]")
                # logger.info(f"[优先级偏好系统] Epoch {epoch+1} 统计 - 有效偏好对: {len(batch_losses)}/{len(sampled_pairs)}, 成功率: {len(batch_losses)/len(sampled_pairs):.1%}")
        
        # 更新统计信息并输出损失变化日志
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            training_time = time.time() - start_time
            
            # 计算损失变化
            previous_loss = self.stats['avg_training_loss']
            loss_change = avg_loss - previous_loss if previous_loss > 0 else 0.0
            loss_change_percent = (loss_change / previous_loss * 100) if previous_loss > 0 else 0.0
            
            self.stats['total_training_steps'] += 1
            self.stats['avg_training_loss'] = (
                (self.stats['avg_training_loss'] * (self.stats['total_training_steps'] - 1) + avg_loss) /
                self.stats['total_training_steps']
            )
            self.stats['last_training_time'] = training_time
            
            # 输出详细的损失变化日志
            logger.info(f"[潜空间偏好奖励模型] 训练完成 - 当前损失: {avg_loss:.6f}, 训练时间: {training_time:.3f}s")
            if previous_loss > 0:
                change_direction = "↓" if loss_change < 0 else "↑" if loss_change > 0 else "→"
                logger.info(f"[潜空间偏好奖励模型] 损失变化: {change_direction} {abs(loss_change):.6f} ({loss_change_percent:+.2f}%), 历史平均: {self.stats['avg_training_loss']:.6f}")
            logger.info(f"[潜空间偏好奖励模型] 训练统计: 批次数={n_batches}, 总训练步数={self.stats['total_training_steps']}, 缓冲池大小={buffer_size}")
            
            return avg_loss
        
        return None

    def _stabilize_training_loss(self, loss: torch.Tensor, confidence: float) -> torch.Tensor:
        """稳定化训练损失
        
        Args:
            loss: 原始损失
            confidence: 偏好对置信度
            
        Returns:
            稳定化后的损失
        """
        # 基于置信度调整损失权重
        if confidence < 0.4:
            # 低置信度样本降低权重
            confidence_weight = 0.5
        elif confidence < 0.7:
            # 中等置信度样本正常权重
            confidence_weight = 1.0
        else:
            # 高置信度样本略微提高权重
            confidence_weight = 1.2
        
        adjusted_loss = loss * confidence_weight
        
        # 损失裁剪，避免极端值
        max_loss = 5.0  # 最大损失阈值
        clipped_loss = torch.clamp(adjusted_loss, max=max_loss)
        
        # 应用平滑处理
        if hasattr(self, '_loss_history'):
            self._loss_history.append(clipped_loss.item())
            if len(self._loss_history) > 20:
                self._loss_history.pop(0)
            
            # 如果当前损失远大于历史平均值，进行平滑
            avg_loss = sum(self._loss_history) / len(self._loss_history)
            if clipped_loss.item() > avg_loss * 2.0:
                smoothing_factor = 0.7
                smoothed_loss = clipped_loss * smoothing_factor + torch.tensor(avg_loss * (1 - smoothing_factor), device=loss.device)
                return smoothed_loss
        else:
            self._loss_history = [clipped_loss.item()]
        
        return clipped_loss

    def should_train_preference_model(self) -> bool:
        """检查是否应该训练偏好模型"""
        # 静默检查训练条件
        
        # 检查偏好模型是否存在
        has_model = self.preference_model is not None
        if not has_model:
            return False
        
        # 检查经验回放池是否就绪
        is_ready = self.prioritized_trainer.is_ready_for_training()
        if not is_ready:
            return False
        
        # 检查缓冲池大小
        buffer_stats = self.prioritized_trainer.get_statistics()
        buffer_size = buffer_stats.get('buffer_stats', {}).get('buffer_size', 0)
        min_required = self.config.min_buffer_size
        
        should_train = buffer_size >= min_required
        return should_train
    
    def get_preference_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """获取偏好奖励"""
        if self.preference_model is None:
            return 0.0
        
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.config.device)
                
                if hasattr(self.preference_model, 'compute_reward'):
                    reward = self.preference_model.compute_reward(obs_tensor, action_tensor)
                else:
                    reward = self.preference_model(obs_tensor, action_tensor)
                
                return float(reward.cpu().item())
        except Exception as e:
            logger.warning(f"计算偏好奖励时出错: {e}")
            return 0.0
    
    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """保存检查点"""
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"{self.config.save_dir}/prioritized_preference_system_{timestamp}.pt"
        
        checkpoint = {
            'config': self.config,
            'stats': self.stats,
            'preference_model_state': self.preference_model.state_dict() if self.preference_model else None,
            'optimizer_state': self.preference_optimizer.state_dict() if self.preference_optimizer else None,
            'buffer_stats': self.prioritized_trainer.get_statistics()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        if 'preference_model_state' in checkpoint and checkpoint['preference_model_state'] is not None:
            if self.preference_model is not None:
                self.preference_model.load_state_dict(checkpoint['preference_model_state'])
        
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            if self.preference_optimizer is not None:
                self.preference_optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if 'stats' in checkpoint:
            self.stats.update(checkpoint['stats'])
        
        logger.info(f"检查点已从 {filepath} 加载")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        buffer_stats = self.prioritized_trainer.get_statistics()
        trajectory_stats = {
            'current_trajectories': len(self.trajectory_collector.current_trajectories),
            'completed_trajectories': len(self.trajectory_collector.completed_trajectories),
            'cache_utilization': f"{len(self.trajectory_collector.completed_trajectories)}/{self.config.train_every_n_episodes}"
        }
        
        stats = self.stats.copy()
        stats['prioritized_trainer_stats'] = buffer_stats
        stats['trajectory_collector_stats'] = trajectory_stats
        stats['config_info'] = {
            'buffer_capacity': self.config.buffer_capacity,
            'batch_size': self.config.batch_size,
            'min_buffer_size': self.config.min_buffer_size,
            'train_every_n_episodes': self.config.train_every_n_episodes,
            'priority_alpha': self.config.priority_alpha,
            'priority_beta': self.config.priority_beta
        }
        
        if hasattr(self.preference_labeling_engine, 'get_statistics'):
            stats['labeling_engine_stats'] = self.preference_labeling_engine.get_statistics()
        
        # 静默处理统计信息，不输出详细日志
        buffer_size = buffer_stats.get('buffer_stats', {}).get('buffer_size', 0)
        buffer_utilization = buffer_size / self.config.buffer_capacity
        
        cached_trajectories = len(self.trajectory_collector.completed_trajectories)
        cache_progress = cached_trajectories / self.config.train_every_n_episodes
        
        # 计算效率指标（静默处理）
        if self.stats['total_trajectories_collected'] > 0:
            pairs_per_traj = self.stats['total_preference_pairs_generated'] / self.stats['total_trajectories_collected']
            if self.stats['total_training_steps'] > 0:
                pairs_per_training = self.stats['total_preference_pairs_generated'] / self.stats['total_training_steps']
        
        return stats
    
    def cleanup(self) -> None:
        """清理资源"""
        self.trajectory_collector.clear_old_trajectories(keep_last_n=50)
        logger.info("优先级偏好系统资源清理完成")

# 工厂函数
def create_prioritized_preference_system(task_name: str, 
                                        config: Optional[PrioritizedSystemConfig] = None,
                                        preference_model: Optional[nn.Module] = None,
                                        tdmpc2_agent=None,
                                        tdmpc2_cfg=None) -> PrioritizedPreferenceSystem:
    """
    创建优先级偏好系统的工厂函数
    
    Args:
        task_name: 任务名称
        config: 系统配置，如果为None则从tdmpc2_cfg创建优化配置
        preference_model: 偏好模型
        tdmpc2_agent: TD-MPC2 agent实例
        tdmpc2_cfg: TD-MPC2配置对象
        
    Returns:
        优先级偏好系统实例
    """
    # 如果没有提供配置，从TD-MPC2配置创建优化配置
    if config is None and tdmpc2_cfg is not None:
        config = PrioritizedSystemConfig.from_tdmpc2_config(tdmpc2_cfg)
        logger.info(f"[工厂函数] 使用TD-MPC2配置创建优化的偏好系统配置")
    
    return PrioritizedPreferenceSystem(task_name, config, preference_model, tdmpc2_agent, tdmpc2_cfg)

if __name__ == "__main__":
    # 测试代码
    print("测试优先级偏好系统...")
    
    # 创建系统
    config = PrioritizedSystemConfig(
        buffer_capacity=500,
        batch_size=16,
        min_buffer_size=50,
        train_every_n_episodes=3
    )
    
    system = create_prioritized_preference_system(
        task_name="test_task",
        config=config
    )
    
    # 模拟轨迹收集
    for episode in range(10):
        traj_id = system.start_new_trajectory()
        
        # 模拟轨迹步骤
        for step in range(50):
            obs = np.random.randn(10)
            action = np.random.randn(5)
            reward = np.random.rand()
            done = step == 49
            
            system.collect_trajectory_step(traj_id, obs, action, reward, done)
        
        system.complete_trajectory(traj_id)
        print(f"完成轨迹 {episode + 1}")
        
        # 检查是否应该训练
        if system.should_train_preference_model():
            print("准备训练偏好模型，但没有实际模型")
    
    # 获取统计信息
    stats = system.get_statistics()
    print(f"\n系统统计信息:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    print(f"    {sub_key}:")
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        print(f"      {sub_sub_key}: {sub_sub_value}")
                else:
                    print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print("测试完成！")