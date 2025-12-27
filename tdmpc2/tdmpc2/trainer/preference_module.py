import os
import sys
import time
import traceback
import yaml
from collections import deque
import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize_scalar
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import random
from dataclasses import dataclass
from enum import Enum
from collections import deque
import math

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from grpo.reward_model import PreferenceRewardModel
from grpo.train_reward_model import train_reward_model_once
from prm.preference_data_engine import PreferenceDataEngine
# 偏好学习相关导入


class PreferenceModule:
    """偏好奖励模块 - 可插拔的偏好学习组件（优化版）"""
    
    def __init__(self, cfg, env, agent, task_name):
        """初始化偏好模块
        
        Args:
            cfg: 配置对象
            env: 环境实例
            agent: 智能体实例
            task_name: 任务名称
        """
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.task_name = task_name
        
        # 偏好引擎
        self.preference_engine = None
        
    def is_enabled(self):
        """检查偏好学习是否启用
        
        Returns:
            bool: 如果偏好学习启用返回True，否则返回False
        """
        # 检查多个配置参数来确定是否启用偏好学习
        preference_enabled = getattr(self.cfg, 'preference_enabled', True)
        use_preference_engine = getattr(self.cfg, 'use_preference_engine', True)
        
        return preference_enabled and use_preference_engine
    
    def should_update_preference_model(self, episode_idx):
        """检查是否应该更新偏好模型
        
        Args:
            episode_idx: 当前回合索引
            
        Returns:
            bool: 如果应该更新偏好模型返回True，否则返回False
        """
        if not self.is_enabled():
            return False
            
        preference_start_episode = getattr(self.cfg, 'preference_start_episode', 50)
        preference_update_interval = getattr(self.cfg, 'preference_update_interval_episode', 10)
        
        return (episode_idx >= preference_start_episode and
                episode_idx % preference_update_interval == 0)
    
    def should_calculate_preference(self, step):
        """检查是否应该计算偏好奖励
        
        Args:
            step: 当前步数
            
        Returns:
            bool: 如果应该计算偏好奖励返回True，否则返回False
        """
        if not self.is_enabled():
            return False
            
        # 可以添加其他条件，比如步数阈值等
        return True
        
        # 奖励模型相关
        self.reward_model = None
        self.reward_model_path = self._build_reward_model_path()
        self.reward_model_mtime = None
        
        # 从配置读取所有参数（取消硬编码）
        self._load_config_parameters()
        
        # 偏好奖励缓冲区
        self.pref_reward_buffer = deque(maxlen=self.buffer_size)
        
        # 偏好奖励统计
        self.pref_reward_mean = 0.0
        self.pref_reward_std = 1.0
        
        # 调试计数器
        self._debug_counter = 0
        self._stats_print_interval = 1000
        
        # 智能缓存系统（优化版）
        self._data_cache = {}
        self._cache_metadata = {}  # 存储缓存元数据
        self._cache_timestamps = {}
        self._cache_max_age = getattr(cfg, 'preference_cache_max_age', 300)  # 从config读取缓存最大年龄
        self._cache_max_size = getattr(cfg, 'preference_cache_size', 2000)
        self._similarity_threshold = getattr(cfg, 'cache_similarity_threshold', 0.15)
        self._quantization_precision = getattr(cfg, 'cache_quantization_precision', 2)
        
        # 自适应频率控制
        self._current_frequency = getattr(cfg, 'preference_frequency', 20)
        self._min_frequency = getattr(cfg, 'preference_min_frequency', 5)
        self._max_frequency = getattr(cfg, 'preference_max_frequency', 50)
        self._step_counter = 0
        self._computation_times = deque(maxlen=getattr(cfg, 'computation_times_maxlen', 100))
        self._learning_qualities = deque(maxlen=getattr(cfg, 'learning_qualities_maxlen', 100))
        
        # 时间窗口策略
        self._window_size = getattr(cfg, 'preference_window_size', 5)
        self._window_calculations = 0
        
        # 缓存统计
        self._cache_hits = 0
        self._exact_hits = 0
        self._similarity_hits = 0
        self._total_cache_queries = 0
        
        # 偏好学习监控参数
        self._monitoring_log_interval = getattr(cfg, 'monitoring_log_interval', 500)
        
        # 偏好学习统计
        self._preference_stats = {
            'total_preferences': 0,
            'preference_rewards': deque(maxlen=1000)
        }
        
        # 验证配置参数
        self._validate_config()
        
        # 初始化偏好引擎
        self._initialize_preference_engine()
        
        # 偏好对临时缓存（用于训练）
        self._preference_pairs_cache = {
            'rule_based': [],
            'collected': []
        }
        
        # 奖励模型缓存系统
        self._reward_model_cache = None  # 缓存的奖励模型
        self._cached_model_timestamp = None  # 缓存模型的时间戳
        self._cached_model_path = None  # 缓存模型的路径
        
        print(f"[PreferenceModule] 初始化完成 - 偏好奖励模型独立运行")
        
        # 加载奖励模型
        self._load_reward_model()
    
    def _load_config_parameters(self):
        """从配置文件加载所有参数"""
        # 基础参数 - 使用config.yaml中的值作为默认值
        self.history_length = getattr(self.cfg, "history_length", 5)
        self.preference_start_episode = getattr(self.cfg, "preference_start_episode", 100)
        self.preference_update_interval_episode = getattr(self.cfg, "preference_update_interval_episode", 10)
        
        # 缓冲区参数
        self.buffer_size = getattr(self.cfg, 'r_pref_buffer_size', 100)
        
        # 奖励混合参数 - 从config.yaml读取
        self.env_reward_weight = getattr(self.cfg, "env_reward_weight", 0.8)  # 默认值与config.yaml一致
        self.pref_reward_weight = getattr(self.cfg, "pref_reward_weight", 0.2)  # 默认值与config.yaml一致
        self.pref_reward_scale = getattr(self.cfg, "pref_reward_scale", 100)  # 使用config.yaml中的值
        self.pref_reward_clip = getattr(self.cfg, "pref_reward_clip", 10.0)  # 使用config.yaml中的值
        
        # 环境奖励统计
        self.env_reward_buffer = deque(maxlen=1000)
        self.env_reward_short_buffer = deque(maxlen=100)
        self.env_reward_mean = 0.0
        self.env_reward_std = 1.0
        self.env_reward_short_mean = 0.0
        self.env_reward_short_std = 1.0
        
        # 数据收集参数
        self.preference_num_trajs_collect = getattr(self.cfg, "preference_num_trajs_collect", 15)
        self.preference_num_trajs_rules = getattr(self.cfg, "preference_num_trajs_rules", 15)
        
        print(f"[PreferenceModule] 配置参数加载完成:")
        print(f"  偏好学习开始回合: {self.preference_start_episode}")
        print(f"  偏好更新间隔: {self.preference_update_interval_episode}")
        print(f"  数据收集轨迹数: {self.preference_num_trajs_collect}")
        print(f"  规则轨迹数: {self.preference_num_trajs_rules}")
    
    def _validate_config(self):
        """验证配置参数的合理性"""
        # 验证基本参数
        if self.preference_start_episode < 0:
            print(f"[PreferenceModule] 警告: 偏好学习开始回合不能为负数: {self.preference_start_episode}")
        
        if self.preference_update_interval_episode <= 0:
            print(f"[PreferenceModule] 警告: 偏好更新间隔必须为正数: {self.preference_update_interval_episode}")
        
        if self.preference_num_trajs_collect <= 0 or self.preference_num_trajs_rules <= 0:
            print(f"[PreferenceModule] 警告: 轨迹收集数量必须为正数")
            print(f"  收集轨迹数: {self.preference_num_trajs_collect}, 规则轨迹数: {self.preference_num_trajs_rules}")
    
    def _initialize_preference_engine(self):
        """初始化偏好数据引擎"""
        try:
            # 动态设置偏好数据的保存路径
            prm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../prm'))
            
            # 标准化任务名称
            task_name_for_path = self.task_name
            if task_name_for_path.startswith('humanoid_'):
                task_name_for_path = task_name_for_path.replace('humanoid_', '')
            
            rule_data_save_path = os.path.join(prm_root, "rules", task_name_for_path)
            collect_data_save_path = os.path.join(prm_root, "prefer", task_name_for_path)
            
            # 确保目录存在
            os.makedirs(rule_data_save_path, exist_ok=True)
            os.makedirs(collect_data_save_path, exist_ok=True)
            
            # 设置环境变量供其他模块使用
            os.environ["TASK"] = task_name_for_path
            
            self.preference_engine = PreferenceDataEngine(
                env=self.env,
                agent=self.agent,
                task_name=self.task_name,
                num_trajectories_per_policy=self.preference_num_trajs_collect,
                max_trajectory_length=getattr(self.cfg, "episode_length", 1000),
                rule_data_save_path=rule_data_save_path,
                collect_data_save_path=collect_data_save_path
            )
            
            print(f"[PreferenceModule] 偏好数据引擎已初始化，任务: {task_name_for_path}")
            print(f"[PreferenceModule] 规则数据路径: {rule_data_save_path}")
            print(f"[PreferenceModule] 采集数据路径: {collect_data_save_path}")
            
        except Exception as e:
            print(f"[PreferenceModule] 偏好引擎初始化失败: {e}")
            traceback.print_exc()
            self.preference_engine = None
    
    def _build_reward_model_path(self):
        """构建奖励模型路径"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        task_name_for_path = self.task_name
        if task_name_for_path.startswith('humanoid_'):
            task_name_for_path = task_name_for_path.replace('humanoid_', '')
        
        model_path = os.path.join(
            project_root,
            "prm", "prefer", task_name_for_path, "preference_reward_model.pt"
        )
        print(f"[PreferenceModule] 奖励模型路径: {model_path}")
        return model_path
    
    def _load_transformer_config(self):
        """从config.yaml加载Transformer配置"""
        # 默认配置
        transformer_config = {
            'transformer_hidden_dim': 256,
            'transformer_n_heads': 4,
            'transformer_n_layers': 3,
            'transformer_dropout': 0.3,
            'transformer_max_seq_len': 1000,  # 增加到500以支持更长的轨迹
            'transformer_learning_rate': 1e-4
        }
        
        # 从cfg对象读取配置
        for key in transformer_config.keys():
            if hasattr(self.cfg, key):
                transformer_config[key] = getattr(self.cfg, key)
                
        print(f"[PreferenceModule] Transformer配置加载成功")
        return transformer_config
    
    def _load_reward_model(self):
        """加载偏好奖励模型（优先从缓存加载）
        
        Returns:
            PreferenceRewardModel or None: 加载的模型对象，如果加载失败则返回None
        """
        try:
            # 1. 优先检查缓存中的模型
            if self._reward_model_cache is not None:
                # 检查缓存模型是否仍然有效
                if (self._cached_model_path == self.reward_model_path and 
                    os.path.exists(self.reward_model_path)):
                    
                    current_mtime = os.path.getmtime(self.reward_model_path)
                    if self._cached_model_timestamp == current_mtime:
                        print(f"[PreferenceModule] 使用缓存中的偏好奖励模型")
                        self.reward_model = self._reward_model_cache
                        self.reward_model_mtime = current_mtime
                        return self._reward_model_cache
                    else:
                        print(f"[PreferenceModule] 缓存模型已过期，重新加载")
                        self._clear_model_cache()
                else:
                    print(f"[PreferenceModule] 缓存模型路径不匹配，重新加载")
                    self._clear_model_cache()
            
            # 2. 检查文件是否存在
            if not os.path.exists(self.reward_model_path):
                print(f"[PreferenceModule] 偏好奖励模型文件不存在: {self.reward_model_path}")
                self.reward_model = None
                return None
            
            # 3. 检查文件修改时间（如果已有模型且非缓存）
            current_mtime = os.path.getmtime(self.reward_model_path)
            if (self.reward_model_mtime == current_mtime and 
                self.reward_model is not None and 
                self._reward_model_cache is None):
                print(f"[PreferenceModule] 偏好奖励模型已是最新版本")
                return self.reward_model
            
            # 4. 动态获取环境的观测和动作空间维度
            try:
                # 从环境获取观测空间维度
                if hasattr(self.env, 'observation_space'):
                    if hasattr(self.env.observation_space, 'shape'):
                        state_dim = self.env.observation_space.shape[0]
                    else:
                        state_dim = self.env.observation_space.n
                else:
                    # 备用方案：从agent获取
                    state_dim = getattr(self.agent, 'obs_dim', 151)
                
                # 从环境获取动作空间维度
                if hasattr(self.env, 'action_space'):
                    if hasattr(self.env.action_space, 'shape'):
                        action_dim = self.env.action_space.shape[0]
                    else:
                        action_dim = self.env.action_space.n
                else:
                    # 备用方案：从agent获取
                    action_dim = getattr(self.agent, 'action_dim', 61)
                
                print(f"[PreferenceModule] 动态获取维度: state_dim={state_dim}, action_dim={action_dim}")
                
            except Exception as e:
                print(f"[PreferenceModule] 无法动态获取维度，使用默认值: {e}")
                state_dim = 151
                action_dim = 61
            
            # 5. 加载Transformer配置
            transformer_config = self._load_transformer_config()
            
            # 6. 创建模型实例
            self.reward_model = PreferenceRewardModel(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=transformer_config['transformer_hidden_dim'],
                n_heads=transformer_config['transformer_n_heads'],
                n_layers=transformer_config['transformer_n_layers'],
                dropout=transformer_config['transformer_dropout'],
                max_seq_len=transformer_config['transformer_max_seq_len']
            ).to(self.agent.device)
            
            # 7. 加载模型权重
            self.reward_model.load_state_dict(torch.load(self.reward_model_path, map_location=self.agent.device))
            self.reward_model.eval()
            
            # 8. 更新文件修改时间
            self.reward_model_mtime = current_mtime
            
            # 9. 将模型保存到缓存中
            self._cache_reward_model(self.reward_model, current_mtime, self.reward_model_path)
            
            print(f"[PreferenceModule] 偏好奖励模型加载成功并已缓存")
            return self.reward_model
            
        except Exception as e:
            print(f"[PreferenceModule] 偏好奖励模型加载失败: {e}")
            traceback.print_exc()
            self.reward_model = None
            self._clear_model_cache()
            return None
    
    def _is_cache_valid(self, cache_key):
        """检查缓存是否有效"""
        if cache_key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[cache_key] < self._cache_max_age
    
    def _clear_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self._cache_max_age:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._data_cache.pop(key, None)
            self._cache_metadata.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def get_cache_statistics(self):
        """获取缓存统计信息"""
        hit_rate = self._cache_hits / self._total_cache_queries if self._total_cache_queries > 0 else 0.0
        exact_hit_rate = self._exact_hits / self._total_cache_queries if self._total_cache_queries > 0 else 0.0
        similarity_hit_rate = self._similarity_hits / self._total_cache_queries if self._total_cache_queries > 0 else 0.0
        
        return {
            'total_queries': self._total_cache_queries,
            'hit_rate': hit_rate,
            'exact_hit_rate': exact_hit_rate,
            'similarity_hit_rate': similarity_hit_rate,
            'cache_size': len(self._data_cache),
            'cache_utilization': len(self._data_cache) / self._cache_max_size,
            'current_frequency': self._current_frequency,
            'avg_computation_time': np.mean(self._computation_times) if self._computation_times else 0,
            'window_calculations': self._window_calculations
        }
    
    def should_calculate_preference(self, step: int = None) -> bool:
        """自适应频率控制：判断是否应该计算偏好奖励（优化后）"""
        if step is not None:
            # 在训练早期更频繁地计算偏好奖励
            if step < 1000:
                early_frequency = max(1, self._current_frequency // 2)
                return step % early_frequency == 0
            # 在学习质量较低时提高频率
            elif len(self._learning_qualities) > 0 and np.mean(list(self._learning_qualities)[-5:]) < 0.6:
                boost_frequency = max(1, self._current_frequency // 1.5)
                return step % int(boost_frequency) == 0
            else:
                return step % self._current_frequency == 0
        else:
            self._step_counter += 1
            # 应用相同的早期和质量提升逻辑
            if self._step_counter < 1000:
                early_frequency = max(1, self._current_frequency // 2)
                return self._step_counter % early_frequency == 0
            elif len(self._learning_qualities) > 0 and np.mean(list(self._learning_qualities)[-5:]) < 0.6:
                boost_frequency = max(1, self._current_frequency // 1.5)
                return self._step_counter % int(boost_frequency) == 0
            else:
                return self._step_counter % self._current_frequency == 0
    
    def _compute_quantized_cache_key(self, obs: np.ndarray, action: np.ndarray, episode_idx: int) -> str:
        """计算量化缓存键以提高命中率"""
        try:
            # 量化观测和动作
            obs_quantized = np.round(obs * (10 ** self._quantization_precision)) / (10 ** self._quantization_precision)
            action_quantized = np.round(action * (10 ** self._quantization_precision)) / (10 ** self._quantization_precision)
            
            # 使用更大的episode窗口
            episode_window = episode_idx // 10
            
            combined = np.concatenate([
                obs_quantized.flatten(), 
                action_quantized.flatten(), 
                [episode_window]
            ])
            return str(hash(combined.tobytes()))
        except:
            return f"fallback_{hash(str(obs))}_{hash(str(action))}_{episode_idx}"
    
    def _compute_similarity(self, obs1: np.ndarray, action1: np.ndarray, 
                          obs2: np.ndarray, action2: np.ndarray) -> float:
        """改进的观测-动作对相似度计算"""
        try:
            # 确保输入是numpy数组
            obs1, action1 = np.array(obs1), np.array(action1)
            obs2, action2 = np.array(obs2), np.array(action2)
            
            # 展平数组
            obs1_flat, action1_flat = obs1.flatten(), action1.flatten()
            obs2_flat, action2_flat = obs2.flatten(), action2.flatten()
            
            # 检查维度匹配
            if obs1_flat.shape != obs2_flat.shape or action1_flat.shape != action2_flat.shape:
                return 0.0
            
            # 计算余弦相似度
            obs1_norm = obs1_flat / (np.linalg.norm(obs1_flat) + 1e-8)
            obs2_norm = obs2_flat / (np.linalg.norm(obs2_flat) + 1e-8)
            action1_norm = action1_flat / (np.linalg.norm(action1_flat) + 1e-8)
            action2_norm = action2_flat / (np.linalg.norm(action2_flat) + 1e-8)
            
            obs_cosine = np.dot(obs1_norm, obs2_norm)
            action_cosine = np.dot(action1_norm, action2_norm)
            
            # 计算欧几里得距离相似度（归一化）
            obs_l2_dist = np.linalg.norm(obs1_flat - obs2_flat)
            action_l2_dist = np.linalg.norm(action1_flat - action2_flat)
            
            # 将距离转换为相似度（使用指数衰减）
            obs_l2_sim = np.exp(-obs_l2_dist / (np.linalg.norm(obs1_flat) + 1e-8))
            action_l2_sim = np.exp(-action_l2_dist / (np.linalg.norm(action1_flat) + 1e-8))
            
            # 计算曼哈顿距离相似度
            obs_l1_dist = np.sum(np.abs(obs1_flat - obs2_flat))
            action_l1_dist = np.sum(np.abs(action1_flat - action2_flat))
            
            obs_l1_sim = np.exp(-obs_l1_dist / (np.sum(np.abs(obs1_flat)) + 1e-8))
            action_l1_sim = np.exp(-action_l1_dist / (np.sum(np.abs(action1_flat)) + 1e-8))
            
            # 综合相似度计算（多指标融合）
            obs_similarity = (
                0.5 * obs_cosine +      # 余弦相似度
                0.3 * obs_l2_sim +      # L2距离相似度
                0.2 * obs_l1_sim        # L1距离相似度
            )
            
            action_similarity = (
                0.5 * action_cosine +   # 余弦相似度
                0.3 * action_l2_sim +   # L2距离相似度
                0.2 * action_l1_sim     # L1距离相似度
            )
            
            # 动态权重调整（基于数据特征）
            obs_magnitude = np.linalg.norm(obs1_flat)
            action_magnitude = np.linalg.norm(action1_flat)
            
            # 如果观测或动作幅度很小，降低其权重
            obs_weight = 0.7 * min(1.0, obs_magnitude / 0.1)
            action_weight = 0.3 * min(1.0, action_magnitude / 0.1)
            
            # 归一化权重
            total_weight = obs_weight + action_weight
            if total_weight > 0:
                obs_weight /= total_weight
                action_weight /= total_weight
            else:
                obs_weight, action_weight = 0.7, 0.3
            
            # 最终加权相似度
            weighted_similarity = obs_weight * obs_similarity + action_weight * action_similarity
            
            return np.clip(weighted_similarity, 0.0, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _adapt_frequency(self):
        """改进的自适应频率调整（使用新的学习质量评估）"""
        if len(self._computation_times) < 5:  # 减少所需数据量，更快响应
            return
        
        avg_computation_time = np.mean(list(self._computation_times)[-10:])  # 使用更少的历史数据
        avg_learning_quality = np.mean(list(self._learning_qualities)[-10:]) if self._learning_qualities else 0.7
        
        # 计算统计量
        computation_time_std = np.std(list(self._computation_times)[-10:]) if len(self._computation_times) > 1 else 0
        quality_std = np.std(list(self._learning_qualities)[-10:]) if len(self._learning_qualities) > 1 else 0
        
        # 目标参数（更保守的设置）
        target_computation_time = 0.06  # 60ms，更严格的时间要求
        target_learning_quality = 0.65  # 稍微提高质量要求
        
        # 计算调整因子
        time_factor = target_computation_time / max(avg_computation_time, 0.001)
        quality_factor = avg_learning_quality / target_learning_quality
        
        # 稳定性因子（基于方差）
        time_stability = 1.0 / (1.0 + computation_time_std * 10.0)
        quality_stability = 1.0 / (1.0 + quality_std * 5.0)
        stability_factor = (time_stability + quality_stability) / 2.0
        
        # 综合调整因子（多因素权衡）
        adjustment_factor = (
            0.4 * time_factor +           # 时间因子权重
            0.3 * quality_factor +        # 质量因子权重
            0.3 * stability_factor        # 稳定性因子权重
        )
        
        # 自适应敏感度（基于历史稳定性）
        base_sensitivity = 0.15
        adaptive_sensitivity = base_sensitivity * stability_factor
        
        # 平滑调整
        new_frequency = self._current_frequency * (1 + adaptive_sensitivity * (adjustment_factor - 1))
        
        # 限制频率范围并应用渐进式调整
        max_change_ratio = 0.3  # 单次最大变化30%
        max_increase = self._current_frequency * (1 + max_change_ratio)
        max_decrease = self._current_frequency * (1 - max_change_ratio)
        
        new_frequency = np.clip(new_frequency, max_decrease, max_increase)
        self._current_frequency = int(np.clip(new_frequency, self._min_frequency, self._max_frequency))
    
    def _estimate_learning_quality(self, preference_reward: float, reward_variance=None, trajectory_length=None) -> float:
        """改进的学习质量估计（优化后，增强方差和轨迹长度考虑）"""
        try:
            # 基础质量评估（基于奖励幅度）
            reward_magnitude = abs(preference_reward)
            if reward_magnitude < 0.005:
                base_quality = 0.2  # 极低质量
            elif reward_magnitude < 0.02:
                base_quality = 0.4  # 低质量
            elif reward_magnitude < 0.08:
                base_quality = 0.65  # 中等质量
            elif reward_magnitude < 1.5:
                base_quality = 0.9   # 高质量
            elif reward_magnitude < 3.0:
                base_quality = 0.85  # 很高但可能不稳定
            else:
                base_quality = 0.7   # 超高奖励，降低置信度
            
            # 增强的方差调整（如果提供）
            if reward_variance is not None:
                # 低方差提高质量，高方差降低质量，使用更敏感的调整
                variance_factor = 1.0 / (1.0 + reward_variance * 15.0)  # 增加敏感度
                base_quality *= (0.7 + 0.5 * variance_factor)  # 增强方差影响
            
            # 增强的轨迹长度调整（如果提供）
            if trajectory_length is not None:
                # 更细致的轨迹长度评估
                if trajectory_length < 5:
                    length_factor = 0.6  # 极短，质量很低
                elif trajectory_length < 10:
                    length_factor = 0.8  # 太短
                elif trajectory_length < 30:
                    length_factor = 1.0  # 适中短
                elif trajectory_length < 100:
                    length_factor = 1.05  # 适中长，稍微提高质量
                elif trajectory_length < 200:
                    length_factor = 0.95  # 较长
                elif trajectory_length < 500:
                    length_factor = 0.85  # 过长
                else:
                    length_factor = 0.75  # 极长，可能包含噪声
                base_quality *= length_factor
            
            # 添加奖励一致性检查
            if hasattr(self, 'pref_reward_buffer') and len(self.pref_reward_buffer) > 5:
                recent_rewards = list(self.pref_reward_buffer)[-5:]
                reward_consistency = 1.0 - np.std(recent_rewards) / (np.mean(np.abs(recent_rewards)) + 1e-6)
                consistency_factor = np.clip(reward_consistency, 0.5, 1.2)
                base_quality *= consistency_factor
            
            return np.clip(base_quality, 0.1, 0.98)
        except:
            return 0.7
    
    def get_preference_reward(self, obs, action, buffer, episode_idx):
        """获取偏好奖励（优化版：集成智能缓存、频率控制和相似性匹配）
        
        Args:
            obs: 当前观测
            action: 当前动作
            buffer: 经验缓冲区
            episode_idx: 当前episode索引
            
        Returns:
            float: 偏好奖励值
        """
        # 检查偏好学习是否启用
        if not self.is_enabled():
            return 0.0
            
        start_time = time.time()
        
        # 早期episode或模型未加载时返回0
        if episode_idx < self.preference_start_episode:
            return 0.0
        
        if self.reward_model is None:
            # 每100个episode打印一次模型状态
            if episode_idx % 100 == 0:
                print(f"[PreferenceModule] Episode {episode_idx}: 偏好奖励模型未加载")
            return 0.0
        
        self._total_cache_queries += 1
        
        # 智能缓存：生成量化缓存键
        try:
            cache_key = self._compute_quantized_cache_key(
                np.array(obs), np.array(action), episode_idx
            )
            
            # 精确缓存匹配
            if cache_key in self._data_cache:
                self._cache_hits += 1
                self._exact_hits += 1
                return self._data_cache[cache_key]
            
            # 相似性匹配（仅在精确匹配失败且查询频率不高时进行）
            if len(self._data_cache) > 0 and self._total_cache_queries % 5 == 0:
                obs_array = np.array(obs)
                action_array = np.array(action)
                
                # 只搜索最近的50个缓存项
                for cached_key in list(self._data_cache.keys())[-50:]:
                    if cached_key in self._cache_metadata:
                        cached_obs = self._cache_metadata[cached_key]['obs']
                        cached_action = self._cache_metadata[cached_key]['action']
                        similarity = self._compute_similarity(
                            obs_array, action_array, cached_obs, cached_action
                        )
                        
                        if similarity > self._similarity_threshold:
                            self._cache_hits += 1
                            self._similarity_hits += 1
                            return self._data_cache[cached_key]
            
            # 定期清理过期缓存
            if self._debug_counter % 100 == 0:
                self._clear_expired_cache()
        except Exception as e:
            print(f"[PreferenceModule] 缓存处理失败: {e}")
            cache_key = None
        
        try:
            # 检查buffer是否为None
            if buffer is None:
                if not hasattr(self, '_buffer_none_warning_count'):
                    self._buffer_none_warning_count = 0
                    self._last_buffer_none_warning_time = 0
                
                current_time = time.time()
                # 每10秒最多输出一次警告，或者前5次都输出
                if (self._buffer_none_warning_count < 5 or 
                    current_time - self._last_buffer_none_warning_time > 10.0):
                    # print(f"[WARNING] 偏好奖励计算失败 (第{self._buffer_none_warning_count + 1}次): buffer为None")
                    self._last_buffer_none_warning_time = current_time
                
                self._buffer_none_warning_count += 1
                return 0.0
            
            # 获取历史数据
            history_obs, history_act = buffer.get_history(self.history_length)
            if history_obs is None or history_act is None or len(history_obs) == 0:
                # 调试信息
                if episode_idx % 100 == 0:
                    print(f"[PreferenceModule] Episode {episode_idx}: 历史数据为空，历史缓冲区大小: {len(buffer.history_buffer)}")
                return 0.0
            device = self.agent.device
            
            # 转换历史数据为tensor
            if history_obs and history_act:
                # 确保每个obs和action都是正确的维度
                processed_obs = []
                processed_act = []
                
                for obs_item in history_obs:
                    obs_tensor = torch.as_tensor(obs_item, dtype=torch.float32, device=device)
                    # 确保obs是1维的
                    if obs_tensor.dim() == 0:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    elif obs_tensor.dim() > 1:
                        obs_tensor = obs_tensor.flatten()
                    processed_obs.append(obs_tensor)
                
                for act_item in history_act:
                    act_tensor = torch.as_tensor(act_item, dtype=torch.float32, device=device)
                    # 确保action是1维的
                    if act_tensor.dim() == 0:
                        act_tensor = act_tensor.unsqueeze(0)
                    elif act_tensor.dim() > 1:
                        act_tensor = act_tensor.flatten()
                    processed_act.append(act_tensor)
                
                history_obs_tensor = torch.stack(processed_obs)
                history_act_tensor = torch.stack(processed_act)
            else:
                # 如果历史为空，创建空的Tensor
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
                
                # 确保obs和action是1维的
                if obs_tensor.dim() == 0:
                    obs_tensor = obs_tensor.unsqueeze(0)
                elif obs_tensor.dim() > 1:
                    obs_tensor = obs_tensor.flatten()
                    
                if action_tensor.dim() == 0:
                    action_tensor = action_tensor.unsqueeze(0)
                elif action_tensor.dim() > 1:
                    action_tensor = action_tensor.flatten()
                
                history_obs_tensor = torch.empty(0, obs_tensor.shape[0], dtype=torch.float32, device=device)
                history_act_tensor = torch.empty(0, action_tensor.shape[0], dtype=torch.float32, device=device)
            
            # 转换当前观测和动作
            current_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            current_act_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
            
            # 确保当前obs和action是1维的
            if current_obs_tensor.dim() == 0:
                current_obs_tensor = current_obs_tensor.unsqueeze(0)
            elif current_obs_tensor.dim() > 1:
                current_obs_tensor = current_obs_tensor.flatten()
                
            if current_act_tensor.dim() == 0:
                current_act_tensor = current_act_tensor.unsqueeze(0)
            elif current_act_tensor.dim() > 1:
                current_act_tensor = current_act_tensor.flatten()
            
            current_obs_tensor = current_obs_tensor.unsqueeze(0)
            current_act_tensor = current_act_tensor.unsqueeze(0)
            
            # 拼接历史和当前数据
            full_obs_tensor = torch.cat([history_obs_tensor, current_obs_tensor], dim=0).unsqueeze(0)
            full_act_tensor = torch.cat([history_act_tensor, current_act_tensor], dim=0).unsqueeze(0)
            
            # 获取偏好奖励
            with torch.no_grad():
                pref_reward = self.reward_model(full_obs_tensor, full_act_tensor)
                raw_pref_reward = pref_reward.item()
            
            # 添加调试信息：记录原始模型输出
            if not hasattr(self, '_debug_output_count'):
                self._debug_output_count = 0
            
            self._debug_output_count += 1
            # # 每100次输出一次调试信息
            # if self._debug_output_count % 100 == 0:
            #     print(f"[PreferenceModule] 调试信息 - Episode {episode_idx}: 原始模型输出={raw_pref_reward:.6f}")
            
            # 数值稳定性检查（减少冗余日志输出）
            if not np.isfinite(raw_pref_reward):
                # 使用计数器限制日志频率，避免大量重复输出
                if not hasattr(self, '_nan_warning_count'):
                    self._nan_warning_count = 0
                    self._last_nan_warning_time = 0
                
                current_time = time.time()
                # 每10秒最多输出一次警告，或者前5次都输出
                if (self._nan_warning_count < 5 or 
                    current_time - self._last_nan_warning_time > 10.0):
                    print(f"[PreferenceModule] 检测到非有限偏好奖励: {raw_pref_reward}，设为0 (已发生{self._nan_warning_count + 1}次)")
                    self._last_nan_warning_time = current_time
                
                self._nan_warning_count += 1
                pref_reward = 0.0
            else:
                pref_reward = raw_pref_reward
            
            # 更新统计信息
            self._update_pref_reward_stats(pref_reward)
            
            # 缓存结果（包含元数据）
            if cache_key is not None:
                # 如果缓存满了，删除最旧的条目
                if len(self._data_cache) >= self._cache_max_size:
                    oldest_key = next(iter(self._data_cache))
                    self._data_cache.pop(oldest_key)
                    self._cache_metadata.pop(oldest_key, None)
                    self._cache_timestamps.pop(oldest_key, None)
                
                self._data_cache[cache_key] = pref_reward
                self._cache_metadata[cache_key] = {
                    'obs': np.array(obs).copy(),
                    'action': np.array(action).copy(),
                    'timestamp': time.time()
                }
                self._cache_timestamps[cache_key] = time.time()
            
            # 记录性能指标
            computation_time = time.time() - start_time
            self._computation_times.append(computation_time)
            
            # 计算增强的学习质量
            # 计算奖励方差（如果有足够历史数据）
            reward_variance = None
            trajectory_length = None
            
            if len(self.pref_reward_buffer) > 3:
                recent_rewards = list(self.pref_reward_buffer)[-10:]
                reward_variance = np.var(recent_rewards)
            
            # 估算轨迹长度（基于buffer信息）
            if buffer is not None and hasattr(buffer, '_size'):
                trajectory_length = min(buffer._size, 200)  # 限制最大长度
            
            # 估算学习质量并自适应调整频率（更频繁的调整）
            if len(self._computation_times) % 15 == 0:  # 从20改为15，更频繁调整
                learning_quality = self._estimate_learning_quality(
                    pref_reward, reward_variance, trajectory_length
                )
                self._learning_qualities.append(learning_quality)
                self._adapt_frequency()
            
            # 渐进优化：应用信号增强处理
            enhanced_reward = self._apply_progressive_signal_enhancement(pref_reward, episode_idx)
            
            # 更新偏好学习统计
            self._preference_stats['total_preferences'] += 1
            self._preference_stats['preference_rewards'].append(enhanced_reward)
            
            return enhanced_reward
            
        except Exception as e:
            print(f"[PreferenceModule] 偏好奖励计算失败: {e}")
            return 0.0
    
    def _update_pref_reward_stats(self, pref_reward):
        """更新偏好奖励统计信息"""
        # 统计所有奖励值
        self.pref_reward_buffer.append(float(pref_reward))
        
        # 更新统计信息
        if len(self.pref_reward_buffer) >= 10:
            self.pref_reward_mean = np.mean(self.pref_reward_buffer)
            raw_std = np.std(self.pref_reward_buffer)
            # 确保标准差有合理的最小值
            self.pref_reward_std = max(raw_std, 0.1)
            
            # 减少调试信息频率
            if len(self.pref_reward_buffer) % self._stats_print_interval == 0:
                print(f"[PreferenceModule] 偏好奖励统计: 均值={self.pref_reward_mean:.3f}, 标准差={self.pref_reward_std:.3f}, 样本数={len(self.pref_reward_buffer)}")
    
    def _apply_progressive_signal_enhancement(self, pref_reward, episode_idx):
        """渐进式偏好信号增强处理
        
        Args:
            pref_reward: 原始偏好奖励
            episode_idx: 当前回合索引
            
        Returns:
            float: 增强处理后的偏好奖励
        """
        try:
            # 数值稳定性保护
            processed_reward = np.clip(pref_reward, -10.0, 10.0)
            
            # 获取增强配置
            hybrid_config = getattr(self.cfg, 'hybrid_value_estimation', {})
            enable_enhancement = hybrid_config.get('enable_signal_enhancement', True)
            
            if not enable_enhancement:
                return float(processed_reward)
            
            # 1. 训练阶段自适应增强
            stage_factor = self._compute_stage_enhancement_factor(episode_idx, hybrid_config)
            
            # 2. 信号质量评估和过滤
            quality_factor = self._compute_signal_quality_factor(processed_reward)
            
            # 3. 动态信号放大
            amplification_factor = self._compute_amplification_factor(processed_reward, episode_idx)
            
            # 4. 噪声抑制
            denoised_reward = self._apply_noise_suppression(processed_reward)
            
            # 5. 渐进式信号强化
            enhanced_reward = denoised_reward * stage_factor * quality_factor * amplification_factor
            
            # 6. 最终数值稳定性保护和范围限制
            enhanced_reward = np.clip(enhanced_reward, -5.0, 5.0)
            
            # 记录增强统计
            if not hasattr(self, '_enhancement_stats'):
                self._enhancement_stats = {
                    'original_rewards': deque(maxlen=100),
                    'enhanced_rewards': deque(maxlen=100),
                    'enhancement_ratios': deque(maxlen=100)
                }
            
            self._enhancement_stats['original_rewards'].append(abs(processed_reward))
            self._enhancement_stats['enhanced_rewards'].append(abs(enhanced_reward))
            if abs(processed_reward) > 1e-6:
                ratio = abs(enhanced_reward) / abs(processed_reward)
                self._enhancement_stats['enhancement_ratios'].append(ratio)
            
            # 定期打印增强统计
            if episode_idx % 200 == 0 and len(self._enhancement_stats['enhancement_ratios']) > 0:
                avg_ratio = np.mean(self._enhancement_stats['enhancement_ratios'])
                print(f"[PreferenceModule] 信号增强统计 (Episode {episode_idx}): 平均增强比例={avg_ratio:.3f}")
            
            return float(enhanced_reward)
            
        except Exception as e:
            print(f"[PreferenceModule] 偏好信号增强失败: {e}")
            return float(processed_reward if 'processed_reward' in locals() else pref_reward)
    
    def _compute_stage_enhancement_factor(self, episode_idx, hybrid_config):
        """计算训练阶段增强因子"""
        early_stage = hybrid_config.get('early_stage_episodes', 500)
        mid_stage = hybrid_config.get('mid_stage_episodes', 1500)
        
        if episode_idx < early_stage:
            # 早期阶段：保守增强，避免过度放大噪声
            return 0.8
        elif episode_idx < mid_stage:
            # 中期阶段：逐渐增强
            progress = (episode_idx - early_stage) / (mid_stage - early_stage)
            return 0.8 + 0.4 * progress  # 从0.8增长到1.2
        else:
            # 后期阶段：积极增强偏好信号
            return 1.3
    
    def _compute_signal_quality_factor(self, reward):
        """计算信号质量因子"""
        # 基于信号强度和历史统计评估质量
        signal_strength = abs(reward)
        
        # 弱信号抑制
        if signal_strength < 0.01:
            return 0.5  # 抑制噪声
        
        # 中等信号适度增强
        elif signal_strength < 0.1:
            return 1.0 + 0.3 * (signal_strength / 0.1)  # 1.0到1.3
        
        # 强信号保持或轻微增强
        else:
            return min(1.5, 1.0 + 0.2 * np.log10(signal_strength + 1))
    
    def _compute_amplification_factor(self, reward, episode_idx):
        """计算动态放大因子"""
        # 基于奖励历史和当前训练状态
        base_factor = 1.0
        
        # 如果有历史数据，基于方差调整
        if hasattr(self, 'pref_reward_buffer') and len(self.pref_reward_buffer) > 10:
            recent_rewards = list(self.pref_reward_buffer)[-20:]
            reward_std = np.std(recent_rewards)
            reward_mean = abs(np.mean(recent_rewards))
            
            # 如果信号稳定且有意义，增加放大
            if reward_std < 0.3 and reward_mean > 0.02:
                base_factor *= 1.2
            # 如果信号不稳定，减少放大
            elif reward_std > 0.8:
                base_factor *= 0.8
        
        # 基于当前奖励强度的自适应放大
        signal_strength = abs(reward)
        if signal_strength > 0.05:
            # 对强信号适度放大
            strength_factor = 1.0 + 0.3 * np.tanh(signal_strength * 5)
            base_factor *= strength_factor
        
        return np.clip(base_factor, 0.5, 2.0)
    
    def _apply_noise_suppression(self, reward):
        """应用噪声抑制"""
        # 使用移动平均进行噪声抑制
        if not hasattr(self, '_reward_smoothing_buffer'):
            self._reward_smoothing_buffer = deque(maxlen=5)
        
        self._reward_smoothing_buffer.append(reward)
        
        if len(self._reward_smoothing_buffer) >= 3:
            # 使用加权移动平均，给最近的值更高权重
            weights = np.array([0.1, 0.2, 0.3, 0.4])[-len(self._reward_smoothing_buffer):]
            weights = weights / weights.sum()
            smoothed = np.average(list(self._reward_smoothing_buffer), weights=weights)
            
            # 如果当前值与平滑值差异过大，可能是噪声
            if abs(reward - smoothed) > 0.5 and abs(reward) < 0.1:
                return smoothed * 0.7 + reward * 0.3  # 更多依赖平滑值
            else:
                return smoothed * 0.3 + reward * 0.7  # 更多依赖当前值
        else:
            return reward
    

    
    def get_combined_reward(self, env_reward, obs, action, buffer, episode_idx):
        """获取融合奖励（实现真正的奖励融合）
        
        Args:
            env_reward: 环境奖励
            obs: 当前观测
            action: 当前动作
            buffer: 经验缓冲区
            episode_idx: 当前episode索引
            
        Returns:
            tuple: (融合奖励, 偏好奖励)
        """
        if episode_idx < self.preference_start_episode:
            # 即使在偏好学习开始前也要更新环境奖励统计
            self._update_env_reward_stats(env_reward)
            return env_reward, 0.0
        
        # 获取原始偏好奖励
        raw_pref_reward = self.get_preference_reward(obs, action, buffer, episode_idx)
        
        # 计算自适应权重
        env_weight, pref_weight = self._compute_adaptive_weights(episode_idx, raw_pref_reward, env_reward)
        
        # 执行奖励融合
        fused_reward = env_weight * env_reward + pref_weight * raw_pref_reward
        
        # 记录融合信息用于调试
        if episode_idx % 500 == 0:
            print(f"[PreferenceModule] Episode {episode_idx}: 奖励融合")
            print(f"  环境奖励: {env_reward:.4f}, 偏好奖励: {raw_pref_reward:.4f}")
            print(f"  权重 - 环境: {env_weight:.3f}, 偏好: {pref_weight:.3f}")
            print(f"  融合奖励: {fused_reward:.4f}")
        
        # 更新融合统计
        self._update_fusion_stats(env_reward, raw_pref_reward, fused_reward, env_weight, pref_weight)
        
        # 返回融合奖励和原始偏好奖励
        return fused_reward, raw_pref_reward
    
    def _compute_adaptive_weights(self, episode_idx, pref_reward, env_reward):
        """计算自适应权重
        
        Args:
            episode_idx: 当前episode索引
            pref_reward: 偏好奖励
            env_reward: 环境奖励
            
        Returns:
            tuple: (环境权重, 偏好权重)
        """
        # 从配置中获取混合价值估计配置
        hybrid_config = getattr(self.cfg, 'hybrid_value_estimation', {})
        
        # 从配置中获取基础权重
        base_env_weight = hybrid_config.get('base_mpc_weight', 0.6)
        base_pref_weight = hybrid_config.get('base_preference_weight', 0.4)
        
        # 获取自适应参数
        enable_adaptive = hybrid_config.get('enable_adaptive_weighting', True)
        if not enable_adaptive:
            return base_env_weight, base_pref_weight
        
        # 获取敏感度参数
        quality_sensitivity = hybrid_config.get('quality_sensitivity', 0.3)
        uncertainty_sensitivity = hybrid_config.get('uncertainty_sensitivity', 0.2)
        stage_sensitivity = hybrid_config.get('stage_sensitivity', 0.15)
        confidence_sensitivity = hybrid_config.get('confidence_sensitivity', 0.25)
        trend_sensitivity = hybrid_config.get('trend_sensitivity', 0.15)
        
        # 训练阶段调整
        early_stage = hybrid_config.get('early_stage_episodes', 500)
        mid_stage = hybrid_config.get('mid_stage_episodes', 1500)
        
        stage_adjustment = 0.0
        if episode_idx < early_stage:
            # 早期阶段：降低偏好权重，让智能体先学习基础行为
            stage_adjustment = -0.1 * stage_sensitivity
        elif episode_idx > mid_stage:
            # 后期阶段：增加偏好权重，强化偏好学习
            stage_adjustment = 0.1 * stage_sensitivity
        
        # 偏好信号质量调整
        quality_adjustment = 0.0
        if hasattr(self, 'pref_reward_buffer') and len(self.pref_reward_buffer) > 10:
            recent_rewards = list(self.pref_reward_buffer)[-10:]
            reward_std = np.std(recent_rewards)
            reward_mean = np.mean(recent_rewards)
            
            # 如果偏好信号稳定且有意义，增加权重
            if abs(reward_mean) > 0.01 and reward_std < 0.5:
                quality_adjustment = 0.2 * quality_sensitivity  # 提高质量调整幅度
            elif abs(reward_mean) < 0.005:  # 偏好信号微弱
                quality_adjustment = -0.1 * quality_sensitivity
        
        # 置信度调整（基于奖励一致性和强度）
        confidence_adjustment = 0.0
        if abs(pref_reward) > 0.01:  # 有明显偏好信号
            # 根据偏好奖励强度调整
            signal_strength = min(abs(pref_reward), 1.0)  # 限制在[0,1]范围
            confidence_adjustment = signal_strength * 0.15 * confidence_sensitivity
        
        # 奖励趋势调整
        trend_adjustment = 0.0
        if hasattr(self, 'env_reward_buffer') and len(self.env_reward_buffer) > 5:
            recent_env_rewards = list(self.env_reward_buffer)[-5:]
            if len(recent_env_rewards) >= 3:
                # 计算奖励趋势（简单的线性回归斜率）
                x = np.arange(len(recent_env_rewards))
                y = np.array(recent_env_rewards)
                if np.std(y) > 1e-6:  # 避免除零
                    slope = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                    trend_adjustment = np.tanh(slope) * 0.1 * trend_sensitivity
        
        # 计算最终权重
        pref_weight_adjustment = (
            stage_adjustment + 
            quality_adjustment + 
            confidence_adjustment + 
            trend_adjustment
        )
        final_pref_weight = base_pref_weight + pref_weight_adjustment
        
        # 应用权重范围限制
        min_pref_weight = hybrid_config.get('min_preference_weight', 0.2)
        max_pref_weight = hybrid_config.get('max_preference_weight', 0.7)
        final_pref_weight = np.clip(final_pref_weight, min_pref_weight, max_pref_weight)
        
        # 环境权重 = 1 - 偏好权重
        final_env_weight = 1.0 - final_pref_weight
        
        return final_env_weight, final_pref_weight
    
    def _update_fusion_stats(self, env_reward, pref_reward, fused_reward, env_weight, pref_weight):
        """更新融合统计信息
        
        Args:
            env_reward: 环境奖励
            pref_reward: 偏好奖励
            fused_reward: 融合奖励
            env_weight: 环境权重
            pref_weight: 偏好权重
        """
        # 初始化融合统计
        if not hasattr(self, '_fusion_stats'):
            self._fusion_stats = {
                'env_weights': deque(maxlen=1000),
                'pref_weights': deque(maxlen=1000),
                'env_rewards': deque(maxlen=1000),
                'pref_rewards': deque(maxlen=1000),
                'fused_rewards': deque(maxlen=1000),
                'fusion_count': 0
            }
        
        # 更新统计
        self._fusion_stats['env_weights'].append(env_weight)
        self._fusion_stats['pref_weights'].append(pref_weight)
        self._fusion_stats['env_rewards'].append(env_reward)
        self._fusion_stats['pref_rewards'].append(pref_reward)
        self._fusion_stats['fused_rewards'].append(fused_reward)
        self._fusion_stats['fusion_count'] += 1
        
        # 每1000次融合输出统计信息
        if self._fusion_stats['fusion_count'] % 1000 == 0:
            avg_env_weight = np.mean(self._fusion_stats['env_weights'])
            avg_pref_weight = np.mean(self._fusion_stats['pref_weights'])
            avg_env_reward = np.mean(self._fusion_stats['env_rewards'])
            avg_pref_reward = np.mean(self._fusion_stats['pref_rewards'])
            avg_fused_reward = np.mean(self._fusion_stats['fused_rewards'])
            
            print(f"\n[融合统计] 最近1000次融合:")
            print(f"  平均权重 - 环境: {avg_env_weight:.3f}, 偏好: {avg_pref_weight:.3f}")
            print(f"  平均奖励 - 环境: {avg_env_reward:.4f}, 偏好: {avg_pref_reward:.4f}, 融合: {avg_fused_reward:.4f}")

    def print_preference_status(self, episode_idx):
        """打印偏好模块状态信息
        
        Args:
            episode_idx: 当前episode索引
        """
        print(f"\n[偏好模块状态] Episode {episode_idx}:")
        print(f"  偏好奖励缓冲区长度: {len(self.pref_reward_buffer)}")
        print(f"  环境奖励缓冲区长度: {len(self.env_reward_buffer)}")
        
        if len(self.pref_reward_buffer) >= 10:
            recent_pref_rewards = list(self.pref_reward_buffer)[-10:]
            print(f"  最近偏好奖励: 均值={np.mean(recent_pref_rewards):.4f}, 标准差={np.std(recent_pref_rewards):.4f}")
        
        if len(self.env_reward_buffer) >= 10:
            recent_env_rewards = list(self.env_reward_buffer)[-10:]
            print(f"  最近环境奖励: 均值={np.mean(recent_env_rewards):.4f}, 标准差={np.std(recent_env_rewards):.4f}")
        
        print(f"  缓存统计: 命中率={self._cache_hits}/{self._total_cache_queries} ({100*self._cache_hits/max(1,self._total_cache_queries):.1f}%)")
        print(f"  偏好对缓存: 规则={len(self._preference_pairs_cache['rule_based'])}, 采集={len(self._preference_pairs_cache['collected'])}")
        print()

    
    def update_preference_model(self, episode_idx):
        """更新偏好模型
        
        Args:
            episode_idx: 当前episode索引
        """
        # 检查偏好学习是否启用
        if not self.is_enabled():
            return
            
        if not (self.preference_engine and 
                episode_idx >= self.preference_start_episode and
                episode_idx % self.preference_update_interval_episode == 0):
            return
        
        try:
            print(f"[PreferenceModule] Episode {episode_idx}: 开始偏好模型更新")
            
            # 检查缓存数量，如果达到规定数量的10倍就清空
            self._check_and_clear_cache_if_needed()
            
            # 收集轨迹数据
            trajectories = self.preference_engine.collect_trajectories()
            print(f"[PreferenceModule] 收集到 {len(trajectories)} 条轨迹")
            
            # 持续生成偏好对直到达到指定数量
            rule_based_pairs = self._generate_sufficient_rule_pairs(trajectories)
            collected_pairs = self._generate_sufficient_collected_pairs(trajectories)
            
            print(f"[PreferenceModule] 生成 {len(rule_based_pairs)} 个规则偏好对")
            print(f"[PreferenceModule] 生成 {len(collected_pairs)} 个采集偏好对")
            
            # 将新生成的偏好对添加到缓存中（不清除之前的）
            if rule_based_pairs or collected_pairs:
                # 将偏好对添加到持久缓存中
                self._add_preference_pairs_to_cache(rule_based_pairs, collected_pairs)
                
                print(f"[PreferenceModule] 开始训练偏好奖励模型")
                final_acc_rule, final_acc_collected = train_reward_model_once(preference_module=self)
                print(f"[PreferenceModule] 训练完成 - 规则准确率: {final_acc_rule:.3f}, 采集准确率: {final_acc_collected:.3f}")
                self._load_reward_model()
                
                print(f"[PreferenceModule] 偏好模型更新完成，episode {episode_idx}")
                print(f"[PreferenceModule] 当前缓存数量: 规则={len(self._preference_pairs_cache['rule_based'])}, 采集={len(self._preference_pairs_cache['collected'])}")
            else:
                print(f"[PreferenceModule] 没有生成偏好对，跳过模型训练")
                
        except Exception as e:
            print(f"[PreferenceModule] 偏好模型更新失败: {e}")
            traceback.print_exc()
    
    def _store_preference_pairs_in_cache(self, rule_based_pairs, collected_pairs):
        """将偏好对存储到临时缓存中（保留原方法以兼容性）
        
        Args:
            rule_based_pairs: 规则偏好对列表
            collected_pairs: 采集偏好对列表
        """
        self._preference_pairs_cache['rule_based'] = rule_based_pairs.copy() if rule_based_pairs else []
        self._preference_pairs_cache['collected'] = collected_pairs.copy() if collected_pairs else []
        
        print(f"[PreferenceModule] 偏好对已存储到缓存: 规则={len(self._preference_pairs_cache['rule_based'])}, 采集={len(self._preference_pairs_cache['collected'])}")
    
    def _add_preference_pairs_to_cache(self, rule_based_pairs, collected_pairs):
        """将偏好对添加到持久缓存中
        
        Args:
            rule_based_pairs: 规则偏好对列表
            collected_pairs: 采集偏好对列表
        """
        if rule_based_pairs:
            self._preference_pairs_cache['rule_based'].extend(rule_based_pairs)
        if collected_pairs:
            self._preference_pairs_cache['collected'].extend(collected_pairs)
        
        print(f"[PreferenceModule] 偏好对已添加到缓存: 新增规则={len(rule_based_pairs) if rule_based_pairs else 0}, 新增采集={len(collected_pairs) if collected_pairs else 0}")
        print(f"[PreferenceModule] 缓存总数: 规则={len(self._preference_pairs_cache['rule_based'])}, 采集={len(self._preference_pairs_cache['collected'])}")
    
    def _generate_sufficient_rule_pairs(self, trajectories):
        """持续生成规则偏好对直到达到指定数量
        
        Args:
            trajectories: 轨迹数据列表
            
        Returns:
            list: 生成的规则偏好对列表
        """
        target_count = getattr(self.cfg, "preference_num_trajs_rules", 20)
        rule_based_pairs = []
        max_total_attempts = target_count * 4  # 增加最大总尝试次数
        attempt_count = 0
        
        print(f"[PreferenceModule] 开始生成规则偏好对，目标数量: {target_count}")
        
        while len(rule_based_pairs) < target_count and attempt_count < max_total_attempts:
            # 每次尝试生成一批偏好对
            batch_pairs = self.preference_engine._generate_rule_based_pairs_optimized(trajectories, None)
            
            # 去重并添加到结果中
            for pair in batch_pairs:
                if len(rule_based_pairs) >= target_count:
                    break
                # 简单的去重检查
                pair_exists = any(
                    (id(pair[0]) == id(existing[0]) and id(pair[1]) == id(existing[1])) or
                    (id(pair[0]) == id(existing[1]) and id(pair[1]) == id(existing[0]))
                    for existing in rule_based_pairs
                )
                if not pair_exists:
                    rule_based_pairs.append(pair)
            
            attempt_count += 1
            if len(rule_based_pairs) < target_count:
                print(f"[PreferenceModule] 规则偏好对生成进度: {len(rule_based_pairs)}/{target_count}, 尝试次数: {attempt_count}")
        
        if len(rule_based_pairs) < target_count:
            print(f"[WARNING] 规则偏好对生成不足({len(rule_based_pairs)}/{target_count})，已尝试{attempt_count}次")
        else:
            print(f"[PreferenceModule] 成功生成 {len(rule_based_pairs)} 个规则偏好对")
        
        return rule_based_pairs
    
    def _generate_sufficient_collected_pairs(self, trajectories):
        """持续生成采集偏好对直到达到指定数量
        
        Args:
            trajectories: 轨迹数据列表
            
        Returns:
            list: 生成的采集偏好对列表
        """
        target_count = getattr(self.cfg, "preference_num_trajs_collect", 15)
        collected_pairs = []
        max_total_attempts = target_count * 4  # 增加最大总尝试次数
        attempt_count = 0
        
        print(f"[PreferenceModule] 开始生成采集偏好对，目标数量: {target_count}")
        
        while len(collected_pairs) < target_count and attempt_count < max_total_attempts:
            # 每次尝试生成一批偏好对
            batch_pairs = self.preference_engine._generate_collected_pairs_optimized(trajectories, None)
            
            # 去重并添加到结果中
            for pair in batch_pairs:
                if len(collected_pairs) >= target_count:
                    break
                # 简单的去重检查
                pair_exists = any(
                    (id(pair[0]) == id(existing[0]) and id(pair[1]) == id(existing[1])) or
                    (id(pair[0]) == id(existing[1]) and id(pair[1]) == id(existing[0]))
                    for existing in collected_pairs
                )
                if not pair_exists:
                    collected_pairs.append(pair)
            
            attempt_count += 1
            if len(collected_pairs) < target_count:
                print(f"[PreferenceModule] 采集偏好对生成进度: {len(collected_pairs)}/{target_count}, 尝试次数: {attempt_count}")
        
        if len(collected_pairs) < target_count:
            print(f"[WARNING] 采集偏好对生成不足({len(collected_pairs)}/{target_count})，已尝试{attempt_count}次")
        else:
            print(f"[PreferenceModule] 成功生成 {len(collected_pairs)} 个采集偏好对")
        
        return collected_pairs
    
    def _check_and_clear_cache_if_needed(self):
        """检查缓存数量，如果达到规定数量的10倍就清空全部缓存"""
        # 从配置获取规定数量
        preference_num_trajs_rules = getattr(self.cfg, "preference_num_trajs_rules", 20)
        preference_num_trajs_collect = getattr(self.cfg, "preference_num_trajs_collect", 15)
        
        # 计算阈值（规定数量的10倍）
        rule_threshold = preference_num_trajs_rules * 3
        collect_threshold = preference_num_trajs_collect * 3
        
        current_rule_count = len(self._preference_pairs_cache['rule_based'])
        current_collect_count = len(self._preference_pairs_cache['collected'])
        
        # 检查是否需要清空缓存
        should_clear = (current_rule_count >= rule_threshold or 
                       current_collect_count >= collect_threshold)
        
        if should_clear:
            print(f"[PreferenceModule] 缓存数量达到阈值，清空缓存")
            print(f"[PreferenceModule] 当前数量: 规则={current_rule_count}/{rule_threshold}, 采集={current_collect_count}/{collect_threshold}")
            self._clear_preference_pairs_cache()
        else:
            print(f"[PreferenceModule] 缓存数量检查: 规则={current_rule_count}/{rule_threshold}, 采集={current_collect_count}/{collect_threshold}")
    
    def _clear_preference_pairs_cache(self):
        """清除偏好对缓存"""
        self._preference_pairs_cache['rule_based'].clear()
        self._preference_pairs_cache['collected'].clear()
        print(f"[PreferenceModule] 偏好对缓存已清除")
    
    def _cache_reward_model(self, model, timestamp, model_path):
        """将奖励模型保存到缓存中
        
        Args:
            model: 要缓存的模型
            timestamp: 模型文件的时间戳
            model_path: 模型文件路径
        """
        try:
            # 深拷贝模型以避免引用问题
            import copy
            self._reward_model_cache = copy.deepcopy(model)
            self._cached_model_timestamp = timestamp
            self._cached_model_path = model_path
            print(f"[PreferenceModule] 奖励模型已缓存: {model_path}")
        except Exception as e:
            print(f"[PreferenceModule] 缓存奖励模型失败: {e}")
            self._clear_model_cache()
    
    def _clear_model_cache(self):
        """清除奖励模型缓存"""
        self._reward_model_cache = None
        self._cached_model_timestamp = None
        self._cached_model_path = None
        print(f"[PreferenceModule] 奖励模型缓存已清除")
    
    def get_cached_preference_pairs(self):
        """获取缓存中的偏好对（供训练使用）
        
        Returns:
            tuple: (rule_based_pairs, collected_pairs)
        """
        return self._preference_pairs_cache['rule_based'], self._preference_pairs_cache['collected']
    
    def is_enabled(self):
        """检查偏好模块是否启用
        
        Returns:
            bool: 是否启用偏好模块
        """
        return self.preference_engine is not None
    
    def _update_env_reward_stats(self, env_reward):
        """更新环境奖励统计信息"""
        self.env_reward_buffer.append(float(env_reward))
        self.env_reward_short_buffer.append(float(env_reward))
        
        # 更新长期统计
        if len(self.env_reward_buffer) >= 10:
            self.env_reward_mean = np.mean(self.env_reward_buffer)
            self.env_reward_std = max(np.std(self.env_reward_buffer), 0.1)
        
        # 更新短期统计
        if len(self.env_reward_short_buffer) >= 10:
            self.env_reward_short_mean = np.mean(self.env_reward_short_buffer)
            self.env_reward_short_std = max(np.std(self.env_reward_short_buffer), 0.1)
    

    

    
    def print_optimization_status(self, episode_idx):
        """打印优化状态信息"""
        if episode_idx % 500 == 0 and self._total_cache_queries > 0:
            stats = self.get_cache_statistics()
            print(f"[PreferenceModule] Episode {episode_idx} 优化状态:")
            print(f"  缓存命中率: {stats['hit_rate']:.2%} (精确: {stats['exact_hit_rate']:.2%}, 相似: {stats['similarity_hit_rate']:.2%})")
            print(f"  缓存利用率: {stats['cache_utilization']:.2%} ({stats['cache_size']}/{self._cache_max_size})")
            print(f"  当前计算频率: 每{stats['current_frequency']}步")
            print(f"  平均计算时间: {stats['avg_computation_time']*1000:.1f}ms")

            if hasattr(self, 'env_reward_mean'):
                print(f"  环境奖励统计: 均值={self.env_reward_mean:.3f}, 标准差={self.env_reward_std:.3f}")
            
            # 打印偏好学习状态
            self.print_preference_status(episode_idx)
