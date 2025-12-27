#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好数据引擎 (Preference Data Engine)

兼容性模块：为现有的 grpo 模块提供向后兼容的接口。
实际功能由 preference_labeling_engine.py 提供。

主要功能：
- 提供向后兼容的API接口
- 轨迹数据包装和处理
- 偏好标签生成的兼容性封装
- 轨迹质量评估

作者：AI Assistant
日期：2025-01-11
版本：2.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import warnings
import time

# 导入实际的实现
from .preference_labeling_engine import (
    PreferenceLabelingEngine, 
    LabelType, 
    create_preference_labeling_engine
)
from .trajectory_metrics import TrajectoryMetrics

# 导入已弃用的规则注册表（仅用于兼容性）
try:
    from .rule_registry import RULE_REGISTRY
except ImportError:
    # 如果rule_registry不可用，使用空字典
    RULE_REGISTRY = {}

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 版本信息
__version__ = "2.0.0"
__author__ = "AI Assistant"
__email__ = "assistant@ai.com"

class TrajectoryWrapper:
    """轨迹包装器 - 兼容性类
    
    为现有代码提供统一的轨迹数据接口，支持多种数据格式的轨迹。
    
    Attributes:
        data: 原始轨迹数据字典
        states: 状态序列 (obs的别名)
        actions: 动作序列 (action的别名)
        rewards: 奖励序列 (reward的别名)
        dones: 结束标志序列 (done的别名)
        length: 轨迹长度
        total_reward: 总奖励
    """
    
    def __init__(self, data: Dict[str, Any]):
        """初始化轨迹包装器
        
        Args:
            data: 轨迹数据字典，包含 obs, action, reward, done 等键
            
        Raises:
            ValueError: 当数据格式不正确时
        """
        if not isinstance(data, dict):
            raise ValueError("轨迹数据必须是字典格式")
            
        self.data = data.copy()  # 创建副本避免修改原数据
        
        # 设置标准属性
        self.states = np.array(data.get('obs', [])) if data.get('obs') is not None else np.array([])
        self.actions = np.array(data.get('action', [])) if data.get('action') is not None else np.array([])
        self.rewards = np.array(data.get('reward', [])) if data.get('reward') is not None else np.array([])
        self.dones = np.array(data.get('done', [])) if data.get('done') is not None else np.array([])
        
        # 兼容性属性
        self.obs = self.states
        self.action = self.actions
        self.reward = self.rewards
        self.done = self.dones
        
        # 计算轨迹长度
        self.length = len(self.states) if len(self.states) > 0 else 0
        
        # 计算总奖励
        self.total_reward = float(np.sum(self.rewards)) if len(self.rewards) > 0 else 0.0
        
        # 验证数据一致性
        self._validate_data()
    
    def _validate_data(self) -> None:
        """验证轨迹数据的一致性
        
        Raises:
            ValueError: 当数据不一致时
        """
        lengths = []
        if len(self.states) > 0:
            lengths.append(len(self.states))
        if len(self.actions) > 0:
            lengths.append(len(self.actions))
        if len(self.rewards) > 0:
            lengths.append(len(self.rewards))
        if len(self.dones) > 0:
            lengths.append(len(self.dones))
            
        if lengths and len(set(lengths)) > 1:
            logger.warning(f"轨迹数据长度不一致: {lengths}，使用最小长度")
            self.length = min(lengths)
        elif lengths:
            self.length = lengths[0]
        else:
            self.length = 0
    
    def __len__(self) -> int:
        """返回轨迹长度"""
        return self.length
    
    def get_state(self, index: int) -> Optional[np.ndarray]:
        """获取指定索引的状态
        
        Args:
            index: 状态索引
            
        Returns:
            状态数组或None
        """
        if 0 <= index < len(self.states):
            return self.states[index]
        return None
    
    def get_action(self, index: int) -> Optional[np.ndarray]:
        """获取指定索引的动作
        
        Args:
            index: 动作索引
            
        Returns:
            动作数组或None
        """
        if 0 <= index < len(self.actions):
            return self.actions[index]
        return None
    
    def get_reward(self, index: int) -> float:
        """获取指定索引的奖励
        
        Args:
            index: 奖励索引
            
        Returns:
            奖励值，如果索引无效则返回0.0
        """
        if 0 <= index < len(self.rewards):
            return float(self.rewards[index])
        return 0.0
    
    def is_done(self, index: int) -> bool:
        """检查指定索引是否结束
        
        Args:
            index: 检查索引
            
        Returns:
            是否结束，如果索引无效则返回False
        """
        if 0 <= index < len(self.dones):
            return bool(self.dones[index])
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            轨迹数据字典的副本
        """
        return self.data.copy()
    
    def __repr__(self) -> str:
        """返回轨迹的字符串表示"""
        return f"TrajectoryWrapper(length={self.length}, total_reward={self.total_reward:.3f})"

class PreferenceDataEngine:
    """偏好数据引擎 - 兼容性类
    
    为现有的 grpo 模块提供向后兼容的接口，实际功能由 PreferenceLabelingEngine 提供。
    
    这个类主要用于保持向后兼容性，新项目建议直接使用 PreferenceLabelingEngine。
    
    Attributes:
        task_name: 任务名称
        config_path: 配置文件路径
        labeling_engine: 底层的 PreferenceLabelingEngine 实例
        trajectory_metrics: 轨迹指标计算器
        stats: 统计信息字典
    """
    
    def __init__(self, task_name: Optional[str] = None, config_path: Optional[str] = None):
        """初始化偏好数据引擎
        
        Args:
            task_name: 任务名称，如果为None则使用"default"
            config_path: 配置文件路径，如果为None则使用默认配置
            
        Raises:
            FileNotFoundError: 当配置文件不存在时
            ValueError: 当配置参数无效时
        """
        try:
            self.task_name = task_name or "default"
            self.config_path = config_path
            
            # 创建实际的标签引擎
            self.labeling_engine = create_preference_labeling_engine(
                task_name=self.task_name,
                config_path=self.config_path
            )
            
            # 创建轨迹指标计算器
            self.trajectory_metrics = TrajectoryMetrics()
            
            # 统计信息
            self.stats = {
                'total_comparisons': 0,
                'successful_comparisons': 0,
                'failed_comparisons': 0,
                'preference_labels_generated': 0,
                'quality_evaluations': 0,
                'trajectory_comparisons': 0
            }
            
            logger.info(f"[偏好数据引擎] 初始化完成，任务: {self.task_name} (兼容性模式)")
            
        except Exception as e:
            logger.error(f"[偏好数据引擎] 初始化失败: {e}")
            raise
    
    def generate_preference_labels(self, 
                                 obs_a: np.ndarray, 
                                 act_a: np.ndarray,
                                 obs_b: np.ndarray, 
                                 act_b: np.ndarray,
                                 label_type: str = "rule_based",
                                 **kwargs) -> Tuple[float, Dict[str, Any]]:
        """生成偏好标签 - 兼容性方法
        
        Args:
            obs_a: 轨迹A的观测序列
            act_a: 轨迹A的动作序列
            obs_b: 轨迹B的观测序列
            act_b: 轨迹B的动作序列
            label_type: 标签类型，支持 'rule_based', 'dpo_binary', 'quality_based'
            **kwargs: 额外的配置参数
            
        Returns:
            (preference_score, metadata): 偏好分数和元数据
            
        Raises:
            ValueError: 当标签类型不支持或输入数据无效时
        """
        # 验证输入参数
        if not isinstance(obs_a, np.ndarray) or not isinstance(act_a, np.ndarray):
            raise ValueError("轨迹A的观测和动作序列必须是numpy数组")
        if not isinstance(obs_b, np.ndarray) or not isinstance(act_b, np.ndarray):
            raise ValueError("轨迹B的观测和动作序列必须是numpy数组")
        if label_type not in ["rule_based", "dpo_binary", "quality_based"]:
            raise ValueError(f"不支持的标签类型: {label_type}，支持的类型: rule_based, dpo_binary, quality_based")
        
        try:
            # 转换标签类型
            if label_type == "rule_based":
                lt = LabelType.RULE_BASED
            elif label_type == "dpo_binary":
                lt = LabelType.DPO_BINARY
            elif label_type == "quality_based":
                lt = LabelType.QUALITY_BASED
            else:
                lt = LabelType.RULE_BASED
            
            # 使用标签引擎生成标签
            label = self.labeling_engine.generate_preference_labels(
                obs_a, act_a, obs_b, act_b, lt, **kwargs
            )
            
            # 构造兼容的返回格式
            metadata = {
                'confidence': float(label.metadata.confidence) if label.metadata else 0.5,
                'label_type': label.metadata.label_type.value if label.metadata else label_type,
                'quality_score_a': float(label.metadata.quality_score_a) if label.metadata else 0.5,
                'quality_score_b': float(label.metadata.quality_score_b) if label.metadata else 0.5,
                'is_valid': bool(label.is_valid),
                'validation_errors': list(label.validation_errors) if label.validation_errors else [],
                'trajectory_lengths': [len(obs_a), len(obs_b)]
            }
            
            # 更新统计信息
            self.stats['preference_labels_generated'] += 1
            if label.is_valid:
                self.stats['successful_comparisons'] += 1
            else:
                self.stats['failed_comparisons'] += 1
            self.stats['total_comparisons'] += 1
            
            return float(label.preference_score), metadata
            
        except Exception as e:
            logger.error(f"生成偏好标签失败: {e}")
            self.stats['failed_comparisons'] += 1
            self.stats['total_comparisons'] += 1
            
            # 返回默认值
            return 0.5, {
                'confidence': 0.1,
                'label_type': label_type,
                'quality_score_a': 0.5,
                'quality_score_b': 0.5,
                'is_valid': False,
                'validation_errors': [str(e)],
                'trajectory_lengths': [len(obs_a) if isinstance(obs_a, np.ndarray) else 0, 
                                     len(obs_b) if isinstance(obs_b, np.ndarray) else 0]
            }
    
    def evaluate_trajectory_quality(self, 
                                   obs_seq: Union[np.ndarray, Dict[str, Any]], 
                                   act_seq: Optional[np.ndarray] = None,
                                   method: str = "comprehensive",
                                   **kwargs) -> Tuple[float, Dict[str, float]]:
        """评估轨迹质量 - 兼容性方法
        
        Args:
            obs_seq: 观测序列或轨迹数据字典
            act_seq: 动作序列（当obs_seq为数组时必需）
            method: 评估方法，支持 'comprehensive', 'reward_based', 'success_based'
            **kwargs: 额外的评估参数
            
        Returns:
            (quality_score, feature_scores): 质量分数和特征分数
            
        Raises:
            ValueError: 当输入参数无效时
        """
        if method not in ["comprehensive", "reward_based", "success_based"]:
            logger.warning(f"未知的评估方法: {method}，使用默认方法 'comprehensive'")
            method = "comprehensive"
            
        try:
            # 处理不同的输入格式
            if isinstance(obs_seq, dict):
                # 输入是轨迹字典
                trajectory_data = obs_seq
                obs_array = np.array(trajectory_data.get('obs', []))
                act_array = np.array(trajectory_data.get('action', []))
            elif isinstance(obs_seq, np.ndarray) and act_seq is not None:
                # 输入是分离的观测和动作序列
                obs_array = obs_seq
                act_array = act_seq
            else:
                raise ValueError("必须提供观测序列和动作序列，或者提供完整的轨迹数据字典")
            
            if len(obs_array) == 0 or len(act_array) == 0:
                logger.warning("轨迹数据为空，返回默认质量分数")
                return 0.1, {}
            
            # 使用轨迹指标计算器
            quality_score, feature_scores = self.trajectory_metrics.evaluate_trajectory_quality(
                obs_array, act_array, method=method, **kwargs
            )
            
            # 更新统计信息
            self.stats['quality_evaluations'] += 1
            
            # 确保返回值在有效范围内
            quality_score = max(0.0, min(1.0, float(quality_score)))
            
            logger.debug(f"轨迹质量评估完成: {quality_score:.3f} (方法: {method})")
            return quality_score, feature_scores
            
        except Exception as e:
            logger.error(f"评估轨迹质量失败: {e}")
            self.stats['quality_evaluations'] += 1  # 仍然计入统计
            return 0.1, {}
    
    def compare_trajectories(self, 
                           traj_a: Union[TrajectoryWrapper, Dict[str, Any]], 
                           traj_b: Union[TrajectoryWrapper, Dict[str, Any]],
                           rule_name: Optional[str] = None,
                           method: str = "auto",
                           **kwargs) -> Dict[str, Any]:
        """比较两个轨迹 - 兼容性方法
        
        Args:
            traj_a: 轨迹A（TrajectoryWrapper或字典）
            traj_b: 轨迹B（TrajectoryWrapper或字典）
            rule_name: 比较规则名称
            method: 比较方法，支持 'auto', 'rule_based', 'quality_based'
            **kwargs: 额外的比较参数
            
        Returns:
            比较结果字典，包含preference, confidence, method等字段
            
        Raises:
            ValueError: 当输入参数无效时
        """
        # 验证输入参数
        if method not in ["auto", "rule_based", "quality_based"]:
            logger.warning(f"未知的比较方法: {method}，使用默认方法 'auto'")
            method = "auto"
        
        try:
            # 确保输入是TrajectoryWrapper类型
            if isinstance(traj_a, dict):
                traj_a = TrajectoryWrapper(traj_a)
            if isinstance(traj_b, dict):
                traj_b = TrajectoryWrapper(traj_b)
                
            if not isinstance(traj_a, TrajectoryWrapper) or not isinstance(traj_b, TrajectoryWrapper):
                raise ValueError(f"轨迹数据必须是TrajectoryWrapper或字典格式，得到: {type(traj_a)}, {type(traj_b)}")
            
            if rule_name and rule_name in RULE_REGISTRY:
                # 使用指定的规则
                compare_func = RULE_REGISTRY[rule_name]
                better_traj, worse_traj = compare_func(traj_a, traj_b, None)  # goal=None
                
                if better_traj is not None:
                    preference = 0 if better_traj == traj_a else 1
                    confidence = 0.8  # 规则比较的默认置信度
                else:
                    preference = -1  # 无法比较
                    confidence = 0.0
                
                result = {
                    'preference': preference,
                    'confidence': confidence,
                    'method': 'rule_based',
                    'rule_name': rule_name,
                    'trajectory_lengths': [len(traj_a), len(traj_b)]
                }
            else:
                # 使用质量评估进行比较
                quality_a, _ = self.evaluate_trajectory_quality(traj_a.states, traj_a.actions, **kwargs)
                quality_b, _ = self.evaluate_trajectory_quality(traj_b.states, traj_b.actions, **kwargs)
                
                if quality_a > quality_b + 0.05:  # 5% 阈值
                    preference = 0  # traj_a更好
                elif quality_b > quality_a + 0.05:
                    preference = 1  # traj_b更好
                else:
                    preference = -1  # 无法区分
                
                confidence = float(abs(quality_a - quality_b))
                
                result = {
                    'preference': preference,
                    'confidence': confidence,
                    'method': 'quality_based',
                    'quality_scores': [float(quality_a), float(quality_b)],
                    'trajectory_lengths': [len(traj_a), len(traj_b)]
                }
            
            # 添加通用信息
            result.update({
                'timestamp': time.time(),
                'task_name': getattr(self, 'task_name', 'unknown')
            })
            
            # 更新统计信息
            self.stats['trajectory_comparisons'] += 1
            self.stats['total_comparisons'] += 1
            if result.get('preference', -1) != -1:
                self.stats['successful_comparisons'] += 1
            else:
                self.stats['failed_comparisons'] += 1
            
            logger.debug(f"轨迹比较完成: preference={result['preference']}, confidence={result['confidence']:.3f}, method={result['method']}")
            return result
                    
        except Exception as e:
            logger.error(f"比较轨迹失败: {e}")
            self.stats['failed_comparisons'] += 1
            self.stats['total_comparisons'] += 1
            
            return {
                'preference': -1,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e),
                'timestamp': time.time(),
                'trajectory_lengths': [0, 0]
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息 - 兼容性方法
        
        Returns:
            统计信息字典，包含各种操作的计数和成功率
        """
        try:
            # 首先尝试从底层引擎获取统计信息
            engine_stats = self.labeling_engine.get_statistics()
            
            # 计算成功率
            total_ops = self.stats.get('total_comparisons', 0)
            success_rate = (self.stats.get('successful_comparisons', 0) / total_ops * 100) if total_ops > 0 else 0.0
            
            # 合并统计信息
            combined_stats = {
                'task_name': getattr(self, 'task_name', 'unknown'),
                'config_path': getattr(self, 'config_path', None),
                
                # 基本统计
                'total_comparisons': self.stats.get('total_comparisons', 0),
                'successful_comparisons': self.stats.get('successful_comparisons', 0),
                'failed_comparisons': self.stats.get('failed_comparisons', 0),
                'success_rate_percent': round(success_rate, 2),
                
                # 详细统计
                'preference_labels_generated': self.stats.get('preference_labels_generated', 0),
                'quality_evaluations': self.stats.get('quality_evaluations', 0),
                'trajectory_comparisons': self.stats.get('trajectory_comparisons', 0),
                
                # 系统信息
                'api_rules_loaded': len(self.labeling_engine.api_rules) if hasattr(self.labeling_engine, 'api_rules') else 0,
                'engine_type': type(self.labeling_engine).__name__ if hasattr(self, 'labeling_engine') else 'unknown',
                'trajectory_metrics_available': hasattr(self, 'trajectory_metrics'),
                
                # 时间戳
                'last_updated': time.time()
            }
            
            # 合并底层引擎的统计信息
            if isinstance(engine_stats, dict):
                combined_stats.update(engine_stats)
                
            return combined_stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                'task_name': getattr(self, 'task_name', 'unknown'),
                'error': str(e),
                'last_updated': time.time()
            }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_comparisons': 0,
            'successful_comparisons': 0,
            'failed_comparisons': 0,
            'preference_labels_generated': 0,
            'quality_evaluations': 0,
            'trajectory_comparisons': 0
        }
        
        # 也重置底层引擎的统计信息（如果支持）
        try:
            if hasattr(self.labeling_engine, 'reset_statistics'):
                self.labeling_engine.reset_statistics()
        except Exception as e:
            logger.warning(f"重置底层引擎统计信息失败: {e}")
            
        logger.info("统计信息已重置")
    
    def __repr__(self) -> str:
        """返回引擎的字符串表示"""
        try:
            stats = self.get_statistics()
            return (f"PreferenceDataEngine(task='{stats.get('task_name', 'unknown')}', "
                    f"comparisons={stats.get('total_comparisons', 0)}, "
                    f"success_rate={stats.get('success_rate_percent', 0):.1f}%)")
        except:
            return f"PreferenceDataEngine(task='{getattr(self, 'task_name', 'unknown')}')"
    
    def __del__(self):
        """析构函数，记录最终统计信息"""
        try:
            if hasattr(self, 'stats') and self.stats.get('total_comparisons', 0) > 0:
                stats = self.get_statistics()
                logger.info(f"PreferenceDataEngine 销毁，最终统计: {stats.get('total_comparisons', 0)} 次比较，"
                           f"成功率 {stats.get('success_rate_percent', 0):.1f}%")
        except:
            pass  # 忽略析构时的错误

# 全局比较规则字典 - 兼容性变量
global_compare_rules = RULE_REGISTRY.copy() if RULE_REGISTRY else {}

# 兼容性函数
def global_compare_rules() -> Dict[str, Any]:
    """获取全局比较规则 - 兼容性函数
    
    Returns:
        规则注册表的副本
        
    Warning:
        此函数已弃用，建议使用 PreferenceLabelingEngine.api_rules
    """
    warnings.warn(
        "global_compare_rules() 已弃用，请使用 PreferenceLabelingEngine.api_rules。"
        "新系统提供更好的性能和更多功能。",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        return RULE_REGISTRY.copy()
    except Exception as e:
        logger.error(f"获取全局比较规则失败: {e}")
        return {}

def auto_register_rules_for_task(task_name: str) -> bool:
    """为指定任务自动注册规则 - 兼容性函数
    
    Args:
        task_name: 任务名称
        
    Returns:
        是否成功注册规则
        
    Warning:
        此函数已弃用，建议使用 PreferenceLabelingEngine
    """
    warnings.warn(
        "auto_register_rules_for_task() 已弃用，请使用 PreferenceLabelingEngine。"
        "新系统提供自动规则发现和更好的任务适配。",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not isinstance(task_name, str) or not task_name.strip():
        logger.warning("任务名称无效，返回False")
        return False
    
    try:
        # 检查规则注册表是否已有规则
        if len(RULE_REGISTRY) > 0:
            logger.info(f"[规则注册] 任务 {task_name} 的规则已注册，共 {len(RULE_REGISTRY)} 个规则")
            return True
        else:
            logger.warning(f"[规则注册] 任务 {task_name} 没有可用的规则")
            return False
    except Exception as e:
        logger.error(f"[规则注册] 为任务 {task_name} 注册规则失败: {e}")
        return False

# 工厂函数 - 兼容性函数
def create_preference_data_engine(task_name: Optional[str] = None, 
                                config_path: Optional[str] = None,
                                **kwargs) -> PreferenceDataEngine:
    """创建偏好数据引擎 - 兼容性函数
    
    Args:
        task_name: 任务名称，如果为None则使用"default"
        config_path: 配置文件路径，如果为None则使用默认配置
        **kwargs: 额外的配置参数
        
    Returns:
        PreferenceDataEngine实例
        
    Raises:
        ValueError: 当参数无效时
        FileNotFoundError: 当配置文件不存在时
    """
    try:
        return PreferenceDataEngine(task_name=task_name, config_path=config_path)
    except Exception as e:
        logger.error(f"创建偏好数据引擎失败: {e}")
        raise

def create_trajectory_wrapper(obs_seq: np.ndarray, 
                            act_seq: np.ndarray, 
                            rewards: Optional[np.ndarray] = None,
                            dones: Optional[np.ndarray] = None) -> TrajectoryWrapper:
    """创建轨迹包装器 - 兼容性函数
    
    Args:
        obs_seq: 观测序列
        act_seq: 动作序列
        rewards: 奖励序列（可选）
        dones: 结束标志序列（可选）
        
    Returns:
        轨迹包装器实例
        
    Raises:
        ValueError: 当输入数据格式无效时
    """
    if not isinstance(obs_seq, np.ndarray) or not isinstance(act_seq, np.ndarray):
        raise ValueError(f"观测和动作序列必须是numpy数组，得到: {type(obs_seq)}, {type(act_seq)}")
    
    if len(obs_seq) == 0 or len(act_seq) == 0:
        raise ValueError("观测和动作序列不能为空")
    
    try:
        data = {
            'obs': obs_seq,
            'action': act_seq,
            'reward': rewards if rewards is not None else np.zeros(len(obs_seq)),
            'done': dones if dones is not None else np.zeros(len(obs_seq), dtype=bool)
        }
        return TrajectoryWrapper(data)
    except Exception as e:
        logger.error(f"创建轨迹包装器失败: {e}")
        raise

if __name__ == "__main__":
    # 测试代码 - 兼容性验证
    import sys
    import traceback
    
    def test_basic_functionality():
        """测试基本功能"""
        print("=" * 60)
        print("测试偏好数据引擎基本功能...")
        print("=" * 60)
        
        try:
            # 创建引擎
            print("1. 创建偏好数据引擎...")
            engine = create_preference_data_engine("test_task")
            print(f"   ✓ 引擎创建成功: {engine}")
            
            # 生成测试数据
            print("2. 生成测试轨迹数据...")
            obs_a = np.random.randn(50, 151)
            act_a = np.random.randn(50, 61)
            obs_b = np.random.randn(45, 151)
            act_b = np.random.randn(45, 61)
            print(f"   ✓ 轨迹A: obs{obs_a.shape}, act{act_a.shape}")
            print(f"   ✓ 轨迹B: obs{obs_b.shape}, act{act_b.shape}")
            
            # 测试轨迹包装器
            print("3. 测试轨迹包装器...")
            traj_wrapper_a = create_trajectory_wrapper(obs_a, act_a)
            traj_wrapper_b = create_trajectory_wrapper(obs_b, act_b)
            print(f"   ✓ 轨迹A包装器: {traj_wrapper_a}")
            print(f"   ✓ 轨迹B包装器: {traj_wrapper_b}")
            
            # 测试质量评估
            print("4. 测试轨迹质量评估...")
            quality_a, features_a = engine.evaluate_trajectory_quality(obs_a, act_a)
            quality_b, features_b = engine.evaluate_trajectory_quality(obs_b, act_b)
            print(f"   ✓ 轨迹A质量: {quality_a:.3f}")
            print(f"   ✓ 轨迹B质量: {quality_b:.3f}")
            
            # 测试轨迹比较
            print("5. 测试轨迹比较...")
            traj_dict_a = {'obs': obs_a, 'action': act_a}
            traj_dict_b = {'obs': obs_b, 'action': act_b}
            comparison_result = engine.compare_trajectories(traj_dict_a, traj_dict_b)
            print(f"   ✓ 比较结果: preference={comparison_result['preference']}, "
                  f"confidence={comparison_result['confidence']:.3f}, "
                  f"method={comparison_result['method']}")
            
            # 测试偏好标签生成
            print("6. 测试偏好标签生成...")
            trajectory_pairs = [(traj_dict_a, traj_dict_b)]
            labels = engine.generate_preference_labels(trajectory_pairs, "quality_based")
            if labels:
                label = labels[0]
                print(f"   ✓ 偏好标签: preference={label['preference']}, "
                      f"confidence={label['confidence']:.3f}, "
                      f"type={label['label_type']}")
            
            # 获取统计信息
            print("7. 获取统计信息...")
            stats = engine.get_statistics()
            print(f"   ✓ 统计信息: {stats['total_comparisons']} 次比较, "
                  f"成功率 {stats['success_rate_percent']:.1f}%")
            
            print("\n" + "=" * 60)
            print("✓ 所有基本功能测试通过！")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            print("错误详情:")
            traceback.print_exc()
            return False
    
    def test_compatibility():
        """测试兼容性功能"""
        print("\n" + "=" * 60)
        print("测试兼容性功能...")
        print("=" * 60)
        
        try:
            # 测试全局规则获取
            print("1. 测试全局规则获取...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                rules = global_compare_rules()
            print(f"   ✓ 获取到 {len(rules)} 个规则")
            
            # 测试任务规则注册
            print("2. 测试任务规则注册...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                success = auto_register_rules_for_task("test_task")
            print(f"   ✓ 任务规则注册: {success}")
            
            print("\n" + "=" * 60)
            print("✓ 兼容性功能测试通过！")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n❌ 兼容性测试失败: {e}")
            print("错误详情:")
            traceback.print_exc()
            return False
    
    # 运行测试
    print("开始偏好数据引擎测试套件...")
    
    success_basic = test_basic_functionality()
    success_compat = test_compatibility()
    
    if success_basic and success_compat:
        print("\n🎉 所有测试通过！偏好数据引擎工作正常。")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查代码。")
        sys.exit(1)