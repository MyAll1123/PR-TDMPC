#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先级偏好集成器 (Prioritized Preference Integrator)

将优先级经验回放系统集成到现有的TD-MPC2训练流程中，提供：
1. 与现有偏好系统的兼容性接口
2. 训练流程的无缝集成
3. 性能监控和统计
4. 配置管理和持久化

作者：AI Assistant
日期：2025-01-19
版本：1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import time
import threading
from pathlib import Path
import yaml

# 导入优先级系统
from .prioritized_preference_system import (
    PrioritizedPreferenceSystem,
    PrioritizedSystemConfig,
    create_prioritized_preference_system
)
from .prioritized_experience_replay import PreferencePair
from .trajectory_encoder import TrajectoryEncoder, create_trajectory_encoder

# 导入现有模块
try:
    from .optimized_preference_integrator import OptimizedPreferenceIntegrator
    from .optimized_preference_trainer import OptimizedPreferenceTrainer
    from .optimized_latent_preference_model import OptimizedLatentPreferenceModel
except ImportError:
    # 如果导入失败，定义占位符类
    class OptimizedPreferenceIntegrator:
        pass
    class OptimizedPreferenceTrainer:
        pass
    class OptimizedLatentPreferenceModel:
        pass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """集成配置"""
    # 启用/禁用标志
    enable_prioritized_replay: bool = True
    enable_legacy_compatibility: bool = True
    enable_performance_monitoring: bool = True
    
    # 集成模式
    integration_mode: str = "hybrid"  # "hybrid", "prioritized_only", "legacy_only"
    
    # 性能配置
    max_memory_usage_mb: float = 1024.0  # 最大内存使用量(MB)
    performance_check_interval: int = 100  # 性能检查间隔
    
    # 兼容性配置
    fallback_to_legacy: bool = True  # 出错时回退到传统方法
    legacy_weight: float = 0.3  # 传统方法权重
    prioritized_weight: float = 0.7  # 优先级方法权重
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # 保存配置
    save_integration_stats: bool = True
    stats_save_interval: int = 1000

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.metrics = {
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'training_time_ms': 0.0,
            'sampling_time_ms': 0.0,
            'priority_update_time_ms': 0.0,
            'total_operations': 0,
            'error_count': 0,
            'last_check_time': time.time()
        }
        self.history = []
        self.lock = threading.RLock()
    
    def start_operation(self, operation_name: str) -> Dict[str, Any]:
        """开始操作计时"""
        return {
            'operation': operation_name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
    
    def end_operation(self, operation_context: Dict[str, Any]) -> None:
        """结束操作计时"""
        with self.lock:
            end_time = time.time()
            duration_ms = (end_time - operation_context['start_time']) * 1000
            
            operation_name = operation_context['operation']
            if operation_name == 'training':
                self.metrics['training_time_ms'] = duration_ms
            elif operation_name == 'sampling':
                self.metrics['sampling_time_ms'] = duration_ms
            elif operation_name == 'priority_update':
                self.metrics['priority_update_time_ms'] = duration_ms
            
            self.metrics['total_operations'] += 1
            
            # 定期检查性能
            if self.metrics['total_operations'] % self.config.performance_check_interval == 0:
                self._update_system_metrics()
    
    def record_error(self, error_type: str, error_message: str) -> None:
        """记录错误"""
        with self.lock:
            self.metrics['error_count'] += 1
            logger.warning(f"性能监控记录错误 [{error_type}]: {error_message}")
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量(MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _update_system_metrics(self) -> None:
        """更新系统指标"""
        self.metrics['memory_usage_mb'] = self._get_memory_usage()
        self.metrics['last_check_time'] = time.time()
        
        # 保存历史记录
        self.history.append(self.metrics.copy())
        
        # 限制历史记录长度
        if len(self.history) > 1000:
            self.history = self.history[-500:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.lock:
            return self.metrics.copy()
    
    def is_performance_acceptable(self) -> bool:
        """检查性能是否可接受"""
        return (
            self.metrics['memory_usage_mb'] < self.config.max_memory_usage_mb and
            self.metrics['error_count'] < 10  # 错误数量阈值
        )

class PrioritizedPreferenceIntegrator:
    """优先级偏好集成器主类"""
    
    def __init__(self,
                 task_name: str,
                 cfg: Any,  # TD-MPC2配置对象
                 integration_config: Optional[IntegrationConfig] = None,
                 legacy_integrator: Optional[OptimizedPreferenceIntegrator] = None,
                 tdmpc2_agent=None):
        """
        初始化优先级偏好集成器
        
        Args:
            task_name: 任务名称
            cfg: TD-MPC2配置对象
            integration_config: 集成配置
            legacy_integrator: 传统偏好集成器
            tdmpc2_agent: TD-MPC2 agent实例
        """
        self.task_name = task_name
        self.cfg = cfg
        self.integration_config = integration_config or IntegrationConfig()
        self.legacy_integrator = legacy_integrator
        self.tdmpc2_agent = tdmpc2_agent
        
        # 创建优先级偏好系统
        self.prioritized_system = None
        if self.integration_config.enable_prioritized_replay:
            try:
                # 从配置中读取优先级经验回放相关参数
                per_config = getattr(cfg, 'prioritized_experience_replay', {})
                training_config = per_config.get('training_config', {})
                
                # 获取preference_model配置
                preference_model_config = per_config.get('preference_model', {})
                
                prioritized_config = PrioritizedSystemConfig(
                    buffer_capacity=per_config.get('buffer_capacity', 10000),
                    batch_size=per_config.get('batch_size', 64),
                    min_buffer_size=per_config.get('min_buffer_size', 200),  # 从配置文件读取，不硬编码
                    train_every_n_episodes=training_config.get('train_every_n_episodes', 10),
                    device=getattr(cfg, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'),
                    # 从preference_model配置中读取action_dim
                    action_dim=preference_model_config.get('action_dim', 61),
                    # 其他训练相关参数
                    preference_model_lr=training_config.get('preference_model_lr', 3e-4),
                    preference_model_weight_decay=training_config.get('preference_model_weight_decay', 1e-5),
                    preference_model_grad_clip=training_config.get('preference_model_grad_clip', 1.0)
                )
                
                # 创建轨迹编码器
                trajectory_encoder = None
                encoder_config = per_config.get('trajectory_encoder', {})
                if encoder_config.get('enabled', True):  # 默认启用轨迹编码器
                    try:
                        # 从配置中获取编码器参数
                        # 优先从preference_model配置中读取action_dim
                        preference_model_config = per_config.get('preference_model', {})
                        
                        # 动态获取观测维度：优先从TD-MPC2配置中获取，然后从编码器配置，最后使用默认值
                        obs_dim = 151  # 默认值
                        if hasattr(cfg, 'obs_shape') and cfg.obs_shape:
                            # 导入OmegaConf类型以处理配置对象
                            try:
                                from omegaconf import DictConfig, ListConfig
                            except ImportError:
                                DictConfig = dict
                                ListConfig = list
                            
                            if isinstance(cfg.obs_shape, (dict, DictConfig)):
                                # 处理字典格式的obs_shape，如{'state': (213,)} 或 {'state': [213]}
                                for obs_type, shape in cfg.obs_shape.items():
                                    if isinstance(shape, (tuple, list, ListConfig)) and len(shape) > 0:
                                        try:
                                            obs_dim = int(shape[0])
                                            break
                                        except (TypeError, ValueError):
                                            continue
                            elif isinstance(cfg.obs_shape, (tuple, list, ListConfig)) and len(cfg.obs_shape) > 0:
                                # 处理元组/列表格式的obs_shape
                                try:
                                    obs_dim = int(cfg.obs_shape[0])
                                except (TypeError, ValueError):
                                    pass
                        elif encoder_config.get('obs_dim'):
                            obs_dim = encoder_config.get('obs_dim')
                        
                        action_dim = preference_model_config.get('action_dim', encoder_config.get('action_dim', 61))  # 从preference_model读取action_dim
                        latent_dim = preference_model_config.get('latent_dim', encoder_config.get('latent_dim', 512))  # TD-MPC2潜空间维度
                        
                        trajectory_encoder = create_trajectory_encoder(
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            latent_dim=latent_dim,
                            device=prioritized_config.device
                        )
                        logger.info(f"优先级偏好集成器创建轨迹编码器成功 (obs_dim={obs_dim}, action_dim={action_dim}, latent_dim={latent_dim})")
                    except Exception as e:
                        logger.warning(f"创建轨迹编码器失败: {e}，将使用默认编码器")
                        trajectory_encoder = None
                
                self.prioritized_system = create_prioritized_preference_system(
                    task_name=task_name,
                    config=prioritized_config,
                    preference_model=None,  # 初始化时为None，后续通过train_preference_model传递
                    tdmpc2_agent=self.tdmpc2_agent,
                    tdmpc2_cfg=self.cfg
                )
                
                logger.info("优先级偏好系统创建成功")
            except Exception as e:
                logger.error(f"创建优先级偏好系统失败: {e}")
                if not self.integration_config.fallback_to_legacy:
                    raise
        
        # 性能监控器
        self.performance_monitor = None
        if self.integration_config.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor(self.integration_config)
        
        # 当前活跃轨迹
        self.active_trajectories = {}
        self.trajectory_counter = 0
        
        # 统计信息
        self.integration_stats = {
            'total_episodes': 0,
            'prioritized_training_steps': 0,
            'legacy_training_steps': 0,
            'hybrid_training_steps': 0,
            'fallback_count': 0,
            'error_count': 0,
            'start_time': time.time()
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"优先级偏好集成器初始化完成")
        logger.info(f"  - 任务: {task_name}")
        logger.info(f"  - 集成模式: {self.integration_config.integration_mode}")
        logger.info(f"  - 优先级系统: {'启用' if self.prioritized_system else '禁用'}")
        logger.info(f"  - 传统系统: {'启用' if self.legacy_integrator else '禁用'}")
    
    def start_episode(self) -> int:
        """开始新的训练回合"""
        with self.lock:
            episode_id = self.trajectory_counter
            self.trajectory_counter += 1
            
            # 在优先级系统中开始新轨迹
            if self.prioritized_system:
                trajectory_id = self.prioritized_system.start_new_trajectory()
                self.active_trajectories[episode_id] = trajectory_id
            
            self.integration_stats['total_episodes'] += 1
            return episode_id
    
    def reset_trajectory_stats(self):
        """重置轨迹级别的统计信息"""
        try:
            # 如果有优先级系统，重置其统计信息
            if self.prioritized_system and hasattr(self.prioritized_system, 'reset_stats'):
                # 重置优先级系统的统计信息
                pass  # 优先级系统有自己的统计管理
            
            # 如果有传统集成器，调用其reset_trajectory_stats方法
            if self.legacy_integrator and hasattr(self.legacy_integrator, 'reset_trajectory_stats'):
                self.legacy_integrator.reset_trajectory_stats()
            
            # 重置本地轨迹统计
            if not hasattr(self, 'trajectory_stats'):
                self.trajectory_stats = {
                    'positive_preference_count': 0,
                    'negative_preference_count': 0,
                    'total_steps': 0,
                    'avg_confidence': 0.0
                }
            else:
                self.trajectory_stats.update({
                    'positive_preference_count': 0,
                    'negative_preference_count': 0,
                    'total_steps': 0,
                    'avg_confidence': 0.0
                })
                
        except Exception as e:
            logger.warning(f"重置轨迹统计失败: {e}")
    
    def collect_step(self, episode_id: int, obs: np.ndarray, action: np.ndarray, 
                    reward: float, done: bool, info: Dict = None) -> None:
        """收集训练步骤数据"""
        if self.prioritized_system and episode_id in self.active_trajectories:
            trajectory_id = self.active_trajectories[episode_id]
            self.prioritized_system.collect_trajectory_step(
                trajectory_id, obs, action, reward, done, info
            )
    
    def end_episode(self, episode_id: int) -> None:
        """结束训练回合"""
        with self.lock:
            if self.prioritized_system and episode_id in self.active_trajectories:
                trajectory_id = self.active_trajectories[episode_id]
                self.prioritized_system.complete_trajectory(trajectory_id)
                del self.active_trajectories[episode_id]
    
    def should_train_preference_model(self) -> bool:
        """检查是否应该训练偏好模型"""
        if self.integration_config.integration_mode == "legacy_only":
            return self.legacy_integrator and hasattr(self.legacy_integrator, 'should_train')
        
        elif self.integration_config.integration_mode == "prioritized_only":
            return self.prioritized_system and self.prioritized_system.should_train_preference_model()
        
        else:  # hybrid mode
            prioritized_ready = (
                self.prioritized_system and 
                self.prioritized_system.should_train_preference_model()
            )
            legacy_ready = (
                self.legacy_integrator and 
                hasattr(self.legacy_integrator, 'should_train') and
                self.legacy_integrator.should_train()
            )
            return prioritized_ready or legacy_ready
    
    def train_preference_model(self, preference_model: Optional[nn.Module] = None) -> Optional[Union[float, Dict[str, Any]]]:
        """训练偏好模型"""
        if not self.should_train_preference_model():
            return None
        
        # 性能监控
        perf_context = None
        if self.performance_monitor:
            perf_context = self.performance_monitor.start_operation('training')
        
        try:
            # 根据集成模式选择训练策略
            if self.integration_config.integration_mode == "prioritized_only":
                # 在prioritized_only模式下，返回字典格式的结果
                result = self.prioritized_system.train_preference_model() if self.prioritized_system else None
                if result is not None:
                    self.integration_stats['prioritized_training_steps'] += 1
                    # 确保返回字典格式
                    if isinstance(result, (int, float)):
                        return {
                            'preference_loss': float(result),
                            'training_steps': 1,
                            'training_mode': 'prioritized_only'
                        }
                    elif isinstance(result, dict):
                        return result
                    else:
                        return {
                            'preference_loss': 0.0,
                            'training_steps': 1,
                            'training_mode': 'prioritized_only',
                            'error': f'Unexpected result type: {type(result)}'
                        }
                return None
            
            elif self.integration_config.integration_mode == "legacy_only":
                loss = self._train_with_legacy_system(preference_model)
                if loss is not None:
                    self.integration_stats['legacy_training_steps'] += 1
                return loss
            
            # 混合模式处理
            total_loss = 0.0
            training_count = 0
            
            # 优先级系统训练
            prioritized_loss = self._train_with_prioritized_system(preference_model)
            if prioritized_loss is not None:
                total_loss += prioritized_loss * self.integration_config.prioritized_weight
                training_count += 1
                self.integration_stats['prioritized_training_steps'] += 1
            
            # 传统系统训练
            legacy_loss = self._train_with_legacy_system(preference_model)
            if legacy_loss is not None:
                total_loss += legacy_loss * self.integration_config.legacy_weight
                training_count += 1
                self.integration_stats['legacy_training_steps'] += 1
            
            if training_count > 0:
                self.integration_stats['hybrid_training_steps'] += 1
                # 计算平均损失
                avg_loss = total_loss / training_count
                
                if perf_context and self.performance_monitor:
                    self.performance_monitor.end_operation(perf_context)
                
                return avg_loss
            
            if perf_context and self.performance_monitor:
                self.performance_monitor.end_operation(perf_context)
            
            return None
            
        except Exception as e:
            self.integration_stats['error_count'] += 1
            if self.performance_monitor:
                self.performance_monitor.record_error('training_error', str(e))
            
            logger.error(f"偏好模型训练出错: {e}")
            
            # 尝试回退到传统方法
            if (self.integration_config.fallback_to_legacy and 
                self.legacy_integrator and 
                self.integration_config.integration_mode != "legacy_only"):
                
                try:
                    self.integration_stats['fallback_count'] += 1
                    return self._train_with_legacy_system(preference_model)
                except Exception as fallback_error:
                    logger.error(f"回退训练也失败: {fallback_error}")
            
            return None
    
    def _train_with_prioritized_system(self, preference_model: Optional[nn.Module]) -> Optional[float]:
        """使用优先级系统训练"""
        if not self.prioritized_system:
            return None
        
        # 设置偏好模型
        if preference_model is not None:
            print(f"[PrioritizedPreferenceIntegrator] 正在设置偏好模型到优先级系统")
            self.prioritized_system.preference_model = preference_model
            print(f"[PrioritizedPreferenceIntegrator] 偏好模型已设置")
        else:
            print(f"[PrioritizedPreferenceIntegrator] ❌ 偏好模型为None，无法设置到优先级系统")
        
        result = self.prioritized_system.train_preference_model()
        # 如果返回的是字典格式，提取损失值；否则直接返回
        if isinstance(result, dict) and 'preference_loss' in result:
            return result['preference_loss']
        return result
    
    def _train_with_legacy_system(self, preference_model: Optional[nn.Module]) -> Optional[float]:
        """使用传统系统训练"""
        if not self.legacy_integrator:
            return None
        
        try:
            # 调用传统系统的训练方法
            if hasattr(self.legacy_integrator, 'train_preference_model'):
                return self.legacy_integrator.train_preference_model(preference_model)
            elif hasattr(self.legacy_integrator, 'update_preference_model'):
                return self.legacy_integrator.update_preference_model()
            else:
                logger.warning("传统集成器没有可用的训练方法")
                return None
        except Exception as e:
            logger.error(f"传统系统训练失败: {e}")
            return None
    
    def get_preference_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """获取偏好奖励"""
        total_reward = 0.0
        weight_sum = 0.0
        
        # 优先级系统奖励
        if (self.prioritized_system and 
            self.integration_config.integration_mode != "legacy_only"):
            try:
                prioritized_reward = self.prioritized_system.get_preference_reward(obs, action)
                total_reward += prioritized_reward * self.integration_config.prioritized_weight
                weight_sum += self.integration_config.prioritized_weight
            except Exception as e:
                logger.warning(f"获取优先级奖励失败: {e}")
        
        # 传统系统奖励
        if (self.legacy_integrator and 
            self.integration_config.integration_mode != "prioritized_only"):
            try:
                if hasattr(self.legacy_integrator, 'get_preference_reward'):
                    legacy_reward = self.legacy_integrator.get_preference_reward(obs, action)
                    total_reward += legacy_reward * self.integration_config.legacy_weight
                    weight_sum += self.integration_config.legacy_weight
            except Exception as e:
                logger.warning(f"获取传统奖励失败: {e}")
        
        # 归一化奖励
        if weight_sum > 0:
            return total_reward / weight_sum
        else:
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取集成统计信息"""
        stats = self.integration_stats.copy()
        
        # 优先级系统统计
        if self.prioritized_system:
            stats['prioritized_system_stats'] = self.prioritized_system.get_statistics()
        
        # 传统系统统计
        if self.legacy_integrator and hasattr(self.legacy_integrator, 'get_statistics'):
            stats['legacy_system_stats'] = self.legacy_integrator.get_statistics()
        
        # 性能监控统计
        if self.performance_monitor:
            stats['performance_stats'] = self.performance_monitor.get_metrics()
        
        # 轨迹统计
        if hasattr(self, 'trajectory_stats'):
            stats['trajectory_stats'] = self.trajectory_stats.copy()
        
        # 计算运行时间
        stats['total_runtime_seconds'] = time.time() - stats['start_time']
        
        return stats
    
    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """保存检查点"""
        if filepath is None:
            timestamp = int(time.time())
            filepath = f"./prioritized_integrator_checkpoint_{timestamp}.pt"
        
        checkpoint = {
            'integration_config': self.integration_config,
            'integration_stats': self.integration_stats,
            'task_name': self.task_name
        }
        
        # 保存优先级系统检查点
        if self.prioritized_system:
            prioritized_checkpoint_path = filepath.replace('.pt', '_prioritized.pt')
            self.prioritized_system.save_checkpoint(prioritized_checkpoint_path)
            checkpoint['prioritized_checkpoint_path'] = prioritized_checkpoint_path
        
        torch.save(checkpoint, filepath)
        logger.info(f"集成器检查点已保存到: {filepath}")
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.prioritized_system:
            self.prioritized_system.cleanup()
        
        # 清理活跃轨迹
        self.active_trajectories.clear()
        
        logger.info("优先级偏好集成器资源清理完成")
    
    def is_healthy(self) -> bool:
        """检查系统健康状态"""
        if self.performance_monitor:
            return self.performance_monitor.is_performance_acceptable()
        return True

# 工厂函数
def create_prioritized_preference_integrator(
    task_name: str,
    cfg: Any,
    integration_config: Optional[IntegrationConfig] = None,
    legacy_integrator: Optional[OptimizedPreferenceIntegrator] = None,
    tdmpc2_agent=None
) -> PrioritizedPreferenceIntegrator:
    """
    创建优先级偏好集成器的工厂函数
    
    Args:
        task_name: 任务名称
        cfg: TD-MPC2配置对象
        integration_config: 集成配置
        legacy_integrator: 传统偏好集成器
        tdmpc2_agent: TD-MPC2 agent实例
        
    Returns:
        优先级偏好集成器实例
    """
    return PrioritizedPreferenceIntegrator(
        task_name=task_name,
        cfg=cfg,
        integration_config=integration_config,
        legacy_integrator=legacy_integrator,
        tdmpc2_agent=tdmpc2_agent
    )

# 配置加载函数
def load_integration_config(config_path: str) -> IntegrationConfig:
    """从YAML文件加载集成配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return IntegrationConfig(**config_dict)
    except Exception as e:
        logger.warning(f"加载集成配置失败: {e}，使用默认配置")
        return IntegrationConfig()

if __name__ == "__main__":
    # 测试代码
    print("测试优先级偏好集成器...")
    
    # 模拟配置对象
    class MockConfig:
        def __init__(self):
            self.preference_buffer_capacity = 500
            self.preference_batch_size = 16
            self.min_preference_buffer_size = 50
            self.preference_train_interval = 3
            self.device = "cpu"
    
    cfg = MockConfig()
    
    # 创建集成配置
    integration_config = IntegrationConfig(
        integration_mode="prioritized_only",
        enable_performance_monitoring=True
    )
    
    # 创建集成器
    integrator = create_prioritized_preference_integrator(
        task_name="test_task",
        cfg=cfg,
        integration_config=integration_config
    )
    
    # 模拟训练过程
    for episode in range(10):
        episode_id = integrator.start_episode()
        
        # 模拟轨迹步骤
        for step in range(50):
            obs = np.random.randn(10)
            action = np.random.randn(5)
            reward = np.random.rand()
            done = step == 49
            
            integrator.collect_step(episode_id, obs, action, reward, done)
        
        integrator.end_episode(episode_id)
        print(f"完成回合 {episode + 1}")
        
        # 检查是否应该训练
        if integrator.should_train_preference_model():
            loss = integrator.train_preference_model()
            if loss is not None:
                print(f"训练损失: {loss:.4f}")
    
    # 获取统计信息
    stats = integrator.get_statistics()
    print(f"\n集成器统计信息:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    print(f"    {sub_key}: [嵌套字典]")
                else:
                    print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # 清理资源
    integrator.cleanup()
    
    print("测试完成！")