
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化偏好模型包装器 (Optimized Preference Model Wrapper)

该文件提供与现有代码兼容的接口，用于无缝集成优化后的偏好模型

自动生成时间: 2025-08-12T22:36:56.047368
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# 导入优化的模块
from ..optimized_preference_integrator import OptimizedPreferenceIntegrator
from ..optimized_preference_trainer import OptimizedPreferenceTrainer

class OptimizedPreferenceWrapper:
    """优化偏好模型包装器
    
    提供与现有LatentPreferenceIntegrator兼容的接口
    """
    
    def __init__(self, trainer: OptimizedPreferenceTrainer, integrator: OptimizedPreferenceIntegrator):
        self.trainer = trainer
        self.integrator = integrator
        self.episode_count = 0
        
        # 兼容性属性
        self.preference_weight = integrator.current_preference_weight
        self.environment_weight = integrator.current_environment_weight
    
    def compute_integrated_reward(self, 
                                latent_state: torch.Tensor, 
                                action: torch.Tensor, 
                                environment_reward: float) -> Dict[str, Any]:
        """计算集成奖励（兼容接口）"""
        return self.integrator.compute_integrated_reward(latent_state, action, environment_reward)
    
    def get_preference_reward(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[float, float]:
        """获取偏好奖励（兼容接口）"""
        return self.trainer.get_preference_reward(latent_state, action)
    
    def should_train(self, episode: int, data_size: int) -> bool:
        """判断是否应该训练（兼容接口）"""
        return self.trainer.should_train(episode, data_size)
    
    def train_step(self, chosen_data: List[Dict], rejected_data: List[Dict]) -> Dict[str, float]:
        """执行训练步骤（兼容接口）"""
        return self.trainer.train_step(chosen_data, rejected_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息（兼容接口）"""
        trainer_stats = self.trainer.get_training_stats()
        integrator_stats = self.integrator.get_stats()
        
        # 合并统计信息
        combined_stats = {}
        combined_stats.update(trainer_stats)
        combined_stats.update(integrator_stats)
        
        return combined_stats
    
    def save_checkpoint(self, path: str, additional_info: Optional[Dict] = None):
        """保存检查点（兼容接口）"""
        self.trainer.save_checkpoint(self.episode_count, additional_info)
    
    def load_checkpoint(self, path: str):
        """加载检查点（兼容接口）"""
        self.trainer.load_checkpoint(path)
    
    def update_episode(self, episode: int):
        """更新episode计数"""
        self.episode_count = episode
        
        # 更新兼容性属性
        self.preference_weight = self.integrator.current_preference_weight
        self.environment_weight = self.integrator.current_environment_weight

# 全局实例（用于兼容现有代码）
_global_wrapper = None

def create_optimized_preference_system(model_config=None, integration_config=None, tdmpc2_cfg=None):
    """创建优化偏好系统"""
    from ..optimized_preference_integrator import create_optimized_preference_system as _create_system
    
    trainer, integrator = _create_system(model_config, integration_config, tdmpc2_cfg)
    return trainer, integrator

def initialize_optimized_preference_system(model_config=None, training_config=None, integration_config=None):
    """初始化优化偏好系统"""
    global _global_wrapper
    
    trainer, integrator = create_optimized_preference_system(
        model_config, integration_config
    )
    
    _global_wrapper = OptimizedPreferenceWrapper(trainer, integrator)
    return _global_wrapper

def get_optimized_preference_system():
    """获取全局优化偏好系统"""
    global _global_wrapper
    if _global_wrapper is None:
        _global_wrapper = initialize_optimized_preference_system()
    return _global_wrapper

# 兼容性函数
def compute_integrated_reward(latent_state, action, environment_reward):
    """兼容性函数：计算集成奖励"""
    wrapper = get_optimized_preference_system()
    return wrapper.compute_integrated_reward(latent_state, action, environment_reward)

def get_preference_reward(latent_state, action):
    """兼容性函数：获取偏好奖励"""
    wrapper = get_optimized_preference_system()
    return wrapper.get_preference_reward(latent_state, action)

def should_train_preference_model(episode, data_size):
    """兼容性函数：判断是否应该训练"""
    wrapper = get_optimized_preference_system()
    return wrapper.should_train(episode, data_size)

def train_preference_model(chosen_data, rejected_data):
    """兼容性函数：训练偏好模型"""
    wrapper = get_optimized_preference_system()
    return wrapper.train_step(chosen_data, rejected_data)

def get_preference_stats():
    """兼容性函数：获取偏好统计"""
    wrapper = get_optimized_preference_system()
    return wrapper.get_stats()
