
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Any
from collections import deque

# 导入依赖
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.common import math

from prm.optimized_preference_integrator import OptimizedPreferenceIntegrator
from prm.optimized_preference_trainer import OptimizedPreferenceTrainer
from prm.optimized_models.optimized_preference_wrapper import create_optimized_preference_system
from prm.hybrid_value_estimator import HybridValueEstimator, HybridValueConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreferenceAwareTDMPC2(TDMPC2):
    """偏好感知的TD-MPC2智能体"""
    
    def __init__(self, *args, **kwargs):
        # 提取偏好相关参数，避免传递给父类
        preference_integrator = kwargs.pop('preference_integrator', None)
        preference_trainer = kwargs.pop('preference_trainer', None)
        hybrid_config = kwargs.pop('hybrid_config', None)
        enable_preference_planning = kwargs.pop('enable_preference_planning', True)
        
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 偏好系统组件
        self.preference_integrator = preference_integrator
        self.preference_trainer = preference_trainer
        self.hybrid_estimator = None
        self.hybrid_config = hybrid_config
        self.enable_preference_planning = enable_preference_planning
        
        # 偏好统计信息
        self.preference_stats = {
            'total_plans': 0,
            'preference_plans': 0,
            'hybrid_estimates': 0,
            'fallback_plans': 0,
            'average_preference_value': 0.0,
            'preference_integration_time': 0.0
        }
        
        # 滑动窗口统计（替代全局累计）
        self.sliding_window_size = 1000
        self.hybrid_value_window = deque(maxlen=self.sliding_window_size)
        
        logger.info("[偏好感知TD-MPC2] 初始化完成")
    
    def setup_preference_system(self, 
                              preference_integrator: OptimizedPreferenceIntegrator,
                              preference_trainer: OptimizedPreferenceTrainer,
                              hybrid_config: Optional[HybridValueConfig] = None):
        """设置偏好系统"""
        self.preference_integrator = preference_integrator
        self.preference_trainer = preference_trainer
        
        # 创建混合价值估计器
        if hybrid_config is None:
            hybrid_config = HybridValueConfig(
                enable_value_calibration=True,
                enable_value_caching=True,
                enable_confidence_weighting=True
            )
        
        # 获取偏好模型
        preference_model = None
        if hasattr(preference_trainer, 'get_preference_model'):
            preference_model = preference_trainer.get_preference_model()
        elif hasattr(preference_trainer, 'preference_model'):
            preference_model = preference_trainer.preference_model
        
        self.hybrid_estimator = HybridValueEstimator(
            mpc_agent=self,
            preference_model=preference_model,
            config=hybrid_config
        )
        
        logger.info("[偏好感知TD-MPC2] 偏好系统设置完成")
    
    @torch.no_grad()
    def _estimate_value_with_preference(self, 
                                      z: torch.Tensor, 
                                      actions: torch.Tensor, 
                                      task: int,
                                      context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """使用混合价值估计（MPC + 偏好）
        
        Args:
            z: 潜在状态
            actions: 动作序列
            task: 任务ID
            context: 上下文信息
            
        Returns:
            混合价值估计
        """
        if self.hybrid_estimator is not None:
            try:
                hybrid_value = self.hybrid_estimator.estimate_hybrid_value(
                    z, actions, task, context
                )
                self.preference_stats['hybrid_estimates'] += 1
                
                # 更新滑动窗口统计
                current_value = hybrid_value.mean().item()
                self.hybrid_value_window.append(current_value)
                
                # 计算滑动窗口平均值
                sliding_avg = np.mean(self.hybrid_value_window) if self.hybrid_value_window else 0.0
                
                # 精简日志输出（每1000次估计记录一次）
                if self.preference_stats['hybrid_estimates'] % 1000 == 0:
                    logger.info(f"[偏好感知TD-MPC2] 混合价值估计 #{self.preference_stats['hybrid_estimates']} - 滑动窗口平均: {sliding_avg:.4f}")
                
                return hybrid_value
            except Exception as e:
                logger.warning(f"[偏好感知TD-MPC2] 混合价值估计失败，回退到原生估计: {e}")
                self.preference_stats['fallback_plans'] += 1
        
        # 回退到原生价值估计
        return self._estimate_value(z, actions, task)
    
    @torch.no_grad()
    def plan(self, z, task, t0=False, eval_mode=False, **kwargs):
        """偏好感知的MPPI规划"""
        self.preference_stats['total_plans'] += 1
        
        # 如果有偏好系统，使用偏好感知规划
        if self.hybrid_estimator is not None and not eval_mode:
            self.preference_stats['preference_plans'] += 1
            
            # 创建上下文信息
            context = {
                'task': task,
                'timestep': t0,
                'eval_mode': eval_mode
            }
            
            # 添加环境奖励信息（如果可用）
            if hasattr(self, '_last_env_reward'):
                context['env_reward'] = self._last_env_reward
            
            return self._plan_with_preference(z, task, t0, context, **kwargs)
        else:
            # 使用原生规划
            return super().plan(z, task, t0, eval_mode, **kwargs)
    
    def _plan_with_preference(self, z, task, t0, context, **kwargs):
        """使用偏好的MPPI规划"""
        # 获取原始的MPPI参数
        horizon = self.cfg.horizon
        num_samples = self.cfg.num_samples
        num_iterations = self.cfg.iterations
        
        # 记录规划开始
        if self.preference_stats['preference_plans'] % 200 == 0:  # 减少日志频率
            logger.info(f"[偏好感知TD-MPC2] 偏好感知MPPI规划 #{self.preference_stats['preference_plans']} - 采样数: {num_samples}, 迭代数: {num_iterations}")
        
        # 初始化动作分布
        if not hasattr(self, '_prev_mean') or self._prev_mean is None:
            mean = torch.zeros(horizon, self.cfg.action_dim, device=z.device, dtype=z.dtype)
        else:
            mean = self._prev_mean.clone()
        
        std = self.cfg.max_std * torch.ones_like(mean)
        
        # MPPI迭代
        for iteration in range(num_iterations):
            # 采样动作序列
            noise = torch.randn(num_samples, horizon, self.cfg.action_dim, 
                              device=z.device, dtype=z.dtype)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            
            # 使用混合价值估计评估动作序列
            values = self._estimate_value_with_preference(
                z.repeat(num_samples, 1), actions, task, context
            )
            
            # 记录价值范围（减少频率）
            if iteration == 0 and self.preference_stats['preference_plans'] % 200 == 0:
                logger.info(f"  - 价值范围=[{values.min().item():.4f}, {values.max().item():.4f}], 均值: {values.mean().item():.4f}")
            
            # 计算权重
            weights = torch.softmax(values / self.cfg.temperature, dim=0)
            
            # 更新均值和标准差
            mean = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * actions, dim=0)
            
            if iteration < num_iterations - 1:
                variance = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * 
                                   (actions - mean.unsqueeze(0))**2, dim=0)
                std = torch.sqrt(variance + 1e-6)
                std = torch.clamp(std, self.cfg.min_std, self.cfg.max_std)
        
        # 保存均值用于下次规划
        self._prev_mean = mean
        
        # 返回第一个动作
        return mean[0]
    
    def update_env_reward(self, env_reward: float):
        """更新环境奖励信息"""
        self._last_env_reward = env_reward
        
        # 传递给权重控制器
        if (self.hybrid_estimator is not None and 
            hasattr(self.hybrid_estimator, 'weight_controller')):
            self.hybrid_estimator.weight_controller.env_reward_history.append(env_reward)
    
    def get_preference_stats(self) -> Dict[str, Any]:
        """获取偏好统计信息"""
        stats = self.preference_stats.copy()
        
        # 添加滑动窗口统计
        if self.hybrid_value_window:
            stats['sliding_window_average'] = np.mean(self.hybrid_value_window)
            stats['sliding_window_std'] = np.std(self.hybrid_value_window)
            stats['sliding_window_size'] = len(self.hybrid_value_window)
        
        # 添加混合估计器统计
        if self.hybrid_estimator is not None:
            estimator_stats = self.hybrid_estimator.get_stats()
            stats.update({f'estimator_{k}': v for k, v in estimator_stats.items()})
        
        return stats
    
    def reset_preference_stats(self):
        """重置偏好统计信息"""
        self.preference_stats = {
            'total_plans': 0,
            'preference_plans': 0,
            'hybrid_estimates': 0,
            'fallback_plans': 0,
            'average_preference_value': 0.0,
            'preference_integration_time': 0.0
        }
        self.hybrid_value_window.clear()
        
        if self.hybrid_estimator is not None:
            self.hybrid_estimator.reset_statistics()
        
        logger.info("[偏好感知TD-MPC2] 偏好统计信息已重置")
    
    def update_preference_model(self, preference_model):
        """更新偏好模型
        
        Args:
            preference_model: 新的偏好模型
        """
        try:
            # 更新偏好训练器的模型
            if self.preference_trainer is not None:
                if hasattr(self.preference_trainer, 'preference_model'):
                    self.preference_trainer.preference_model = preference_model
                elif hasattr(self.preference_trainer, 'set_preference_model'):
                    self.preference_trainer.set_preference_model(preference_model)
                
                logger.info("[偏好感知TD-MPC2] 偏好训练器模型已更新")
            
            # 更新混合价值估计器的偏好模型
            if self.hybrid_estimator is not None:
                if hasattr(self.hybrid_estimator, 'preference_model'):
                    self.hybrid_estimator.preference_model = preference_model
                elif hasattr(self.hybrid_estimator, 'update_preference_model'):
                    self.hybrid_estimator.update_preference_model(preference_model)
                
                logger.info("[偏好感知TD-MPC2] 混合价值估计器模型已更新")
            
            # 更新偏好集成器的模型（如果存在）
            if self.preference_integrator is not None:
                if hasattr(self.preference_integrator, 'preference_model'):
                    self.preference_integrator.preference_model = preference_model
                elif hasattr(self.preference_integrator, 'update_preference_model'):
                    self.preference_integrator.update_preference_model(preference_model)
                
                logger.info("[偏好感知TD-MPC2] 偏好集成器模型已更新")
            
            logger.info("[偏好感知TD-MPC2] ✅ 偏好模型更新完成")
            
        except Exception as e:
            logger.error(f"[偏好感知TD-MPC2] ❌ 偏好模型更新失败: {e}")
            import traceback
            traceback.print_exc()


def create_preference_aware_tdmpc2(*args, **kwargs):
    """创建偏好感知TD-MPC2智能体的工厂函数
    
    Args:
        *args: 传递给PreferenceAwareTDMPC2的位置参数
        **kwargs: 传递给PreferenceAwareTDMPC2的关键字参数
        
    Returns:
        PreferenceAwareTDMPC2: 偏好感知TD-MPC2智能体实例
    """
    return PreferenceAwareTDMPC2(*args, **kwargs)
