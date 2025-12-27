#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成偏好学习的TD-MPC2训练器 - 内存缓存版本

功能：
1. 继承自原有的OnlineTrainer，保持TD-MPC2的核心训练流程
2. 集成HistoricalPreferenceManager，实现历史数据收集和偏好模型训练
3. 最小化对原有训练流程的影响
4. 使用内存缓存，避免文件IO操作

特点：
- 无缝集成到现有训练流程
- 自动收集历史轨迹数据
- 自动创建和更新偏好模型
- 提供偏好奖励增强（可选）
- 高性能内存缓存
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from termcolor import colored
from tensordict.tensordict import TensorDict

# 导入TD-MPC2核心模块
try:
    from .online_trainer import OnlineTrainer
    from .historical_preference_manager import HistoricalPreferenceManager
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from online_trainer import OnlineTrainer
    from historical_preference_manager import HistoricalPreferenceManager

# 尝试导入优化后的偏好系统
try:
    import sys
    import os
    # 添加SPE项目根路径
    spe_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if spe_root not in sys.path:
        sys.path.insert(0, spe_root)
    
    from prm.optimized_preference_integrator import OptimizedPreferenceIntegrator
    from prm.optimized_preference_trainer import OptimizedPreferenceTrainer
    from prm.optimized_models.optimized_preference_wrapper import create_optimized_preference_system
    OPTIMIZED_PREFERENCE_AVAILABLE = True
    print("[IntegratedPreferenceTrainer] ✅ 优化偏好系统导入成功")
except ImportError as e:
    OPTIMIZED_PREFERENCE_AVAILABLE = False
    print(f"[IntegratedPreferenceTrainer] ⚠️ 优化偏好系统导入失败: {e}")
    print("[IntegratedPreferenceTrainer] 将使用基础偏好学习功能")

class IntegratedPreferenceTrainer(OnlineTrainer):
    """集成偏好学习的TD-MPC2训练器 - 内存缓存版本"""
    
    def __init__(self, *args, **kwargs):
        """初始化集成偏好学习训练器"""
        
        # 先调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 优先级偏好集成器（外部注入）
        self.prioritized_integrator = None
        self.current_episode_id = None  # 当前episode的ID
        
        # 训练频率控制
        self.episodes_completed = 0  # 已完成的episode数量
        self.last_preference_training_episode = 0  # 上次偏好模型训练的episode
        
        # 初始化偏好学习相关组件
        self._init_historical_preference_manager()
        
        # 初始化优化偏好系统
        self._init_optimized_preference_system()
        
        # 当前episode的数据缓冲（内存）
        self.current_episode_obs = []
        self.current_episode_actions = []
        self.current_episode_rewards = []
        self.current_episode_latent_states = []  # 新增：潜空间状态缓冲
        
        # 统计信息（添加潜空间偏好奖励统计）
        self.preference_stats = {
            'historical_data_collections': 0,
            'preference_model_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            # 双路径奖励融合统计
            'latent_preference_computations': 0,
            'reward_fusions': 0,
            'total_preference_reward': 0.0,
            'total_environment_reward': 0.0,
            'total_integrated_reward': 0.0,
        }
        
        # 性能监控（添加潜空间偏好计算时间监控）
        self.performance_monitor = {
            'data_collection_time': deque(maxlen=100),
            'model_update_time': deque(maxlen=20),
            'model_training_time': deque(maxlen=20),
            # 双路径奖励融合性能监控
            'reward_fusion_time': deque(maxlen=100),
            'latent_preference_computation_time': deque(maxlen=100),
        }
        
        print(f"[IntegratedPreferenceTrainer] 初始化完成 - 内存缓存模式")
        if self.historical_preference_manager:
            print(f"  - 历史偏好管理器: 已启用")
            print(f"  - 内存缓存模式: 启用")
            print(f"  - 文件IO操作: 已禁用")
        else:
            print(f"  - 历史偏好管理器: 未启用")
        
        if hasattr(self, 'latent_preference_integrator') and self.latent_preference_integrator:
            print(f"  - 潜空间偏好系统: 已启用")
            print(f"  - 双路径奖励融合: 启用")
        else:
            print(f"  - 潜空间偏好系统: 未启用")
            
        if self.prioritized_integrator:
            print(f"  - 优先级偏好系统: 已启用")
        else:
            print(f"  - 优先级偏好系统: 未启用")
    
    def set_prioritized_integrator(self, integrator):
        """设置优先级偏好集成器"""
        self.prioritized_integrator = integrator
        print(f"[IntegratedPreferenceTrainer] 优先级偏好集成器已设置")
        
        # 如果已经有偏好训练器，立即将偏好模型传递给优先级系统
        if self.preference_trainer and hasattr(self.preference_trainer, 'models') and len(self.preference_trainer.models) > 0:
            preference_model = self.preference_trainer.models[0]  # 使用第一个模型
            print(f"[IntegratedPreferenceTrainer] 正在将偏好模型传递给优先级系统: {type(preference_model).__name__}")
            
            # 直接设置到优先级系统中
            if hasattr(integrator, 'prioritized_system') and integrator.prioritized_system:
                integrator.prioritized_system.preference_model = preference_model
                print(f"[IntegratedPreferenceTrainer] ✅ 偏好模型已成功传递给优先级系统")
            else:
                print(f"[IntegratedPreferenceTrainer] ⚠️ 优先级集成器没有prioritized_system属性")
        else:
            print(f"[IntegratedPreferenceTrainer] ⚠️ 偏好训练器或模型尚未就绪，稍后传递偏好模型")
    
    def _init_historical_preference_manager(self):
        """初始化历史偏好管理器 - 已禁用原始偏好学习流程"""
        # 原始偏好学习流程已禁用，不再初始化历史偏好管理器
        self.historical_preference_manager = None
        print(f"[IntegratedPreferenceTrainer] ⚠️ 历史偏好管理器已禁用 - 使用优先级偏好系统")
    
    def _init_optimized_preference_system(self):
        """初始化优化偏好系统"""
        self.preference_integrator = None
        self.preference_trainer = None
        
        if not OPTIMIZED_PREFERENCE_AVAILABLE:
            print(f"[IntegratedPreferenceTrainer] 优化偏好系统不可用，跳过初始化")
            return
        
        try:
            # 获取任务配置信息
            task_name = getattr(self.cfg, 'task', 'unknown')
            
            # 根据任务确定动作维度
            action_dim = 61  # humanoid_h1hand 默认动作维度
            if hasattr(self.env, 'action_space'):
                if hasattr(self.env.action_space, 'shape'):
                    action_dim = self.env.action_space.shape[0]
                elif hasattr(self.env.action_space, 'n'):
                    action_dim = self.env.action_space.n
            
            # 获取潜空间维度
            latent_dim = getattr(self.cfg, 'latent_dim', 512)
            
            print(f"[IntegratedPreferenceTrainer] 初始化优化偏好系统...")
            print(f"  - 任务: {task_name}")
            print(f"  - 潜空间维度: {latent_dim}")
            print(f"  - 动作维度: {action_dim}")
            
            # 创建优化偏好系统，传递TD-MPC2配置
            self.preference_trainer, self.preference_integrator = create_optimized_preference_system(
                tdmpc2_cfg=self.cfg
            )
            
            # 为兼容性创建别名
            self.latent_preference_integrator = self.preference_integrator
            
            # 如果智能体有偏好系统，更新它
            if hasattr(self.agent, 'preference_integrator') and hasattr(self.agent, 'preference_trainer'):
                self.agent.preference_integrator = self.preference_integrator
                self.agent.preference_trainer = self.preference_trainer
                print(f"[IntegratedPreferenceTrainer] ✅ 智能体偏好系统已更新")
            
            # 如果优先级集成器已经存在，将偏好模型传递给它
            if self.prioritized_integrator and self.preference_trainer and hasattr(self.preference_trainer, 'models') and len(self.preference_trainer.models) > 0:
                preference_model = self.preference_trainer.models[0]  # 使用第一个模型
                print(f"[IntegratedPreferenceTrainer] 正在将偏好模型传递给已存在的优先级系统: {type(preference_model).__name__}")
                
                # 直接设置到优先级系统中
                if hasattr(self.prioritized_integrator, 'prioritized_system') and self.prioritized_integrator.prioritized_system:
                    self.prioritized_integrator.prioritized_system.preference_model = preference_model
                    print(f"[IntegratedPreferenceTrainer] ✅ 偏好模型已成功传递给已存在的优先级系统")
                else:
                    print(f"[IntegratedPreferenceTrainer] ⚠️ 优先级集成器没有prioritized_system属性")
            else:
                print(f"[IntegratedPreferenceTrainer] ℹ️ 优先级集成器尚未设置或偏好模型尚未就绪")
            
            print(f"[IntegratedPreferenceTrainer] ✅ 优化偏好系统初始化成功")
            print(f"  - 偏好集成器: {type(self.preference_integrator).__name__}")
            print(f"  - 偏好训练器: {type(self.preference_trainer).__name__}")
            print(f"  - 智能体集成: {'是' if hasattr(self.agent, 'preference_integrator') else '否'}")
            
        except Exception as e:
            print(f"[ERROR] 优化偏好系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.preference_integrator = None
            self.preference_trainer = None
            self.latent_preference_integrator = None
    
    def _collect_step_data(self, obs, action, reward, done=False):
        """收集单步数据到偏好学习系统"""
        if self.historical_preference_manager is None and self.preference_integrator is None and self.prioritized_integrator is None:
            return
        
        start_time = time.time()
        
        try:
            # 添加到当前episode缓冲
            self.current_episode_obs.append(obs.copy() if isinstance(obs, np.ndarray) else np.array(obs))
            self.current_episode_actions.append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
            self.current_episode_rewards.append(reward)
            
            # 如果有优先级集成器，收集步骤数据
            if self.prioritized_integrator and self.current_episode_id is not None:
                try:
                    # 确保obs和action是numpy数组
                    obs_np = obs.copy() if isinstance(obs, np.ndarray) else np.array(obs.cpu().numpy() if hasattr(obs, 'cpu') else obs)
                    action_np = action.copy() if isinstance(action, np.ndarray) else np.array(action.cpu().numpy() if hasattr(action, 'cpu') else action)
                    
                    self.prioritized_integrator.collect_step(
                        self.current_episode_id, obs_np, action_np, reward, done
                    )
                except Exception as e:
                    print(f"[WARNING] 优先级集成器collect_step失败: {e}")
            
            # 如果有潜空间偏好系统，收集潜空间状态
            if self.latent_preference_integrator and hasattr(self.agent, 'model') and hasattr(self.agent.model, 'encode'):
                try:
                    # 将观测转换为潜空间状态
                    with torch.no_grad():
                        if isinstance(obs, np.ndarray):
                            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                        else:
                            obs_tensor = obs.float().unsqueeze(0) if obs.dim() == 1 else obs.float()
                        
                        # 移动到正确的设备
                        obs_tensor = obs_tensor.to(self.agent.device)
                        
                        # 使用TD-MPC2的encode方法获取潜空间表示
                        # 对于单任务场景，task参数为None
                        task = None
                        if hasattr(self.cfg, 'multitask') and self.cfg.multitask:
                            # 多任务场景下需要提供task参数
                            task = torch.tensor([0], device=self.agent.device)  # 默认使用第一个任务
                        
                        latent_state = self.agent.model.encode(obs_tensor, task)
                        if latent_state.dim() > 1:
                            latent_state = latent_state.squeeze(0)  # 移除批次维度
                        
                        self.current_episode_latent_states.append(latent_state.cpu())
                        
                except Exception as e:
                    # 潜空间转换失败不影响训练
                    if self._step % 1000 == 0:
                        print(f"[WARNING] 潜空间状态转换失败: {e}")
            
            # 原始偏好学习流程已禁用，不再向历史偏好管理器添加数据
            # if self.historical_preference_manager:
            #     self.historical_preference_manager.add_step_data(...)
            
            # 记录性能
            collection_time = time.time() - start_time
            self.performance_monitor['data_collection_time'].append(collection_time)
            
        except Exception as e:
            print(f"[WARNING] 收集步骤数据失败: {e}")
    
    def _finalize_episode_data(self):
        """完成episode数据收集"""
        # 如果有优先级集成器，调用其end_episode方法
        if self.prioritized_integrator and self.current_episode_id is not None:
            try:
                self.prioritized_integrator.end_episode(self.current_episode_id)
            except Exception as e:
                pass  # 静默处理失败
        
        # 清空缓冲区
        self.current_episode_obs.clear()
        self.current_episode_actions.clear()
        self.current_episode_rewards.clear()
        self.current_episode_latent_states.clear()
        
        # 重置episode_id
        self.current_episode_id = None
    
    def _check_and_create_preference_model(self):
        """检查并创建偏好模型 - 已禁用原始偏好学习流程"""
        # 原始偏好学习流程已禁用，只使用优先级偏好系统
        pass
    
    def _update_agent_preference_model(self):
        """更新智能体的偏好模型"""
        if self.historical_preference_manager is None:
            return
        
        try:
            # 获取最新的偏好模型
            preference_model = self.historical_preference_manager.get_preference_model()
            
            if preference_model is not None and hasattr(self.agent, 'update_preference_model'):
                print(f"[IntegratedPreferenceTrainer] 更新智能体偏好模型...")
                self.agent.update_preference_model(preference_model)
                
                # 启用偏好感知规划
                if hasattr(self.agent, 'enable_preference_planning'):
                    self.agent.enable_preference_planning = True
                    print(f"[IntegratedPreferenceTrainer] ✅ 偏好感知规划已启用")
                
                print(f"[IntegratedPreferenceTrainer] ✅ 智能体偏好模型更新完成")
            else:
                print(f"[IntegratedPreferenceTrainer] ⚠️ 无法更新智能体偏好模型 (模型为空或智能体不支持)")
                
        except Exception as e:
            print(f"[ERROR] 更新智能体偏好模型失败: {e}")
            import traceback
            traceback.print_exc()
    
    # _get_preference_reward方法已移除（恢复原始TD-MPC2流程）
    
    def _log_preference_stats(self):
        """记录偏好学习统计信息"""
        if self.historical_preference_manager is None:
            return
        
        try:
            # 获取统计信息
            manager_stats = self.historical_preference_manager.get_stats()
            
            # 合并统计信息
            combined_stats = {
                **self.preference_stats,
                **manager_stats,
                'performance': {
                    'avg_data_collection_time': np.mean(self.performance_monitor['data_collection_time']) if self.performance_monitor['data_collection_time'] else 0,
                    'avg_model_training_time': np.mean(self.performance_monitor['model_training_time']) if self.performance_monitor['model_training_time'] else 0,
                    'avg_reward_fusion_time': np.mean(self.performance_monitor['reward_fusion_time']) if self.performance_monitor.get('reward_fusion_time') else 0,
                    'avg_latent_preference_computation_time': np.mean(self.performance_monitor['latent_preference_computation_time']) if self.performance_monitor.get('latent_preference_computation_time') else 0,
                }
            }
            
            # 记录到日志
            if hasattr(self, 'logger'):
                # 使用 train category 代替 preference_stats
                self.logger.log(combined_stats, "train")
            
            # 精简统计信息打印（每500个episode打印一次）
            if self._ep_idx % 500 == 0:
                print(f"\n=== 偏好学习统计 (Episode {self._ep_idx}) ===")
                print(f"数据收集: {self.preference_stats['historical_data_collections']}, 模型更新: {self.preference_stats['preference_model_updates']}, 缓存轨迹: {manager_stats.get('total_trajectories', 0)}")
                if self.latent_preference_integrator:
                    print(f"奖励融合: {self.preference_stats.get('reward_fusions', 0)} 次")
                    
                    if self.preference_stats.get('reward_fusions', 0) > 0:
                        # 计算平均奖励
                        avg_pref_reward = self.preference_stats['total_preference_reward'] / self.preference_stats['reward_fusions']
                        avg_env_reward = self.preference_stats['total_environment_reward'] / self.preference_stats['reward_fusions']
                        avg_integrated_reward = self.preference_stats['total_integrated_reward'] / self.preference_stats['reward_fusions']
                        
                        print(f"  📊 平均偏好奖励: {avg_pref_reward:.4f}")
                        print(f"  📊 平均环境奖励: {avg_env_reward:.4f}")
                        print(f"  📊 平均集成奖励: {avg_integrated_reward:.4f}")
                        print(f"  📈 奖励提升: {avg_integrated_reward - avg_env_reward:+.4f}")
                        
                        # 融合效果分析
                        improvement_ratio = (avg_integrated_reward - avg_env_reward) / abs(avg_env_reward) * 100 if avg_env_reward != 0 else 0
                        if improvement_ratio > 1:
                            print(f"  ✅ 偏好系统显著提升性能 (+{improvement_ratio:.2f}%)")
                        elif improvement_ratio > 0:
                            print(f"  ✅ 偏好系统轻微提升性能 (+{improvement_ratio:.2f}%)")
                        elif improvement_ratio < -1:
                            print(f"  ⚠️ 偏好系统显著降低性能 ({improvement_ratio:.2f}%)")
                        else:
                            print(f"  ➖ 偏好系统影响微弱 ({improvement_ratio:.2f}%)")
                    else:
                        print(f"  ⚠️ 尚未进行奖励融合")
                    
                    # 奖励融合性能统计
                    if self.performance_monitor.get('reward_fusion_time'):
                        avg_fusion_time = np.mean(self.performance_monitor['reward_fusion_time'])
                        print(f"  ⏱️ 平均奖励融合时间: {avg_fusion_time*1000:.2f}ms")
                        print(f"  📊 融合频率: {len(self.performance_monitor['reward_fusion_time'])/100:.2f} 次/episode")
                
                print("=" * 60)
                
        except Exception as e:
            print(f"[WARNING] 记录偏好学习统计信息失败: {e}")
    
    def _cleanup_cache_if_needed(self):
        """根据需要清理缓存"""
        if self.historical_preference_manager is None:
            return
        
        try:
            # 每1000个episode清理一次缓存
            if self._ep_idx % 1000 == 0 and self._ep_idx > 0:
                print(f"[IntegratedPreferenceTrainer] 定期清理缓存 (Episode {self._ep_idx})")
                self.historical_preference_manager.cleanup_cache()
                
        except Exception as e:
            print(f"[WARNING] 清理缓存失败: {e}")
    
    def train(self):
        """主训练循环（重写父类方法以集成偏好学习）"""
        print(f"[IntegratedPreferenceTrainer] 开始训练 - 集成偏好学习模式")
        print(f"  - 偏好学习系统: {'启用' if self.historical_preference_manager else '禁用'}")
        
        # 使用父类的训练逻辑，但在关键点插入偏好学习功能
        train_metrics, done, eval_next = {}, True, True
        
        while self._step <= self.cfg.steps:
            # 评估
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # 重置环境
            if done:
                if eval_next:
                    eval_metrics = self.eval()  # 不保存视频
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    # 计算episode奖励（如果环境没有跟踪episode奖励）
                    episode_env_reward = getattr(self, '_episode_env_reward', 0.0)
                    
                    if episode_env_reward == 0.0:
                        episode_env_reward = torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum().item()
                    
                    # 计算集成奖励统计
                    episode_integrated_reward = getattr(self, '_episode_integrated_reward', episode_env_reward)
                    
                    # === Episode结束时的双路径奖励融合统计 ===
                    if self.latent_preference_integrator and hasattr(self, '_episode_integrated_reward'):
                        reward_difference = episode_integrated_reward - episode_env_reward
                        
                        # 输出episode融合汇总（每个episode结束时输出一次）
                        if hasattr(self, '_episode_fusion_data') and self._episode_fusion_data['fusion_count'] > 0:
                            data = self._episode_fusion_data
                            count = data['fusion_count']
                            
                            # 计算平均值
                            avg_env_reward = data['total_env_reward'] / count
                            avg_pref_reward = data['total_pref_reward'] / count
                            avg_integrated_reward = data['total_integrated_reward'] / count
                            avg_fusion_time = data['total_fusion_time'] / count
                            avg_confidence = data['avg_confidence'] / count
                            avg_pref_weight = data['avg_pref_weight'] / count
                            avg_env_weight = data['avg_env_weight'] / count
                            print(f"  🌍 平均环境奖励: {avg_env_reward:.4f}")
                            print(f"  🧠 平均偏好奖励: {avg_pref_reward:.4f}")
                            print(f"  🎯 平均置信度: {avg_confidence:.4f}")
                            print(f"  ⏱️ 平均融合耗时: {avg_fusion_time*1000:.2f}ms")
                            print(f"  ✅ 正向偏好: {data['positive_preference_count']} | ⚠️ 负向偏好: {data['negative_preference_count']}")
                            
                            # 智能融合状态指示
                            if avg_pref_reward > 0.01:
                                print(f"  ✅ Episode整体符合偏好 (+{avg_pref_reward:.4f})")
                            elif avg_pref_reward < -0.01:
                                print(f"  ⚠️ Episode整体偏离偏好 ({avg_pref_reward:.4f})")
                            else:
                                print(f"  ➖ Episode偏好信号微弱")
                            
                            # 重置episode融合数据
                            self._episode_fusion_data = {
                                'fusion_count': 0,
                                'total_env_reward': 0.0,
                                'total_pref_reward': 0.0,
                                'total_integrated_reward': 0.0,
                                'total_fusion_time': 0.0,
                                'positive_preference_count': 0,
                                'negative_preference_count': 0,
                                'avg_confidence': 0.0,
                                'avg_pref_weight': 0.0,
                                'avg_env_weight': 0.0
                            }
                    
                    # 完成episode数据收集（偏好学习）
                    self._finalize_episode_data()
                    
                    # 检查是否需要创建偏好模型
                    self._check_and_create_preference_model()
                    
                    # 记录偏好学习统计信息  
                    self._log_preference_stats()
                    
                    # 清理缓存
                    self._cleanup_cache_if_needed()
                    
                    # 原有的episode结束处理（恢复原始TD-MPC2流程）
                    
                    train_metrics.update(
                        episode_env_reward=episode_env_reward,
                        episode_success=info.get("success", False),
                    )
                    train_metrics.update(self.common_metrics())

                    results_metrics = {
                        'return': train_metrics['episode_env_reward'],
                        'episode_length': len(self._tds[1:]),
                        'success': train_metrics['episode_success'],
                        'success_subtasks': info.get('success_subtasks', []),
                        'step': self._step,
                    }
                
                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                # 结束上一个episode（如果有的话）
                if hasattr(self, '_step') and self._step > 0:
                    self._finalize_episode_data()
                    # 更新episode计数器
                    self.episodes_completed += 1
                    print(f"[IntegratedPreferenceTrainer] Episode {self.episodes_completed} 完成")
                
                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]
                
                # 开始新的episode
                if self.prioritized_integrator:
                    try:
                        self.current_episode_id = self.prioritized_integrator.start_episode()
                        # 重置轨迹级别的偏好统计
                        self.prioritized_integrator.reset_trajectory_stats()
                        print(f"[IntegratedPreferenceTrainer] 🚀 开始新episode: {self.current_episode_id}")
                    except Exception as e:
                        print(f"[WARNING] 优先级集成器start_episode失败: {e}")
                        self.current_episode_id = None
                
                # 初始化episode奖励记录（恢复原始TD-MPC2流程）
                self._episode_env_reward = 0.0
                if hasattr(self, '_episode_integrated_reward'):
                    self._episode_integrated_reward = 0.0

            # 收集经验
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
                
                # 移除混合价值估计相关的日志输出（恢复原始TD-MPC2流程）
                # 不再记录混合价值估计使用情况
            else:
                action = self.env.rand_act()
            
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            
            # 收集步骤数据到偏好学习系统
            self._collect_step_data(obs, action, reward, done)
            
            # === 双路径奖励融合处理 ===
            final_reward = reward  # 默认使用环境奖励
            preference_reward = 0.0
            confidence = 0.0
            integrated_reward = reward
            
            # 如果潜空间偏好系统可用且有足够的历史数据
            if (self.preference_integrator and 
                len(self.current_episode_latent_states) > 0 and 
                len(self.current_episode_actions) > 0):
                
                try:
                    fusion_start_time = time.time()
                    
                    # 获取当前序列的潜空间状态和动作
                    if len(self.current_episode_latent_states) >= 1:
                        # 使用最近的状态序列进行偏好奖励计算
                        seq_len = min(20, len(self.current_episode_latent_states))  # 使用最近20步
                        recent_latent_states = torch.stack(self.current_episode_latent_states[-seq_len:])
                        recent_actions = torch.stack([torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a.float() 
                                                    for a in self.current_episode_actions[-seq_len:]])
                        
                        # 计算集成奖励 - 使用最新的状态和动作
                        latest_latent_state = recent_latent_states[-1]  # 取最新的状态
                        latest_action = recent_actions[-1]  # 取最新的动作
                        
                        # 确保张量在正确的设备上
                        device = self.agent.device if hasattr(self.agent, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
                        latest_latent_state = latest_latent_state.to(device)
                        latest_action = latest_action.to(device)
                        
                        reward_details = self.latent_preference_integrator.compute_integrated_reward(
                            latent_state=latest_latent_state,
                            action=latest_action,
                            environment_reward=reward
                        )
                        integrated_reward = reward_details['integrated_reward']
                        
                        # 提取详细信息
                        preference_reward = reward_details['preference_reward']
                        confidence = reward_details['confidence']
                        final_reward = integrated_reward
                        
                        # 更新统计信息
                        self.preference_stats['latent_preference_computations'] += 1
                        self.preference_stats['reward_fusions'] += 1
                        self.preference_stats['total_preference_reward'] += preference_reward
                        self.preference_stats['total_environment_reward'] += reward
                        self.preference_stats['total_integrated_reward'] += integrated_reward
                        
                        # 记录性能
                        fusion_time = time.time() - fusion_start_time
                        self.performance_monitor['reward_fusion_time'].append(fusion_time)
                        
                        # 累积episode内的融合数据，只在episode结束时输出汇总
                        if not hasattr(self, '_episode_fusion_data'):
                            self._episode_fusion_data = {
                                'fusion_count': 0,
                                'total_env_reward': 0.0,
                                'total_pref_reward': 0.0,
                                'total_integrated_reward': 0.0,
                                'total_fusion_time': 0.0,
                                'positive_preference_count': 0,
                                'negative_preference_count': 0,
                                'avg_confidence': 0.0,
                                'avg_pref_weight': 0.0,
                                'avg_env_weight': 0.0
                            }
                        
                        # 累积数据
                        self._episode_fusion_data['fusion_count'] += 1
                        self._episode_fusion_data['total_env_reward'] += reward
                        self._episode_fusion_data['total_pref_reward'] += preference_reward
                        self._episode_fusion_data['total_integrated_reward'] += integrated_reward
                        self._episode_fusion_data['total_fusion_time'] += fusion_time
                        self._episode_fusion_data['avg_confidence'] += confidence
                        self._episode_fusion_data['avg_pref_weight'] += reward_details['preference_weight']
                        self._episode_fusion_data['avg_env_weight'] += reward_details['environment_weight']
                        
                        # 偏好分类统计（基于正负数判断）
                        # 正数偏好奖励计为正向偏好+1，负数偏好奖励计为负向偏好+1
                        if preference_reward > 0:
                            self._episode_fusion_data['positive_preference_count'] += 1
                        elif preference_reward < 0:
                            self._episode_fusion_data['negative_preference_count'] += 1
                        
                except Exception as e:
                    # 奖励融合失败时回退到环境奖励
                    # 静默处理融合失败，使用环境奖励
                    final_reward = reward
            
            # 更新episode奖励统计
            self._episode_env_reward += reward  # 环境奖励统计
            if hasattr(self, '_episode_integrated_reward'):
                self._episode_integrated_reward += final_reward
            else:
                self._episode_integrated_reward = final_reward
            
            # 使用融合后的奖励进行训练
            self._tds.append(self.to_td(obs, action, final_reward))
            
            # 更新步数
            self._step += 1
            
            # 定期保存模型
            save_freq = getattr(self.cfg, 'save_freq', 100000)
            save_agent = getattr(self.cfg, 'save_agent', True)
            if save_agent and self._step > 0 and self._step % save_freq == 0:
                checkpoint_path = os.path.join(self.cfg.work_dir, f'checkpoint_step_{self._step}.pt')
                try:
                    self.agent.save(checkpoint_path)
                    # 同时保存一个最新的检查点
                    latest_path = os.path.join(self.cfg.work_dir, 'latest_checkpoint.pt')
                    self.agent.save(latest_path)
                except Exception as e:
                    pass  # 静默处理保存失败
            
            # 训练智能体
            if self._step > self.cfg.seed_steps and len(self._tds) > 1:
                if hasattr(self.agent, 'update'):
                    _train_metrics = self.agent.update(self.buffer)
                    train_metrics.update(_train_metrics)
                
                # 检查并训练优先级偏好模型（基于episode频率控制）
                # 只在episode结束时检查，避免每个训练步骤都检查
                if self.prioritized_integrator and done:
                    try:
                        # 获取训练频率配置
                        train_every_n_episodes = getattr(self.cfg, 'train_every_n_episodes', 10)
                        
                        # 检查是否到了训练时机
                        episodes_since_last_training = self.episodes_completed - self.last_preference_training_episode
                        should_check_training = episodes_since_last_training >= train_every_n_episodes
                        
                        if should_check_training:
                            if self.prioritized_integrator.should_train_preference_model():
                                
                                # 获取偏好模型实例
                                preference_model = None
                                if self.preference_trainer and hasattr(self.preference_trainer, 'models') and len(self.preference_trainer.models) > 0:
                                    preference_model = self.preference_trainer.models[0]  # 使用第一个模型
                                    print(f"[IntegratedPreferenceTrainer] 传递偏好模型: {type(preference_model).__name__}")
                                else:
                                    print(f"[IntegratedPreferenceTrainer] ⚠️ 未找到偏好模型实例")
                                
                                preference_metrics = self.prioritized_integrator.train_preference_model(preference_model)
                                if preference_metrics:
                                    train_metrics.update(preference_metrics)
                                    print(f"[IntegratedPreferenceTrainer] ✅ 优先级偏好模型训练完成")
                                    # 更新最后训练的episode
                                    self.last_preference_training_episode = self.episodes_completed
                            else:
                                print(f"[IntegratedPreferenceTrainer] Episode {self.episodes_completed}: 训练条件未满足，跳过偏好模型训练")
                    except Exception as e:
                        print(f"[WARNING] 优先级偏好模型训练失败: {e}")
                        import traceback
                        traceback.print_exc()
        
        # 训练结束时，结束最后一个episode
        if self.current_episode_id is not None:
            self._finalize_episode_data()
        
        # 训练完成
    
    def save(self, fp):
        """保存模型（重写以包含偏好学习状态）"""
        # 如果父类有save方法则调用，否则跳过
        if hasattr(super(), 'save'):
            super().save(fp)
        
        try:
            # 保存偏好学习统计信息到日志
            if self.historical_preference_manager:
                stats = self.historical_preference_manager.get_stats()
                print(f"[IntegratedPreferenceTrainer] 保存时的偏好学习状态:")
                print(f"  - 缓存轨迹数: {stats.get('total_trajectories', 0)}")
                print(f"  - 缓存偏好对数: {stats.get('total_preference_pairs', 0)}")
                print(f"  - 模型版本: {stats.get('model_info', {}).get('version', 0) if stats.get('model_info') else 0}")
                print(f"  - 内存使用: {stats.get('memory_usage_estimate', 'N/A')}")
                
        except Exception as e:
            print(f"[WARNING] 保存偏好学习状态失败: {e}")
    
    def eval(self):
        """评估模式（恢复原始TD-MPC2流程）"""
        # 直接调用父类评估，不添加偏好相关指标
        return super().eval()
    
    def get_preference_stats(self) -> Dict[str, Any]:
        """获取偏好学习统计信息"""
        if self.historical_preference_manager is None:
            return {'preference_learning_enabled': False}
        
        try:
            manager_stats = self.historical_preference_manager.get_stats()
            combined_stats = {
                'preference_learning_enabled': True,
                **self.preference_stats,
                **manager_stats,
                'performance_metrics': {
                    'data_collection_times': list(self.performance_monitor['data_collection_time']),
                    'model_training_times': list(self.performance_monitor['model_training_time']),
                    'reward_fusion_times': list(self.performance_monitor.get('reward_fusion_time', [])),
                    'latent_preference_computation_times': list(self.performance_monitor.get('latent_preference_computation_time', [])),
                }
            }
            return combined_stats
            
        except Exception as e:
            print(f"[WARNING] 获取偏好学习统计信息失败: {e}")
            return {'preference_learning_enabled': True, 'error': str(e)}

def create_integrated_preference_trainer(*args, **kwargs) -> IntegratedPreferenceTrainer:
    """创建集成偏好学习训练器的便捷函数"""
    return IntegratedPreferenceTrainer(*args, **kwargs)

# 示例用法和测试
if __name__ == "__main__":
    print("IntegratedPreferenceTrainer - 内存缓存版本")
    print("该模块已移除所有文件IO操作，使用纯内存缓存")
    print("特点:")
    print("1. 零文件IO开销")
    print("2. 高性能内存缓存")
    print("3. 自动缓存管理")
    print("4. 实时统计监控")
    print("5. 无缝集成TD-MPC2训练流程")
