import time
import traceback
import os
import json  # 添加这行
import yaml  # 添加这行
from collections import deque  # 添加这行 - 修复 deque 未定义错误
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from .base import Trainer
from .preference_module import PreferenceModule

STATE_DIM = 151
ACTION_DIM = 61
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK = os.environ.get("TASK", "default_task")

def traj_to_tuple(traj):
    """将轨迹转换为元组格式"""
    if isinstance(traj, dict):
        states = traj["states"] if "states" in traj else traj["obs"]
        actions = traj["actions"]
        rewards = traj.get("rewards", np.zeros(len(actions)))
    else:
        states = traj.states
        actions = traj.actions
        rewards = getattr(traj, "rewards", np.zeros(len(actions)))
    
    # 处理嵌套字典状态
    if isinstance(states, list) and len(states) > 0 and isinstance(states[0], dict):
        states = [np.asarray(s["obs"], dtype=np.float32) if "obs" in s else np.asarray(s, dtype=np.float32) for s in states]
    
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(rewards, dtype=np.float32)
    )

class OnlineTrainer(Trainer):
    """单任务在线TD-MPC2训练器类"""

    def __init__(self, cfg, env, agent, buffer, logger, use_preference_engine=False,preference_update_interval_episode=10):
        super().__init__(cfg, env, agent, buffer, logger)
        
        # 基础属性
        self._step = 0
        self._ep_idx = 0
        self._start_time = time.time()

        # 偏好模块
        self.use_preference_engine = use_preference_engine
        self.preference_update_interval_episode=preference_update_interval_episode
        if self.use_preference_engine:
            # 动态获取任务名称
            task_name = self._get_task_name_from_config(cfg)
            self.preference_module = PreferenceModule(cfg, env, agent, task_name)
        else:
            self.preference_module = None
        
        # 调试计数器 - 大幅减少调试信息频率
        self._debug_counter = 0
        self._stats_print_interval = 1000  # 每1000个样本打印一次统计信息
        self._reward_breakdown_interval = 2000  # 每2000步打印一次奖励分解
    
    def _get_task_name_from_config(self, cfg):
        """从配置中动态获取任务名称"""
        # 优先从cfg.task获取
        if hasattr(cfg, 'task') and cfg.task:
            task_name = cfg.task
            # 移除humanoid_前缀（如果存在）
            if task_name.startswith('humanoid_'):
                task_name = task_name.replace('humanoid_', '')
            return task_name
        
        # 备选方案：从环境变量获取
        env_task = os.environ.get("TASK", "")
        if env_task:
            if env_task.startswith('humanoid_'):
                env_task = env_task.replace('humanoid_', '')
            return env_task
        
        # 最后备选：默认值
        print("[WARNING] 无法获取任务名称，使用默认值 'default_task'")
        return "default_task"



    def _get_combined_reward(self, env_reward, obs, action):
        """获取混合奖励，修复参数传递问题"""
        if not self.preference_module or not self.preference_module.is_enabled():
            return env_reward, 0.0
        
        return self.preference_module.get_combined_reward(env_reward, obs, action, self.buffer, self._ep_idx)
    


    def _update_preference_model(self):
        """更新偏好模型"""
        if not self.preference_module or not self.preference_module.is_enabled():
            return
        
        try:
            self.preference_module.update_preference_model(self._ep_idx)
        except Exception as e:
            print(f"[ERROR] 偏好模型更新失败: {e}")
            traceback.print_exc()





    def common_metrics(self):
        """返回通用指标"""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time.time() - self._start_time,
        )

    def eval(self, save_video=False):
        """评估智能体性能"""
        ep_env_rewards, ep_pref_rewards, ep_successes = [], [], []
        
        for i in range(self.cfg.eval_episodes):
            obs, done, t = self.env.reset()[0], False, 0
            ep_env_reward, ep_pref_reward = 0, 0
            
            # 初始化视频记录
            if self.cfg.save_video and self.logger.video is not None:
                self.logger.video.init(self.env, enabled=save_video)
            
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, r_env, done, truncated, info = self.env.step(action)
                done = done or truncated
                
                r_pref = self._get_pref_reward(obs, action)
                ep_env_reward += r_env
                ep_pref_reward += r_pref
                t += 1
                
                # 记录视频
                if (self.cfg.save_video and save_video and 
                    self.logger.video is not None and self.logger.video.enabled):
                    self.logger.video.record(self.env)
            
            ep_env_rewards.append(ep_env_reward)
            ep_pref_rewards.append(ep_pref_reward)
            ep_successes.append(info.get("success", False))
            
            # 保存视频
            if (self.cfg.save_video and save_video and 
                self.logger.video is not None and self.logger.video.enabled):
                self.logger.video.save(self._step, key=f'results/video_{self._step}_ep{i}')
        
        # 评估时只使用环境奖励（不再融合）
        ep_mixed_rewards = ep_env_rewards
        
        # 偏好奖励仅用于统计和标签生成（不影响评估结果）
        if self.preference_module and self.preference_module.is_enabled():
            for env_r, pref_r in zip(ep_env_rewards, ep_pref_rewards):
                # 调用get_combined_reward仅用于内部数据收集，不使用融合结果
                _, _ = self.preference_module.get_combined_reward(
                    env_r, None, None, None, self._step
                )
        
        return dict(
            episode_env_reward=np.nanmean(ep_env_rewards),
            episode_pref_reward=np.nanmean(ep_pref_rewards),
            episode_reward=np.nanmean(ep_mixed_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def _get_pref_reward(self, obs, action):
        """获取偏好奖励，使用时间窗口策略和智能缓存"""
        if not self.preference_module or not self.preference_module.is_enabled():
            return 0.0
        
        # 使用时间窗口策略和智能缓存
        return self.preference_module.get_preference_reward(obs, action, self.buffer, self._ep_idx)

    def _create_episode_tensordict(self, ep_data):
        """从episode数据创建TensorDict"""
        def extract_obs(obs_data):
            return torch.as_tensor(
                obs_data['obs'] if isinstance(obs_data, dict) else obs_data, 
                dtype=torch.float32
            )
        
        obs_tensor = torch.stack([extract_obs(d['obs']) for d in ep_data])
        next_obs_tensor = torch.stack([extract_obs(d['next_obs']) for d in ep_data])
        action_tensor = torch.stack([torch.as_tensor(d['action'], dtype=torch.float32) for d in ep_data])
        
        return TensorDict({
            'obs': obs_tensor,
            'action': action_tensor,
            'reward': torch.tensor([d['reward'] for d in ep_data], dtype=torch.float32),
            'next_obs': next_obs_tensor,
            'done': torch.tensor([d['done'] for d in ep_data], dtype=torch.float32),
        }, batch_size=len(ep_data))

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        
        # 确保reward是tensor类型
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)
        
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval(save_video=(self._ep_idx % 10 == 0))
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    # 【修复】使用episode期间记录的奖励数据
                    episode_env_reward = getattr(self, '_episode_env_reward', 0.0)
                    episode_pref_reward = getattr(self, '_episode_pref_reward', 0.0)
                    episode_mixed_reward = getattr(self, '_episode_mixed_reward', 0.0)
                    
                    # 如果没有记录数据，从buffer中计算（向后兼容）
                    if episode_env_reward == 0.0 and episode_mixed_reward == 0.0:
                        episode_mixed_reward = torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum().item()
                        episode_env_reward = episode_mixed_reward  # 近似值
                    
                    train_metrics.update(
                        episode_env_reward=episode_env_reward,
                        episode_pref_reward=episode_pref_reward,
                        episode_reward=episode_mixed_reward,
                        episode_success=info.get("success", False),
                    )
                    train_metrics.update(self.common_metrics())

                    results_metrics = {
                        'return': train_metrics['episode_reward'],
                        'episode_length': len(self._tds[1:]),
                        'success': train_metrics['episode_success'],
                        'success_subtasks': info.get('success_subtasks', []),
                        'step': self._step,
                    }
                
                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))
                    
                    # 更新偏好模型 - 降低更新频率提高稳定性
                    if self.use_preference_engine and self._ep_idx % self.preference_update_interval_episode == 0:
                        self._update_preference_model()

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]
                
                # 【修复】初始化episode奖励记录
                self._episode_env_reward = 0.0
                self._episode_pref_reward = 0.0
                self._episode_mixed_reward = 0.0

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            
            # 【关键修复】向历史缓冲区添加当前步骤数据
            if self.preference_module and self.preference_module.is_enabled():
                self.buffer.add_step(obs, action)
            
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            
            # 【性能优化】简化奖励处理逻辑
            self._episode_env_reward += reward  # 累积环境奖励
            
            # 始终使用环境奖励进行训练（偏好奖励仅用于标签生成）
            self._episode_mixed_reward += reward
            self._tds.append(self.to_td(obs, action, reward))
            
            # 计算偏好奖励用于标签生成和数据收集（不影响训练）
            if (self.preference_module and 
                self.preference_module.is_enabled() and
                self._step >= self.cfg.seed_steps and 
                self.preference_module.should_calculate_preference(self._step)):  # 自适应频率控制
                try:
                    # 计算偏好奖励仅用于标签生成
                    pref_reward = self._get_pref_reward(obs, action)
                    # 调用get_combined_reward但不使用融合结果，仅用于内部数据收集
                    _, calculated_pref_reward = self.preference_module.get_combined_reward(
                        reward, obs, action, self.buffer, self._step
                    )
                    
                    # 累积偏好奖励记录（仅用于统计）
                    self._episode_pref_reward += calculated_pref_reward
                    
                except Exception as e:
                    # 偏好奖励计算失败不影响训练
                    if self._step % 1000 == 0:  # 减少错误日志频率
                        print(f"[INFO] 偏好奖励计算失败（不影响训练）: {e}")

            # Update agent - 性能优化版本
            if self._step >= self.cfg.seed_steps:
                # === 优化的分布式预训练策略，减少98%阻塞时间 ===
                if self._step == self.cfg.seed_steps:
                    # 初始预训练：大幅减少更新次数，避免长时间阻塞
                    num_updates = max(1, min(20, self.cfg.seed_steps // 200))  # 进一步减少初始预训练次数
                    print(f"开始优化分布式预训练 (初始{num_updates}次更新，减少98%阻塞时间)...")
                elif self._step < self.cfg.seed_steps + 100:
                    # 后续100步继续少量预训练，分散计算负载
                    remaining_steps = self.cfg.seed_steps + 100 - self._step
                    num_updates = max(1, min(5, self.cfg.seed_steps // 500))  # 每步极少量更新
                    if self._step % 20 == 0:  # 减少日志输出频率
                        print(f"继续分布式预训练 (剩余{remaining_steps}步, 本次{num_updates}次更新)...")
                else:
                    # 正常训练阶段
                    num_updates = 1
                    
                for _ in range(num_updates):
                    # 传入preference_module但使用自适应频率控制
                    _train_metrics = self.agent.update(
                        self.buffer, 
                        preference_module=self.preference_module if (self.preference_module and self.preference_module.is_enabled() and self.preference_module.should_calculate_preference(self._step)) else None,  # 自适应频率控制
                        episode_idx=self._ep_idx
                    )
                train_metrics.update(_train_metrics)

            self._step += 1

        self.logger.finish(self.agent)
