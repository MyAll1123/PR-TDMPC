import time
import traceback
import os
import json
import yaml
from collections import deque
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from .base import Trainer

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
    """单任务在线TD-MPC2训练器类 - 干净版本（无preference_module）"""

    def __init__(self, cfg, env, agent, buffer, logger, use_preference_engine=False):
        super().__init__(cfg, env, agent, buffer, logger)
        
        # 基础属性
        self._step = 0
        self._ep_idx = 0
        self._start_time = time.time()

        # 注意：use_preference_engine参数保留但不使用，仅为兼容性
        
        # 调试计数器
        self._debug_counter = 0
        self._stats_print_interval = 1000
        self._reward_breakdown_interval = 2000
    
    def eval(self, save_video=False):
        """评估智能体性能"""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs = self.env.reset()[0]
            done, ep_reward, t = False, 0, 0
            if save_video:
                self.logger.log_video_start()
            while not done:
                action = self.agent.act(obs, t0=t==0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                if save_video:
                    self.logger.log_video_step(obs)
                t += 1
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get("success", False))
            if save_video:
                self.logger.log_video_end()
        
        return dict(
            episode_reward=np.mean(ep_rewards),
            episode_success=np.mean(ep_successes),
        )

    def common_metrics(self):
        """返回通用指标"""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time.time() - self._start_time,
        )

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
                    # 使用episode期间记录的奖励数据
                    episode_env_reward = getattr(self, '_episode_env_reward', 0.0)
                    episode_mixed_reward = getattr(self, '_episode_mixed_reward', 0.0)
                    
                    # 如果没有记录数据，从buffer中计算
                    if episode_env_reward == 0.0 and episode_mixed_reward == 0.0:
                        episode_mixed_reward = torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum().item()
                        episode_env_reward = episode_mixed_reward
                    
                    train_metrics.update(
                        episode_env_reward=episode_env_reward,
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

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]
                
                # 初始化episode奖励记录
                self._episode_env_reward = 0.0
                self._episode_mixed_reward = 0.0

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            
            # 处理奖励
            self._episode_env_reward += reward
            self._episode_mixed_reward += reward
            self._tds.append(self.to_td(obs, action, reward))
            
            self._step += 1

            # Train agent
            if self._step > self.cfg.seed_steps:
                num_updates = self.cfg.num_update_steps
                
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                    train_metrics.update(_train_metrics)
