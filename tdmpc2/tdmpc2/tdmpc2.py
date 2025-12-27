import numpy as np
import torch
import torch.nn.functional as F

from .common import math
from .common.scale import RunningScale
from .common.world_model import WorldModel

# 导入奖励尺度优化器
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../prm'))
    from reward_scale_optimizer import RewardScaleOptimizer
    REWARD_SCALE_OPTIMIZER_AVAILABLE = True
except ImportError:
    REWARD_SCALE_OPTIMIZER_AVAILABLE = False
    print("[WARNING] 奖励尺度优化器不可用，将使用原始RunningScale")


class TDMPC2:
    """
    TD-MPC2 代理类。实现了训练和推理功能。
    支持单任务和多任务实验，适用于状态和像素观测。
    """

    def __init__(self, cfg):
        """
        初始化 TD-MPC2 代理。

        参数:
            cfg: 配置对象，包含超参数和设置。
        """
        self.cfg = cfg
        # 根据硬件可用性选择设备（GPU 或 CPU）
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 初始化世界模型并移动到选定设备
        self.model = WorldModel(cfg).to(self.device)
        
        # 初始化世界模型的优化器
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
                {
                    "params": self.model._task_emb.parameters()
                    if self.cfg.multitask
                    else []
                },
            ],
            lr=self.cfg.lr,
        )
        
        # 初始化策略网络的优化器
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
        )
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 初始化归一化工具，用于奖励和 Q 值的归一化
        self.scale = RunningScale(cfg)
        
        # 初始化奖励尺度优化器（如果可用）
        if REWARD_SCALE_OPTIMIZER_AVAILABLE:
            # 检查配置中是否启用奖励尺度优化
            reward_scale_config = getattr(cfg, 'reward_scale_optimization', {})
            reward_scale_enabled = reward_scale_config.get('enabled', True)
            
            if reward_scale_enabled:
                # 从配置文件读取奖励尺度优化器配置
                from types import SimpleNamespace
                scale_cfg = SimpleNamespace()
                scale_cfg.scale_percentile = reward_scale_config.get('scale_percentile', 95)
                scale_cfg.scale_min = reward_scale_config.get('scale_min', 1e-6)
                scale_cfg.scale_max = reward_scale_config.get('scale_max', 1e6)
                scale_cfg.tau = reward_scale_config.get('scale_tau', 0.005)
                scale_cfg.pref_reward_scale = reward_scale_config.get('pref_reward_scale', 1.0)
                scale_cfg.adaptive_reward_scaling = reward_scale_config.get('adaptive_reward_scaling', True)
                scale_cfg.preference_loss_type = reward_scale_config.get('preference_loss_type', 'adaptive_logistic')
                scale_cfg.preference_loss_temperature = reward_scale_config.get('preference_loss_temperature', 1.0)
                scale_cfg.preference_loss_margin = reward_scale_config.get('preference_loss_margin', 1.0)
                scale_cfg.env_reward_weight = reward_scale_config.get('env_reward_weight', 0.7)
                scale_cfg.pref_reward_weight = reward_scale_config.get('pref_reward_weight', 0.3)
                scale_cfg.reward_adaptation_rate = reward_scale_config.get('reward_adaptation_rate', 0.01)
                scale_cfg.scale_adaptation_rate = reward_scale_config.get('scale_adaptation_rate', 0.01)
                scale_cfg.small_reward_threshold = reward_scale_config.get('small_reward_threshold', 0.01)
                scale_cfg.small_reward_scale_factor = reward_scale_config.get('small_reward_scale_factor', 0.1)
                
                self.reward_scale_optimizer = RewardScaleOptimizer(scale_cfg)
                print(f"[INFO] 奖励尺度优化器已启用，损失类型: {scale_cfg.preference_loss_type}")
            else:
                self.reward_scale_optimizer = None
                print("[INFO] 奖励尺度优化器已在配置中禁用")
        else:
            self.reward_scale_optimizer = None
        
        # 根据动作维度调整迭代次数（启发式）
        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)
        
        # 初始化折扣因子
        self.discount = (
            torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
                device=self.device,
                dtype=torch.float32,
            )
            if self.cfg.multitask
            else self._get_discount(cfg.episode_length)
        )
        self._prev_mean = None  # 确保有这一行

    def _get_discount(self, episode_length):
        """
        根据 episode 长度返回折扣因子。
        使用简单的线性缩放启发式方法。

        参数:
            episode_length (int): episode 的长度，假设是固定长度。

        返回:
            float: 任务的折扣因子。
        """
        frac = episode_length / self.cfg.discount_denom
        return min(
            max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
        )

    def save(self, fp):
        """
        将代理的状态字典保存到指定文件路径。

        参数:
            fp (str): 保存状态字典的文件路径。
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        从文件路径或字典加载保存的状态字典到当前代理。

        参数:
            fp (str 或 dict): 文件路径或状态字典。
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        在世界模型的潜在空间中规划动作。

        参数:
            obs (torch.Tensor): 环境的观测。
            t0 (bool): 是否为 episode 的第一个观测。
            eval_mode (bool): 是否使用动作分布的均值（评估模式）。
            task (int): 任务索引（仅用于多任务实验）。

        返回:
            torch.Tensor: 在环境中执行的动作。
        """
        # 将观测移动到设备并添加批次维度
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        
        # 如果提供了任务索引，将其转换为张量
        if task is not None:
            task = torch.tensor([task], device=self.device, dtype=torch.long)
        
        # 将观测编码到潜在空间
        z = self.model.encode(obs, task)
        
        # 使用 MPC 或策略网络选择动作
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """
        估计从潜在状态 z 开始执行指定动作序列的轨迹价值。

        参数:
            z (torch.Tensor): 初始潜在状态。
            actions (torch.Tensor): 要评估的动作序列。
            task (int): 任务索引（仅用于多任务实验）。

        返回:
            torch.Tensor: 轨迹的估计价值。
        """
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            # 计算奖励和下一个潜在状态
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            
            # 累积折扣奖励
            G += discount * reward
            discount *= (
                self.discount[torch.tensor(task, dtype=torch.long, device=self.device)]
                if self.cfg.multitask
                else self.discount
            )
        
        # 加上最终状态的价值
        return G + discount * self.model.Q(
            z, self.model.pi(z, task)[1], task, return_type="avg"
        )

    @torch.no_grad()
    def plan(self, z, t0=0, eval_mode=False, task=None):
        """
        使用学习到的世界模型规划动作序列。
        """
        # 采样策略轨迹
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # 初始化状态和参数
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(
            self.cfg.horizon, self.cfg.action_dim, device=self.device
        )
        # === 修正：确保 self._prev_mean 初始化在切片操作之前 ===
        if not t0:
            if self._prev_mean is None:
                self._prev_mean = mean.clone().detach()
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, : self.cfg.num_pi_trajs] = pi_actions

        # 使用 MPPI 优化动作序列
        for _ in range(self.cfg.iterations):
            actions[:, self.cfg.num_pi_trajs :] = (
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(
                    self.cfg.horizon,
                    self.cfg.num_samples - self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                    device=std.device,
                )
            ).clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1
                )
                / (score.sum(0) + 1e-9)
            ).clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        if self._prev_mean is None:
            self._prev_mean = mean.clone().detach()
        # 这里不需要再次切片赋值，因为已在上面处理
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1).cpu()

    def update_pi(self, zs, task):
        """
        使用潜在状态序列更新策略。

        参数:
            zs (torch.Tensor): 潜在状态序列。
            task (torch.Tensor): 任务索引（仅用于多任务实验）。

        返回:
            float: 策略更新的损失值。
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type="avg")
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # 损失是 Q 值的加权和
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        从奖励和下一时间步的观测计算 TD 目标。

        参数:
            next_z (torch.Tensor): 下一时间步的潜在状态。
            reward (torch.Tensor): 当前时间步的奖励。
            task (torch.Tensor): 任务索引（仅用于多任务实验）。

        返回:
            torch.Tensor: TD 目标。
        """
        pi = self.model.pi(next_z, task)[1]
        discount = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        return reward + discount * self.model.Q(
            next_z, pi, task, return_type="min", target=True
        )

    def update(self, buffer, preference_dataset=None, lambda_pref=0.0001, margin=0.01, pref_reward_scale=None, preference_module=None, episode_idx=0):
        """
        主更新函数。对应一次模型学习的迭代。

        参数:
            buffer (common.buffer.Buffer): 经验回放缓冲区。
            preference_dataset: 偏好辅助数据集
            lambda_pref: 辅助损失权重
            margin: 偏好排序损失间隔
            pref_reward_scale: 用于归一化辅助损失的尺度（如环境reward均值）
            preference_module: 偏好模块，用于计算偏好奖励
            episode_idx: 当前episode索引

        返回:
            dict: 包含训练统计信息的字典。
        """
        obs, action, reward, task = buffer.sample()
        
        # 计算融合奖励用于训练奖励预测器
        integrated_reward = reward.clone()  # 默认使用环境奖励
        
        # 使用奖励尺度优化器进行奖励归一化（如果可用）
        if self.reward_scale_optimizer is not None:
            # 更新环境奖励尺度
            self.reward_scale_optimizer.update_env_reward_scale(reward)
        
        if (preference_module is not None and 
            preference_module.is_enabled() and 
            episode_idx >= preference_module.preference_start_episode):
            
            try:
                # 计算融合奖励用于奖励预测器训练
                batch_size = obs.shape[1]  # [horizon+1, batch_size, ...]
                
                # 对批次中的每个样本计算融合奖励
                for b in range(min(batch_size, 8)):  # 限制计算量，只处理前8个样本
                    # 获取当前样本的状态和动作序列
                    sample_obs = obs[:, b]  # [horizon+1, obs_dim]
                    sample_action = action[:, b]  # [horizon, action_dim]
                    sample_reward = reward[:, b]  # [horizon]
                    
                    # 编码状态到潜空间
                    with torch.no_grad():
                        sample_z = self.model.encode(sample_obs[:-1], task[b] if task.dim() > 0 else task)  # [horizon, latent_dim]
                    
                    # 对每个时间步计算融合奖励
                    for t in range(sample_z.shape[0]):
                        try:
                            # 获取偏好集成器
                            if hasattr(preference_module, 'latent_preference_integrator') and preference_module.latent_preference_integrator:
                                integrator = preference_module.latent_preference_integrator
                                
                                # 计算融合奖励
                                reward_details = integrator.compute_integrated_reward(
                                    latent_state=sample_z[t],
                                    action=sample_action[t],
                                    environment_reward=sample_reward[t].item()
                                )
                                
                                # 更新融合奖励
                                integrated_reward[t, b] = reward_details['integrated_reward']
                                
                        except Exception as e:
                            # 单个样本融合失败时保持环境奖励
                            continue
                
                # 调试信息
                if not hasattr(self, '_debug_update_counter'):
                    self._debug_update_counter = 0
                self._debug_update_counter += 1
                if self._debug_update_counter % 5000 == 0:
                    env_reward_mean = reward.mean().item()
                    integrated_reward_mean = integrated_reward.mean().item()
                    print(f"[INFO] 奖励融合 (第{self._debug_update_counter}次更新) - 环境奖励均值: {env_reward_mean:.4f}, 融合奖励均值: {integrated_reward_mean:.4f}, 提升: {integrated_reward_mean - env_reward_mean:+.4f}")
                        
            except Exception as e:
                # 偏好奖励计算失败不影响训练，使用环境奖励
                if not hasattr(self, '_pref_error_counter'):
                    self._pref_error_counter = 0
                self._pref_error_counter += 1
                if self._pref_error_counter % 1000 == 0:
                    print(f"[WARNING] 融合奖励计算失败 (第{self._pref_error_counter}次)，使用环境奖励: {e}")

        # 计算目标值（使用融合奖励）
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, integrated_reward, task)

        # 准备更新
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # 潜在空间 rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t + 1] = z

        # 预测
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, task)

        # 计算损失
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            # 使用融合奖励训练奖励预测模型（解决训练目标不一致问题）
            reward_loss += (
                math.soft_ce(reward_preds[t], integrated_reward[t], self.cfg).mean()
                * self.cfg.rho**t
            )
            for q in range(self.cfg.num_q):
                value_loss += (
                    math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
                    * self.cfg.rho**t
                )
        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
        )
        # === 新增辅助损失（使用改进的损失函数）===
        L_pref = 0
        if preference_dataset is not None:
            batch_pref = preference_dataset.sample(batch_size=16)
            
            # 使用改进的偏好损失函数（如果可用）
            if self.reward_scale_optimizer is not None:
                # 提取Q值用于尺度感知损失计算
                z_c, a_c, z_r, a_r, task_pref = zip(*batch_pref)
                z_c = torch.stack(z_c)
                a_c = torch.stack(a_c)
                z_r = torch.stack(z_r)
                a_r = torch.stack(a_r)
                task_pref = torch.stack(task_pref) if isinstance(task_pref[0], torch.Tensor) else None
                
                # 计算Q值
                q_chosen = self.model.Q(z_c, a_c, task_pref, return_type="avg")
                q_rejected = self.model.Q(z_r, a_r, task_pref, return_type="avg")
                
                # 使用改进的偏好损失函数
                L_pref = self.reward_scale_optimizer.compute_preference_loss(q_chosen, q_rejected)
            else:
                # 如果没有奖励尺度优化器，跳过偏好损失计算
                L_pref = torch.tensor(0.0, device=reward.device, dtype=reward.dtype)
            
            total_loss = total_loss + lambda_pref * L_pref
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()
        pi_loss = self.update_pi(zs.detach(), task)
        self.model.soft_update_target_Q()
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pref_loss": float(L_pref) if preference_dataset is not None else 0.0,
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }


# 旧的preference_ranking_loss函数已被奖励尺度优化器中的改进损失函数替代