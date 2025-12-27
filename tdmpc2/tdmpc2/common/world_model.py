from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from . import layers, math, init

class WorldModel(nn.Module):
    """
    TD-MPC2 隐式世界模型架构。
    用于单任务和多任务实验，核心功能包括：
    - 状态编码
    - 状态转移预测
    - 奖励预测
    - 策略先验（动作分布）建模
    - Q值（价值）估计
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 多任务场景下的任务嵌入和动作掩码
        if cfg.multitask:
            # 任务嵌入向量表，每个任务一个嵌入
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            # 动作掩码，用于不同任务的动作维度适配
            self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
            for i in range(len(cfg.tasks)):
                self._action_masks[i, : cfg.action_dims[i]] = 1.0

        # 状态编码器（支持像素/状态输入）
        self._encoder = layers.enc(cfg)

        # 状态转移模型（预测下一个隐状态）
        self._dynamics = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            cfg.latent_dim,
            act=layers.SimNorm(cfg),
        )

        # 奖励预测网络
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )

        # 策略先验网络（输出动作分布的均值和对数方差）
        self._pi = layers.mlp(
            cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim
        )

        # Q值网络集合（Ensemble），用于价值估计和不确定性建模
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )

        # 权重初始化
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])

        # 目标 Q 网络（用于目标值计算，Polyak 平滑更新）
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

        # 策略先验的 log_std 限制参数
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

    @property
    def total_params(self):
        """统计模型参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        重载 `to` 方法，确保所有张量（如 action_masks、log_std）也迁移到目标设备。
        """
        super().to(*args, **kwargs)
        if self.cfg.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        重载 `train` 方法，确保目标 Q 网络始终处于 eval 模式（不参与训练）。
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        控制 Q 网络和任务嵌入的梯度开关。
        用于策略优化时关闭 Q 网络梯度，减少计算量。
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.cfg.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        软更新目标 Q 网络（Polyak 平滑），用于稳定目标值。
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)

    def task_emb(self, x, task):
        """
        多任务场景下，将任务嵌入拼接到输入特征上。
        - x: 输入特征（如状态、隐状态等）
        - task: 任务 ID
        返回拼接后的特征。
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        编码观测为隐状态（latent）。
        - obs: 原始观测（状态或像素）
        - task: 任务 ID
        支持多任务和像素输入。
        """
        # 确保观测在正确的设备上
        if hasattr(self, '_encoder') and len(self._encoder) > 0:
            # 获取编码器的设备
            encoder_device = next(iter(self._encoder.parameters())).device
            if obs.device != encoder_device:
                obs = obs.to(encoder_device)
        
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            # 像素输入时，批量编码
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        预测下一个隐状态。
        - z: 当前隐状态
        - a: 动作
        - task: 任务 ID
        返回下一个隐状态。
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        预测一步奖励。
        - z: 当前隐状态
        - a: 动作
        - task: 任务 ID
        返回奖励预测值。
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        策略先验网络：给定隐状态，采样动作。
        - z: 当前隐状态
        - task: 任务 ID
        返回：
          mu: 动作分布均值
          pi: 采样动作
          log_pi: 对数概率
          log_std: 对数标准差
        说明：
        - 策略先验是一个高斯分布，均值和对数方差由神经网络预测。
        - 多任务时对无效动作维度做掩码处理。
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # 预测动作分布的均值和对数方差
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:
            # 掩码无效动作维度
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:
            action_dims = None

        # 计算对数概率
        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        # 采样动作
        pi = mu + eps * log_std.exp()
        # squash 到动作空间
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type="min", target=False):
        """
        预测状态-动作价值（Q值）。
        - z: 当前隐状态
        - a: 动作
        - task: 任务 ID
        - return_type: 返回类型（min/avg/all）
        - target: 是否使用目标 Q 网络
        返回：
          - min: 两个 Q 网络的最小值
          - avg: 两个 Q 网络的均值
          - all: 所有 Q 网络输出
        """
        assert return_type in {"min", "avg", "all"}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == "all":
            return out

        # 随机采样两个 Q 网络，取 min 或 avg
        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2