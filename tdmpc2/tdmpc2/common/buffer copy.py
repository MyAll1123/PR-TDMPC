import sys

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
import numpy as np

class Buffer:
    """
    TD-MPC2 的经验回放缓冲区（Replay Buffer）实现，基于 torchrl。
    支持自动选择 CUDA/CPU 存储，按 episode 存储和采样，适配多步序列采样。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # 根据操作系统选择默认设备（macOS 只用 CPU，其他优先用 CUDA）
        if sys.platform == "darwin":
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda")
        # 缓冲区容量为 buffer_size 和总步数 steps 的较小值
        self._capacity = min(cfg.buffer_size, cfg.steps)
        # 采样器：按 episode 切片采样，支持多步序列
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
        )
        # 每个 batch 包含 batch_size 个序列，每个序列长度为 horizon+1
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0  # 当前已存 episode 数

        # 新增：用于存储历史（obs, act）的缓冲区
        self.history_buffer = []
        self.cfg.obs_shape = cfg.obs_shape
        self.cfg.action_dim = cfg.action_dim

    @property
    def capacity(self):
        """返回缓冲区容量。"""
        return self._capacity

    @property
    def num_eps(self):
        """返回当前缓冲区中 episode 数。"""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        用指定存储（CPU/CUDA）初始化 ReplayBuffer。
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """
        用首个 episode 的数据结构初始化缓冲区，并根据显存/内存情况选择存储设备。
        """
        print(f"Buffer capacity: {self._capacity:,}")
        if sys.platform == "darwin":
            mem_free = 0
        else:
            # 添加CUDA可用性检查
            if torch.cuda.is_available():
                try:
                    mem_free, _ = torch.cuda.mem_get_info()
                except RuntimeError as e:
                    print(f"CUDA内存检查失败，使用CPU模式: {e}")
                    mem_free = 0
            else:
                print("CUDA不可用，使用CPU模式")
                mem_free = 0
        
        # 估算每步数据占用的字节数
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes/1e9:.2f} GB")
        # 经验法则：显存剩余量的2.5倍大于需求则用 CUDA，否则用 CPU
        storage_device = "cuda" if 2.5 * total_bytes < mem_free else "cpu"
        print(f"Using {storage_device.upper()} memory for storage.")
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        """
        将输入张量批量转移到目标设备（默认缓冲区设备）。
        """
        if device is None:
            device = self._device
        return (
            arg.to(device, non_blocking=True) if arg is not None else None
            for arg in args
        )

    def _prepare_batch(self, td):
        """
        对采样出的 batch 进行后处理，便于训练使用。
        输入 td: TensorDict，shape 为 TxB（时间步×batch）。
        返回：obs, action, reward, task（全部转到目标设备）
        """
        obs = td["obs"]
        action = td["action"][1:]  # 去掉第一个动作（通常是初始动作）
        reward = td["reward"][1:].unsqueeze(-1)  # 奖励 shape: [T, B, 1]
        task = td["task"][0] if "task" in td.keys() else None  # 多任务时取第一个 task
        return self._to_device(obs, action, reward, task)

    def add(self, td):
        """
        向缓冲区添加一个 episode（TensorDict）。
        自动为每步打上 episode 编号。
        """
        td["episode"] = torch.ones_like(td["reward"], dtype=torch.int64) * self._num_eps

        # HumanoidBench 特殊修复：episode 太短则丢弃
        if len(td["episode"]) <= self.cfg.horizon + 1:
            return self._num_eps

        # 首次添加时初始化缓冲区
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def add_step(self, obs, act):
        """
        向历史缓冲区中添加一个 (obs, act) 对。
        """
        if len(self.history_buffer) >= self._capacity:
            self.history_buffer.pop(0)
        self.history_buffer.append((obs, act))

    def get_history(self, length):
        """
        从历史缓冲区中获取最近的 `length` 个 (obs, act) 对。
        """
        num_steps = min(length, len(self.history_buffer))
        if num_steps == 0:
            return [], []

        recent_history = self.history_buffer[-num_steps:]
        obs_history = [item[0] for item in recent_history]
        act_history = [item[1] for item in recent_history]

        # 填充以确保obs和act具有正确的维度
        if num_steps < length:
            if not obs_history:
                # 缓冲区为空，返回零填充的历史记录
                obs_pad_shape = self.cfg.obs_shape
                act_pad_shape = self.cfg.action_dim
                obs_padding = [np.zeros(obs_pad_shape) for _ in range(length)]
                act_padding = [np.zeros(act_pad_shape) for _ in range(length)]
                return obs_padding, act_padding

            obs_pad_shape = obs_history[0].shape
            act_pad_shape = act_history[0].shape
            obs_padding = [np.zeros(obs_pad_shape) for _ in range(length - num_steps)]
            act_padding = [np.zeros(act_pad_shape) for _ in range(length - num_steps)]
            obs_history = obs_padding + obs_history
            act_history = act_padding + act_history

        return obs_history, act_history

    def sample(self):
        """
        从缓冲区采样一批子序列（subsequences），用于训练。
        返回：obs, action, reward, task（全部转到目标设备）
        """
        # 采样后 shape: [batch_size*(horizon+1), ...]，重塑为 [horizon+1, batch_size, ...]
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)
