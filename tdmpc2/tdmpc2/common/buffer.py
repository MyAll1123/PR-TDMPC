import sys
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
import numpy as np
from collections import deque

class RingBuffer:
    """高效的循环缓冲区实现，避免list.pop(0)的O(n)复杂度"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.size = 0
    
    def append(self, item):
        """添加元素，O(1)时间复杂度"""
        self.buffer[self.head] = item
        self.head = (self.head + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def get_recent(self, length):
        """获取最近的length个元素，O(length)时间复杂度"""
        if self.size == 0:
            return []
        
        length = min(length, self.size)
        result = []
        for i in range(length):
            idx = (self.head - 1 - i) % self.capacity
            result.append(self.buffer[idx])
        return result[::-1]  # 返回正序
    
    def __len__(self):
        return self.size

class TensorHistoryBuffer:
    """基于Tensor的历史缓冲区，减少CPU-GPU传输开销"""
    
    def __init__(self, capacity, obs_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        
        # 预分配tensor缓冲区
        if isinstance(obs_shape, (list, tuple)):
            self.obs_buffer = torch.zeros((capacity,) + tuple(obs_shape), device=device, dtype=torch.float32)
        else:
            self.obs_buffer = torch.zeros((capacity, obs_shape), device=device, dtype=torch.float32)
        
        if isinstance(action_dim, (list, tuple)):
            self.action_buffer = torch.zeros((capacity,) + tuple(action_dim), device=device, dtype=torch.float32)
        else:
            self.action_buffer = torch.zeros((capacity, action_dim), device=device, dtype=torch.float32)
        
        self.head = 0
        self.size = 0
    
    def add_step(self, obs, action):
        """添加步骤数据，直接在GPU上操作"""
        try:
            # 转换为tensor并确保在正确设备上
            if not isinstance(obs, torch.Tensor):
                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            else:
                obs_tensor = obs.to(device=self.device, dtype=torch.float32)
            
            if not isinstance(action, torch.Tensor):
                action_tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32)
            else:
                action_tensor = action.to(device=self.device, dtype=torch.float32)
            
            # 确保维度匹配
            if obs_tensor.dim() == 0:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif obs_tensor.dim() > 1:
                obs_tensor = obs_tensor.flatten()
            
            if action_tensor.dim() == 0:
                action_tensor = action_tensor.unsqueeze(0)
            elif action_tensor.dim() > 1:
                action_tensor = action_tensor.flatten()
            
            # 存储到缓冲区
            self.obs_buffer[self.head] = obs_tensor
            self.action_buffer[self.head] = action_tensor
            
            self.head = (self.head + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1
                
        except Exception as e:
            print(f"[TensorHistoryBuffer] 添加步骤时出错: {e}")
            print(f"obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"action shape: {action.shape if hasattr(action, 'shape') else type(action)}")
    
    def get_recent(self, length):
        """获取最近的length个(obs, action)对"""
        if self.size == 0:
            return [], []
        
        length = min(length, self.size)
        obs_list = []
        action_list = []
        
        for i in range(length):
            idx = (self.head - 1 - i) % self.capacity
            obs_list.append(self.obs_buffer[idx])
            action_list.append(self.action_buffer[idx])
        
        # 返回正序
        return obs_list[::-1], action_list[::-1]
    
    def __len__(self):
        return self.size

class OptimizedBuffer:
    """
    优化版本的TD-MPC2经验回放缓冲区
    主要优化：
    1. 使用高效的循环缓冲区替代list操作
    2. 支持tensor预分配减少GPU传输
    3. 可配置的缓冲区类型
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # 根据操作系统选择默认设备
        if sys.platform == "darwin":
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda")
        
        # 缓冲区容量
        self._capacity = min(cfg.buffer_size, cfg.steps)
        
        # 采样器配置
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
        )
        
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0
        
        # 获取性能优化配置
        perf_config = getattr(cfg, 'performance_optimization', {})
        self.history_buffer_type = perf_config.get('history_buffer_type', 'ring')
        self.enable_tensor_prealloc = perf_config.get('enable_tensor_prealloc', True)
        
        # 初始化历史缓冲区
        self._init_history_buffer()
        
        print(f"[OptimizedBuffer] 使用 {self.history_buffer_type} 类型的历史缓冲区")
    
    def _init_history_buffer(self):
        """初始化历史缓冲区"""
        if self.history_buffer_type == 'tensor' and self.enable_tensor_prealloc:
            # 使用tensor缓冲区（需要知道obs和action的形状）
            obs_shape = getattr(self.cfg, 'obs_shape', None)
            action_dim = getattr(self.cfg, 'action_dim', None)
            
            if obs_shape is not None and action_dim is not None:
                self.history_buffer = TensorHistoryBuffer(
                    self._capacity, obs_shape, action_dim, self._device
                )
                print(f"[OptimizedBuffer] 使用TensorHistoryBuffer，obs_shape={obs_shape}, action_dim={action_dim}")
            else:
                print(f"[OptimizedBuffer] obs_shape或action_dim未定义，回退到RingBuffer")
                self.history_buffer = RingBuffer(self._capacity)
        elif self.history_buffer_type == 'deque':
            # 使用deque（双端队列）
            self.history_buffer = deque(maxlen=self._capacity)
        elif self.history_buffer_type == 'ring':
            # 使用循环缓冲区
            self.history_buffer = RingBuffer(self._capacity)
        else:
            # 回退到原始list实现
            self.history_buffer = []
            print(f"[OptimizedBuffer] 使用原始list实现（性能较低）")

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_eps(self):
        return self._num_eps

    def _reserve_buffer(self, storage):
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=True,
            prefetch=1,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """初始化主缓冲区"""
        print(f"Buffer capacity: {self._capacity:,}")
        
        if sys.platform == "darwin":
            mem_free = 0
        else:
            if torch.cuda.is_available():
                try:
                    mem_free, _ = torch.cuda.mem_get_info()
                except RuntimeError as e:
                    print(f"CUDA内存检查失败，使用CPU模式: {e}")
                    mem_free = 0
            else:
                print("CUDA不可用，使用CPU模式")
                mem_free = 0
        
        # 估算内存需求
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
        
        storage_device = "cuda" if 2.5 * total_bytes < mem_free else "cpu"
        print(f"Using {storage_device.upper()} memory for storage.")
        
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=torch.device(storage_device))
        )

    def _to_device(self, *args, device=None):
        if device is None:
            device = self._device
        return (
            arg.to(device, non_blocking=True) if arg is not None else None
            for arg in args
        )

    def _prepare_batch(self, td):
        obs = td["obs"]
        action = td["action"][1:]
        reward = td["reward"][1:].unsqueeze(-1)
        task = td["task"][0] if "task" in td.keys() else None
        return self._to_device(obs, action, reward, task)

    def add(self, td):
        """添加episode到主缓冲区"""
        td["episode"] = torch.ones_like(td["reward"], dtype=torch.int64) * self._num_eps

        if len(td["episode"]) <= self.cfg.horizon + 1:
            return self._num_eps

        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self._num_eps += 1
        return self._num_eps

    def add_step(self, obs, action):
        """优化的步骤添加方法"""
        if isinstance(self.history_buffer, TensorHistoryBuffer):
            self.history_buffer.add_step(obs, action)
        elif isinstance(self.history_buffer, RingBuffer):
            self.history_buffer.append((obs, action))
        elif isinstance(self.history_buffer, deque):
            self.history_buffer.append((obs, action))
        else:
            # 原始list实现（性能较低）
            if len(self.history_buffer) >= self._capacity:
                self.history_buffer.pop(0)
            self.history_buffer.append((obs, action))

    def get_history(self, length):
        """优化的历史获取方法"""
        if isinstance(self.history_buffer, TensorHistoryBuffer):
            return self.history_buffer.get_recent(length)
        elif isinstance(self.history_buffer, (RingBuffer, deque)):
            recent_items = self.history_buffer.get_recent(length) if hasattr(self.history_buffer, 'get_recent') else list(self.history_buffer)[-length:]
            if not recent_items:
                return [], []
            
            obs_history = [item[0] for item in recent_items]
            act_history = [item[1] for item in recent_items]
            
            # 填充逻辑
            num_steps = len(obs_history)
            if num_steps < length:
                if not obs_history:
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
        else:
            # 原始实现
            num_steps = min(length, len(self.history_buffer))
            if num_steps == 0:
                return [], []

            recent_history = self.history_buffer[-num_steps:]
            obs_history = [item[0] for item in recent_history]
            act_history = [item[1] for item in recent_history]

            if num_steps < length:
                if not obs_history:
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
        """从缓冲区采样"""
        td = self._buffer.sample().view(-1, self.cfg.horizon + 1).permute(1, 0)
        return self._prepare_batch(td)

# 为了向后兼容，保留原始Buffer类名
Buffer = OptimizedBuffer