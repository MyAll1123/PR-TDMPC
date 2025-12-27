import numpy as np
import gymnasium as gym
from envs.wrappers.time_limit import TimeLimit

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
    """
    Meta-World 环境的自定义包装器。
    主要功能：
    - 固定相机视角
    - 控制环境随机性
    - 支持多步累积奖励
    - 兼容 TD-MPC2 的 reset/step/render 接口
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_name = "corner2"  # 指定渲染时使用的相机
        # 设置相机位置（z轴），便于视觉观测一致
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False  # 不冻结环境随机性

    def reset(self, **kwargs):
        """
        重置环境，返回初始观测。
        额外：重置后执行一个全零动作，确保观测稳定。
        """
        obs = super().reset(**kwargs).astype(np.float32)
        self.env.step(np.zeros(self.env.action_space.shape))  # 额外执行一步
        return obs

    def step(self, action):
        """
        执行动作，返回累积奖励。
        这里每个 step 实际执行环境两步，并累加奖励，提升训练信号。
        """
        reward = 0
        for _ in range(2):
            obs, r, _, info = self.env.step(action.copy())
            reward += r
        obs = obs.astype(np.float32)
        return obs, reward, False, info  # 不终止 episode，info 保持原样

    @property
    def unwrapped(self):
        """返回底层原始环境对象。"""
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        """
        渲染环境，返回指定分辨率和相机视角的图像。
        只支持离屏渲染（offscreen）。
        """
        return self.env.render(
            offscreen=True, resolution=(384, 384), camera_name=self.camera_name
        ).copy()


def make_env(cfg):
    """
    构建 Meta-World 环境实例。
    步骤：
    1. 根据任务名生成环境 id
    2. 检查任务合法性，只支持 mw- 前缀和 state 观测
    3. 创建环境实例并包装
    4. 限定 episode 最大步数为 100
    """
    env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
    if (
        not cfg.task.startswith("mw-")
        or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    ):
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "This task only supports state observations."
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
    env = MetaWorldWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)  # 限定最大步数
    env.max_episode_steps = env._max_episode_steps  # 兼容外部调用
    return env
