import gymnasium as gym
import numpy as np
from envs.wrappers.time_limit import TimeLimit

import mani_skill2.envs

# ManiSkill2 任务名到官方环境 id 及控制模式的映射表
MANISKILL_TASKS = {
    "lift-cube": dict(
        env="LiftCube-v0",
        control_mode="pd_ee_delta_pos",  # 末端执行器位置控制
    ),
    "pick-cube": dict(
        env="PickCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "stack-cube": dict(
        env="StackCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "pick-ycb": dict(
        env="PickSingleYCB-v0",
        control_mode="pd_ee_delta_pose",  # 末端执行器位姿控制
    ),
    "turn-faucet": dict(
        env="TurnFaucet-v0",
        control_mode="pd_ee_delta_pose",
    ),
}


class ManiSkillWrapper(gym.Wrapper):
    """
    ManiSkill2 环境包装器。
    主要功能：
    - 统一 observation_space 和 action_space，保证与 TD-MPC2 其他环境兼容
    - step 时每步实际执行两次底层环境 step，并累加奖励，增强训练信号
    - 提供 unwrapped 属性和自定义 render 方法
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = self.env.observation_space
        # 统一 action_space 上下界，便于不同任务间兼容
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )

    def reset(self):
        """
        重置环境，返回初始观测。
        """
        return self.env.reset()

    def step(self, action):
        """
        执行动作，返回累积奖励。
        这里每个 step 实际执行环境两步，并累加奖励，提升训练信号。
        """
        reward = 0
        for _ in range(2):
            obs, r, _, info = self.env.step(action)
            reward += r
        return obs, reward, False, info  # 不终止 episode，info 保持原样

    @property
    def unwrapped(self):
        """
        返回底层原始环境对象。
        """
        return self.env.unwrapped

    def render(self, args, **kwargs):
        """
        渲染环境，返回摄像头视角的图像。
        """
        return self.env.render(mode="cameras")


def make_env(cfg):
    """
    创建 ManiSkill2 环境实例。
    步骤：
    1. 检查任务名是否合法，只支持 MANISKILL_TASKS 中定义的任务
    2. 只支持 'state' 观测（低维状态输入）
    3. 用 gym.make 创建 ManiSkill2 环境，设置观测模式、控制模式和渲染参数
    4. 用 ManiSkillWrapper 包装环境
    5. 用 TimeLimit 限制每个 episode 最多 100 步
    6. 返回最终环境对象
    """
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "This task only supports state observations."
    task_cfg = MANISKILL_TASKS[cfg.task]
    env = gym.make(
        task_cfg["env"],
        obs_mode="state",
        control_mode=task_cfg["control_mode"],
        render_camera_cfgs=dict(width=384, height=384),
    )
    env = ManiSkillWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)  # 限定最大步数
    env.max_episode_steps = env._max_episode_steps  # 兼容外部调用
    return env
