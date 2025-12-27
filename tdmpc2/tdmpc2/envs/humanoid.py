import os
import sys

import numpy as np
import gymnasium as gym

from .wrappers.time_limit import TimeLimit


class HumanoidWrapper(gym.Wrapper):
    """
    HumanoidBench 环境包装器。
    主要功能：
    - 处理 MuJoCo 渲染后端和 GPU 设备环境变量，适配集群/服务器运行
    - 统一观测数据类型（float32）
    - 提供 unwrapped 属性和自定义 render 方法
    """

    def __init__(self, env, cfg):
        # 非 macOS 且未设置 MUJOCO_GL 时，默认用 EGL 后端（适合服务器/无显示环境）
        if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        # 若在 SLURM 分布式环境下，设置 EGL_DEVICE_ID 以指定 GPU
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
        if "SLURM_JOB_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

        super().__init__(env)
        self.env = env
        self.cfg = cfg

    def step(self, action):
        """
        执行动作，返回观测、奖励、终止标志、截断标志和信息字典。
        - obs: 环境观测，转为 float32
        - reward: 环境奖励
        - done: episode 是否终止
        - truncated: 是否因超时等被截断
        - info: 额外信息
        """
        obs, reward, done, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        if "x_velocity" in info:
            info["move"] = info["x_velocity"]
        return obs, reward, done, truncated, info

    @property
    def unwrapped(self):
        """
        返回底层原始环境对象。
        """
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        """
        渲染环境，返回图像或可视化窗口。
        """
        return self.env.render()


def make_env(cfg):
    """
    创建 HumanoidBench 环境实例。
    步骤：
    1. 检查任务名是否合法，只支持 humanoid_ 前缀
    2. 动态导入 humanoid_bench
    3. 读取 policy/model 路径等参数
    4. 用 gym.make 创建环境
    5. 用 HumanoidWrapper 包装环境
    6. 设置最大 episode 步数
    7. 返回最终环境对象
    """
    if not cfg.task.startswith("humanoid_"):
        raise ValueError("Unknown task:", cfg.task)
    import humanoid_bench

    policy_path = cfg.get("policy_path", None)
    mean_path = cfg.get("mean_path", None)
    var_path = cfg.get("var_path", None)
    policy_type = cfg.get("policy_type", None)
    small_obs = cfg.get("small_obs", None)
    if small_obs is not None:
        small_obs = str(small_obs)

    print("small obs start:", small_obs)

    env = gym.make(
        cfg.task.removeprefix("humanoid_"),
        policy_path=policy_path,
        mean_path=mean_path,
        var_path=var_path,
        policy_type=policy_type,
        small_obs=small_obs,
    )
    env = HumanoidWrapper(env, cfg)
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps")
    return env
