import os
import sys

# 如果不是 macOS 平台，设置环境变量以使用 EGL 渲染
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

import warnings

# 忽略所有警告信息
warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

# 导入项目中的模块
from .common.parser import parse_cfg  # 配置解析器
from .common.seed import set_seed  # 随机种子设置
from .envs import make_env  # 环境创建函数
from .tdmpc2 import TDMPC2  # TD-MPC2 算法实现

# 启用 CuDNN 的基准模式以优化性能（适用于固定大小的输入）
torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
    """
    用于评估单任务或多任务 TD-MPC2 检查点的脚本。

    主要参数：
        `task`: 任务名称（或 mt30/mt80 表示多任务评估）
        `model_size`: 模型大小，必须是 `[1, 5, 19, 48, 317]` 中的一个值（默认值：5）
        `checkpoint`: 要加载的模型检查点路径
        `eval_episodes`: 每个任务评估的 episode 数量（默认值：10）
        `save_video`: 是否保存评估过程的视频（默认值：True）
        `seed`: 随机种子（默认值：1）

    更多参数请参考 config.yaml。

    示例用法：
    ````
    $ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
    $ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
    $ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
    ```
    """
    # 确保 CUDA 可用
    assert torch.cuda.is_available()
    # 确保评估的 episode 数量大于 0
    assert cfg.eval_episodes > 0, "必须至少评估 1 个 episode。"

    # 解析配置文件
    cfg = parse_cfg(cfg)
    # 设置随机种子
    set_seed(cfg.seed)

    # 打印任务信息
    print(colored(f"任务: {cfg.task}", "blue", attrs=["bold"]))
    print(colored(f"模型大小: {cfg.get('model_size', 'default')}", "blue", attrs=["bold"]))
    print(colored(f"检查点: {cfg.checkpoint}", "blue", attrs=["bold"]))

    # 如果尝试单任务评估多任务模型，打印警告信息
    if not cfg.multitask and ("mt80" in cfg.checkpoint or "mt30" in cfg.checkpoint):
        print(
            colored(
                "警告：当前不支持对多任务模型进行单任务评估。",
                "red",
                attrs=["bold"],
            )
        )
        print(
            colored(
                "要评估多任务模型，请使用 task=mt80 或 task=mt30。",
                "red",
                attrs=["bold"],
            )
        )

    # 创建环境
    env = make_env(cfg)

    # 加载代理
    agent = TDMPC2(cfg)
    # 确保检查点路径存在
    assert os.path.exists(
        cfg.checkpoint
    ), f"检查点 {cfg.checkpoint} 未找到！必须是有效的文件路径。"
    agent.load(cfg.checkpoint)

    # 开始评估
    if cfg.multitask:
        print(
            colored(
                f"正在评估代理在 {len(cfg.tasks)} 个任务上的表现：", "yellow", attrs=["bold"]
            )
        )
    else:
        print(colored(f"正在评估代理在任务 {cfg.task} 上的表现：", "yellow", attrs=["bold"]))

    # 如果启用了视频保存功能，创建视频保存目录
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    # 初始化评估结果
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]

    # 遍历任务列表
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None  # 单任务评估时任务索引为 None
        ep_rewards, ep_successes = [], []

        # 遍历每个任务的评估 episode
        for i in range(cfg.eval_episodes):
            # 重置环境，初始化观测、完成标志、累积奖励和时间步计数
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0

            # 如果启用了视频保存功能，初始化帧列表
            if cfg.save_video:
                frames = [env.render()]

            # 执行当前 episode，直到完成
            while not done:
                # 使用代理选择动作
                action = agent.act(obs, t0=t == 0, task=task_idx)
                # 执行动作并获取环境返回的结果
                obs, reward, done, info = env.step(action)
                # 累加奖励
                ep_reward += reward
                # 增加时间步计数
                t += 1
                # 如果启用了视频保存功能，记录当前帧
                if cfg.save_video:
                    frames.append(env.render())

            # 将当前 episode 的累积奖励和成功标志添加到列表
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])

            # 如果启用了视频保存功能，保存视频
            if cfg.save_video:
                imageio.mimsave(
                    os.path.join(video_dir, f"{task}-{i}.mp4"), frames, fps=15
                )

        # 计算当前任务的平均奖励和成功率
        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)

        # 如果是多任务评估，计算归一化分数
        if cfg.multitask:
            scores.append(
                ep_successes * 100 if task.startswith("mw-") else ep_rewards / 10
            )

        # 打印当前任务的评估结果
        print(
            colored(
                f"  {task:<22}" f"\tR: {ep_rewards:.01f}  " f"\tS: {ep_successes:.02f}",
                "yellow",
            )
        )

    # 如果是多任务评估，打印归一化分数
    if cfg.multitask:
        print(
            colored(
                f"归一化分数: {np.mean(scores):.02f}", "yellow", attrs=["bold"]
            )
        )


# 如果此脚本是主程序，则调用 evaluate 函数
if __name__ == "__main__":
    evaluate()
