import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from . import MODEL_SIZE, TASK_SET


def parse_cfg(cfg: OmegaConf, env) -> OmegaConf:
    """
    解析和处理 Hydra 配置对象（OmegaConf）。
    主要功能：
    - 处理 None 值、简单的字符串表达式
    - 设置工作目录、任务标题、离散回归的 bin 大小等便捷属性
    - 根据模型规模和任务类型补充/修正配置参数
    - 支持多任务实验的特殊参数处理
    返回处理后的配置对象。
    """

    # 1. 逻辑处理：将配置中值为 None 的项设为 True（方便后续判断）
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass

    # 2. 解析简单的代数表达式（如 "8*4"），将其转为数值
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    # 计算表达式结果并赋值
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    # 如果结果是整数（但类型为 float），转为 int
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except:
            pass

    # 3. 便捷属性设置
    # 设置工作目录路径 logs/<task>/<seed>/<exp_name>
    cfg.work_dir = (
        Path(hydra.utils.get_original_cwd())
        / "logs"
        / cfg.task
        / str(cfg.seed)
        / cfg.exp_name
    )
    # 设置任务标题（如 "humanoid-run" -> "Humanoid Run"）
    cfg.task_title = cfg.task.replace("-", " ").title()
    # 计算离散回归的 bin 大小
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (
        cfg.num_bins - 1
    )  # Bin size for discrete regression

    # 4. 根据模型规模补充参数
    if cfg.get("model_size", None) is not None:
        # 检查模型规模是否合法
        assert (
            cfg.model_size in MODEL_SIZE.keys()
        ), f"Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}"
        # 将模型规模对应的参数写入 cfg
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
        # 针对 mt30-19M 检查点的特殊处理
        if cfg.task == "mt30" and cfg.model_size == 19:
            cfg.latent_dim = 512  # 该模型 latent_dim 略小



    # Update config with environment specific attributes
    # Set obs_shape as a dictionary based on observation type
    if cfg.obs == 'state':
        cfg.obs_shape = {'state': env.observation_space.shape}
    elif cfg.obs == 'rgb':
        cfg.obs_shape = {'rgb': env.observation_space.shape}
    else:
        # Fallback: assume state observation
        cfg.obs_shape = {'state': env.observation_space.shape}
    
    cfg.action_dim = env.action_space.shape[0]
    
    # 确保max_episode_steps存在于cfg中
    if not hasattr(cfg, 'max_episode_steps'):
        OmegaConf.set_struct(cfg, False)  # 临时允许添加新键
        cfg.max_episode_steps = env.max_episode_steps
        OmegaConf.set_struct(cfg, True)  # 恢复结构化限制

    return cfg
