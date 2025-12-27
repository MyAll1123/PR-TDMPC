import os
import datetime
import re
import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf
from datetime import datetime, timedelta

from . import TASK_SET

# 控制台输出格式，每个元组为 (字段名, 显示名, 类型)
CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("step", "I", "int"),
    ("episode", "E", "int"),
    ("episode_env_reward", "EnvR", "float"),
    # ("episode_pref_reward", "PrefR", "float"),  # 已移除偏好奖励显示
    # ("episode_reward", "MixR", "float"),  # 已移除混合奖励显示
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
    ("episode_length", "L", "int"),
    ("buffer_size", "B", "int"),
    ("fps", "FPS", "float"),
]

# 日志类别到颜色的映射
CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
    "results": "magenta",
}


def make_dir(dir_path):
    """如果目录不存在则创建目录。"""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception:
        pass
    return dir_path


def print_run(cfg):
    """
    美观地打印当前实验的关键信息。
    在 Logger 初始化时调用。
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        # 字符串截断显示
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        # 打印单行信息
        print(
            prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs),
            _limstr(v),
        )

    # 组装关键信息
    observations = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("task", cfg.task_title),
        ("steps", f"{int(cfg.steps):,}"),
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """
    生成 wandb 日志分组名。
    可选返回列表形式。
    """
    lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


class VideoRecorder:
    """训练/评估时录制视频的工具类。"""

    def __init__(self, cfg, wandb=None, fps=15):
        self.cfg = cfg
        self._save_dir = make_dir(cfg.work_dir / "train_video")
        self._wandb = wandb
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        """初始化视频录制。"""
        self.frames = []
        self.enabled = enabled
        if enabled:
            # 采集初始帧
            try:
                frame = env.render()
            except Exception:
                frame = env.render(mode="rgb_array")
            self.frames.append(frame)

    def record(self, env):
        """每步调用，记录当前环境帧。"""
        if self.enabled:
            try:
                frame = env.render()
            except Exception:
                frame = env.render(mode="rgb_array")
            self.frames.append(frame)

    def save(self, step, key="results/train_video"):
        if self.enabled and len(self.frames) > 1:
            frames = np.stack(self.frames)  # (N, H, W, C)
            # wandb 需要 (N, C, H, W)
            if self._wandb is not None:
                frames_wandb = frames.transpose(0, 3, 1, 2)
                self._wandb.log({key: self._wandb.Video(frames_wandb, fps=self.fps, format="mp4")}, step=step)
            # 本地
            import imageio
            video_path = os.path.join(self._save_dir, f"train_video_{step}.mp4")
            imageio.mimsave(video_path, frames, fps=self.fps)
            print(f"[VideoRecorder] Saved video with {len(self.frames)} frames to {video_path}")
        else:
            print(f"[VideoRecorder] Warning: Not enough frames to save video (frames={len(self.frames)})")
        self.frames = []
        self.enabled = False


class Logger:
    """
    日志主类。支持本地日志和 wandb 云日志。
    """

    def __init__(self, cfg):
        # 生成唯一的工作目录（包含时间戳、实验名、随机种子）
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = f"{cfg.exp_name}_{timestamp}_{cfg.seed}"
        cfg.work_dir = cfg.work_dir / unique_id  # 确保路径唯一
        self._log_dir = make_dir(cfg.work_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_csv = cfg.save_csv
        self._save_agent = cfg.save_agent
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = []
        print_run(cfg)
        self.project = cfg.get("wandb_project", "none")
        self.entity = cfg.get("wandb_entity", "none")

        # === 新增：支持offline参数 ===
        disable_wandb = str(getattr(cfg, "disable_wandb", "")).lower()
        if disable_wandb == "disable":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            self._video = None
            return
        elif disable_wandb == "offline":
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_SERVICE_WAIT"] = "0"
            print(colored("Wandb set to offline mode.", "blue", attrs=["bold"]))
            # 在offline模式下完全禁用wandb以避免服务启动问题
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            self._video = None
            return

        # wandb 配置检查
        if disable_wandb == "disable" or self.project == "none" or self.entity == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            self._video = None
            return

        # wandb 初始化
        os.environ["WANDB_SILENT"] = "true" if getattr(cfg, "wandb_silent", False) else "false"
        import wandb

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=f"{cfg.task}.tdmpc.{cfg.exp_name}.{cfg.seed}",
            group=self._group,
            tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
            dir=self._log_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(colored("Logs will be synced with wandb (offline mode)." if disable_wandb == "offline" else "Logs will be synced with wandb.", "blue", attrs=["bold"]))
        self._wandb = wandb if cfg.save_video else None
        self._video = VideoRecorder(cfg, self._wandb) if cfg.save_video else None

    @property
    def video(self):
        """返回视频录制器对象."""
        return self._video

    @property
    def model_dir(self):
        """返回模型保存目录."""
        return self._model_dir

    def save_agent(self, agent=None, identifier="final"):
        """
        保存智能体模型到本地和 wandb。
        """
        if self._save_agent and agent:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + "-" + str(self._seed) + "-" + str(identifier),
                    type="model",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def finish(self, agent=None):
        """
        结束日志记录，保存模型并关闭 wandb。
        """
        try:
            self.save_agent(agent)
        except Exception as e:
            print(colored(f"Failed to save model: {e}", "red"))
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        """
        按类型格式化日志输出。
        """
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.01f}'
        elif ty == "time":
            value = str(timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        """
        控制台美观打印日志信息。
        """
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def pprint_multitask(self, d, cfg):
        """
        多任务训练时，按任务类别美观打印评估指标。
        """
        print(
            colored(
                f"Evaluated agent on {len(cfg.tasks)} tasks:", "yellow", attrs=["bold"]
            )
        )
        dmcontrol_reward = []
        metaworld_reward = []
        metaworld_success = []
        for k, v in d.items():
            if "+" not in k:
                continue
            task = k.split("+")[1]
            if task in TASK_SET["mt30"] and k.startswith("episode_reward"):  # DMControl
                dmcontrol_reward.append(v)
                print(colored(f"  {task:<22}\tR: {v:.01f}", "yellow"))
            elif (
                task in TASK_SET["mt80"] and task not in TASK_SET["mt30"]
            ):  # Meta-World
                if k.startswith("episode_reward"):
                    metaworld_reward.append(v)
                elif k.startswith("episode_success"):
                    metaworld_success.append(v)
                    print(colored(f"  {task:<22}\tS: {v:.02f}", "yellow"))
        dmcontrol_reward = np.nanmean(dmcontrol_reward)
        d["episode_reward+avg_dmcontrol"] = dmcontrol_reward
        print(
            colored(
                f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}',
                "yellow",
                attrs=["bold"],
            )
        )
        if cfg.task == "mt80":
            metaworld_reward = np.nanmean(metaworld_reward)
            metaworld_success = np.nanmean(metaworld_success)
            d["episode_reward+avg_metaworld"] = metaworld_reward
            d["episode_success+avg_metaworld"] = metaworld_success
            print(
                colored(
                    f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}',
                    "yellow",
                    attrs=["bold"],
                )
            )
            print(
                colored(
                    f'  {"metaworld":<22}\tS: {metaworld_success:.02f}',
                    "yellow",
                    attrs=["bold"],
                )
            )

    def log(self, d, category="train"):
        """
        记录日志到 wandb、本地 csv，并控制台打印。
        d: 日志字典
        category: 日志类别（train/eval/pretrain/results）
        """
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        # wandb 日志
        if self._wandb:
            if category in {"train", "eval", "results"}:
                xkey = "step"
            elif category == "pretrain":
                xkey = "iteration"
            for k, v in d.items():
                if category == "results" and k == "step":
                    continue
                self._wandb.log({category + "/" + k: v}, step=d[xkey])
        # 本地 csv 日志（仅 eval）
        if category == "eval" and self._save_csv:
            keys = list(d.keys())
            # 保存所有指标
            self._eval.append([d[k] for k in keys])
            pd.DataFrame(self._eval, columns=keys).to_csv(
                self._log_dir / "eval.csv", header=True, index=None
            )
        # 控制台打印
        if category != "results":
            self._print(d, category)