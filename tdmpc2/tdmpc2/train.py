import os
import sys
import time

# 设置环境变量以支持不同平台的渲染
if sys.platform != "darwin":  # 如果不是 macOS 平台
    os.environ["MUJOCO_GL"] = "egl"  # 使用 EGL 渲染（适用于 Linux）

os.environ["LAZY_LEGACY_OP"] = "0"  # 禁用 PyTorch 的懒惰操作（可能与性能优化有关）

import warnings

# 忽略所有警告信息
warnings.filterwarnings("ignore")

import torch
import hydra  # 用于配置管理
from termcolor import colored  # 用于打印带颜色的文本

# 导入项目中的模块
from .common.parser import parse_cfg, TASK_SET  # 配置解析器
from .common.seed import set_seed  # 随机种子设置
from .envs import make_env  # 环境创建函数
from .tdmpc2 import TDMPC2  # TD-MPC2 算法实现
from .common.buffer import Buffer  # 经验回放缓冲区
from .common.logger import Logger  # 日志记录器
from .trainer.offline_trainer import OfflineTrainer  # 离线训练器
from .trainer.online_trainer import OnlineTrainer  # 在线训练器

# 导入集成偏好学习训练器
try:
    from .trainer.integrated_preference_trainer import IntegratedPreferenceTrainer
    INTEGRATED_PREFERENCE_AVAILABLE = True
    print(colored("🚀 集成偏好学习训练器可用 - 双路径版本", "green", attrs=["bold"]))
except ImportError as e:
    INTEGRATED_PREFERENCE_AVAILABLE = False
    print(colored(f"⚠️ 集成偏好学习训练器不可用: {e}", "yellow"))
    print(colored("将使用标准训练器", "yellow"))

# 导入优先级偏好集成器
try:
    # 添加项目根路径到sys.path以支持绝对导入
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from prm.prioritized_preference_integrator import (
        PrioritizedPreferenceIntegrator,
        IntegrationConfig,
        create_prioritized_preference_integrator
    )
    PRIORITIZED_PREFERENCE_AVAILABLE = True
    print(colored("🎯 优先级偏好集成器可用 - 经验回放版本", "magenta", attrs=["bold"]))
except ImportError as e:
    PRIORITIZED_PREFERENCE_AVAILABLE = False
    print(colored(f"⚠️ 优先级偏好集成器不可用: {e}", "yellow"))
    print(colored("将使用标准偏好系统", "yellow"))

# 启用 CuDNN 的基准模式以优化性能（适用于固定大小的输入）
torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """训练脚本"""
    # 确保训练步数大于 0
    assert cfg.steps > 0, "Must train for at least 1 step."

    # 5. 多任务相关参数
    cfg.multitask = cfg.task in TASK_SET.keys()  # 是否为多任务
    if cfg.multitask:
        cfg.task_title = cfg.task.upper()  # 多任务标题大写
        # 针对 mt80 任务和部分模型规模的 task_dim 特殊处理
        cfg.task_dim = 96 if cfg.task == "mt80" or cfg.model_size in {1, 317} else 64
    else:
        cfg.task_dim = 0  # 单任务时 task_dim 设为 0
    # 任务列表：多任务为任务集，单任务为自身
    cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])

    # 创建环境
    env = make_env(cfg)

    # 解析配置文件
    cfg = parse_cfg(cfg, env)

    # 设置随机种子以确保实验的可重复性
    set_seed(cfg.seed)

    # 打印工作目录
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    # 检查是否启用偏好引擎
    use_preference_engine = getattr(cfg, "use_preference_engine", False)
    preference_enabled = getattr(cfg, "preference_enabled", False)
    
    # 检查历史数据收集和偏好模型创建配置
    history_enabled = getattr(cfg, "history_data_collection", {}).get("enabled", False)
    preference_model_enabled = getattr(cfg, "preference_model_creation", {}).get("enabled", False)
    
    # 检查优先级经验回放配置
    prioritized_replay_enabled = getattr(cfg, "prioritized_experience_replay", {}).get("enabled", False)
    
    print(colored(f"偏好引擎状态: {'启用' if use_preference_engine else '禁用'}", "yellow", attrs=["bold"]))
    print(colored(f"偏好学习总开关: {'启用' if preference_enabled else '禁用'}", "yellow", attrs=["bold"]))
    print(colored(f"历史数据收集: {'启用' if history_enabled else '禁用'}", "yellow", attrs=["bold"]))
    print(colored(f"偏好模型创建: {'启用' if preference_model_enabled else '禁用'}", "yellow", attrs=["bold"]))
    print(colored(f"优先级经验回放: {'启用' if prioritized_replay_enabled else '禁用'}", "magenta", attrs=["bold"]))
    
    # 优先使用优先级偏好系统，其次使用集成偏好学习训练器
    should_use_prioritized_preference = (
        PRIORITIZED_PREFERENCE_AVAILABLE and
        prioritized_replay_enabled and
        not cfg.multitask  # 仅单任务支持
    )
    
    should_use_integrated_preference = False  # 禁用原始集成偏好学习训练器
    
    print(colored(f"优先级偏好系统: {'启用' if should_use_prioritized_preference else '禁用'}", "magenta", attrs=["bold"]))
    print(colored(f"集成偏好学习训练器: {'启用' if should_use_integrated_preference else '禁用'}", "cyan", attrs=["bold"]))
    
    # 根据配置选择训练器类 - 优先级系统 > 集成偏好学习 > 标准训练器
    if cfg.multitask:
        trainer_cls = OfflineTrainer
        print(colored("📋 使用离线训练器 (多任务)", "blue", attrs=["bold"]))
    elif should_use_prioritized_preference:
        trainer_cls = IntegratedPreferenceTrainer  # 复用集成训练器，但添加优先级系统
        print(colored("🎯 使用优先级偏好系统 - 经验回放版本", "magenta", attrs=["bold"]))
        print(colored("   ✅ 优先级经验回放缓冲池", "magenta"))
        print(colored("   ✅ 置信度 + 时间优先级", "magenta"))
        print(colored("   ✅ 基于损失的优先级更新", "magenta"))
        print(colored("   ✅ TD-MPC2 + 偏好奖励结合", "green"))
    elif should_use_integrated_preference:
        trainer_cls = IntegratedPreferenceTrainer
        print(colored("🚀 使用集成偏好学习训练器 - 双路径版本", "green", attrs=["bold"]))
        print(colored("   ✅ TD-MPC2 + 偏好奖励结合", "green"))
        print(colored("   ✅ 内存缓存系统", "green"))
        print(colored("   ✅ 零文件IO操作", "green"))
    else:
        trainer_cls = OnlineTrainer
        print(colored("📝 使用标准在线训练器 (降级)", "yellow", attrs=["bold"]))

    # 初始化训练器
    if cfg.multitask:
        # 离线训练器（多任务）
        trainer = trainer_cls(
            cfg=cfg,
            env=env,
            agent=TDMPC2(cfg),
            buffer=Buffer(cfg),
            logger=Logger(cfg)
        )
    elif should_use_prioritized_preference:
        # 优先级偏好系统训练器（单任务）
        print(colored("初始化优先级偏好系统...", "magenta"))
        
        # 创建优先级偏好集成器配置
        integration_config = IntegrationConfig(
            enable_prioritized_replay=True,
            enable_legacy_compatibility=True,
            enable_performance_monitoring=True,
            integration_mode="prioritized_only",  # 优先使用优先级系统
            max_memory_usage_mb=getattr(cfg, 'max_memory_usage_mb', 2048.0),
            performance_check_interval=getattr(cfg, 'performance_check_interval', 100),
            fallback_to_legacy=True,
            prioritized_weight=0.8,
            legacy_weight=0.2
        )
        
        # 导入偏好感知TD-MPC2
        try:
            from prm.preference_aware_tdmpc2 import create_preference_aware_tdmpc2
            from prm.hybrid_value_estimator import HybridValueConfig
            
            # 创建偏好感知智能体
            print(colored("创建偏好感知TD-MPC2智能体...", "cyan"))
            agent = create_preference_aware_tdmpc2(
                cfg=cfg,
                preference_integrator=None,  # 初始为None，后续由训练器管理
                preference_trainer=None,  # 初始为None，后续由训练器管理
                hybrid_config=HybridValueConfig.from_config(cfg),
                enable_preference_planning=True
            )
            print(colored("✅ 偏好感知TD-MPC2智能体创建成功", "cyan"))
            
        except ImportError as e:
            print(colored(f"⚠️ 无法导入偏好感知TD-MPC2，使用标准智能体: {e}", "yellow"))
            agent = TDMPC2(cfg)
        
        # 创建优先级偏好集成器
        try:
            prioritized_integrator = create_prioritized_preference_integrator(
                task_name=cfg.task,
                cfg=cfg,
                integration_config=integration_config,
                legacy_integrator=None,  # 可以后续添加传统集成器作为回退
                tdmpc2_agent=agent  # 传递TD-MPC2 agent
            )
            print(colored("✅ 优先级偏好集成器创建成功", "magenta"))
        except Exception as e:
            print(colored(f"⚠️ 创建优先级偏好集成器失败: {e}", "yellow"))
            prioritized_integrator = None
        
        trainer = trainer_cls(
            cfg=cfg,
            env=env,
            agent=agent,
            buffer=Buffer(cfg),
            logger=Logger(cfg)
        )
        
        # 将优先级集成器注入到训练器中
        if prioritized_integrator and hasattr(trainer, 'set_prioritized_integrator'):
            trainer.set_prioritized_integrator(prioritized_integrator)
            print(colored("✅ 优先级集成器已注入训练器", "magenta"))
        elif prioritized_integrator:
            # 如果训练器没有专门的方法，直接设置属性
            trainer.prioritized_integrator = prioritized_integrator
            print(colored("✅ 优先级集成器已设置为训练器属性", "magenta"))
        
        print(colored("✅ 优先级偏好系统初始化完成", "magenta", attrs=["bold"]))
        
    elif should_use_integrated_preference:
        # 集成偏好学习训练器（单任务）
        print(colored("初始化集成偏好学习训练器...", "green"))
        
        # 导入偏好感知TD-MPC2
        try:
            from prm.preference_aware_tdmpc2 import create_preference_aware_tdmpc2
            from prm.hybrid_value_estimator import HybridValueConfig
            
            # 创建偏好感知智能体
            print(colored("创建偏好感知TD-MPC2智能体...", "cyan"))
            agent = create_preference_aware_tdmpc2(
                cfg=cfg,
                preference_integrator=None,  # 初始为None，后续由训练器管理
                preference_trainer=None,  # 初始为None，后续由训练器管理
                hybrid_config=HybridValueConfig.from_config(cfg),
                enable_preference_planning=True
            )
            print(colored("✅ 偏好感知TD-MPC2智能体创建成功", "cyan"))
            
        except ImportError as e:
            print(colored(f"⚠️ 无法导入偏好感知TD-MPC2，使用标准智能体: {e}", "yellow"))
            agent = TDMPC2(cfg)
        
        trainer = trainer_cls(
            cfg=cfg,
            env=env,
            agent=agent,
            buffer=Buffer(cfg),
            logger=Logger(cfg)
        )
        print(colored("✅ 集成偏好学习训练器初始化完成", "green", attrs=["bold"]))
    else:
        # 标准在线训练器（单任务）
        trainer = trainer_cls(
            cfg=cfg,
            env=env,
            agent=TDMPC2(cfg),
            buffer=Buffer(cfg),
            logger=Logger(cfg),
            use_preference_engine=use_preference_engine
        )

    # 主训练循环
    trainer.train()

    # 训练结束后的收尾
    trainer.logger.finish(trainer.agent)

    # 打印偏好系统统计信息
    if should_use_prioritized_preference and hasattr(trainer, 'prioritized_integrator'):
        try:
            integrator = trainer.prioritized_integrator
            if integrator:
                stats = integrator.get_statistics()
                print(colored("\n=== 优先级偏好系统统计信息 ===", "magenta", attrs=["bold"]))
                print(colored(f"总训练回合数: {stats.get('total_episodes', 0)}", "magenta"))
                print(colored(f"优先级训练步数: {stats.get('prioritized_training_steps', 0)}", "magenta"))
                print(colored(f"混合训练步数: {stats.get('hybrid_training_steps', 0)}", "magenta"))
                print(colored(f"回退次数: {stats.get('fallback_count', 0)}", "yellow"))
                print(colored(f"错误次数: {stats.get('error_count', 0)}", "red" if stats.get('error_count', 0) > 0 else "green"))
                print(colored(f"总运行时间: {stats.get('total_runtime_seconds', 0):.2f}秒", "cyan"))
                
                # 静默处理优先级系统和性能监控统计，不输出详细信息
                
                print(colored("=" * 40, "magenta"))
        except Exception as e:
            print(colored(f"获取优先级偏好系统统计信息失败: {e}", "yellow"))
    
    elif should_use_integrated_preference and hasattr(trainer, 'get_preference_stats'):
        try:
            stats = trainer.get_preference_stats()
            print(colored("\n=== 偏好学习统计信息 ===", "cyan", attrs=["bold"]))
            print(colored(f"历史数据收集次数: {stats.get('historical_data_collections', 0)}", "green"))
            print(colored(f"偏好模型更新次数: {stats.get('preference_model_updates', 0)}", "green"))
            print(colored(f"缓存轨迹数: {stats.get('total_trajectories', 0)}", "green"))
            print(colored(f"缓存偏好对数: {stats.get('total_preference_pairs', 0)}", "green"))
            print(colored(f"内存使用: {stats.get('memory_usage_estimate', 'N/A')}", "green"))
            print(colored("=" * 30, "cyan"))
        except Exception as e:
            print(colored(f"获取偏好学习统计信息失败: {e}", "yellow"))

    # 打印训练完成信息
    print(colored("\n🎉 Training completed successfully! 🎉", "green", attrs=["bold"]))


# 如果此脚本是主程序，则调用 train 函数
if __name__ == "__main__":
    train()
