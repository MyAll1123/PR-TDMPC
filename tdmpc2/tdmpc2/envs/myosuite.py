import numpy as np
import gymnasium as gym
from .wrappers.time_limit import TimeLimit

# MyoSuite 任务名到官方环境 id 的映射表
MYOSUITE_TASKS = {
    "myo-reach": "myoHandReachFixed-v0",           # 固定目标的手部到达任务
    "myo-reach-hard": "myoHandReachRandom-v0",     # 随机目标的手部到达任务
    "myo-pose": "myoHandPoseFixed-v0",             # 固定目标的手部姿态任务
    "myo-pose-hard": "myoHandPoseRandom-v0",       # 随机目标的手部姿态任务
    "myo-obj-hold": "myoHandObjHoldFixed-v0",      # 固定目标的手部持物任务
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",# 随机目标的手部持物任务
    "myo-key-turn": "myoHandKeyTurnFixed-v0",      # 固定目标的钥匙旋转任务
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",# 随机目标的钥匙旋转任务
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",    # 固定目标的钢笔旋转任务
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",# 随机目标的钢笔旋转任务
}

class MyoSuiteWrapper(gym.Wrapper):
    """
    MyoSuite 环境包装器。
    主要功能：
    - 统一观测数据类型（float32）
    - 将 info["solved"] 字段重命名为 info["success"]，便于统计
    - 固定渲染相机视角
    - 提供 unwrapped 属性和自定义 render 方法
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_id = "hand_side_inter"  # 指定渲染时的相机视角

    def step(self, action):
        """
        执行动作，返回观测、奖励、终止标志和信息字典。
        - obs: 环境观测，转为 float32
        - reward: 环境奖励
        - done: 这里始终为 False（由 TimeLimit 控制 episode 终止）
        - info: 增加 "success" 字段，等于原始 "solved"
        """
        obs, reward, _, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        info["success"] = info["solved"]
        return obs, reward, False, info

    @property
    def unwrapped(self):
        """
        返回底层原始环境对象。
        """
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        """
        离屏渲染当前环境，返回 384x384 分辨率的图像。
        固定使用 hand_side_inter 相机。
        """
        return self.env.sim.renderer.render_offscreen(
            width=384, height=384, camera_id=self.camera_id
        ).copy()

def make_env(cfg):
    """
    创建 MyoSuite 环境实例。
    步骤：
    1. 检查任务名是否合法，只支持 MYOSUITE_TASKS 中定义的任务
    2. 只支持 'state' 观测（低维状态输入）
    3. 动态导入 myosuite 并用 gym.make 创建环境
    4. 用 MyoSuiteWrapper 包装环境
    5. 用 TimeLimit 限制每个 episode 最多 100 步
    6. 返回最终环境对象
    """
    if not cfg.task in MYOSUITE_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "This task only supports state observations."
    import myosuite  # 动态导入，避免无关任务报错

    env = gym.make(MYOSUITE_TASKS[cfg.task])
    env = MyoSuiteWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)  # 限定最大步数
    env.max_episode_steps = env._max_episode_steps  # 兼容外部调用
    return env
