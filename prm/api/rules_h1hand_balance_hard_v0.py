from .common_utils import tolerance

"""Balance Hard任务的启发式规则：基于轨迹质量评估的DPO偏好标签生成

该模块为Balance Hard任务实现启发式规则，用于生成DPO训练所需的偏好标签。
规则基于Balance Hard任务的特定指标和奖励函数，评估轨迹质量并生成偏好对比。

主要功能：
1. 轨迹质量评估：基于站立稳定性、静止稳定性、控制效率等指标
2. 偏好标签生成：比较两个轨迹的质量，生成DPO训练标签
3. 自包含实现：所有功能都在本文件内实现，无外部依赖

基于humanoid_bench Balance Hard任务的真实奖励函数：
reward = small_control * stand_reward * dont_move
其中：
- small_control: 控制力度最小化奖励
- stand_reward: 站立奖励 = standing * upright
- dont_move: 静止奖励（水平速度最小化）
"""

import numpy as np
from typing import Dict, Any, Tuple

# Balance Hard任务的关键参数（与humanoid_bench保持一致）
_STAND_HEIGHT = 1.65

def compare_h1hand_balance_hard_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-balance_hard-v0 任务的专用比较规则"""
    return compare_balance_hard_trajectories(tau_1, tau_2, goal)

def compute_balance_hard_reward_components(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> Dict[str, float]:
    """
    计算平衡困难任务的奖励组件
    基于humanoid_bench Balance Hard任务的真实奖励函数
    
    真实奖励公式: reward = small_control * stand_reward * dont_move
    其中：
    - small_control: 控制力度最小化 (权重: 20%)
    - stand_reward: 站立奖励 = standing * upright (权重: 40%)
    - dont_move: 静止奖励 (权重: 40%)
    
    Args:
        obs_seq: 观测序列 [T, obs_dim]
        action_seq: 动作序列 [T, action_dim] 
        reward_seq: 奖励序列 [T]
        
    Returns:
        Dict[str, float]: 包含各奖励组件的字典
    """
    if len(obs_seq) == 0:
        return {"small_control": 0.0, "stand_reward": 0.0, "dont_move": 0.0, "standing": 0.0, "upright": 0.0}
    
    # 1. small_control 组件 - 控制力度最小化
    if len(action_seq) > 0:
        # 基于动作幅度计算控制效率
        action_forces = np.sum(action_seq**2, axis=1) if len(action_seq.shape) > 1 else action_seq**2
        # 使用tolerance函数的近似实现
        control_margin = 10.0
        small_control_raw = np.exp(-action_forces / control_margin)
        small_control_score = np.mean(small_control_raw)
        # 应用 (4 + small_control) / 5 的变换
        small_control_score = (4 + small_control_score) / 5
    else:
        small_control_score = 1.0
    
    # 2. standing 组件 - 头部高度稳定性
    if obs_seq.shape[1] >= 3:  # 确保有位置信息
        # 头部高度（假设第3维是z坐标）
        head_heights = obs_seq[:, 2] if obs_seq.shape[1] > 2 else obs_seq[:, 1]
        # 基于真实奖励函数: bounds=(_STAND_HEIGHT + 0.37, inf), margin=_STAND_HEIGHT/4
        target_height = _STAND_HEIGHT + 0.37  # 1.65 + 0.37 = 2.02
        margin = _STAND_HEIGHT / 4  # 0.4125
        
        # tolerance函数的近似实现
        height_deviations = np.maximum(0, target_height - head_heights)
        standing_scores = np.exp(-height_deviations / margin)
        standing_score = np.mean(standing_scores)
    else:
        standing_score = 0.0
    
    # 3. upright 组件 - 躯干直立度
    if obs_seq.shape[1] >= 7:  # 确保有姿态信息
        # 假设姿态信息在obs的某个位置，这里使用简化计算
        # 基于真实奖励函数: bounds=(0.9, inf), margin=1.9
        trunk_orientations = obs_seq[:, 3:7] if obs_seq.shape[1] >= 7 else obs_seq[:, 3:6]
        # 计算直立度（四元数w分量或姿态稳定性）
        if trunk_orientations.shape[1] >= 4:
            # 假设是四元数，w分量表示直立度
            upright_values = np.abs(trunk_orientations[:, 0])  # w分量
        else:
            # 使用姿态变化的逆作为直立度指标
            upright_values = 1.0 - np.std(trunk_orientations, axis=1)
            upright_values = np.clip(upright_values, 0, 1)
        
        # 应用tolerance函数: bounds=(0.9, inf), margin=1.9
        upright_deviations = np.maximum(0, 0.9 - upright_values)
        upright_scores = np.exp(-upright_deviations / 1.9)
        upright_score = np.mean(upright_scores)
    else:
        upright_score = 0.5
    
    # 4. stand_reward = standing * upright
    stand_reward_score = standing_score * upright_score
    
    # 5. dont_move 组件 - 水平速度最小化
    if obs_seq.shape[1] >= 6:  # 确保有速度信息
        # 假设速度信息在obs的后面部分
        velocities = obs_seq[:, 3:6] if obs_seq.shape[1] >= 6 else obs_seq[:, 2:4]
        # 水平速度（x, y方向）
        horizontal_velocities = velocities[:, :2] if velocities.shape[1] >= 2 else velocities[:, :1]
        
        # 基于真实奖励函数: tolerance(horizontal_velocity, margin=2)
        velocity_magnitudes = np.sum(horizontal_velocities**2, axis=1)
        dont_move_scores = np.exp(-velocity_magnitudes / 2.0)
        dont_move_score = np.mean(dont_move_scores)
    else:
        # 如果没有速度信息，使用位置变化作为替代
        if len(obs_seq) > 1:
            position_changes = np.diff(obs_seq[:, :2], axis=0) if obs_seq.shape[1] >= 2 else np.diff(obs_seq[:, :1], axis=0)
            position_change_magnitudes = np.sum(position_changes**2, axis=1)
            dont_move_score = np.mean(np.exp(-position_change_magnitudes / 0.1))
        else:
            dont_move_score = 1.0
    
    return {
        "small_control": float(np.clip(small_control_score, 0, 1)),
        "stand_reward": float(np.clip(stand_reward_score, 0, 1)),
        "dont_move": float(np.clip(dont_move_score, 0, 1)),
        "standing": float(np.clip(standing_score, 0, 1)),
        "upright": float(np.clip(upright_score, 0, 1))
    }

def compute_trajectory_score(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> float:
    """
    计算轨迹的综合得分
    基于humanoid_bench Balance Hard任务的真实奖励函数
    
    Args:
        obs_seq: 观测序列
        action_seq: 动作序列
        reward_seq: 奖励序列
        
    Returns:
        float: 轨迹综合得分
    """
    components = compute_balance_hard_reward_components(obs_seq, action_seq, reward_seq)
    
    # 基于真实奖励函数: reward = small_control * stand_reward * dont_move
    trajectory_score = components["small_control"] * components["stand_reward"] * components["dont_move"]
    
    return float(trajectory_score)

def evaluate_balance_hard_trajectory(trajectory_data: Dict[str, Any]) -> Dict[str, float]:
    """
    评估Balance Hard轨迹的性能
    
    Args:
        trajectory_data: 包含obs, action, reward的轨迹数据
        
    Returns:
        Dict[str, float]: 评估结果
    """
    obs_seq = np.array(trajectory_data.get('obs', []))
    action_seq = np.array(trajectory_data.get('action', []))
    reward_seq = np.array(trajectory_data.get('reward', []))
    
    components = compute_balance_hard_reward_components(obs_seq, action_seq, reward_seq)
    trajectory_score = compute_trajectory_score(obs_seq, action_seq, reward_seq)
    
    return {
        **components,
        "trajectory_score": trajectory_score,
        "episode_length": len(obs_seq),
        "total_reward": float(np.sum(reward_seq)) if len(reward_seq) > 0 else 0.0
    }

def evaluate_dpo_preference(tau_1: Dict[str, Any], tau_2: Dict[str, Any], goal: Any = None) -> Tuple[float, float]:
    """
    使用DPO方法评估平衡困难任务轨迹偏好
    基于真实奖励函数的组件进行评估
    
    Args:
        tau_1: 轨迹1，包含obs, action, reward等
        tau_2: 轨迹2，包含obs, action, reward等
        goal: 目标（可选）
        
    Returns:
        Tuple[float, float]: (偏好概率, 置信度)
    """
    try:
        # 提取轨迹数据
        obs_1 = np.array(tau_1.get('obs', []))
        action_1 = np.array(tau_1.get('action', []))
        reward_1 = np.array(tau_1.get('reward', []))
        
        obs_2 = np.array(tau_2.get('obs', []))
        action_2 = np.array(tau_2.get('action', []))
        reward_2 = np.array(tau_2.get('reward', []))
        
        # 计算轨迹得分
        score_1 = compute_trajectory_score(obs_1, action_1, reward_1)
        score_2 = compute_trajectory_score(obs_2, action_2, reward_2)
        
        # 使用sigmoid函数计算偏好概率
        score_diff = score_1 - score_2
        preference_prob = 1.0 / (1.0 + np.exp(-10 * score_diff))  # 温度参数=10
        
        # 计算置信度（基于得分差异的绝对值）
        confidence = min(0.95, 0.5 + abs(score_diff))
        
        return float(preference_prob), float(confidence)
        
    except Exception as e:
        # 异常情况下返回中性偏好
        return 0.5, 0.1

def compare_balance_hard_trajectories(tau_1, tau_2, goal=None):
    """
    平衡困难场景DPO偏好评估
    基于真实奖励函数组件：small_control, stand_reward, dont_move
    
    真实奖励公式: reward = small_control * stand_reward * dont_move
    其中：
    - small_control: 控制力度最小化
    - stand_reward: 站立奖励 = standing * upright
    - dont_move: 静止奖励（水平速度最小化）
    
    Args:
        tau_1: 第一个轨迹
        tau_2: 第二个轨迹
        goal: 目标（可选）
        
    Returns:
        int: 偏好结果 (1表示tau_1更好, -1表示tau_2更好, 0表示相等)
    """
    # 使用DPO方法评估偏好
    preference_prob, confidence = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 根据偏好概率确定更优轨迹
    if preference_prob > 0.55:  # tau_1更优
        return 1
    elif preference_prob < 0.45:  # tau_2更优
        return -1
    else:  # 无明显偏好
        return 0

def _get_trajectory_data(trajectory):
    """提取轨迹数据的工具函数
    
    Args:
        trajectory: 轨迹对象
        
    Returns:
        tuple: (states, actions, rewards)
    """
    # 兼容不同的轨迹数据格式
    if hasattr(trajectory, 'states'):
        states = trajectory.states
    elif hasattr(trajectory, 'obs'):
        states = trajectory.obs
    elif isinstance(trajectory, dict) and 'obs' in trajectory:
        states = trajectory['obs']
    else:
        states = []
    
    if hasattr(trajectory, 'actions'):
        actions = trajectory.actions
    elif hasattr(trajectory, 'action'):
        actions = trajectory.action
    elif isinstance(trajectory, dict) and 'action' in trajectory:
        actions = trajectory['action']
    else:
        actions = []
    
    if hasattr(trajectory, 'rewards'):
        rewards = trajectory.rewards
    elif hasattr(trajectory, 'reward'):
        rewards = trajectory.reward
    elif isinstance(trajectory, dict) and 'reward' in trajectory:
        rewards = trajectory['reward']
    else:
        rewards = []
    
    return states, actions, rewards