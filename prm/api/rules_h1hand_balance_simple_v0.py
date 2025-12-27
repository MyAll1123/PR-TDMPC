
"""Balance Simple任务的启发式规则：基于轨迹质量评估的DPO偏好标签生成

该模块为Balance Simple任务实现启发式规则，用于生成DPO训练所需的偏好标签。
规则基于Balance Simple任务的特定指标和奖励函数，评估轨迹质量并生成偏好对比。

主要功能：
1. 轨迹质量评估：基于站立稳定性、静止稳定性、控制效率等指标
2. 偏好标签生成：比较两个轨迹的质量，生成DPO训练标签
3. 自包含实现：所有功能都在本文件内实现，无外部依赖
"""

import numpy as np
from typing import Dict, Any, Tuple

def compare_h1hand_balance_simple_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-balance_simple-v0 任务的专用比较规则"""
    return compare_balance_simple_trajectories(tau_1, tau_2, goal)

def compute_balance_simple_reward_components(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> Dict[str, float]:
    """
    计算平衡简单任务的奖励组件
    基于humanoid_bench平衡任务的真实奖励函数
    
    Args:
        obs_seq: 观测序列 [T, obs_dim]
        action_seq: 动作序列 [T, action_dim] 
        reward_seq: 奖励序列 [T]
        
    Returns:
        Dict[str, float]: 包含各奖励组件的字典
    """
    if len(obs_seq) == 0:
        return {"standing_stability": 0.0, "static_stability": 0.0, "control_efficiency": 0.0}
    
    # 1. 站立稳定性组件 (30%权重)
    # 基于头部高度和躯干直立度
    if obs_seq.shape[1] >= 3:  # 确保有位置信息
        # 头部高度稳定性（假设第3维是z坐标）
        head_heights = obs_seq[:, 2] if obs_seq.shape[1] > 2 else obs_seq[:, 1]
        target_height = 1.5  # 目标站立高度
        height_stability = 1.0 - np.mean(np.abs(head_heights - target_height)) / target_height
        height_stability = max(0.0, height_stability)
        
        # 躯干直立度（基于姿态信息）
        if obs_seq.shape[1] >= 7:  # 确保有姿态信息
            trunk_orientation = obs_seq[:, 3:7]  # 四元数或欧拉角
            upright_stability = 1.0 - np.mean(np.std(trunk_orientation, axis=0))
            upright_stability = max(0.0, upright_stability)
        else:
            upright_stability = 0.5
        
        standing_stability_score = 0.6 * height_stability + 0.4 * upright_stability
    else:
        standing_stability_score = 0.0
    
    # 2. 静止稳定性组件 (25%权重)
    # 基于位置和速度的稳定性
    if obs_seq.shape[1] >= 3:
        # 位置变化最小化
        position_changes = np.diff(obs_seq[:, :3], axis=0) if obs_seq.shape[1] >= 3 else np.diff(obs_seq[:, :2], axis=0)
        position_stability = 1.0 / (1.0 + np.mean(np.sum(position_changes**2, axis=1)))
        
        # 速度稳定性（如果有速度信息）
        if obs_seq.shape[1] >= 6:
            velocities = obs_seq[:, 3:6] if obs_seq.shape[1] > 5 else obs_seq[:, 2:5]
            velocity_stability = 1.0 / (1.0 + np.mean(np.sum(velocities**2, axis=1)))
        else:
            velocity_stability = position_stability
        
        static_stability_score = 0.7 * position_stability + 0.3 * velocity_stability
    else:
        static_stability_score = 0.0
    
    # 3. 控制效率组件 (45%权重)
    # 基于动作平滑性、控制幅度和能量效率
    if len(action_seq) > 1:
        # 动作平滑性
        action_smoothness = 1.0 / (1.0 + np.mean(np.sum(np.diff(action_seq, axis=0)**2, axis=1)))
        
        # 控制幅度最小化
        control_magnitude = 1.0 / (1.0 + np.mean(np.sum(action_seq**2, axis=1)))
        
        # 控制一致性
        control_consistency = 1.0 - np.mean(np.std(action_seq, axis=0)) if len(action_seq) > 1 else 1.0
        control_consistency = max(0.0, control_consistency)
        
        control_efficiency_score = 0.4 * action_smoothness + 0.3 * control_magnitude + 0.3 * control_consistency
    else:
        control_efficiency_score = 1.0
    
    return {
        "standing_stability": float(np.clip(standing_stability_score, 0, 1)),
        "static_stability": float(np.clip(static_stability_score, 0, 1)),
        "control_efficiency": float(np.clip(control_efficiency_score, 0, 1))
    }

def evaluate_dpo_preference(tau_1: Dict[str, Any], tau_2: Dict[str, Any], goal: Any = None) -> Tuple[float, float]:
    """
    使用DPO方法评估平衡简单任务轨迹偏好
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
        
        # 计算奖励组件
        components_1 = compute_balance_simple_reward_components(obs_1, action_1, reward_1)
        components_2 = compute_balance_simple_reward_components(obs_2, action_2, reward_2)
        
        # 权重配置（与labeling_config.yaml保持一致）
        weights = {
            "standing_stability": 0.30,
            "static_stability": 0.25,
            "control_efficiency": 0.45
        }
        
        # 计算加权综合奖励
        reward_1_weighted = sum(weights[k] * components_1[k] for k in weights.keys())
        reward_2_weighted = sum(weights[k] * components_2[k] for k in weights.keys())
        
        # 使用sigmoid函数计算偏好概率
        reward_diff = reward_1_weighted - reward_2_weighted
        preference_prob = 1.0 / (1.0 + np.exp(-10 * reward_diff))  # 温度参数=10
        
        # 计算置信度（基于奖励差异的绝对值）
        confidence = min(0.95, 0.5 + abs(reward_diff))
        
        return float(preference_prob), float(confidence)
        
    except Exception as e:
        # 异常情况下返回中性偏好
        return 0.5, 0.1

def compare_balance_simple_trajectories(tau_1, tau_2, goal=None):
    """
    平衡简单场景DPO偏好评估（替代原有的多级优先级策略）
    基于真实奖励函数组件：站立稳定性、静止稳定性、控制效率
    
    权重分配：
    - 站立稳定性: 30% (头部高度稳定性和躯干直立度)
    - 静止稳定性: 25% (位置和速度稳定性)
    - 控制效率: 45% (动作平滑性、控制幅度、控制一致性)
    
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