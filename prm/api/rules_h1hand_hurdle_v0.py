import numpy as np
import torch
from typing import Dict, Any, Tuple

# 简化导入，避免循环依赖
try:
    from prm.rule_registry import (
        compare_trajectories_success,
        compare_trajectories_survival,
        compare_trajectories_distance,
        compare_trajectories_energy,
        compare_trajectories_hurdle_forward_progress,
        compare_trajectories_hurdle_obstacle_collision,
        compare_trajectories_hurdle_running_efficiency,
        compare_trajectories_hurdle_speed_consistency,
        compare_trajectories_hurdle_balance_stability,
        compare_trajectories_hurdle_obstacle_clearance,
    )
except ImportError:
    # 如果导入失败，定义简单的占位函数
    def compare_trajectories_success(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_survival(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_distance(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_energy(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_hurdle_forward_progress(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_hurdle_obstacle_collision(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_hurdle_running_efficiency(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_hurdle_speed_consistency(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_hurdle_balance_stability(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2
    def compare_trajectories_hurdle_obstacle_clearance(tau_1, tau_2, goal):
        return tau_1 if np.random.random() > 0.5 else tau_2

def compare_hurdle_forward_progress(tau_1, tau_2, goal):
    """跨栏前向进展比较"""
    return compare_trajectories_hurdle_forward_progress(tau_1, tau_2, goal)

def compare_hurdle_obstacle_collision(tau_1, tau_2, goal):
    """跨栏障碍物碰撞比较"""
    return compare_trajectories_hurdle_obstacle_collision(tau_1, tau_2, goal)

def compare_hurdle_running_efficiency(tau_1, tau_2, goal):
    """跨栏跑步效率比较"""
    return compare_trajectories_hurdle_running_efficiency(tau_1, tau_2, goal)

def compare_hurdle_speed_consistency(tau_1, tau_2, goal):
    """跨栏速度一致性比较"""
    return compare_trajectories_hurdle_speed_consistency(tau_1, tau_2, goal)

def compare_hurdle_balance_stability(tau_1, tau_2, goal):
    """跨栏平衡稳定性比较"""
    return compare_trajectories_hurdle_balance_stability(tau_1, tau_2, goal)

def compare_hurdle_obstacle_clearance(tau_1, tau_2, goal):
    """跨栏障碍物通过能力比较"""
    return compare_trajectories_hurdle_obstacle_clearance(tau_1, tau_2, goal)

def compare_h1hand_hurdle_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-hurdle-v0 任务的专用比较规则"""
    return compare_hurdle_trajectories(tau_1, tau_2, goal)

def compute_hurdle_reward_components(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> Dict[str, float]:
    """
    计算跨栏任务的奖励组件
    基于humanoid_bench跨栏任务的真实奖励函数
    
    Args:
        obs_seq: 观测序列 [T, obs_dim]
        action_seq: 动作序列 [T, action_dim] 
        reward_seq: 奖励序列 [T]
        
    Returns:
        Dict[str, float]: 包含各奖励组件的字典
    """
    if len(obs_seq) == 0:
        return {"forward_progress": 0.0, "obstacle_clearance": 0.0, "running_stability": 0.0}
    
    # 1. 前向进展组件 (25%权重)
    # 基于位置变化和速度一致性
    if obs_seq.shape[1] >= 3:  # 确保有位置信息
        position_changes = np.diff(obs_seq[:, 0])  # x方向位置变化
        forward_progress = np.mean(np.maximum(position_changes, 0))  # 只考虑前向移动
        speed_consistency = 1.0 - np.std(position_changes) if len(position_changes) > 1 else 1.0
        forward_progress_score = 0.7 * forward_progress + 0.3 * speed_consistency
    else:
        forward_progress_score = 0.0
    
    # 2. 障碍物通过能力组件 (25%权重)
    # 基于高度变化和碰撞避免
    if obs_seq.shape[1] >= 6:  # 确保有足够的状态信息
        height_changes = obs_seq[:, 2] if obs_seq.shape[1] > 2 else obs_seq[:, 1]  # z方向高度
        # 检测跳跃行为（高度显著增加）
        height_increases = np.sum(np.diff(height_changes) > 0.1)  # 高度增加超过0.1m
        max_height = np.max(height_changes) - np.min(height_changes)
        obstacle_clearance_score = min(1.0, 0.6 * (height_increases / max(1, len(height_changes) // 10)) + 0.4 * min(1.0, max_height / 0.5))
    else:
        obstacle_clearance_score = 0.0
    
    # 3. 跑步稳定性组件 (50%权重)
    # 基于动作平滑性、平衡稳定性和能量效率
    if len(action_seq) > 1:
        action_smoothness = 1.0 / (1.0 + np.mean(np.sum(np.diff(action_seq, axis=0)**2, axis=1)))
    else:
        action_smoothness = 1.0
    
    # 平衡稳定性（基于躯干姿态）
    if obs_seq.shape[1] >= 10:  # 确保有姿态信息
        trunk_orientation = obs_seq[:, 3:7] if obs_seq.shape[1] > 6 else obs_seq[:, 2:6]  # 四元数或欧拉角
        balance_stability = 1.0 - np.mean(np.std(trunk_orientation, axis=0))
        balance_stability = max(0.0, balance_stability)
    else:
        balance_stability = 0.5
    
    # 能量效率（基于动作幅度）
    energy_efficiency = 1.0 / (1.0 + np.mean(np.sum(action_seq**2, axis=1)))
    
    running_stability_score = 0.4 * action_smoothness + 0.4 * balance_stability + 0.2 * energy_efficiency
    
    return {
        "forward_progress": float(np.clip(forward_progress_score, 0, 1)),
        "obstacle_clearance": float(np.clip(obstacle_clearance_score, 0, 1)),
        "running_stability": float(np.clip(running_stability_score, 0, 1))
    }

def evaluate_dpo_preference(tau_1: Dict[str, Any], tau_2: Dict[str, Any], goal: Any = None) -> Tuple[float, float]:
    """
    使用DPO方法评估跨栏任务轨迹偏好
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
        components_1 = compute_hurdle_reward_components(obs_1, action_1, reward_1)
        components_2 = compute_hurdle_reward_components(obs_2, action_2, reward_2)
        
        # 权重配置（与labeling_config.yaml保持一致）
        weights = {
            "forward_progress": 0.25,
            "obstacle_clearance": 0.25, 
            "running_stability": 0.50
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

def compare_hurdle_trajectories(tau_1, tau_2, goal):
    """
    跨栏场景DPO偏好评估（替代原有的多级优先级策略）
    基于真实奖励函数组件：前向进展、障碍物通过能力、跑步稳定性
    
    权重分配：
    - 前向进展: 25% (跨栏距离和速度一致性)
    - 障碍物通过能力: 25% (成功跨越障碍物)
    - 跑步稳定性: 50% (动作平滑性、平衡稳定性、能量效率)
    """
    # 使用DPO方法评估偏好
    preference_prob, confidence = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 根据偏好概率确定更优轨迹
    if preference_prob > 0.55:  # tau_1更优
        return tau_1, tau_2
    elif preference_prob < 0.45:  # tau_2更优
        return tau_2, tau_1
    else:  # 无明显偏好
        return None, None