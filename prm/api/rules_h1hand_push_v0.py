import numpy as np
from typing import Dict, Tuple, Union, Optional


def compare_h1hand_push_v0_trajectories(tau_1, tau_2, goal=None):
    """h1hand-push-v0 任务的专用比较规则"""
    preference_prob = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    if preference_prob > 0.5:
        return tau_1, tau_2  # tau_1 更好
    elif preference_prob < 0.5:
        return tau_2, tau_1  # tau_2 更好
    else:
        return None, None  # 两个轨迹相等

def compute_push_reward_components(trajectory: Union[Dict, object], goal: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    基于humanoid_bench Push类的真实奖励函数计算奖励组件
    
    Push任务真实奖励函数分析：
    - goal_dist: 箱子到目标的欧几里得距离
    - hand_dist: 机器人左手到箱子的欧几里得距离  
    - reward_success: 如果 goal_dist < 0.05，奖励 1000
    - 总奖励: -hand_penalty - penalty_dist + reward_success
    
    奖励组件权重分配：
    1. 推送成功率 (50%): 结合goal_dist和reward_success组件
    2. 接触效率 (30%): 基于hand_dist组件
    3. 控制稳定性 (20%): 基于动作平滑性
    
    Args:
        trajectory: 轨迹数据，包含observations和actions
        goal: 目标位置（可选）
        
    Returns:
        Tuple[float, float, float]: (推送成功率, 接触效率, 控制稳定性)
    """
    try:
        # 处理不同格式的轨迹数据
        if trajectory is None:
            return 0.0, 0.0, 0.0
            
        if isinstance(trajectory, dict):
            observations = trajectory.get('observations', [])
            actions = trajectory.get('actions', [])
        else:
            observations = getattr(trajectory, 'observations', [])
            actions = getattr(trajectory, 'actions', [])
        
        if not observations or not actions:
            return 0.0, 0.0, 0.0
        
        # 确保observations和actions是numpy数组
        if isinstance(observations, list):
            observations = np.array(observations)
        if isinstance(actions, list):
            actions = np.array(actions)
            
        # 从观测中提取关键状态信息
        # 假设观测格式：[position_x, position_y, position_z, ...]
        # 手部位置通常在观测的特定索引
        hand_positions = observations[:, 20:23] if observations.shape[1] > 22 else observations[:, :3]
        
        # 箱子位置（假设在观测的特定位置）
        box_positions = observations[:, 23:26] if observations.shape[1] > 25 else observations[:, 3:6]
        
        # 目标位置（默认为[1.0, 0.0, 1.0]）
        if goal is None:
            goal = np.array([1.0, 0.0, 1.0])
        
        # 1. 推送成功率组件
        # 计算箱子到目标的距离
        goal_distances = np.linalg.norm(box_positions - goal, axis=1)
        min_goal_dist = np.min(goal_distances)
        avg_goal_dist = np.mean(goal_distances)
        
        # 成功奖励（距离<0.05m）
        success_reward = 1.0 if min_goal_dist < 0.05 else 0.0
        
        # 距离奖励（越近越好）
        distance_reward = np.exp(-avg_goal_dist)  # 指数衰减
        
        # 综合推送成功率
        push_success = 0.6 * success_reward + 0.4 * distance_reward
        push_success = np.clip(push_success, 0.0, 1.0)
        
        # 2. 接触效率组件
        # 计算手到箱子的距离
        hand_to_box_distances = np.linalg.norm(hand_positions - box_positions, axis=1)
        avg_hand_dist = np.mean(hand_to_box_distances)
        min_hand_dist = np.min(hand_to_box_distances)
        
        # 接触效率（距离越近越好，但不能太远）
        contact_efficiency = np.exp(-avg_hand_dist * 2.0)  # 更快衰减
        
        # 考虑是否有有效接触
        contact_bonus = 1.0 if min_hand_dist < 0.1 else 0.5
        contact_efficiency = contact_efficiency * contact_bonus
        contact_efficiency = np.clip(contact_efficiency, 0.0, 1.0)
        
        # 3. 控制稳定性组件
        # 基于动作的平滑性
        if len(actions) > 1:
            action_changes = np.diff(actions, axis=0)
            action_smoothness = np.mean(np.linalg.norm(action_changes, axis=1))
            
            # 控制稳定性（变化越小越好）
            control_stability = np.exp(-action_smoothness)
        else:
            control_stability = 1.0
            
        # 考虑动作幅度
        action_magnitude = np.mean(np.linalg.norm(actions, axis=1))
        magnitude_penalty = np.exp(-action_magnitude * 0.5)  # 适度惩罚过大动作
        
        control_stability = control_stability * magnitude_penalty
        control_stability = np.clip(control_stability, 0.0, 1.0)
        
        return push_success, contact_efficiency, control_stability
        
    except Exception as e:
        # 异常情况返回默认值
        return 0.0, 0.0, 0.0

def evaluate_dpo_preference(traj1: Union[Dict, object], traj2: Union[Dict, object], goal: Optional[np.ndarray] = None) -> float:
    """
    使用DPO偏好评估方法比较两个轨迹
    
    Args:
        traj1: 第一个轨迹
        traj2: 第二个轨迹
        goal: 目标位置（可选）
        
    Returns:
        float: 偏好概率，>0.5表示偏好traj1，<0.5表示偏好traj2
    """
    try:
        # 计算两个轨迹的奖励组件
        push_success1, contact_efficiency1, control_stability1 = compute_push_reward_components(traj1, goal)
        push_success2, contact_efficiency2, control_stability2 = compute_push_reward_components(traj2, goal)
        
        # 权重分配：推送成功率50%，接触效率30%，控制稳定性20%
        weights = [0.5, 0.3, 0.2]
        
        # 计算加权综合奖励
        reward1 = (weights[0] * push_success1 + 
                  weights[1] * contact_efficiency1 + 
                  weights[2] * control_stability1)
        
        reward2 = (weights[0] * push_success2 + 
                  weights[1] * contact_efficiency2 + 
                  weights[2] * control_stability2)
        
        # 使用sigmoid函数计算偏好概率
        preference_prob = 1 / (1 + np.exp(-(reward1 - reward2)))
        
        return preference_prob
        
    except Exception as e:
        # 异常情况返回中性偏好
        return 0.5