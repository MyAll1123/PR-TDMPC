from .common_utils import tolerance

"""Window任务的启发式规则 - 基于humanoid_bench真实奖励函数

这个模块实现了基于humanoid_bench中window任务真实奖励函数的启发式规则。
Window任务要求机器人使用工具擦拭窗户，包含多个评估维度：
- 站立稳定性 (stand_reward)
- 控制效率 (small_control) 
- 手工具接近度 (hand_tool_proximity_reward)
- 移动擦拭奖励 (moving_wipe_reward)
- 头部窗户距离奖励 (head_window_distance_reward)
- 窗户接触奖励 (window_contact_reward)

真实奖励函数：
manipulation_reward = 0.2 * (stand_reward * small_control * head_window_distance_reward) + 0.4 * moving_wipe_reward + 0.4 * hand_tool_proximity_reward
window_contact_total_reward = window_contact_filter * window_contact_reward
reward = 0.5 * manipulation_reward + 0.5 * window_contact_total_reward
"""

import numpy as np
from typing import Dict, Any, Tuple


def compare_h1hand_window_v0_trajectories(trajectory_a, trajectory_b, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """比较两个window任务轨迹的偏好
    
    Args:
        trajectory_a: 第一个轨迹
        trajectory_b: 第二个轨迹
        **kwargs: 额外参数
    
    Returns:
        Tuple: (better_trajectory, worse_trajectory)
    """
    result = evaluate_dpo_preference(trajectory_a, trajectory_b, **kwargs)
    
    # 根据偏好返回轨迹对
    if result["preference"] > 0:  # 偏好轨迹A
        return (trajectory_a, trajectory_b)
    elif result["preference"] < 0:  # 偏好轨迹B
        return (trajectory_b, trajectory_a)
    else:  # 无明显偏好，返回原顺序
        return (trajectory_a, trajectory_b)


def evaluate_dpo_preference(trajectory_a, trajectory_b, **kwargs) -> Dict[str, Any]:
    """评估两个轨迹的DPO偏好
    
    基于humanoid_bench中window任务的真实奖励函数进行评估：
    - 站立稳定性：standing * upright
    - 控制效率：tolerance(actuator_forces)
    - 手工具接近度：min(left_hand_tool_distance, right_hand_tool_distance)
    - 移动擦拭奖励：tolerance(tool_velocity)
    - 头部窗户距离：tolerance(head_window_distance)
    - 窗户接触奖励：window_contact_filter * window_contact_reward
    
    Args:
        trajectory_a: 第一个轨迹
        trajectory_b: 第二个轨迹
        **kwargs: 额外参数
    
    Returns:
        Dict: 偏好评估结果
    """
    # 计算两个轨迹的综合得分
    score_a = _compute_trajectory_score(trajectory_a)
    score_b = _compute_trajectory_score(trajectory_b)
    
    # 确定偏好
    if abs(score_a - score_b) < 0.05:  # 分数差异很小
        preference = 0  # 无明显偏好
        confidence = 0.6
    elif score_a > score_b:
        preference = 1  # 偏好轨迹A
        confidence = min(0.9, 0.7 + (score_a - score_b) * 2)
    else:
        preference = -1  # 偏好轨迹B
        confidence = min(0.9, 0.7 + (score_b - score_a) * 2)
    
    return {
        "preference": preference,
        "confidence": confidence,
        "score_a": score_a,
        "score_b": score_b,
        "reasoning": f"Window task evaluation: A={score_a:.3f}, B={score_b:.3f}"
    }


def _compute_trajectory_score(trajectory) -> float:
    """计算轨迹的综合得分
    
    基于window任务的真实奖励函数计算得分：
    manipulation_reward = 0.2 * (stand_reward * small_control * head_window_distance_reward) + 0.4 * moving_wipe_reward + 0.4 * hand_tool_proximity_reward
    window_contact_total_reward = window_contact_filter * window_contact_reward
    reward = 0.5 * manipulation_reward + 0.5 * window_contact_total_reward
    
    Args:
        trajectory: 轨迹数据
    
    Returns:
        float: 综合得分 (0-1)
    """
    # 获取轨迹数据
    states, actions = _get_trajectory_data(trajectory)
    
    if len(states) == 0 or len(actions) == 0:
        return 0.0
    
    # 计算各个组件得分
    standing_stability = _evaluate_standing_stability(states)
    control_efficiency = _evaluate_control_efficiency(actions)
    hand_tool_proximity = _evaluate_hand_tool_proximity(states)
    moving_wipe_reward = _evaluate_moving_wipe_reward(states)
    head_window_distance = _evaluate_head_window_distance(states)
    window_contact_reward = _evaluate_window_contact_reward(states)
    
    # 按照真实奖励函数的权重计算
    # manipulation_reward = 0.2 * (stand_reward * small_control * head_window_distance_reward) + 0.4 * moving_wipe_reward + 0.4 * hand_tool_proximity_reward
    manipulation_reward = (
        0.2 * (standing_stability * control_efficiency * head_window_distance) +
        0.4 * moving_wipe_reward +
        0.4 * hand_tool_proximity
    )
    
    # window_contact_total_reward = window_contact_filter * window_contact_reward
    # 简化：假设有接触时window_contact_filter=1
    window_contact_total_reward = window_contact_reward
    
    # reward = 0.5 * manipulation_reward + 0.5 * window_contact_total_reward
    total_score = 0.5 * manipulation_reward + 0.5 * window_contact_total_reward
    
    return float(np.clip(total_score, 0.0, 1.0))


def _evaluate_standing_stability(states) -> float:
    """评估站立稳定性：standing * upright
    
    真实奖励函数中：
    standing = tolerance(head_height, bounds=(_STAND_HEIGHT, inf), margin=_STAND_HEIGHT/4)
    upright = tolerance(torso_upright, bounds=(0.9, inf), sigmoid='linear', margin=1.9, value_at_margin=0)
    stand_reward = standing * upright
    """
    if len(states) == 0:
        return 0.0
    
    stability_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 估算头部高度（质心高度 + 偏移）
        if len(obs) >= 3:
            head_height = obs[2] + 0.67  # 假设头部比质心高0.67米
        else:
            head_height = 1.65  # 默认站立高度
        
        # 计算standing奖励：tolerance(head_height, bounds=(1.65, inf), margin=1.65/4)
        standing = _tolerance_reward(head_height, bounds=(1.65, float('inf')), margin=1.65/4)
        
        # 估算直立度（基于四元数）
        if len(obs) >= 7:
            quaternion = obs[3:7]
            w, x, y, z = quaternion
            # 计算torso_upright：Z轴向上的分量
            upright_component = 2 * (w*w + z*z) - 1
            upright = _tolerance_reward(
                upright_component, 
                bounds=(0.9, float('inf')), 
                margin=1.9, 
                value_at_margin=0, 
                sigmoid='linear'
            )
        else:
            upright = 0.9  # 默认直立度
        
        # 站立稳定性 = standing * upright
        stability = standing * upright
        stability_scores.append(stability)
    
    return float(np.mean(stability_scores))


def _evaluate_control_efficiency(actions) -> float:
    """评估控制效率：tolerance(actuator_forces)
    
    真实奖励函数中：
    small_control = tolerance(actuator_forces, margin=10, value_at_margin=0, sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    """
    if len(actions) == 0:
        return 0.0
    
    control_scores = []
    
    for action in actions:
        action = np.array(action)
        
        # 计算控制力（模拟actuator_forces）
        actuator_forces = np.abs(action)  # 简化：使用动作的绝对值
        
        # 计算small_control奖励
        small_control = _tolerance_reward(
            actuator_forces,
            bounds=(0, 0),  # 期望控制力为0
            margin=10,
            value_at_margin=0,
            sigmoid='quadratic'
        )
        
        # 应用真实公式的变换
        small_control_mean = np.mean(small_control)
        small_control_final = (4 + small_control_mean) / 5
        
        control_scores.append(small_control_final)
    
    return float(np.mean(control_scores))


def _evaluate_hand_tool_proximity(states) -> float:
    """评估手工具接近度：min(left_hand_tool_distance, right_hand_tool_distance)
    
    真实奖励函数中：
    hand_tool_proximity_reward = min([
        tolerance(left_hand_tool_distance, bounds=(0, 0.2), margin=0.5),
        tolerance(right_hand_tool_distance, bounds=(0, 0.2), margin=0.5)
    ])
    """
    if len(states) == 0:
        return 0.0
    
    proximity_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 估算手到工具的距离（基于观测状态）
        if len(obs) >= 20:  # 假设有足够的观测维度
            # 简化：使用观测中的位置信息估算手工具距离
            left_hand_tool_distance = abs(obs[10]) if len(obs) > 10 else 0.3
            right_hand_tool_distance = abs(obs[11]) if len(obs) > 11 else 0.3
        else:
            left_hand_tool_distance = 0.3  # 默认距离
            right_hand_tool_distance = 0.3
        
        # 计算手工具接近度奖励
        left_proximity = _tolerance_reward(
            left_hand_tool_distance,
            bounds=(0, 0.2),
            margin=0.5
        )
        right_proximity = _tolerance_reward(
            right_hand_tool_distance,
            bounds=(0, 0.2),
            margin=0.5
        )
        
        # 取最小值（最好的手）
        hand_tool_proximity = min(left_proximity, right_proximity)
        proximity_scores.append(hand_tool_proximity)
    
    return float(np.mean(proximity_scores))


def _evaluate_moving_wipe_reward(states) -> float:
    """评估移动擦拭奖励：tolerance(tool_velocity)
    
    真实奖励函数中：
    moving_wipe_reward = tolerance(
        abs(window_wiping_tool_subtreelinvel[2]),
        bounds=(0.5, 0.5),
        margin=0.5
    )
    """
    if len(states) == 0:
        return 0.0
    
    wipe_scores = []
    
    for i, state in enumerate(states):
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 估算工具速度（基于位置变化）
        if i > 0 and len(obs) >= 15:
            prev_obs = states[i-1]
            if isinstance(prev_obs, dict):
                prev_obs = prev_obs["obs"]
            prev_obs = np.array(prev_obs)
            
            # 简化：使用观测中的位置变化估算速度
            if len(prev_obs) >= 15:
                tool_velocity = abs(obs[12] - prev_obs[12]) if len(obs) > 12 else 0.0
            else:
                tool_velocity = 0.0
        else:
            tool_velocity = 0.0
        
        # 计算移动擦拭奖励
        moving_wipe = _tolerance_reward(
            tool_velocity,
            bounds=(0.5, 0.5),  # 期望速度为0.5
            margin=0.5
        )
        
        wipe_scores.append(moving_wipe)
    
    return float(np.mean(wipe_scores))


def _evaluate_head_window_distance(states) -> float:
    """评估头部窗户距离奖励：tolerance(head_window_distance)
    
    真实奖励函数中：
    head_window_distance_reward = tolerance(
        norm(head_pos - head_pos0),
        bounds=(0.4, 0.4),
        margin=0.1
    )
    """
    if len(states) == 0:
        return 0.0
    
    distance_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 估算头部位置变化（基于质心位置）
        if len(obs) >= 3:
            head_pos_change = np.linalg.norm(obs[:3])  # 简化：使用质心位置的变化
        else:
            head_pos_change = 0.0
        
        # 计算头部窗户距离奖励
        head_window_distance = _tolerance_reward(
            head_pos_change,
            bounds=(0.4, 0.4),  # 期望距离为0.4
            margin=0.1
        )
        
        distance_scores.append(head_window_distance)
    
    return float(np.mean(distance_scores))


def _evaluate_window_contact_reward(states) -> float:
    """评估窗户接触奖励：window_contact_filter * window_contact_reward
    
    真实奖励函数中：
    window_contact_reward = min([
        tolerance(site_xpos[site_name, 'x'], bounds=(0.92, 0.92), margin=0.4, sigmoid='linear')
        for site_name in ['wipe_contact_site_a', 'wipe_contact_site_b', 'wipe_contact_site_c', 'wipe_contact_site_d', 'wipe_contact_site_e']
    ])
    window_contact_filter = 1 if contact detected else 0
    """
    if len(states) == 0:
        return 0.0
    
    contact_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 估算窗户接触（基于观测状态）
        if len(obs) >= 20:
            # 简化：使用观测中的位置信息估算接触情况
            contact_positions = [obs[i] for i in range(13, min(18, len(obs)))]  # 假设这些是接触位置
            
            # 计算接触奖励（模拟多个接触点）
            contact_rewards = []
            for pos in contact_positions:
                contact_reward = _tolerance_reward(
                    pos,
                    bounds=(0.92, 0.92),  # 期望接触位置
                    margin=0.4,
                    sigmoid='linear'
                )
                contact_rewards.append(contact_reward)
            
            # 取最小值（最差的接触点）
            window_contact = min(contact_rewards) if contact_rewards else 0.0
        else:
            window_contact = 0.0
        
        # 简化：假设有接触时window_contact_filter=1
        window_contact_filter = 1.0 if window_contact > 0.1 else 0.0
        window_contact_total = window_contact_filter * window_contact
        
        contact_scores.append(window_contact_total)
    
    return float(np.mean(contact_scores))


def _get_trajectory_data(trajectory):
    """从轨迹中提取状态和动作数据"""
    states = getattr(trajectory, "states", [])
    actions = getattr(trajectory, "actions", [])
    
    if hasattr(trajectory, "observations"):
        states = trajectory.observations
    if hasattr(trajectory, "obs"):
        states = trajectory.obs
        
    return states, actions


def _tolerance_reward(value, bounds, margin, value_at_margin=1.0, sigmoid='gaussian'):
    """计算tolerance奖励函数（模拟dm_control的tolerance函数）
    
    Args:
        value: 输入值或数组
        bounds: 奖励边界 (lower, upper)
        margin: 容忍边界
        value_at_margin: 边界处的奖励值
        sigmoid: sigmoid函数类型
    
    Returns:
        float or array: 奖励值 (0-1)
    """
    value = np.array(value)
    lower, upper = bounds
    
    # 处理无限边界
    if upper == float('inf'):
        if np.all(value >= lower):
            return np.ones_like(value) if value.shape else 1.0
        distance = np.maximum(0, lower - value)
    elif lower == upper:  # 单点目标
        distance = np.abs(value - lower)
    else:
        # 计算到边界的距离
        distance = np.maximum(0, np.maximum(lower - value, value - upper))
    
    # 使用指数衰减函数模拟tolerance
    if sigmoid == 'linear':
        reward = value_at_margin * np.maximum(0, 1 - distance / margin)
    elif sigmoid == 'quadratic':
        reward = value_at_margin * np.exp(-0.5 * (distance / margin) ** 2)
    else:  # gaussian (default)
        reward = value_at_margin * np.exp(-0.5 * (distance / margin) ** 2)
    
    return np.clip(reward, 0.0, 1.0)


# ============================================================================
# 公共接口函数
# ============================================================================

def compute_window_reward_components(trajectory, **kwargs) -> Dict[str, float]:
    """计算window任务的各个奖励组件
    
    Args:
        trajectory: 轨迹数据
        **kwargs: 额外参数
    
    Returns:
        Dict[str, float]: 各个奖励组件的得分
    """
    states, actions = _get_trajectory_data(trajectory)
    
    return {
        "standing_stability": _evaluate_standing_stability(states),
        "control_efficiency": _evaluate_control_efficiency(actions),
        "hand_tool_proximity": _evaluate_hand_tool_proximity(states),
        "moving_wipe_reward": _evaluate_moving_wipe_reward(states),
        "head_window_distance": _evaluate_head_window_distance(states),
        "window_contact_reward": _evaluate_window_contact_reward(states),
        "overall_score": _compute_trajectory_score({"states": states, "actions": actions})
    }


def compare_window_trajectories(trajectory_a, trajectory_b, **kwargs) -> Dict[str, Any]:
    """比较两个window轨迹（兼容性包装函数）
    
    Args:
        trajectory_a: 第一个轨迹
        trajectory_b: 第二个轨迹
        **kwargs: 额外参数
    
    Returns:
        Dict: 比较结果
    """
    return compare_h1hand_window_v0_trajectories(trajectory_a, trajectory_b, **kwargs)