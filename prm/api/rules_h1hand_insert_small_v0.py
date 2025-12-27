
"""Insert Small任务的启发式规则：基于轨迹质量评估的DPO偏好标签生成

基于humanoid_bench Insert任务的真实奖励函数设计，评估机器人插入操作的能力。
针对insert_small场景，物体尺寸更小，需要更精确的控制。

核心奖励公式: reward = (0.5 * (small_control * stand_reward) + 0.5 * cube_target_reward) * (0.5 * peg_height_reward + 0.5 * hand_tool_proximity_reward)

评估维度:
- 稳定控制 (25%): 基于small_control和stand_reward的乘积
- 目标对齐 (25%): 基于立方体与目标位置的对齐程度
- 高度控制 (25%): 基于peg高度的控制
- 手部接近 (25%): 基于手部与工具的接近程度
"""

import numpy as np
from typing import Tuple, Optional

def compare_h1hand_insert_small_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-insert_small-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Insert任务真实奖励函数的DPO偏好评估
    针对insert_small场景优化，物体更小需要更精确的控制
    
    核心奖励公式: reward = (0.5 * (small_control * stand_reward) + 0.5 * cube_target_reward) * (0.5 * peg_height_reward + 0.5 * hand_tool_proximity_reward)
    
    评估维度权重:
    - 稳定控制 (25%): 基于small_control和stand_reward的乘积
    - 目标对齐 (25%): 基于立方体与目标位置的对齐程度
    - 高度控制 (25%): 基于peg高度的控制
    - 手部接近 (25%): 基于手部与工具的接近程度
    
    Args:
        traj_a: 轨迹A的状态-动作序列
        traj_b: 轨迹B的状态-动作序列
        goal: 目标参数（可选）
    
    Returns:
        (better_idx, worse_idx): 更优轨迹索引和较差轨迹索引
    """
    
    # 计算两个轨迹的综合得分
    score_a = calculate_trajectory_score(traj_a)
    score_b = calculate_trajectory_score(traj_b)
    
    # 设置偏好阈值，对于小物体任务需要更严格的判断
    preference_threshold = 0.03  # 比normal版本更小的阈值
    
    if abs(score_a - score_b) < preference_threshold:
        return None, None  # 差异太小，无明确偏好
    
    if score_a > score_b:
        return 0, 1  # 轨迹A更优
    else:
        return 1, 0  # 轨迹B更优

def calculate_trajectory_score(traj) -> float:
    """
    计算轨迹的综合得分
    
    基于humanoid_bench Insert任务的真实奖励函数:
    reward = (0.5 * (small_control * stand_reward) + 0.5 * cube_target_reward) * (0.5 * peg_height_reward + 0.5 * hand_tool_proximity_reward)
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度
    stable_control_score = stable_control_reward(states, actions)
    cube_target_score = cube_target_reward(states)
    peg_height_score = peg_height_reward(states)
    hand_tool_proximity_score = hand_tool_proximity_reward(states)
    
    # 按照真实奖励函数的结构计算得分
    # reward = (0.5 * (small_control * stand_reward) + 0.5 * cube_target_reward) * (0.5 * peg_height_reward + 0.5 * hand_tool_proximity_reward)
    first_part = 0.5 * stable_control_score + 0.5 * cube_target_score
    second_part = 0.5 * peg_height_score + 0.5 * hand_tool_proximity_score
    
    total_score = first_part * second_part
    
    return min(max(total_score, 0.0), 1.0)

def stable_control_reward(states, actions) -> float:
    """
    评估稳定控制 (对应 small_control * stand_reward)
    
    - small_control: 基于动作力的大小，较小的控制力更好
    - stand_reward: 基于站立稳定性 (standing * upright)
    """
    if len(states) == 0 or len(actions) == 0:
        return 0.0
    
    # 计算small_control得分
    small_control_score = _compute_small_control_reward(actions)
    
    # 计算stand_reward得分
    stand_reward_score = _compute_stand_reward(states)
    
    # 乘积形式（与真实奖励函数一致）
    stable_control_score = small_control_score * stand_reward_score
    
    return min(max(stable_control_score, 0.0), 1.0)

def cube_target_reward(states) -> float:
    """
    评估立方体目标对齐 (对应 cube_target_reward)
    
    基于block_peg与peg之间的距离
    针对insert_small，使用更严格的容差
    """
    if len(states) == 0:
        return 0.0
    
    target_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 20:
            try:
                # 模拟block_peg和peg的位置计算
                # 实际实现中需要根据环境状态结构调整
                
                # 假设状态中包含相关位置信息
                # 这里使用简化的距离计算
                block_peg_a_pos = state[15:18] if len(state) > 17 else [0.8, -0.2, 0.95]
                peg_a_pos = [0.8, -0.2, 0.95]  # 目标位置
                
                block_peg_b_pos = state[18:21] if len(state) > 20 else [0.8, 0.2, 0.95]
                peg_b_pos = [0.8, 0.2, 0.95]  # 目标位置
                
                # 计算距离
                distance_a = np.linalg.norm(np.array(block_peg_a_pos) - np.array(peg_a_pos))
                distance_b = np.linalg.norm(np.array(block_peg_b_pos) - np.array(peg_b_pos))
                
                # 使用tolerance函数逻辑计算奖励，针对小物体使用更严格的margin
                target_a = _compute_tolerance_reward(distance_a, margin=0.3, sigmoid="linear")  # 比normal版本更严格
                target_b = _compute_tolerance_reward(distance_b, margin=0.3, sigmoid="linear")
                
                # 平均得分
                target_score = (target_a + target_b) / 2
                target_scores.append(target_score)
                
            except:
                target_scores.append(0.0)
    
    return np.mean(target_scores) if target_scores else 0.0

def peg_height_reward(states) -> float:
    """
    评估peg高度控制 (对应 peg_height_reward)
    
    基于peg的z坐标与目标高度1.1的差异
    针对insert_small，使用更严格的高度控制
    """
    if len(states) == 0:
        return 0.0
    
    height_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 25:
            try:
                # 假设peg高度信息在状态中
                peg_a_height = state[21] if len(state) > 21 else 1.1
                peg_b_height = state[22] if len(state) > 22 else 1.1
                
                # 计算与目标高度1.1的差异
                height_diff_a = abs(peg_a_height - 1.1)
                height_diff_b = abs(peg_b_height - 1.1)
                
                # 使用tolerance函数逻辑，针对小物体使用更严格的margin
                height_reward_a = _compute_tolerance_reward(height_diff_a, margin=0.1, sigmoid="linear")  # 比normal版本更严格
                height_reward_b = _compute_tolerance_reward(height_diff_b, margin=0.1, sigmoid="linear")
                
                # 平均得分
                height_score = (height_reward_a + height_reward_b) / 2
                height_scores.append(height_score)
                
            except:
                height_scores.append(0.0)
    
    return np.mean(height_scores) if height_scores else 0.0

def hand_tool_proximity_reward(states) -> float:
    """
    评估手部工具接近度 (对应 hand_tool_proximity_reward)
    
    基于左右手与对应peg的距离
    针对insert_small，需要更精确的手部控制
    """
    if len(states) == 0:
        return 0.0
    
    proximity_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 15:
            try:
                # 假设手部位置在状态中
                left_hand_pos = state[7:10] if len(state) > 9 else [0, 0, 0]
                right_hand_pos = state[10:13] if len(state) > 12 else [0, 0, 0]
                
                # peg位置
                peg_a_pos = [0.8, -0.2, 0.95]
                peg_b_pos = [0.8, 0.2, 0.95]
                
                # 计算距离
                left_distance = np.linalg.norm(np.array(left_hand_pos) - np.array(peg_a_pos))
                right_distance = np.linalg.norm(np.array(right_hand_pos) - np.array(peg_b_pos))
                
                # 选择最小距离（与真实奖励函数一致）
                min_distance = min(left_distance, right_distance)
                
                # 使用tolerance函数逻辑，针对小物体使用更严格的接近度要求
                proximity_score = _compute_proximity_reward(min_distance, bounds=(0, 0.15), margin=0.3)  # 比normal版本更严格
                proximity_scores.append(proximity_score)
                
            except:
                proximity_scores.append(0.0)
    
    return np.mean(proximity_scores) if proximity_scores else 0.0

def _compute_small_control_reward(actions) -> float:
    """
    计算small_control奖励
    
    基于动作力的大小，使用二次sigmoid函数
    针对insert_small，对控制精度要求更高
    """
    if len(actions) == 0:
        return 0.0
    
    try:
        # 计算动作力的平均值
        action_forces = []
        for action in actions:
            if isinstance(action, (list, np.ndarray)):
                # 计算动作的L2范数作为力的度量
                force = np.linalg.norm(action)
                action_forces.append(force)
        
        if not action_forces:
            return 0.0
        
        avg_force = np.mean(action_forces)
        
        # 使用tolerance函数逻辑（二次sigmoid），针对小物体使用更严格的控制要求
        small_control = _compute_tolerance_reward(avg_force, margin=8, sigmoid="quadratic")  # 比normal版本更严格
        
        # 归一化处理 (4 + small_control) / 5
        normalized_control = (4 + small_control) / 5
        
        return min(max(normalized_control, 0.0), 1.0)
        
    except:
        return 0.0

def _compute_stand_reward(states) -> float:
    """
    计算stand_reward (standing * upright)
    
    - standing: 基于头部高度
    - upright: 基于躯干直立程度
    """
    if len(states) == 0:
        return 0.0
    
    try:
        stand_scores = []
        
        for state in states:
            if isinstance(state, (list, np.ndarray)) and len(state) > 5:
                # 假设头部高度在状态的第3个位置（z坐标）
                head_height = state[2] if len(state) > 2 else 1.65
                
                # 假设躯干直立度信息在状态中
                torso_upright = state[6] if len(state) > 6 else 1.0
                
                # 计算standing奖励
                standing = _compute_tolerance_reward(
                    head_height, 
                    bounds=(1.65, float('inf')), 
                    margin=1.65/4
                )
                
                # 计算upright奖励
                upright = _compute_tolerance_reward(
                    torso_upright,
                    bounds=(0.9, float('inf')),
                    margin=1.9,
                    sigmoid="linear"
                )
                
                # 乘积形式
                stand_score = standing * upright
                stand_scores.append(stand_score)
        
        return np.mean(stand_scores) if stand_scores else 0.0
        
    except:
        return 0.0

def _compute_tolerance_reward(value, bounds=None, margin=1.0, sigmoid="gaussian") -> float:
    """
    模拟dm_control.utils.rewards.tolerance函数
    """
    try:
        if bounds is None:
            # 无边界情况，使用高斯函数
            if sigmoid == "quadratic":
                return max(0.0, 1.0 - (value / margin) ** 2)
            else:
                return np.exp(-(value / margin) ** 2)
        else:
            lower, upper = bounds
            if lower <= value <= upper:
                return 1.0
            elif value < lower:
                diff = lower - value
            else:
                diff = value - upper
            
            if sigmoid == "linear":
                return max(0.0, 1.0 - diff / margin)
            elif sigmoid == "quadratic":
                return max(0.0, 1.0 - (diff / margin) ** 2)
            else:
                return np.exp(-(diff / margin) ** 2)
    except:
        return 0.0

def _compute_proximity_reward(distance, bounds=(0, 0.15), margin=0.3) -> float:
    """
    计算接近度奖励
    针对insert_small使用更严格的参数
    """
    return _compute_tolerance_reward(distance, bounds=bounds, margin=margin)

def _get_trajectory_data(traj, key):
    """
    从轨迹中提取指定类型的数据
    
    Args:
        traj: 轨迹数据
        key: 数据类型 ('obs', 'action', 'reward'等)
    
    Returns:
        提取的数据列表
    """
    try:
        if isinstance(traj, dict):
            return traj.get(key, [])
        elif isinstance(traj, (list, tuple)):
            # 假设轨迹是(state, action, reward, ...)的序列
            if key == 'obs' or key == 'state':
                return [step[0] for step in traj if len(step) > 0]
            elif key == 'action':
                return [step[1] for step in traj if len(step) > 1]
            elif key == 'reward':
                return [step[2] for step in traj if len(step) > 2]
        return []
    except:
        return []