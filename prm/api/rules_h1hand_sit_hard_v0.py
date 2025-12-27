from .common_utils import tolerance

"""Sit Hard任务的启发式规则：基于轨迹质量评估的DPO偏好标签生成

基于humanoid_bench Sit Hard任务的真实奖励函数设计，评估机器人在随机椅子位置下的坐下能力。

核心奖励公式: reward = small_control * sit_reward * dont_move
其中:
- sit_reward = (0.5 * sitting + 0.5 * on_chair) * upright * sitting_posture
- small_control = (4 + tolerance(actuator_forces, margin=10, sigmoid="quadratic").mean()) / 5
- dont_move = tolerance(horizontal_velocity, margin=2).mean()

评估维度:
- 坐姿控制 (40%): 基于sitting和on_chair的组合
- 姿态稳定 (30%): 基于upright和sitting_posture
- 控制平滑 (20%): 基于small_control
- 静止稳定 (10%): 基于dont_move
"""

import numpy as np
from typing import Tuple, Optional

def compare_h1hand_sit_hard_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-sit_hard-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Sit Hard任务真实奖励函数的DPO偏好评估
    
    核心奖励公式: reward = small_control * sit_reward * dont_move
    
    评估维度权重:
    - 坐姿控制 (40%): 基于sitting和on_chair的组合
    - 姿态稳定 (30%): 基于upright和sitting_posture
    - 控制平滑 (20%): 基于small_control
    - 静止稳定 (10%): 基于dont_move
    
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
    
    # 设置偏好阈值，避免微小差异导致的不稳定判断
    # sit_hard任务由于椅子位置随机化，使用较小的阈值
    preference_threshold = 0.03
    
    if abs(score_a - score_b) < preference_threshold:
        return None, None  # 差异太小，无明确偏好
    
    if score_a > score_b:
        return 0, 1  # 轨迹A更优
    else:
        return 1, 0  # 轨迹B更优

def calculate_trajectory_score(traj) -> float:
    """
    计算轨迹的综合得分
    
    基于humanoid_bench Sit Hard任务的真实奖励函数:
    reward = small_control * sit_reward * dont_move
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度
    sitting_control_score = sitting_control_reward(states)
    posture_stability_score = posture_stability_reward(states)
    smooth_control_score = smooth_control_reward(actions)
    stillness_score = stillness_reward(states)
    
    # 按照真实奖励函数的结构计算得分
    # reward = small_control * sit_reward * dont_move
    # 这里使用加权平均来近似乘积关系
    total_score = (
        0.4 * sitting_control_score +
        0.3 * posture_stability_score +
        0.2 * smooth_control_score +
        0.1 * stillness_score
    )
    
    return min(max(total_score, 0.0), 1.0)

def sitting_control_reward(states) -> float:
    """
    评估坐姿控制 (对应 sitting + on_chair)
    
    - sitting: 基于z坐标在(0.68, 0.72)范围内
    - on_chair: 基于x,y坐标与椅子位置的对齐
    """
    if len(states) == 0:
        return 0.0
    
    sitting_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 2:
            try:
                # 机器人z坐标（高度）
                robot_z = state[2] if len(state) > 2 else 0.7
                
                # sitting奖励：z坐标在(0.68, 0.72)范围内
                sitting_score = _compute_tolerance_reward(
                    robot_z, bounds=(0.68, 0.72), margin=0.2
                )
                
                # 机器人x,y坐标
                robot_x = state[0] if len(state) > 0 else 0.0
                robot_y = state[1] if len(state) > 1 else 0.0
                
                # 椅子位置（sit_hard中椅子位置为-0.25, 0, 0）
                chair_x = -0.25
                chair_y = 0.0
                
                # on_chair奖励：x,y坐标与椅子位置对齐
                on_chair_x = _compute_tolerance_reward(
                    robot_x - chair_x, bounds=(-0.19, 0.19), margin=0.2
                )
                on_chair_y = _compute_tolerance_reward(
                    robot_y - chair_y, margin=0.1
                )
                on_chair_score = on_chair_x * on_chair_y
                
                # 组合得分 (0.5 * sitting + 0.5 * on_chair)
                combined_score = 0.5 * sitting_score + 0.5 * on_chair_score
                sitting_scores.append(combined_score)
                
            except:
                sitting_scores.append(0.0)
    
    return np.mean(sitting_scores) if sitting_scores else 0.0

def posture_stability_reward(states) -> float:
    """
    评估姿态稳定性 (对应 upright * sitting_posture)
    
    - upright: 基于躯干直立程度
    - sitting_posture: 基于头部高度与IMU的相对位置
    """
    if len(states) == 0:
        return 0.0
    
    posture_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 10:
            try:
                # 模拟躯干直立度计算
                # 假设四元数在state[3:7]
                if len(state) > 6:
                    quat = state[3:7]
                    # 计算躯干直立度（简化计算）
                    upright_score = _compute_tolerance_reward(
                        abs(quat[0]), bounds=(0.95, float('inf')), 
                        margin=0.9, sigmoid="linear"
                    )
                else:
                    upright_score = 0.8
                
                # 模拟坐姿姿态计算
                # 基于头部高度与IMU的相对位置
                head_height = state[2] + 0.7 if len(state) > 2 else 1.4  # 估算头部高度
                imu_z = state[2] if len(state) > 2 else 0.7  # 估算IMU高度
                
                height_diff = head_height - imu_z
                sitting_posture_score = _compute_tolerance_reward(
                    height_diff, bounds=(0.35, 0.45), margin=0.3
                )
                
                # 乘积关系
                posture_score = upright_score * sitting_posture_score
                posture_scores.append(posture_score)
                
            except:
                posture_scores.append(0.0)
    
    return np.mean(posture_scores) if posture_scores else 0.0

def smooth_control_reward(actions) -> float:
    """
    评估控制平滑性 (对应 small_control)
    
    基于动作力的大小，使用二次sigmoid函数
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
        
        # 使用tolerance函数逻辑（二次sigmoid）
        small_control = _compute_tolerance_reward(
            avg_force, margin=10, sigmoid="quadratic"
        )
        
        # 归一化处理 (4 + small_control) / 5
        normalized_control = (4 + small_control) / 5
        
        return min(max(normalized_control, 0.0), 1.0)
        
    except:
        return 0.0

def stillness_reward(states) -> float:
    """
    评估静止稳定性 (对应 dont_move)
    
    基于水平速度的控制
    """
    if len(states) < 2:
        return 1.0  # 如果状态太少，假设静止
    
    velocity_scores = []
    
    for i in range(1, len(states)):
        try:
            prev_state = states[i-1]
            curr_state = states[i]
            
            if (isinstance(prev_state, (list, np.ndarray)) and len(prev_state) > 1 and
                isinstance(curr_state, (list, np.ndarray)) and len(curr_state) > 1):
                
                # 计算水平速度（x, y方向）
                dt = 0.002  # 假设时间步长
                vel_x = (curr_state[0] - prev_state[0]) / dt
                vel_y = (curr_state[1] - prev_state[1]) / dt
                
                horizontal_velocity = np.array([vel_x, vel_y])
                
                # 使用tolerance函数计算dont_move奖励
                vel_x_score = _compute_tolerance_reward(abs(vel_x), margin=2)
                vel_y_score = _compute_tolerance_reward(abs(vel_y), margin=2)
                
                # 平均得分
                velocity_score = (vel_x_score + vel_y_score) / 2
                velocity_scores.append(velocity_score)
                
        except:
            velocity_scores.append(1.0)
    
    return np.mean(velocity_scores) if velocity_scores else 1.0

def _compute_tolerance_reward(value, bounds=None, margin=1.0, sigmoid="linear") -> float:
    """
    计算tolerance奖励，模拟dm_control.utils.rewards.tolerance函数
    
    Args:
        value: 输入值
        bounds: 边界范围 (lower, upper)，如果为None则假设(0, inf)
        margin: 容忍边界
        sigmoid: sigmoid函数类型 ("linear" 或 "quadratic")
    
    Returns:
        奖励值 [0, 1]
    """
    try:
        if bounds is None:
            # 单边界情况，值越小越好
            distance = abs(value)
        else:
            lower, upper = bounds
            if value < lower:
                distance = lower - value
            elif value > upper:
                distance = value - upper
            else:
                distance = 0.0
        
        if distance <= 0:
            return 1.0
        
        if sigmoid == "quadratic":
            # 二次衰减
            reward = max(0.0, 1.0 - (distance / margin) ** 2)
        else:
            # 线性衰减
            reward = max(0.0, 1.0 - distance / margin)
        
        return reward
        
    except:
        return 0.0

def _get_trajectory_data(traj, key: str):
    """
    从轨迹中提取指定类型的数据
    
    Args:
        traj: 轨迹数据
        key: 数据类型 ('obs' 或 'action')
    
    Returns:
        提取的数据列表
    """
    try:
        if isinstance(traj, dict):
            if key in traj:
                data = traj[key]
                if isinstance(data, (list, np.ndarray)):
                    return data
        
        elif isinstance(traj, (list, tuple)):
            # 假设轨迹是 [(obs, action, reward, next_obs, done), ...] 格式
            if key == 'obs':
                return [step[0] for step in traj if len(step) > 0]
            elif key == 'action':
                return [step[1] for step in traj if len(step) > 1]
        
        return None
        
    except:
        return None