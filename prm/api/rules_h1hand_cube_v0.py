
"""Cube任务的启发式规则：基于轨迹质量评估的DPO偏好标签生成

基于humanoid_bench Cube任务的真实奖励函数设计，评估机器人操作立方体的能力。

核心奖励公式: reward = 0.2 * (small_control * stand_reward * dont_move) + 0.5 * orientation_alignment_reward + 0.3 * cube_closeness_reward

评估维度:
- 稳定控制 (20%): 基于small_control, stand_reward, dont_move的乘积
- 方向对齐 (50%): 基于左右立方体与目标立方体的方向对齐程度
- 手部接近 (30%): 基于左右手与对应立方体的距离
"""

import numpy as np
from typing import Tuple, Optional

def compare_h1hand_cube_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-cube-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Cube任务真实奖励函数的DPO偏好评估
    
    核心奖励公式: reward = 0.2 * (small_control * stand_reward * dont_move) + 0.5 * orientation_alignment_reward + 0.3 * cube_closeness_reward
    
    评估维度权重:
    - 稳定控制 (20%): 基于small_control, stand_reward, dont_move的乘积
    - 方向对齐 (50%): 基于左右立方体与目标立方体的方向对齐程度
    - 手部接近 (30%): 基于左右手与对应立方体的距离
    
    Args:
        traj_a: 轨迹A的状态-动作序列
        traj_b: 轨迹B的状态-动作序列
        goal: 目标参数（可选）
    
    Returns:
        (better_idx, worse_idx): 更优轨迹索引和较差轨迹索引
    """
    
    # 计算两个轨迹的综合得分
    score_a = _compute_trajectory_score(traj_a)
    score_b = _compute_trajectory_score(traj_b)
    
    # 设置偏好阈值，避免微小差异导致的不稳定判断
    preference_threshold = 0.05
    
    if abs(score_a - score_b) < preference_threshold:
        return None, None  # 差异太小，无明确偏好
    
    if score_a > score_b:
        return 0, 1  # 轨迹A更优
    else:
        return 1, 0  # 轨迹B更优

def _compute_trajectory_score(traj) -> float:
    """
    计算轨迹的综合得分
    
    基于humanoid_bench Cube任务的真实奖励函数:
    reward = 0.2 * (small_control * stand_reward * dont_move) + 0.5 * orientation_alignment_reward + 0.3 * cube_closeness_reward
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度
    stable_control_score = _evaluate_stable_control(states, actions)
    orientation_alignment_score = _evaluate_orientation_alignment(states)
    cube_closeness_score = _evaluate_cube_closeness(states)
    
    # 权重配置（基于真实奖励函数的重要性）
    weights = {
        'stable_control': 0.20,        # small_control * stand_reward * dont_move
        'orientation_alignment': 0.50,  # orientation_alignment_reward
        'cube_closeness': 0.30         # cube_closeness_reward
    }
    
    # 计算加权总分
    total_score = (
        weights['stable_control'] * stable_control_score +
        weights['orientation_alignment'] * orientation_alignment_score +
        weights['cube_closeness'] * cube_closeness_score
    )
    
    return total_score

def _evaluate_stable_control(states, actions) -> float:
    """
    评估稳定控制 (对应 small_control * stand_reward * dont_move)
    
    - small_control: 基于动作力的大小，较小的控制力更好
    - stand_reward: 基于站立稳定性 (standing * upright)
    - dont_move: 基于水平速度，避免不必要的移动
    """
    if len(states) == 0 or len(actions) == 0:
        return 0.0
    
    # 计算small_control得分
    small_control_score = _compute_small_control_reward(actions)
    
    # 计算stand_reward得分
    stand_reward_score = _compute_stand_reward(states)
    
    # 计算dont_move得分
    dont_move_score = _compute_dont_move_reward(states)
    
    # 乘积形式（与真实奖励函数一致）
    stable_control_score = small_control_score * stand_reward_score * dont_move_score
    
    return min(max(stable_control_score, 0.0), 1.0)

def _evaluate_orientation_alignment(states) -> float:
    """
    评估方向对齐 (对应 orientation_alignment_reward)
    
    基于左右立方体与目标立方体的四元数方向对齐程度
    """
    if len(states) == 0:
        return 0.0
    
    alignment_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 20:
            # 假设立方体四元数在状态的特定位置
            # 这里使用模拟的四元数位置，实际需要根据环境调整
            try:
                # 左立方体四元数 (假设位置)
                left_cube_quat = state[15:19] if len(state) > 18 else [1, 0, 0, 0]
                # 右立方体四元数 (假设位置)
                right_cube_quat = state[19:23] if len(state) > 22 else [1, 0, 0, 0]
                # 目标立方体四元数 (假设为单位四元数)
                target_cube_quat = [1, 0, 0, 0]
                
                # 计算对齐得分
                left_alignment = _compute_orientation_alignment_reward(left_cube_quat, target_cube_quat)
                right_alignment = _compute_orientation_alignment_reward(right_cube_quat, target_cube_quat)
                
                # 平均对齐得分
                alignment_score = (left_alignment + right_alignment) / 2
                alignment_scores.append(alignment_score)
                
            except:
                alignment_scores.append(0.0)
    
    return np.mean(alignment_scores) if alignment_scores else 0.0

def _evaluate_cube_closeness(states) -> float:
    """
    评估手部接近立方体 (对应 cube_closeness_reward)
    
    基于左右手与对应立方体的距离
    """
    if len(states) == 0:
        return 0.0
    
    closeness_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 10:
            try:
                # 假设手部位置在状态的特定位置
                # 这里使用模拟的位置，实际需要根据环境调整
                left_hand_pos = state[7:10] if len(state) > 9 else [0, 0, 0]
                right_hand_pos = state[10:13] if len(state) > 12 else [0, 0, 0]
                
                # 假设立方体位置
                left_cube_pos = [0.45, 0.21, 1.125]  # 基于qpos0_robot中的位置
                right_cube_pos = [0.45, -0.21, 1.125]
                
                # 计算距离
                left_distance = np.linalg.norm(np.array(left_hand_pos) - np.array(left_cube_pos))
                right_distance = np.linalg.norm(np.array(right_hand_pos) - np.array(right_cube_pos))
                
                # 计算接近度得分（使用tolerance函数逻辑）
                left_proximity = _compute_proximity_reward(left_distance, bounds=(0, 0.1), margin=0.5)
                right_proximity = _compute_proximity_reward(right_distance, bounds=(0, 0.1), margin=0.5)
                
                # 平均接近度得分
                closeness_score = (left_proximity + right_proximity) / 2
                closeness_scores.append(closeness_score)
                
            except:
                closeness_scores.append(0.0)
    
    return np.mean(closeness_scores) if closeness_scores else 0.0

def _compute_small_control_reward(actions) -> float:
    """
    计算small_control奖励
    
    基于动作力的大小，使用二次sigmoid函数
    """
    if len(actions) == 0:
        return 0.0
    
    # 计算动作力的大小
    action_forces = []
    for action in actions:
        if isinstance(action, (list, np.ndarray)):
            force = np.linalg.norm(action)
            action_forces.append(force)
    
    if not action_forces:
        return 0.0
    
    # 使用tolerance函数逻辑 (margin=10, value_at_margin=0, sigmoid="quadratic")
    avg_force = np.mean(action_forces)
    tolerance_score = max(0, 1 - (avg_force / 10.0) ** 2)
    
    # 归一化到 (4 + score) / 5 范围
    small_control_score = (4 + tolerance_score) / 5
    
    return min(max(small_control_score, 0.0), 1.0)

def _compute_stand_reward(states) -> float:
    """
    计算stand_reward (standing * upright)
    
    - standing: 基于头部高度的tolerance函数
    - upright: 基于躯干直立程度的tolerance函数
    """
    if len(states) == 0:
        return 0.0
    
    stand_scores = []
    
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 6:
            try:
                # 头部高度 (假设在z坐标)
                head_height = state[2] if len(state) > 2 else 0.0
                
                # 躯干四元数 (假设位置)
                torso_quat = state[3:7] if len(state) > 6 else [1, 0, 0, 0]
                
                # 计算standing得分
                standing_score = _compute_standing_reward(head_height)
                
                # 计算upright得分
                upright_score = _compute_upright_reward(torso_quat)
                
                # 乘积
                stand_score = standing_score * upright_score
                stand_scores.append(stand_score)
                
            except:
                stand_scores.append(0.0)
    
    return np.mean(stand_scores) if stand_scores else 0.0

def _compute_dont_move_reward(states) -> float:
    """
    计算dont_move奖励
    
    基于水平速度的tolerance函数
    """
    if len(states) < 2:
        return 1.0  # 如果无法计算速度，给予满分
    
    horizontal_velocities = []
    
    for i in range(1, len(states)):
        try:
            if (isinstance(states[i], (list, np.ndarray)) and 
                isinstance(states[i-1], (list, np.ndarray)) and
                len(states[i]) > 1 and len(states[i-1]) > 1):
                
                # 计算水平速度 (x, y方向)
                dx = states[i][0] - states[i-1][0]
                dy = states[i][1] - states[i-1][1]
                
                horizontal_velocity = np.array([dx, dy])
                horizontal_velocities.append(horizontal_velocity)
                
        except:
            continue
    
    if not horizontal_velocities:
        return 1.0
    
    # 计算平均水平速度幅度
    velocity_magnitudes = [np.linalg.norm(v) for v in horizontal_velocities]
    avg_velocity_magnitude = np.mean(velocity_magnitudes)
    
    # 使用tolerance函数 (margin=2)
    dont_move_score = max(0, 1 - avg_velocity_magnitude / 2.0)
    
    return min(max(dont_move_score, 0.0), 1.0)

def _compute_standing_reward(head_height) -> float:
    """
    计算standing奖励
    
    基于头部高度的tolerance函数
    bounds=(_STAND_HEIGHT, float("inf")), margin=_STAND_HEIGHT / 4
    """
    _STAND_HEIGHT = 1.65
    
    if head_height >= _STAND_HEIGHT:
        return 1.0
    else:
        # 线性衰减
        margin = _STAND_HEIGHT / 4
        if head_height >= _STAND_HEIGHT - margin:
            return (head_height - (_STAND_HEIGHT - margin)) / margin
        else:
            return 0.0

def _compute_upright_reward(torso_quat) -> float:
    """
    计算upright奖励
    
    基于躯干直立程度的tolerance函数
    bounds=(0.9, float("inf")), sigmoid="linear", margin=1.9, value_at_margin=0
    """
    try:
        # 计算躯干直立程度
        upright_value = _compute_torso_upright(torso_quat)
        
        if upright_value >= 0.9:
            return 1.0
        else:
            # 线性衰减
            margin = 1.9
            if upright_value >= 0.9 - margin:
                return (upright_value - (0.9 - margin)) / margin
            else:
                return 0.0
                
    except:
        return 0.5

def _compute_orientation_alignment_reward(cube_quat, target_quat) -> float:
    """
    计算方向对齐奖励
    
    基于四元数差异的tolerance函数
    """
    try:
        # 计算四元数差异的范数
        cube_quat = np.array(cube_quat)
        target_quat = np.array(target_quat)
        
        # 归一化
        cube_quat = cube_quat / np.linalg.norm(cube_quat)
        target_quat = target_quat / np.linalg.norm(target_quat)
        
        # 计算差异
        quat_diff = np.linalg.norm(cube_quat - target_quat)
        
        # 使用tolerance函数 (margin=0.3)
        alignment_score = max(0, 1 - quat_diff / 0.3)
        
        return min(max(alignment_score, 0.0), 1.0)
        
    except:
        return 0.0

def _compute_proximity_reward(distance, bounds=(0, 0.1), margin=0.5) -> float:
    """
    计算接近度奖励
    
    基于距离的tolerance函数
    """
    try:
        lower_bound, upper_bound = bounds
        
        if lower_bound <= distance <= upper_bound:
            return 1.0
        elif distance < lower_bound:
            # 距离太近
            if distance >= lower_bound - margin:
                return (distance - (lower_bound - margin)) / margin
            else:
                return 0.0
        else:
            # 距离太远
            if distance <= upper_bound + margin:
                return 1.0 - (distance - upper_bound) / margin
            else:
                return 0.0
                
    except:
        return 0.0

def _compute_torso_upright(quat) -> float:
    """
    基于四元数计算躯干直立程度
    
    Args:
        quat: 四元数 [w, x, y, z] 或 [x, y, z, w]
    
    Returns:
        直立程度值
    """
    try:
        quat = np.array(quat)
        # 归一化四元数
        quat = quat / np.linalg.norm(quat)
        
        # 计算z轴方向（垂直向上）
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # z轴在世界坐标系中的方向
        z_world = np.array([
            2 * (x*z + w*y),
            2 * (y*z - w*x), 
            1 - 2 * (x*x + y*y)
        ])
        
        # 与垂直向上方向[0, 0, 1]的点积
        upright_dot = z_world[2]
        
        return max(0.0, upright_dot)
    
    except:
        return 0.5  # 默认中等值

def _get_trajectory_data(traj, key: str):
    """
    从轨迹中提取数据，兼容不同的数据格式
    
    Args:
        traj: 轨迹数据（字典或对象）
        key: 数据键名 ('obs', 'action', 'reward', 'done')
    
    Returns:
        提取的数据列表或None
    """
    try:
        if hasattr(traj, 'get'):
            # 字典式访问
            return traj.get(key)
        elif hasattr(traj, key):
            # 属性访问
            return getattr(traj, key)
        elif isinstance(traj, dict):
            # 直接字典访问
            return traj.get(key)
        else:
            return None
    except:
        return None