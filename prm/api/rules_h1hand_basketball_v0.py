import numpy as np
from typing import Tuple, Optional, Dict

# ============================================================================
# H1Hand Basketball V0 任务启发式规则 - 自包含实现
# ============================================================================
# 基于 humanoid_bench Basketball 任务的真实奖励函数设计
# Basketball任务分为两个阶段：
# 1. catch阶段: reward = 0.5 * (stand_reward * small_control) + 0.5 * reward_hand_proximity
# 2. throw阶段: reward = 0.15 * (stand_reward * small_control) + 0.05 * reward_hand_proximity + 0.8 * reward_ball_success
# 成功投篮奖励: +1000
# ============================================================================

from .common_utils import tolerance

def compare_h1hand_basketball_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-basketball-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

# 自包含设计：移除外部依赖，所有功能集成在此文件中
# 如需注册到全局系统，由调用方负责

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Basketball任务真实奖励函数的DPO偏好评估
    
    Basketball任务包含两个阶段：
    1. catch阶段：接球阶段，重点是站立稳定性和手部接近篮球
    2. throw阶段：投篮阶段，重点是球到篮筐的距离
    
    核心奖励公式:
    - catch阶段: reward = 0.5 * (stand_reward * small_control) + 0.5 * reward_hand_proximity
    - throw阶段: reward = 0.15 * (stand_reward * small_control) + 0.05 * reward_hand_proximity + 0.8 * reward_ball_success
    - 成功投篮: +1000 bonus
    
    评估维度权重:
    - 站立稳定性 (25%): 基于standing和upright奖励
    - 控制效率 (25%): 基于small_control奖励
    - 手部接近度 (25%): 基于手到篮球的距离
    - 投篮成功度 (25%): 基于篮球到篮筐的距离
    
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
        return None, None
    
    if score_a > score_b:
        return 0, 1  # traj_a更优
    else:
        return 1, 0  # traj_b更优

def _compute_trajectory_score(traj) -> float:
    """
    计算轨迹的综合得分，基于Basketball任务的真实奖励函数
    
    Args:
        traj: 轨迹数据，包含states和actions
    
    Returns:
        float: 综合得分 (0-1范围)
    """
    
    # 提取轨迹数据
    states = _get_trajectory_data(traj, 'states')
    actions = _get_trajectory_data(traj, 'actions')
    
    if states is None or actions is None:
        return 0.0
    
    # 计算各个评估维度的得分
    standing_stability = _evaluate_standing_stability(states)
    control_efficiency = _evaluate_control_efficiency(actions)
    hand_proximity = _evaluate_hand_proximity(states)
    ball_success = _evaluate_ball_success(states)
    
    # 加权组合得分 (基于Basketball任务的奖励权重)
    total_score = (
        0.25 * standing_stability +
        0.25 * control_efficiency +
        0.25 * hand_proximity +
        0.25 * ball_success
    )
    
    return np.clip(total_score, 0.0, 1.0)

def _evaluate_standing_stability(states) -> float:
    """
    评估站立稳定性，基于Basketball任务的standing和upright奖励
    
    Basketball任务中的站立奖励:
    - standing = tolerance(head_height, bounds=(1.65, inf), margin=1.65/4)
    - upright = tolerance(torso_upright, bounds=(0.9, inf), margin=1.9)
    - stand_reward = standing * upright
    
    Args:
        states: 状态序列
    
    Returns:
        float: 站立稳定性得分 (0-1)
    """
    
    if len(states) == 0:
        return 0.0
    
    stability_scores = []
    
    for state in states:
        # 估算头部高度 (基于躯干位置)
        head_height = _estimate_head_height_from_state(state)
        
        # 估算躯干直立度 (基于四元数)
        upright = _estimate_upright_from_state(state)
        
        # 计算standing奖励
        standing = _tolerance_reward(
            head_height,
            bounds=(1.65, float('inf')),
            margin=1.65/4
        )
        
        # 计算upright奖励
        upright_reward = _tolerance_reward(
            upright,
            bounds=(0.9, float('inf')),
            margin=1.9,
            sigmoid='linear'
        )
        
        # 站立稳定性 = standing * upright
        stability = standing * upright_reward
        stability_scores.append(stability)
    
    return np.mean(stability_scores)

def _evaluate_control_efficiency(actions) -> float:
    """
    评估控制效率，基于Basketball任务的small_control奖励
    
    Basketball任务中的控制奖励:
    small_control = tolerance(actuator_forces, margin=10, sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    
    Args:
        actions: 动作序列
    
    Returns:
        float: 控制效率得分 (0-1)
    """
    
    if len(actions) == 0:
        return 0.0
    
    # 计算动作变化率作为控制效率的代理指标
    action_changes = []
    for i in range(1, len(actions)):
        change = np.linalg.norm(actions[i] - actions[i-1])
        action_changes.append(change)
    
    if len(action_changes) == 0:
        return 1.0
    
    # 使用tolerance函数评估控制平滑性
    avg_change = np.mean(action_changes)
    control_score = _tolerance_reward(
        avg_change,
        margin=10.0,
        sigmoid='quadratic'
    )
    
    # 应用Basketball任务的变换: (4 + score) / 5
    control_efficiency = (4 + control_score) / 5
    
    return np.clip(control_efficiency, 0.0, 1.0)

def _evaluate_hand_proximity(states) -> float:
    """
    评估手部接近度，基于Basketball任务的reward_hand_proximity
    
    Basketball任务中的手部接近度奖励:
    reward_hand_proximity = tolerance(
        max(left_hand_distance, right_hand_distance),
        bounds=(0, 0.2),
        margin=1
    )
    
    Args:
        states: 状态序列
    
    Returns:
        float: 手部接近度得分 (0-1)
    """
    
    if len(states) == 0:
        return 0.0
    
    proximity_scores = []
    
    for state in states:
        # 估算手到篮球的距离
        hand_ball_distance = _estimate_hand_ball_distance(state)
        
        # 计算接近度奖励
        proximity = _tolerance_reward(
            hand_ball_distance,
            bounds=(0, 0.2),
            margin=1.0
        )
        
        proximity_scores.append(proximity)
    
    return np.mean(proximity_scores)

def _evaluate_ball_success(states) -> float:
    """
    评估投篮成功度，基于Basketball任务的reward_ball_success
    
    Basketball任务中的投篮成功奖励:
    reward_ball_success = tolerance(ball_hoop_distance, margin=7, sigmoid='linear')
    成功投篮 (ball_hoop_distance < 0.05): +1000
    
    Args:
        states: 状态序列
    
    Returns:
        float: 投篮成功度得分 (0-1)
    """
    
    if len(states) == 0:
        return 0.0
    
    success_scores = []
    
    for state in states:
        # 估算篮球到篮筐的距离
        ball_hoop_distance = _estimate_ball_hoop_distance(state)
        
        # 计算投篮成功奖励
        success = _tolerance_reward(
            ball_hoop_distance,
            margin=7.0,
            sigmoid='linear'
        )
        
        # 检查是否成功投篮
        if ball_hoop_distance < 0.05:
            success = 1.0  # 成功投篮的最高奖励
        
        success_scores.append(success)
    
    return np.mean(success_scores)

def _get_trajectory_data(traj, key: str):
    """
    从轨迹中提取指定类型的数据
    
    支持多种轨迹数据格式:
    1. 字典格式: {'states': [...], 'actions': [...]}
    2. 元组格式: (states, actions)
    3. 列表格式: [state_action_pairs]
    
    Args:
        traj: 轨迹数据
        key: 数据类型 ('states' 或 'actions')
    
    Returns:
        numpy.ndarray: 提取的数据序列
    """
    
    try:
        if isinstance(traj, dict):
            return np.array(traj.get(key, []))
        elif isinstance(traj, (list, tuple)) and len(traj) >= 2:
            if key == 'states':
                return np.array(traj[0])
            elif key == 'actions':
                return np.array(traj[1])
        elif isinstance(traj, (list, tuple)) and len(traj) > 0:
            # 假设是state-action对的列表
            if key == 'states':
                return np.array([item[0] if isinstance(item, (list, tuple)) else item for item in traj])
            elif key == 'actions':
                return np.array([item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else np.zeros(10) for item in traj])
        
        return np.array([])
    except Exception:
        return np.array([])

def _estimate_head_height_from_state(state) -> float:
    """
    从状态向量估算头部高度
    
    假设状态向量的前几个元素包含躯干位置信息
    
    Args:
        state: 状态向量
    
    Returns:
        float: 估算的头部高度
    """
    
    if len(state) < 3:
        return 1.0  # 默认值
    
    # 假设state[2]是躯干的z坐标，头部高度约为躯干高度 + 0.2m
    torso_height = state[2]
    head_height = torso_height + 0.2
    
    return max(head_height, 0.5)  # 确保合理的最小值

def _estimate_upright_from_state(state) -> float:
    """
    从状态向量估算躯干直立度
    
    假设状态向量包含四元数信息
    
    Args:
        state: 状态向量
    
    Returns:
        float: 估算的直立度 (0-1)
    """
    
    if len(state) < 7:
        return 0.9  # 默认值
    
    # 假设state[3:7]是躯干的四元数 [w, x, y, z]
    quat = state[3:7]
    
    # 归一化四元数
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0:
        quat = quat / quat_norm
    else:
        return 0.9
    
    # 计算z轴方向的直立度
    # 对于直立姿态，z轴应该指向上方
    w, x, y, z = quat
    upright = 2 * (w*w + z*z) - 1  # z轴分量
    
    return max(upright, 0.0)

def _estimate_hand_ball_distance(state) -> float:
    """
    从状态向量估算手到篮球的距离
    
    Args:
        state: 状态向量
    
    Returns:
        float: 估算的手到篮球距离
    """
    
    # 这是一个简化的估算，实际实现需要根据具体的状态向量格式调整
    # 假设状态向量的后几个元素包含篮球位置信息
    if len(state) < 10:
        return 1.0  # 默认距离
    
    # 简化估算：基于状态向量的变化来推断距离
    # 实际应用中需要根据环境的具体状态表示来实现
    estimated_distance = np.linalg.norm(state[-3:] - state[:3]) if len(state) >= 6 else 1.0
    
    return max(estimated_distance, 0.0)

def _estimate_ball_hoop_distance(state) -> float:
    """
    从状态向量估算篮球到篮筐的距离
    
    Args:
        state: 状态向量
    
    Returns:
        float: 估算的篮球到篮筐距离
    """
    
    # 这是一个简化的估算，实际实现需要根据具体的状态向量格式调整
    if len(state) < 6:
        return 5.0  # 默认距离
    
    # 简化估算：假设篮筐位置相对固定
    # 实际应用中需要根据环境的具体状态表示来实现
    estimated_distance = np.linalg.norm(state[-3:]) if len(state) >= 3 else 5.0
    
    return max(estimated_distance, 0.0)

def _tolerance_reward(value, bounds=None, margin=1.0, sigmoid='linear'):
    """
    计算tolerance奖励，模拟dm_control.utils.rewards.tolerance函数
    
    Args:
        value: 输入值
        bounds: 奖励边界 (lower, upper)
        margin: 容忍边界
        sigmoid: 激活函数类型 ('linear', 'quadratic')
    
    Returns:
        float: 奖励值 (0-1)
    """
    
    if bounds is None:
        bounds = (0, float('inf'))
    
    lower, upper = bounds
    
    if lower <= value <= upper:
        return 1.0
    
    if value < lower:
        distance = lower - value
    else:
        distance = value - upper
    
    if sigmoid == 'quadratic':
        reward = np.exp(-(distance / margin) ** 2)
    else:  # linear
        reward = max(0.0, 1.0 - distance / margin)
    
    return np.clip(reward, 0.0, 1.0)

def compute_basketball_reward_components(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> Dict[str, float]:
    """
    计算Basketball任务的各个奖励组件
    
    Args:
        obs_seq: 观测序列
        action_seq: 动作序列
        reward_seq: 奖励序列
    
    Returns:
        Dict[str, float]: 各奖励组件的平均值
    """
    
    # 构造轨迹数据
    traj = {'states': obs_seq, 'actions': action_seq}
    
    # 计算各个组件
    standing_stability = _evaluate_standing_stability(obs_seq)
    control_efficiency = _evaluate_control_efficiency(action_seq)
    hand_proximity = _evaluate_hand_proximity(obs_seq)
    ball_success = _evaluate_ball_success(obs_seq)
    
    # 计算综合得分
    overall_score = _compute_trajectory_score(traj)
    
    return {
        'standing_stability': standing_stability,
        'control_efficiency': control_efficiency,
        'hand_proximity': hand_proximity,
        'ball_success': ball_success,
        'overall_score': overall_score,
        'mean_reward': np.mean(reward_seq) if len(reward_seq) > 0 else 0.0
    }

def compare_basketball_trajectories(tau_1, tau_2, goal):
    """
    Basketball轨迹比较的兼容性包装函数
    
    Args:
        tau_1: 轨迹1
        tau_2: 轨迹2
        goal: 目标参数
    
    Returns:
        Tuple: (更优轨迹, 较差轨迹) 或 (None, None)
    """
    result = compare_h1hand_basketball_v0_trajectories(tau_1, tau_2, goal)
    if result[0] is not None:
        return (tau_1, tau_2) if result[0] == 0 else (tau_2, tau_1)
    else:
        return None, None