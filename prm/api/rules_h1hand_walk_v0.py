import numpy as np
from typing import Tuple, Optional


def compare_h1hand_walk_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-walk-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Walk任务真实奖励函数的DPO偏好评估
    
    核心奖励公式: reward = small_control × standing × upright × move
    
    评估维度权重:
    - 存活稳定性 (40%): 基于standing和upright奖励
    - 前进运动 (35%): 基于move奖励和水平速度
    - 动作效率 (25%): 基于small_control奖励
    
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
    
    基于humanoid_bench Walk任务的真实奖励函数:
    reward = small_control × standing × upright × move
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度
    survival_stability_score = _evaluate_survival_stability(states, actions)
    forward_motion_score = _evaluate_forward_motion(states)
    action_efficiency_score = _evaluate_action_efficiency(actions)
    
    # 权重配置（基于真实奖励函数的重要性）
    weights = {
        'survival_stability': 0.40,  # standing × upright 的重要性
        'forward_motion': 0.35,      # move 的重要性
        'action_efficiency': 0.25    # small_control 的重要性
    }
    
    # 计算加权总分
    total_score = (
        weights['survival_stability'] * survival_stability_score +
        weights['forward_motion'] * forward_motion_score +
        weights['action_efficiency'] * action_efficiency_score
    )
    
    return total_score

def _evaluate_survival_stability(states, actions) -> float:
    """
    评估存活稳定性 (对应 standing × upright)
    
    - standing: 基于头部高度 (torso_height)
    - upright: 基于躯干姿态的直立程度
    """
    if len(states) == 0:
        return 0.0
    
    # 计算存活时间得分（轨迹长度归一化）
    survival_score = min(len(states) / 1000.0, 1.0)  # 假设1000步为满分
    
    # 计算头部高度稳定性（standing奖励）
    head_heights = []
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 0:
            # 假设头部高度在状态的前几个维度中
            head_height = state[2] if len(state) > 2 else 0.0
            head_heights.append(head_height)
    
    if head_heights:
        # 期望头部高度约为1.3米（humanoid站立高度）
        target_height = 1.3
        height_scores = [max(0, 1 - abs(h - target_height) / target_height) for h in head_heights]
        standing_score = np.mean(height_scores)
    else:
        standing_score = 0.0
    
    # 计算姿态直立程度（upright奖励）
    upright_scores = []
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 6:
            # 假设四元数在状态的3-6位置
            quat = state[3:7]
            if len(quat) == 4:
                # 计算与垂直方向的偏差
                upright_score = _compute_upright_score(quat)
                upright_scores.append(upright_score)
    
    upright_score = np.mean(upright_scores) if upright_scores else 0.5
    
    # 综合存活稳定性得分
    stability_score = 0.4 * survival_score + 0.4 * standing_score + 0.2 * upright_score
    
    return min(max(stability_score, 0.0), 1.0)

def _evaluate_forward_motion(states) -> float:
    """
    评估前进运动能力 (对应 move 奖励)
    
    基于水平速度和位移计算前进运动得分
    """
    if len(states) < 2:
        return 0.0
    
    # 计算水平位移
    positions = []
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 1:
            # 假设x, y位置在状态的前两个维度
            pos = [state[0], state[1]]
            positions.append(pos)
    
    if len(positions) < 2:
        return 0.0
    
    # 计算总的前进距离（主要是x方向）
    total_forward_distance = positions[-1][0] - positions[0][0]
    
    # 计算平均前进速度
    forward_distances = []
    for i in range(1, len(positions)):
        forward_dist = positions[i][0] - positions[i-1][0]
        forward_distances.append(forward_dist)
    
    avg_forward_speed = np.mean(forward_distances) if forward_distances else 0.0
    
    # 目标前进速度约为1.0 m/s（WALK_SPEED）
    target_speed = 1.0
    speed_score = max(0, 1 - abs(avg_forward_speed - target_speed) / target_speed)
    
    # 总距离得分（鼓励更长的前进距离）
    distance_score = min(total_forward_distance / 10.0, 1.0)  # 10米为满分
    
    # 运动一致性（避免来回摆动）
    consistency_score = 1.0
    if len(forward_distances) > 1:
        speed_variance = np.var(forward_distances)
        consistency_score = max(0, 1 - speed_variance)
    
    # 综合前进运动得分
    motion_score = 0.4 * speed_score + 0.4 * distance_score + 0.2 * consistency_score
    
    return min(max(motion_score, 0.0), 1.0)

def _evaluate_action_efficiency(actions) -> float:
    """
    评估动作效率 (对应 small_control 奖励)
    
    基于动作幅度和平滑性评估控制效率
    """
    if len(actions) == 0:
        return 0.0
    
    # 计算动作幅度（控制力大小）
    action_magnitudes = []
    for action in actions:
        if isinstance(action, (list, np.ndarray)):
            magnitude = np.linalg.norm(action)
            action_magnitudes.append(magnitude)
    
    if not action_magnitudes:
        return 0.0
    
    # 控制力得分（较小的控制力更好）
    avg_magnitude = np.mean(action_magnitudes)
    # 假设动作范围在[-1, 1]，理想幅度约为0.5
    target_magnitude = 0.5
    magnitude_score = max(0, 1 - abs(avg_magnitude - target_magnitude) / target_magnitude)
    
    # 动作平滑性得分
    smoothness_score = 1.0
    if len(actions) > 1:
        action_diffs = []
        for i in range(1, len(actions)):
            if (isinstance(actions[i], (list, np.ndarray)) and 
                isinstance(actions[i-1], (list, np.ndarray))):
                diff = np.linalg.norm(np.array(actions[i]) - np.array(actions[i-1]))
                action_diffs.append(diff)
        
        if action_diffs:
            avg_diff = np.mean(action_diffs)
            # 较小的动作变化表示更平滑
            smoothness_score = max(0, 1 - avg_diff)
    
    # 综合动作效率得分
    efficiency_score = 0.6 * magnitude_score + 0.4 * smoothness_score
    
    return min(max(efficiency_score, 0.0), 1.0)

def _compute_upright_score(quat) -> float:
    """
    基于四元数计算直立程度得分
    
    Args:
        quat: 四元数 [w, x, y, z] 或 [x, y, z, w]
    
    Returns:
        直立程度得分 [0, 1]
    """
    try:
        quat = np.array(quat)
        # 归一化四元数
        quat = quat / np.linalg.norm(quat)
        
        # 计算z轴方向（垂直向上）
        # 四元数旋转矩阵的z轴分量
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # z轴在世界坐标系中的方向
        z_world = np.array([
            2 * (x*z + w*y),
            2 * (y*z - w*x), 
            1 - 2 * (x*x + y*y)
        ])
        
        # 与垂直向上方向[0, 0, 1]的点积
        upright_dot = z_world[2]
        
        # 转换为得分 [0, 1]
        upright_score = (upright_dot + 1) / 2
        
        return max(0.0, min(1.0, upright_score))
    
    except:
        return 0.5  # 默认中等得分

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