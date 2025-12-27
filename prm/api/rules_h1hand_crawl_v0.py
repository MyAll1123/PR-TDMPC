import numpy as np
from typing import Tuple, Optional, Dict

# ============================================================================
# H1Hand Crawl V0 任务启发式规则 - 自包含实现
# ============================================================================
# 基于 humanoid_bench Crawl 任务的真实奖励函数设计
# 核心奖励公式: reward = (0.1*small_control + 0.25*min(crawling, crawling_head) + 0.4*move + 0.25*reward_xquat) * in_tunnel
# ============================================================================

from .common_utils import tolerance

def compare_h1hand_crawl_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-crawl-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Crawl任务真实奖励函数的DPO偏好评估
    
    核心奖励公式: reward = (0.1*small_control + 0.25*min(crawling, crawling_head) + 0.4*move + 0.25*reward_xquat) * in_tunnel
    
    评估维度权重:
    - 爬行姿态稳定性 (40%): 基于crawling和crawling_head奖励
    - 前进运动能力 (35%): 基于move奖励和水平速度
    - 控制效率 (15%): 基于small_control奖励
    - 姿态方向 (10%): 基于reward_xquat奖励
    
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
    
    基于humanoid_bench Crawl任务的真实奖励函数:
    reward = (0.1*small_control + 0.25*min(crawling, crawling_head) + 0.4*move + 0.25*reward_xquat) * in_tunnel
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度
    crawling_posture_score = _evaluate_crawling_posture(states)
    forward_motion_score = _evaluate_forward_motion(states)
    control_efficiency_score = _evaluate_control_efficiency(actions)
    orientation_score = _evaluate_orientation(states)
    
    # 权重配置（基于真实奖励函数的重要性）
    weights = {
        'crawling_posture': 0.40,    # crawling + crawling_head 的重要性
        'forward_motion': 0.35,      # move 的重要性
        'control_efficiency': 0.15,  # small_control 的重要性
        'orientation': 0.10          # reward_xquat 的重要性
    }
    
    # 计算加权总分
    total_score = (
        weights['crawling_posture'] * crawling_posture_score +
        weights['forward_motion'] * forward_motion_score +
        weights['control_efficiency'] * control_efficiency_score +
        weights['orientation'] * orientation_score
    )
    
    return total_score

def _evaluate_crawling_posture(states) -> float:
    """
    评估爬行姿态稳定性 (对应 crawling + crawling_head)
    
    - crawling: 基于IMU传感器高度在爬行范围内 (0.6-1.0m)
    - crawling_head: 基于头部高度在爬行范围内 (0.6-1.0m)
    """
    if len(states) == 0:
        return 0.0
    
    # 爬行目标高度范围
    _CRAWL_HEIGHT = 0.8
    crawl_height_bounds = (_CRAWL_HEIGHT - 0.2, _CRAWL_HEIGHT + 0.2)  # (0.6, 1.0)
    
    # 计算存活时间得分（轨迹长度归一化）
    survival_score = min(len(states) / 1000.0, 1.0)  # 假设1000步为满分
    
    # 计算头部高度稳定性（crawling_head奖励）
    head_height_scores = []
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 2:
            # 假设头部高度在状态的第3个维度
            head_height = state[2] if len(state) > 2 else 0.0
            # 使用tolerance函数计算得分
            if crawl_height_bounds[0] <= head_height <= crawl_height_bounds[1]:
                score = 1.0 - abs(head_height - _CRAWL_HEIGHT) / 0.2
            else:
                score = max(0.0, 1.0 - abs(head_height - _CRAWL_HEIGHT) / 1.0)
            head_height_scores.append(max(0.0, min(1.0, score)))
    
    crawling_head_score = np.mean(head_height_scores) if head_height_scores else 0.0
    
    # 计算躯干高度稳定性（crawling奖励）
    # 假设IMU传感器高度信息在状态中
    torso_height_scores = []
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 5:
            # 假设躯干高度在状态的某个维度，这里使用头部高度作为近似
            torso_height = state[2] if len(state) > 2 else 0.0
            # 使用tolerance函数计算得分
            if crawl_height_bounds[0] <= torso_height <= crawl_height_bounds[1]:
                score = 1.0 - abs(torso_height - _CRAWL_HEIGHT) / 0.2
            else:
                score = max(0.0, 1.0 - abs(torso_height - _CRAWL_HEIGHT) / 1.0)
            torso_height_scores.append(max(0.0, min(1.0, score)))
    
    crawling_score = np.mean(torso_height_scores) if torso_height_scores else 0.0
    
    # 综合爬行姿态得分（取最小值，如真实奖励函数中的min(crawling, crawling_head)）
    posture_score = min(crawling_score, crawling_head_score)
    
    # 结合存活时间
    final_score = 0.7 * posture_score + 0.3 * survival_score
    
    return min(max(final_score, 0.0), 1.0)

def _evaluate_forward_motion(states) -> float:
    """
    评估前进运动能力 (对应 move 奖励)
    
    基于质心水平速度计算前进运动得分，目标速度为1 m/s以上
    """
    if len(states) < 2:
        return 0.0
    
    # 计算水平位移和速度
    positions = []
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 1:
            # 假设x, y位置在状态的前两个维度
            pos = [state[0], state[1]]
            positions.append(pos)
    
    if len(positions) < 2:
        return 0.0
    
    # 计算质心速度（主要是x方向）
    velocities = []
    for i in range(1, len(positions)):
        velocity_x = positions[i][0] - positions[i-1][0]
        velocities.append(velocity_x)
    
    if not velocities:
        return 0.0
    
    # 计算平均前进速度
    avg_velocity = np.mean(velocities)
    
    # 基于真实奖励函数的move计算：rewards.tolerance(com_velocity, bounds=(1, inf), margin=1, sigmoid="linear")
    # 目标速度为1 m/s以上
    target_speed = 1.0
    if avg_velocity >= target_speed:
        speed_score = 1.0
    else:
        # 线性衰减
        speed_score = max(0.0, avg_velocity / target_speed)
    
    # 应用真实奖励函数的缩放：(5 * move + 1) / 6
    scaled_speed_score = (5 * speed_score + 1) / 6
    
    # 运动一致性（避免来回摆动）
    consistency_score = 1.0
    if len(velocities) > 1:
        velocity_variance = np.var(velocities)
        consistency_score = max(0.0, 1.0 - velocity_variance)
    
    # 综合前进运动得分
    motion_score = 0.8 * scaled_speed_score + 0.2 * consistency_score
    
    return min(max(motion_score, 0.0), 1.0)

def _evaluate_control_efficiency(actions) -> float:
    """
    评估控制效率 (对应 small_control 奖励)
    
    基于动作幅度评估控制效率，较小的控制力更好
    """
    if len(actions) == 0:
        return 0.0
    
    # 计算动作幅度（控制力大小）
    action_forces = []
    for action in actions:
        if isinstance(action, (list, np.ndarray)):
            # 计算动作的L2范数作为控制力
            force = np.linalg.norm(action)
            action_forces.append(force)
    
    if not action_forces:
        return 0.0
    
    # 基于真实奖励函数的small_control计算
    # rewards.tolerance(forces, margin=10, value_at_margin=0, sigmoid="quadratic").mean()
    control_scores = []
    for force in action_forces:
        # 使用二次sigmoid函数，margin=10
        if force <= 10:
            score = 1.0 - (force / 10) ** 2
        else:
            score = 0.0
        control_scores.append(max(0.0, score))
    
    avg_control_score = np.mean(control_scores)
    
    # 应用真实奖励函数的缩放：(4 + small_control) / 5
    scaled_control_score = (4 + avg_control_score) / 5
    
    return min(max(scaled_control_score, 0.0), 1.0)

def _evaluate_orientation(states) -> float:
    """
    评估姿态方向 (对应 reward_xquat)
    
    基于四元数与目标姿态的偏差计算得分
    目标四元数为 [0.75, 0, 0.65, 0]（爬行姿态）
    """
    if len(states) == 0:
        return 0.0
    
    target_quat = np.array([0.75, 0, 0.65, 0])
    target_quat = target_quat / np.linalg.norm(target_quat)  # 归一化
    
    orientation_scores = []
    for state in states:
        if isinstance(state, (list, np.ndarray)) and len(state) > 6:
            # 假设四元数在状态的3-6位置
            quat = np.array(state[3:7])
            if len(quat) == 4:
                # 归一化四元数
                quat = quat / np.linalg.norm(quat)
                
                # 计算与目标四元数的距离
                quat_diff = np.linalg.norm(quat - target_quat)
                
                # 基于真实奖励函数：rewards.tolerance(quat_diff, margin=1)
                if quat_diff <= 1.0:
                    score = 1.0 - quat_diff
                else:
                    score = 0.0
                
                orientation_scores.append(max(0.0, score))
    
    if not orientation_scores:
        return 0.5  # 默认中等得分
    
    return np.mean(orientation_scores)

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

def compute_crawl_reward_components(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> Dict[str, float]:
    """
    计算爬行任务的奖励组件
    基于humanoid_bench爬行任务的真实奖励函数
    
    Args:
        obs_seq: 观测序列 [T, obs_dim]
        action_seq: 动作序列 [T, action_dim] 
        reward_seq: 奖励序列 [T]
        
    Returns:
        Dict[str, float]: 包含各奖励组件的字典
    """
    if len(obs_seq) == 0:
        return {"crawling_posture": 0.0, "forward_motion": 0.0, "control_efficiency": 0.0, "orientation": 0.0}
    
    # 1. 爬行姿态组件 (40%权重)
    crawling_posture_score = _evaluate_crawling_posture(obs_seq)
    
    # 2. 前进运动组件 (35%权重)
    forward_motion_score = _evaluate_forward_motion(obs_seq)
    
    # 3. 控制效率组件 (15%权重)
    control_efficiency_score = _evaluate_control_efficiency(action_seq)
    
    # 4. 姿态方向组件 (10%权重)
    orientation_score = _evaluate_orientation(obs_seq)
    
    return {
        "crawling_posture": crawling_posture_score,
        "forward_motion": forward_motion_score,
        "control_efficiency": control_efficiency_score,
        "orientation": orientation_score
    }

def compare_crawl_trajectories(tau_1, tau_2, goal):
    """
    基于humanoid_bench真实奖励函数的Crawl场景启发式偏好对比规则
    
    使用DPO偏好评估方法，基于真实奖励函数的四个核心组件：
    1. 爬行姿态稳定性 (40%): crawling + crawling_head
    2. 前进运动能力 (35%): move
    3. 控制效率 (15%): small_control
    4. 姿态方向 (10%): reward_xquat
    """
    preference_result = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 基于偏好结果确定更好的轨迹
    if preference_result[0] == 0:  # tau_1更优
        return tau_1, tau_2
    elif preference_result[0] == 1:  # tau_2更优
        return tau_2, tau_1
    else:
        return None, None