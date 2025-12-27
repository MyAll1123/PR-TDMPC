import numpy as np
from typing import Tuple, Optional, Dict

# ============================================================================
# H1Hand Bookshelf Simple V0 任务启发式规则 - 自包含实现
# ============================================================================
# 基于 humanoid_bench Bookshelf 任务的真实奖励函数设计
# 核心奖励公式: reward = 0.2 * (stand_reward * small_control) + 0.4 * reward_proximity + 0.4 * reward_hand_proximity + completion_bonus
# ============================================================================


def compare_h1hand_bookshelf_simple_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-bookshelf_simple-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

# 自包含设计：移除外部依赖，所有功能集成在此文件中
# 如需注册到全局系统，由调用方负责

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Bookshelf任务真实奖励函数的DPO偏好评估
    
    Bookshelf任务包含5个子任务：
    1. 将第1个物体放置到目标位置
    2. 将第2个物体放置到目标位置
    3. 将第3个物体放置到目标位置
    4. 将第4个物体放置到目标位置
    5. 将第5个物体放置到目标位置
    
    核心奖励公式: reward = 0.2 * (stand_reward * small_control) + 0.4 * reward_proximity + 0.4 * reward_hand_proximity + completion_bonus
    
    评估维度权重:
    - 站立稳定性 (20%): 基于standing和upright奖励
    - 控制效率 (20%): 基于small_control奖励
    - 物体接近度 (30%): 基于物体到目标位置的距离
    - 手部接近度 (30%): 基于手到物体的距离
    
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
    
    基于humanoid_bench Bookshelf任务的真实奖励函数:
    reward = 0.2 * (stand_reward * small_control) + 0.4 * reward_proximity + 0.4 * reward_hand_proximity + completion_bonus
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度
    standing_stability_score = _evaluate_standing_stability(states)
    control_efficiency_score = _evaluate_control_efficiency(actions)
    object_proximity_score = _evaluate_object_proximity(states)
    hand_proximity_score = _evaluate_hand_proximity(states)
    
    # 权重配置（基于真实奖励函数的重要性）
    weights = {
        'standing_stability': 0.20,     # stand_reward 的重要性
        'control_efficiency': 0.20,     # small_control 的重要性
        'object_proximity': 0.30,       # reward_proximity 的重要性
        'hand_proximity': 0.30          # reward_hand_proximity 的重要性
    }
    
    # 计算加权总分
    total_score = (
        weights['standing_stability'] * standing_stability_score +
        weights['control_efficiency'] * control_efficiency_score +
        weights['object_proximity'] * object_proximity_score +
        weights['hand_proximity'] * hand_proximity_score
    )
    
    return total_score

def _evaluate_standing_stability(states) -> float:
    """
    评估站立稳定性
    
    基于humanoid_bench的standing和upright奖励:
    - standing: 基于头部高度的容忍度奖励
    - upright: 基于躯干直立度的容忍度奖励
    - stand_reward = standing * upright
    """
    if len(states) == 0:
        return 0.0
    
    # 假设状态向量中包含头部高度和躯干姿态信息
    # 这里使用启发式方法估算稳定性
    stability_scores = []
    
    for state in states:
        # 估算头部高度稳定性（假设在状态向量的特定位置）
        # 目标高度为1.65米，容忍范围为±0.4米
        head_height_estimate = _estimate_head_height_from_state(state)
        standing_score = _tolerance_reward(head_height_estimate, bounds=(1.65, float('inf')), margin=0.4)
        
        # 估算躯干直立度（基于状态向量中的姿态信息）
        upright_estimate = _estimate_upright_from_state(state)
        upright_score = _tolerance_reward(upright_estimate, bounds=(0.9, float('inf')), margin=1.9)
        
        # 站立奖励 = standing * upright
        stability_score = standing_score * upright_score
        stability_scores.append(stability_score)
    
    return np.mean(stability_scores)

def _evaluate_control_efficiency(actions) -> float:
    """
    评估控制效率
    
    基于humanoid_bench的small_control奖励:
    - 基于执行器力的容忍度奖励
    - small_control = (4 + tolerance_reward) / 5
    """
    if len(actions) == 0:
        return 0.0
    
    control_scores = []
    
    for action in actions:
        # 计算动作的L2范数作为控制力的代理
        action_magnitude = np.linalg.norm(action)
        
        # 使用容忍度奖励评估控制效率
        control_tolerance = _tolerance_reward(action_magnitude, margin=10, sigmoid='quadratic')
        control_score = (4 + control_tolerance) / 5
        
        control_scores.append(control_score)
    
    return np.mean(control_scores)

def _evaluate_object_proximity(states) -> float:
    """
    评估物体接近度
    
    基于humanoid_bench的reward_proximity:
    - 基于物体到目标位置距离的容忍度奖励
    - 目标距离范围: (0, 0.15)，边际: 1
    """
    if len(states) == 0:
        return 0.0
    
    proximity_scores = []
    
    for state in states:
        # 估算当前物体到目标的距离
        # 这里使用启发式方法，基于状态变化推断接近程度
        object_goal_distance = _estimate_object_goal_distance(state)
        
        # 使用容忍度奖励评估接近度
        proximity_score = _tolerance_reward(object_goal_distance, bounds=(0, 0.15), margin=1, sigmoid='linear')
        proximity_scores.append(proximity_score)
    
    return np.mean(proximity_scores)

def _evaluate_hand_proximity(states) -> float:
    """
    评估手部接近度
    
    基于humanoid_bench的reward_hand_proximity:
    - 基于手到物体距离的指数奖励
    - reward_hand_proximity = exp(-min(left_hand_distance, right_hand_distance))
    """
    if len(states) == 0:
        return 0.0
    
    hand_proximity_scores = []
    
    for state in states:
        # 估算手到物体的距离
        hand_object_distance = _estimate_hand_object_distance(state)
        
        # 使用指数奖励评估手部接近度
        hand_proximity_score = np.exp(-hand_object_distance)
        hand_proximity_scores.append(hand_proximity_score)
    
    return np.mean(hand_proximity_scores)

def _get_trajectory_data(traj, key: str):
    """
    从轨迹中提取指定类型的数据
    
    Args:
        traj: 轨迹数据
        key: 数据类型 ('obs', 'action', 'reward')
    
    Returns:
        提取的数据序列
    """
    try:
        if hasattr(traj, key):
            data = getattr(traj, key)
        elif isinstance(traj, dict) and key in traj:
            data = traj[key]
        elif isinstance(traj, (list, tuple)) and len(traj) > 0:
            # 尝试从轨迹步骤中提取数据
            if isinstance(traj[0], dict) and key in traj[0]:
                data = [step[key] for step in traj]
            else:
                return None
        else:
            return None
        
        # 确保数据是numpy数组格式
        if isinstance(data, (list, tuple)):
            data = [np.array(item) if not isinstance(item, np.ndarray) else item for item in data]
        elif not isinstance(data, (list, tuple, np.ndarray)):
            return None
        
        return data
    except Exception:
        return None

def _estimate_head_height_from_state(state) -> float:
    """
    从状态向量估算头部高度
    
    这是一个启发式方法，实际实现需要根据具体的状态向量格式调整
    """
    # 假设状态向量中包含机器人的位置信息
    # 这里使用简化的估算方法
    if len(state) >= 3:
        # 假设前3个元素是位置信息，z坐标代表高度
        base_height = state[2] if len(state) > 2 else 0.98
        # 估算头部相对于基座的高度偏移
        head_offset = 0.67  # 典型的头部高度偏移
        return base_height + head_offset
    return 1.65  # 默认目标高度

def _estimate_upright_from_state(state) -> float:
    """
    从状态向量估算躯干直立度
    
    这是一个启发式方法，实际实现需要根据具体的状态向量格式调整
    """
    # 假设状态向量中包含四元数姿态信息
    if len(state) >= 7:
        # 假设位置3-6是四元数 (w, x, y, z)
        quat = state[3:7]
        # 计算躯干的直立度（z轴方向的余弦值）
        # 这里使用简化计算
        w, x, y, z = quat
        # 计算旋转矩阵的z轴分量
        upright = 1 - 2 * (x*x + y*y)
        return max(0, upright)
    return 0.95  # 默认较好的直立度

def _estimate_object_goal_distance(state) -> float:
    """
    从状态向量估算物体到目标的距离
    
    这是一个启发式方法，基于状态变化推断接近程度
    """
    # 这里使用简化的估算方法
    # 实际实现需要根据具体的状态向量格式和任务信息调整
    
    # 假设状态向量的后部包含任务相关信息
    if len(state) > 50:
        # 使用状态向量的变化作为距离的代理
        # 这里使用一个启发式计算
        task_progress = state[-1] if len(state) > 0 else 0
        # 将任务进度转换为距离估算
        estimated_distance = max(0.01, 1.0 - task_progress)
        return estimated_distance
    
    return 0.5  # 默认中等距离

def _estimate_hand_object_distance(state) -> float:
    """
    从状态向量估算手到物体的距离
    
    这是一个启发式方法，基于状态信息推断手部接近程度
    """
    # 这里使用简化的估算方法
    # 实际实现需要根据具体的状态向量格式调整
    
    # 假设可以从状态向量推断手部位置
    if len(state) > 20:
        # 使用状态向量的某些分量作为手部接近度的代理
        # 这里使用一个启发式计算
        hand_activity = np.std(state[10:20]) if len(state) > 20 else 0.1
        # 将手部活动度转换为距离估算
        estimated_distance = max(0.1, 2.0 - hand_activity * 10)
        return estimated_distance
    
    return 1.0  # 默认中等距离

def _tolerance_reward(value, bounds=None, margin=1.0, sigmoid='linear'):
    """
    实现类似dm_control.utils.rewards.tolerance的容忍度奖励函数
    
    Args:
        value: 输入值
        bounds: 奖励为1的边界范围 (lower, upper)
        margin: 边际值
        sigmoid: sigmoid函数类型 ('linear', 'quadratic')
    
    Returns:
        奖励值 [0, 1]
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
    
    if sigmoid == 'linear':
        reward = max(0, 1 - distance / margin)
    elif sigmoid == 'quadratic':
        reward = max(0, 1 - (distance / margin) ** 2)
    else:
        reward = max(0, 1 - distance / margin)
    
    return reward

def compute_bookshelf_reward_components(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> Dict[str, float]:
    """
    计算bookshelf任务的奖励组件
    
    基于humanoid_bench Bookshelf任务的真实奖励函数分解
    
    Args:
        obs_seq: 观测序列
        action_seq: 动作序列  
        reward_seq: 奖励序列
    
    Returns:
        各奖励组件的字典
    """
    
    # 计算各个奖励组件
    standing_stability = _evaluate_standing_stability(obs_seq)
    control_efficiency = _evaluate_control_efficiency(action_seq)
    object_proximity = _evaluate_object_proximity(obs_seq)
    hand_proximity = _evaluate_hand_proximity(obs_seq)
    
    # 计算任务完成度（基于奖励序列的增长）
    task_completion = 0.0
    if len(reward_seq) > 0:
        # 检测大幅奖励增长（完成子任务的标志）
        reward_jumps = np.diff(reward_seq)
        completion_events = np.sum(reward_jumps > 50)  # 检测100分奖励跳跃
        task_completion = min(1.0, completion_events / 5.0)  # 5个子任务
    
    return {
        'standing_stability': standing_stability,
        'control_efficiency': control_efficiency,
        'object_proximity': object_proximity,
        'hand_proximity': hand_proximity,
        'task_completion': task_completion,
        'total_score': (
            0.20 * standing_stability +
            0.20 * control_efficiency +
            0.30 * object_proximity +
            0.30 * hand_proximity
        )
    }

def compare_bookshelf_trajectories(tau_1, tau_2, goal):
    """
    比较两个bookshelf任务轨迹的质量
    
    这是一个兼容性包装函数，调用主要的偏好评估逻辑
    """
    try:
        return evaluate_dpo_preference(tau_1, tau_2, goal)
    except Exception as e:
        print(f"Error in bookshelf trajectory comparison: {e}")
        return None, None