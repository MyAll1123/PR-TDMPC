import numpy as np
from dm_control.utils import rewards


from .common_utils import tolerance
from typing import Dict, Tuple, Any

def compare_h1hand_room_v0_trajectories(
    trajectory_a: Dict[str, Any],
    trajectory_b: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compare two room task trajectories and return preference.
    
    Args:
        trajectory_a: First trajectory data
        trajectory_b: Second trajectory data  
        config: Configuration parameters (optional)
        
    Returns:
        Tuple of (better_trajectory, worse_trajectory)
    """
    if config is None:
        config = {}
    
    try:
        # Extract states and actions from trajectories
        states_a = trajectory_a.get('observations', [])
        actions_a = trajectory_a.get('actions', [])
        states_b = trajectory_b.get('observations', [])
        actions_b = trajectory_b.get('actions', [])
        
        score_a = compute_trajectory_score(states_a, actions_a)
        score_b = compute_trajectory_score(states_b, actions_b)
        
        # Ensure scores are scalar values
        if hasattr(score_a, 'item'):
            score_a = score_a.item()
        if hasattr(score_b, 'item'):
            score_b = score_b.item()
        
        if float(score_a) > float(score_b):
            return trajectory_a, trajectory_b
        else:
            return trajectory_b, trajectory_a
    except Exception as e:
        # Fallback: return trajectories in original order if comparison fails
        return trajectory_a, trajectory_b

def evaluate_dpo_preference(
    trajectory_a: Dict[str, Any],
    trajectory_b: Dict[str, Any], 
    config: Dict[str, Any]
) -> Tuple[str, float]:
    """
    Evaluate DPO preference between two trajectories.
    
    Args:
        trajectory_a: First trajectory data
        trajectory_b: Second trajectory data
        config: Configuration parameters
        
    Returns:
        Tuple of (preferred_trajectory, confidence_score)
    """
    try:
        # Extract states and actions from trajectories
        states_a = trajectory_a.get('observations', [])
        actions_a = trajectory_a.get('actions', [])
        states_b = trajectory_b.get('observations', [])
        actions_b = trajectory_b.get('actions', [])
        
        score_a = compute_trajectory_score(states_a, actions_a)
        score_b = compute_trajectory_score(states_b, actions_b)
        
        # Calculate confidence based on score difference
        score_diff = abs(score_a - score_b)
        max_possible_diff = 10.0  # Estimated maximum score difference
        confidence = min(score_diff / max_possible_diff, 1.0)
        
        preferred = 'A' if score_a > score_b else 'B'
        return preferred, confidence
    except Exception as e:
        # Fallback: return 'A' with low confidence
        return 'A', 0.1

def compute_room_reward_components(states, actions, goal=None):
    """
    计算room任务的奖励组件 - 基于humanoid_bench/envs/room.py的get_reward方法
    
    Args:
        states: 状态序列 (T, state_dim)
        actions: 动作序列 (T, action_dim)
        goal: 目标信息 (可选)
    
    Returns:
        dict: 包含各个奖励组件的字典
    """
    if len(states) == 0 or len(actions) == 0:
        return {
            'stand_reward': 0.0,
            'small_control': 0.0,
            'standing': 0.0,
            'upright': 0.0,
            'room_object_organized': 0.0
        }
    
    states = np.array(states)
    actions = np.array(actions)
    
    # 常量定义 (基于room.py)
    STAND_HEIGHT = 1.65
    
    # 1. Standing reward - 基于头部高度
    # 假设头部高度在状态的第3个维度 (Z坐标)
    head_heights = states[:, 2]
    standing_scores = []
    for height in head_heights:
        standing = rewards.tolerance(
            height,
            bounds=(STAND_HEIGHT, float("inf")),
            margin=STAND_HEIGHT / 4,
        )
        standing_scores.append(standing)
    standing = np.mean(standing_scores)
    
    # 2. Upright reward - 基于躯干直立度
    # 假设躯干四元数在状态的第4-7个维度
    if states.shape[1] > 6:
        torso_quats = states[:, 3:7]  # 四元数 [w, x, y, z]
        # 计算直立度 (w分量接近1表示直立)
        upright_scores = []
        for quat in torso_quats:
            # 计算躯干直立度 (基于四元数的w分量)
            torso_upright = np.abs(quat[0])  # w分量
            upright = rewards.tolerance(
                torso_upright,
                bounds=(0.9, float("inf")),
                sigmoid="linear",
                margin=1.9,
                value_at_margin=0,
            )
            upright_scores.append(upright)
        upright = np.mean(upright_scores)
    else:
        upright = 0.5
    
    # 3. Stand reward (组合standing和upright)
    stand_reward = standing * upright
    
    # 4. Small control reward - 基于执行器力
    control_forces = np.linalg.norm(actions, axis=1)
    small_control_scores = []
    for force in control_forces:
        small_control = rewards.tolerance(
            force,
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        )
        small_control_scores.append(small_control)
    small_control = np.mean(small_control_scores)
    small_control = (4 + small_control) / 5
    
    # 5. Room object organized reward
    # 这里使用简化的实现，因为我们无法直接访问物体位置
    # 基于状态的方差来估计物体组织程度
    if states.shape[1] > 10:
        # 假设物体位置信息在状态的后面部分
        object_positions = states[:, -12:].reshape(-1, 6, 2)  # 6个物体，每个2D位置
        object_entropies = []
        for i in range(2):  # x, y坐标
            col_variance = np.var(object_positions[:, :, i], axis=1)
            object_entropies.append(np.mean(col_variance))
        max_entropy = np.max(object_entropies)
        room_object_organized = rewards.tolerance(
            max_entropy,
            margin=3,
        )
    else:
        # 如果状态维度不足，使用基于动作平滑度的近似
        action_smoothness = 1.0 / (1.0 + np.var(actions))
        room_object_organized = action_smoothness
    
    return {
        'stand_reward': float(stand_reward),
        'small_control': float(small_control),
        'standing': float(standing),
        'upright': float(upright),
        'room_object_organized': float(room_object_organized)
    }


def compute_trajectory_score(states, actions, goal=None):
    """
    计算room任务的轨迹得分 - 基于humanoid_bench/envs/room.py的总奖励
    
    Args:
        states: 状态序列
        actions: 动作序列
        goal: 目标信息
    
    Returns:
        float: 轨迹得分
    """
    components = compute_room_reward_components(states, actions, goal)
    
    # 基于room.py的奖励计算: 0.2 * (small_control * stand_reward) + 0.8 * room_object_organized
    total_reward = (
        0.2 * (components['small_control'] * components['stand_reward']) +
        0.8 * components['room_object_organized']
    )
    
    return float(total_reward)


def evaluate_room_trajectory(states, actions, goal=None):
    """
    评估room任务的轨迹性能
    
    Args:
        states: 状态序列
        actions: 动作序列
        goal: 目标信息
    
    Returns:
        dict: 评估结果
    """
    if len(states) == 0:
        return {
            'success': False,
            'score': 0.0,
            'components': {},
            'metrics': {
                'trajectory_length': 0,
                'average_height': 0.0,
                'control_efficiency': 0.0,
                'organization_score': 0.0
            }
        }
    
    states = np.array(states)
    actions = np.array(actions) if len(actions) > 0 else np.zeros((len(states), 1))
    
    # 计算奖励组件
    components = compute_room_reward_components(states, actions, goal)
    
    # 计算总得分
    score = compute_trajectory_score(states, actions, goal)
    
    # 成功标准 (基于room.py的success_bar = 400)
    # 这里使用相对标准，因为我们的得分范围不同
    success = score > 0.6  # 60%的得分阈值
    
    # 计算额外指标
    trajectory_length = len(states)
    average_height = np.mean(states[:, 2]) if states.shape[1] > 2 else 0.0
    control_efficiency = components['small_control']
    organization_score = components['room_object_organized']
    
    return {
        'success': success,
        'score': score,
        'components': components,
        'metrics': {
            'trajectory_length': trajectory_length,
            'average_height': float(average_height),
            'control_efficiency': control_efficiency,
            'organization_score': organization_score
        }
    }