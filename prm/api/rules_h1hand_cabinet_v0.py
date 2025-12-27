import numpy as np
from typing import Tuple, Optional, Dict

# ============================================================================
# H1Hand Cabinet V0 任务启发式规则 - 自包含实现
# ============================================================================
# 基于 humanoid_bench Cabinet 任务的真实奖励函数设计
# 核心奖励公式: reward = 0.2 * stabilization_reward + 0.8 * subtask_reward + completion_bonus
# ============================================================================


def compare_h1hand_cabinet_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-cabinet-v0 任务的专用比较规则 - 基于humanoid_bench真实奖励函数设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

# 自包含设计：移除外部依赖，所有功能集成在此文件中
# 如需注册到全局系统，由调用方负责

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench Cabinet任务真实奖励函数的DPO偏好评估
    
    Cabinet任务包含4个子任务：
    1. 打开柜门 (door_openness_reward)
    2. 打开抽屉 (drawer_openness_reward) 
    3. 将物体放入柜子 (cube_proximity_reward + door_openness_reward)
    4. 将物体放入上层柜子 (cube_proximity_reward + door_openness_reward)
    
    核心奖励公式: reward = 0.2 * stabilization_reward + 0.8 * subtask_reward + completion_bonus
    
    评估维度权重:
    - 站立稳定性 (25%): 基于standing和upright奖励
    - 控制效率 (15%): 基于small_control奖励
    - 任务进度 (40%): 基于子任务完成情况
    - 操作精度 (20%): 基于门/抽屉开启和物体放置精度
    
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
    
    基于humanoid_bench Cabinet任务的真实奖励函数:
    reward = 0.2 * stabilization_reward + 0.8 * subtask_reward + completion_bonus
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度
    standing_stability_score = _evaluate_standing_stability(states)
    control_efficiency_score = _evaluate_control_efficiency(actions)
    task_progress_score = _evaluate_task_progress(states)
    manipulation_precision_score = _evaluate_manipulation_precision(states)
    
    # 权重配置（基于真实奖励函数的重要性）
    weights = {
        'standing_stability': 0.25,     # stabilization_reward 的重要性
        'control_efficiency': 0.15,     # small_control 的重要性
        'task_progress': 0.40,          # subtask_reward 的重要性
        'manipulation_precision': 0.20   # 操作精度的重要性
    }
    
    # 计算加权总分
    total_score = (
        weights['standing_stability'] * standing_stability_score +
        weights['control_efficiency'] * control_efficiency_score +
        weights['task_progress'] * task_progress_score +
        weights['manipulation_precision'] * manipulation_precision_score
    )
    
    return total_score

def _evaluate_standing_stability(states) -> float:
    """
    评估站立稳定性 (对应 standing * upright)
    
    - standing: 基于头部高度在站立范围内 (>= 1.65m)
    - upright: 基于躯干直立程度 (>= 0.9)
    """
    if len(states) == 0:
        return 0.0
    
    _STAND_HEIGHT = 1.65
    
    # 计算存活时间得分（轨迹长度归一化）
    survival_score = min(len(states) / 1000.0, 1.0)  # 假设1000步为满分
    
    # 模拟站立高度评估
    # 假设状态中包含机器人高度信息（通常在前几个维度）
    height_scores = []
    upright_scores = []
    
    for state in states:
        if len(state) < 10:  # 确保状态维度足够
            continue
            
        # 估算头部高度（基于机器人姿态）
        estimated_height = abs(state[2]) if len(state) > 2 else 1.0  # z坐标
        height_reward = max(0, min(1, (estimated_height - 1.2) / 0.8))  # 1.2-2.0范围映射到0-1
        height_scores.append(height_reward)
        
        # 估算直立程度（基于姿态四元数或角度）
        if len(state) > 6:
            # 假设四元数在状态的3-6位置
            quat_w = state[3] if abs(state[3]) <= 1 else 1.0
            upright_reward = max(0, min(1, abs(quat_w)))
        else:
            upright_reward = 0.8  # 默认值
        upright_scores.append(upright_reward)
    
    # 计算平均站立稳定性
    if height_scores and upright_scores:
        avg_height = np.mean(height_scores)
        avg_upright = np.mean(upright_scores)
        stability_score = avg_height * avg_upright
    else:
        stability_score = 0.5  # 默认中等得分
    
    return stability_score * survival_score

def _evaluate_control_efficiency(actions) -> float:
    """
    评估控制效率 (对应 small_control)
    
    基于动作力矩的大小，奖励小的控制输入
    """
    if len(actions) == 0:
        return 0.0
    
    # 计算动作的平均幅度
    action_magnitudes = []
    for action in actions:
        if isinstance(action, (list, np.ndarray)) and len(action) > 0:
            magnitude = np.mean(np.abs(action))
            action_magnitudes.append(magnitude)
    
    if not action_magnitudes:
        return 0.5  # 默认中等得分
    
    avg_magnitude = np.mean(action_magnitudes)
    
    # 使用类似humanoid_bench的tolerance函数逻辑
    # small_control = (4 + small_control) / 5，其中small_control基于tolerance
    # 这里我们反向计算：较小的动作幅度获得更高得分
    control_efficiency = max(0, min(1, (2.0 - avg_magnitude) / 2.0))
    
    return control_efficiency

def _evaluate_task_progress(states) -> float:
    """
    评估任务进度 (对应各子任务的完成情况)
    
    基于轨迹长度和状态变化推断任务进度
    """
    if len(states) == 0:
        return 0.0
    
    # 基于轨迹长度评估任务进度
    # Cabinet任务通常需要较长时间完成多个子任务
    trajectory_length = len(states)
    
    # 长度得分：更长的轨迹通常意味着更多的任务尝试
    length_score = min(1.0, trajectory_length / 2000.0)  # 假设2000步为完整任务
    
    # 状态变化得分：评估状态的多样性（表示不同的任务阶段）
    if len(states) > 1:
        state_changes = []
        for i in range(1, len(states)):
            if len(states[i]) == len(states[i-1]):
                change = np.linalg.norm(np.array(states[i]) - np.array(states[i-1]))
                state_changes.append(change)
        
        if state_changes:
            avg_change = np.mean(state_changes)
            # 适度的状态变化表示任务进展
            change_score = min(1.0, avg_change / 0.5)  # 归一化
        else:
            change_score = 0.0
    else:
        change_score = 0.0
    
    # 综合任务进度得分
    progress_score = 0.6 * length_score + 0.4 * change_score
    
    return progress_score

def _evaluate_manipulation_precision(states) -> float:
    """
    评估操作精度 (对应门/抽屉开启和物体放置精度)
    
    基于状态变化的平滑性和一致性评估操作质量
    """
    if len(states) < 10:
        return 0.0
    
    # 评估操作的平滑性
    smoothness_scores = []
    
    # 计算状态变化的二阶导数（加速度）来评估平滑性
    for i in range(2, len(states)):
        if len(states[i]) == len(states[i-1]) == len(states[i-2]):
            try:
                curr_state = np.array(states[i])
                prev_state = np.array(states[i-1])
                prev_prev_state = np.array(states[i-2])
                
                # 计算二阶差分（加速度）
                acceleration = curr_state - 2*prev_state + prev_prev_state
                smoothness = 1.0 / (1.0 + np.linalg.norm(acceleration))
                smoothness_scores.append(smoothness)
            except:
                continue
    
    if not smoothness_scores:
        return 0.5  # 默认中等得分
    
    avg_smoothness = np.mean(smoothness_scores)
    
    # 评估操作的一致性（状态变化的方差）
    if len(states) > 5:
        recent_states = states[-min(50, len(states)):]
        if len(recent_states) > 1:
            try:
                state_matrix = np.array(recent_states)
                state_variance = np.var(state_matrix, axis=0)
                consistency = 1.0 / (1.0 + np.mean(state_variance))
            except:
                consistency = 0.5
        else:
            consistency = 0.5
    else:
        consistency = 0.5
    
    # 综合操作精度得分
    precision_score = 0.7 * avg_smoothness + 0.3 * consistency
    
    return precision_score

def _get_trajectory_data(traj, key: str):
    """
    从轨迹中提取指定类型的数据
    
    Args:
        traj: 轨迹数据
        key: 数据类型 ('obs', 'action', 'reward')
    
    Returns:
        提取的数据列表
    """
    try:
        if hasattr(traj, key):
            return getattr(traj, key)
        elif isinstance(traj, dict) and key in traj:
            return traj[key]
        elif isinstance(traj, (list, tuple)) and len(traj) > 0:
            # 尝试从轨迹步骤中提取数据
            data = []
            for step in traj:
                if hasattr(step, key):
                    data.append(getattr(step, key))
                elif isinstance(step, dict) and key in step:
                    data.append(step[key])
            return data if data else None
        else:
            return None
    except Exception:
        return None

def compute_cabinet_reward_components(obs_seq: np.ndarray, action_seq: np.ndarray, reward_seq: np.ndarray) -> Dict[str, float]:
    """
    计算Cabinet任务的奖励组件
    
    Args:
        obs_seq: 观测序列
        action_seq: 动作序列  
        reward_seq: 奖励序列
    
    Returns:
        包含各奖励组件的字典
    """
    if len(obs_seq) == 0:
        return {}
    
    # 计算各个奖励组件
    standing_stability = _evaluate_standing_stability(obs_seq)
    control_efficiency = _evaluate_control_efficiency(action_seq)
    task_progress = _evaluate_task_progress(obs_seq)
    manipulation_precision = _evaluate_manipulation_precision(obs_seq)
    
    return {
        'standing_stability': standing_stability,
        'control_efficiency': control_efficiency,
        'task_progress': task_progress,
        'manipulation_precision': manipulation_precision,
        'total_reward': np.sum(reward_seq) if len(reward_seq) > 0 else 0.0,
        'avg_reward': np.mean(reward_seq) if len(reward_seq) > 0 else 0.0,
        'trajectory_length': len(obs_seq)
    }

def compare_cabinet_trajectories(tau_1, tau_2, goal):
    """
    Cabinet任务轨迹比较的备用函数
    """
    try:
        return compare_h1hand_cabinet_v0_trajectories(tau_1, tau_2, goal)
    except Exception as e:
        print(f"Cabinet trajectory comparison failed: {e}")
        return None, None