
"""Door任务的启发式规则：基于轨迹质量评估的DPO偏好标签生成

该模块为Door任务实现启发式规则，用于生成DPO训练所需的偏好标签。
规则基于Door任务的特定指标和奖励函数，评估轨迹质量并生成偏好对比。

主要功能：
1. 轨迹质量评估：基于站立稳定性、控制效率、门操作进度等指标
2. 偏好标签生成：比较两个轨迹的质量，生成DPO训练标签
3. 规则注册：将规则注册到全局规则注册表中

自包含设计，无外部依赖，基于humanoid_bench Door任务的奖励函数实现轨迹比较。

使用方式：
```python
from prm.api.rules_h1hand_door_v0 import compare_h1hand_door_v0_trajectories
preference = compare_h1hand_door_v0_trajectories(trajectory1, trajectory2)
```
"""

import numpy as np


def evaluate_dpo_preference(trajectory_a, trajectory_b, detailed=False):
    """评估两个轨迹的DPO偏好
    
    Args:
        trajectory_a: 第一个轨迹
        trajectory_b: 第二个轨迹
        detailed: 是否返回详细的评估信息
        
    Returns:
        int: 偏好结果 (1表示A更好, -1表示B更好, 0表示相等)
        dict: 详细评估信息（如果detailed=True）
    """
    # 计算轨迹A的综合得分
    score_a = _compute_comprehensive_score(trajectory_a)
    
    # 计算轨迹B的综合得分
    score_b = _compute_comprehensive_score(trajectory_b)
    
    # 确定偏好
    threshold = 0.05  # 最小差异阈值
    if abs(score_a - score_b) < threshold:
        preference = 0  # 相等
    elif score_a > score_b:
        preference = 1  # A更好
    else:
        preference = -1  # B更好
    
    if detailed:
        return preference, {
            'score_a': score_a,
            'score_b': score_b,
            'difference': score_a - score_b,
            'threshold': threshold
        }
    
    return preference


def _compute_comprehensive_score(trajectory):
    """计算轨迹的综合得分 - 基于humanoid_bench真实奖励函数
    
    对应真实奖励函数的各个组件:
    - standing: 站立稳定性 (权重2.0)
    - control: 控制效率 (权重1.0)
    - door_openness: 门开启度 (权重5.0)
    - hatch_openness: 门闩开启度 (权重5.0)
    - hand_hatch_proximity: 手部到门闩距离 (权重1.0)
    - passage_completion: 通过完成度 (权重10.0)
    
    Args:
        trajectory: 轨迹数据
        
    Returns:
        float: 综合得分 [0, 1]
    """
    states, actions, rewards = _get_trajectory_data(trajectory)
    
    # 各维度评估 - 对应真实奖励函数
    standing_score = _evaluate_standing_reward(states)
    control_score = _evaluate_control_reward(actions)
    door_openness_score = _evaluate_door_openness_reward(states)
    hatch_openness_score = _evaluate_hatch_openness_reward(states)
    hand_hatch_proximity_score = _evaluate_hand_hatch_proximity_reward(states)
    passage_completion_score = _evaluate_passage_completion_reward(states)
    
    # 权重设置 - 基于真实奖励函数权重，归一化到[0,1]范围
    # 原始权重总和: 2.0 + 1.0 + 5.0 + 5.0 + 1.0 + 10.0 = 24.0
    weights = {
        'standing': 2.0 / 24.0,              # 2.0 / 24.0 ≈ 0.083
        'control': 1.0 / 24.0,               # 1.0 / 24.0 ≈ 0.042
        'door_openness': 5.0 / 24.0,         # 5.0 / 24.0 ≈ 0.208
        'hatch_openness': 5.0 / 24.0,        # 5.0 / 24.0 ≈ 0.208
        'hand_hatch_proximity': 1.0 / 24.0,  # 1.0 / 24.0 ≈ 0.042
        'passage_completion': 10.0 / 24.0     # 10.0 / 24.0 ≈ 0.417
    }
    
    # 计算加权得分
    comprehensive_score = (
        weights['standing'] * standing_score +
        weights['control'] * control_score +
        weights['door_openness'] * door_openness_score +
        weights['hatch_openness'] * hatch_openness_score +
        weights['hand_hatch_proximity'] * hand_hatch_proximity_score +
        weights['passage_completion'] * passage_completion_score
    )
    
    return float(np.clip(comprehensive_score, 0.0, 1.0))


def _evaluate_standing_reward(states):
    """评估站立奖励 - 对应真实奖励函数中的standing奖励
    
    使用tolerance函数: bounds=(0.9, inf), sigmoid="linear", margin=1.9, value_at_margin=0
    基于四元数计算躯干直立度
    
    Args:
        states: 状态序列
        
    Returns:
        float: 站立奖励得分 [0, 1]
    """
    if not states:
        return 0.0
    
    standing_scores = []
    for state in states:
        if len(state) >= 7:  # 确保有四元数数据
            # 提取四元数 (假设在状态的前7个元素中，位置3-6是四元数)
            quat = state[3:7]
            upright_score = _compute_upright_score(quat)
            
            # 应用tolerance函数: bounds=(0.9, inf), margin=1.9
            if upright_score >= 0.9:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=1.9，value_at_margin=0
                tolerance_score = max(0.0, (upright_score - (0.9 - 1.9)) / 1.9)
            
            standing_scores.append(tolerance_score)
        else:
            standing_scores.append(0.0)
    
    return np.mean(standing_scores) if standing_scores else 0.0


def _evaluate_control_reward(actions):
    """评估控制奖励 - 对应真实奖励函数中的control奖励
    
    基于动作幅度计算控制效率，使用平方惩罚
    
    Args:
        actions: 动作序列
        
    Returns:
        float: 控制奖励得分 [0, 1]
    """
    if not actions:
        return 0.0
    
    control_scores = []
    for action in actions:
        if len(action) > 0:
            # 计算动作的平方和 (对应真实奖励函数中的control惩罚)
            action_squared = np.sum(np.array(action) ** 2)
            # 转换为奖励得分 (动作越小奖励越高)
            # 使用指数衰减函数，使得小动作得到高奖励
            control_score = np.exp(-0.1 * action_squared)  # 系数可调整
            control_scores.append(control_score)
        else:
            control_scores.append(0.0)
    
    return np.mean(control_scores) if control_scores else 0.0


def _evaluate_hand_hatch_proximity_reward(states):
    """评估手部到门闩距离奖励 - 对应真实奖励函数中的hand_hatch_proximity奖励
    
    使用tolerance函数: bounds=(0, 0.1), sigmoid="linear", margin=0.4, value_at_margin=0
    
    Args:
        states: 状态序列
        
    Returns:
        float: 手部门闩距离奖励得分 [0, 1]
    """
    if not states:
        return 0.0
    
    proximity_scores = []
    for state in states:
        if len(state) >= 10:  # 确保有手部位置数据
            # 假设手部位置在状态的特定位置 (需要根据实际环境调整)
            hand_pos = np.array(state[7:10]) if len(state) >= 10 else np.array([0, 0, 0])
            
            # 门闩位置 (需要根据实际环境调整)
            hatch_pos = np.array([1.0, 0.0, 1.0])  # 示例位置
            
            # 计算距离
            distance = np.linalg.norm(hand_pos - hatch_pos)
            
            # 应用tolerance函数: bounds=(0, 0.1), margin=0.4
            if distance <= 0.1:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=0.4，value_at_margin=0
                tolerance_score = max(0.0, 1.0 - (distance - 0.1) / 0.4)
            
            proximity_scores.append(tolerance_score)
        else:
            proximity_scores.append(0.0)
    
    return np.mean(proximity_scores) if proximity_scores else 0.0


def _evaluate_passage_completion_reward(states):
    """评估通过完成度奖励 - 对应真实奖励函数中的passage_completion奖励
    
    基于机器人是否成功通过门的位置变化评估
    
    Args:
        states: 状态序列
        
    Returns:
        float: 通过完成度奖励得分 [0, 1]
    """
    if not states or len(states) < 2:
        return 0.0
    
    # 获取初始和最终位置
    start_pos = np.array(states[0][:3]) if len(states[0]) >= 3 else np.array([0, 0, 0])
    end_pos = np.array(states[-1][:3]) if len(states[-1]) >= 3 else np.array([0, 0, 0])
    
    # 计算前进距离 (假设门在x方向，需要根据实际环境调整)
    forward_distance = end_pos[0] - start_pos[0]
    
    # 通过完成度评估
    # 假设需要前进至少1.5米才算成功通过门
    target_distance = 1.5
    completion_score = max(0.0, min(1.0, forward_distance / target_distance))
    
    # 额外考虑轨迹的稳定性
    if len(states) > 10:
        # 计算轨迹后期的位置稳定性
        late_states = states[-10:]
        position_variance = 0.0
        if len(late_states) > 1:
            positions = [np.array(s[:3]) for s in late_states if len(s) >= 3]
            if len(positions) > 1:
                position_variance = np.var([p[0] for p in positions])  # x方向的方差
        
        # 稳定性奖励 (方差越小越稳定)
        stability_bonus = max(0.0, 1.0 - position_variance)
        completion_score = 0.8 * completion_score + 0.2 * stability_bonus
    
    return completion_score


def _get_trajectory_data(trajectory):
    """提取轨迹数据的工具函数
    
    Args:
        trajectory: 轨迹对象
        
    Returns:
        tuple: (states, actions, rewards)
    """
    states = getattr(trajectory, 'states', [])
    actions = getattr(trajectory, 'actions', [])
    rewards = getattr(trajectory, 'rewards', [])
    
    return states, actions, rewards


def compute_door_reward_components(trajectory):
    """计算Door任务的各个奖励组件
    
    该函数计算Door任务中的各个关键指标，用于轨迹质量评估。
    
    Args:
        trajectory: 轨迹数据
        
    Returns:
        dict: 包含各个奖励组件的字典
    """
    try:
        components = {
            'standing_stability': _evaluate_standing_stability(trajectory),
            'control_efficiency': _evaluate_control_efficiency(trajectory),
            'openness_progress': _evaluate_door_openness(_get_trajectory_data(trajectory)[0]),
            'hatch_manipulation': _evaluate_hatch_manipulation(_get_trajectory_data(trajectory)[0]),
            'hand_proximity': _evaluate_manipulation_precision(trajectory),
            'passage_completion': _evaluate_passage_completion(_get_trajectory_data(trajectory)[0], _get_trajectory_data(trajectory)[2])
        }
        
        # 计算综合得分
        components['comprehensive_score'] = _compute_comprehensive_score(trajectory)
        
        return components
    except Exception as e:
        print(f"Error computing door reward components: {e}")
        return {
            'standing_stability': 0.0,
            'control_efficiency': 0.0,
            'openness_progress': 0.0,
            'hatch_manipulation': 0.0,
            'hand_proximity': 0.0,
            'passage_completion': 0.0,
            'comprehensive_score': 0.0
        }


def compare_h1hand_door_v0_trajectories(trajectory_a, trajectory_b):
    """比较两个Door任务轨迹的质量
    
    这是Door任务的主要比较函数，用于DPO偏好标签生成。
    
    Args:
        trajectory_a: 第一个轨迹
        trajectory_b: 第二个轨迹
        
    Returns:
        int: 偏好结果 (1表示A更好, -1表示B更好, 0表示相等)
    """
    return evaluate_dpo_preference(trajectory_a, trajectory_b)


def compare_door_trajectories(trajectory_a, trajectory_b):
    """Door轨迹比较的备用函数
    
    提供与主比较函数相同的功能，作为备用接口。
    
    Args:
        trajectory_a: 第一个轨迹
        trajectory_b: 第二个轨迹
        
    Returns:
        int: 偏好结果 (1表示A更好, -1表示B更好, 0表示相等)
    """
    return compare_h1hand_door_v0_trajectories(trajectory_a, trajectory_b)


# 辅助函数实现

def _compute_upright_score(quat):
    """计算直立得分
    
    Args:
        quat: 四元数 [w, x, y, z]
        
    Returns:
        float: 直立得分 [0, 1]
    """
    try:
        # 确保四元数归一化
        quat = np.array(quat)
        quat = quat / np.linalg.norm(quat)
        
        # 计算旋转矩阵的z轴分量
        w, x, y, z = quat
        # z轴向上的分量
        z_up = 2 * (w*w + z*z) - 1
        
        # 转换为得分 (z_up接近1时得分高)
        upright_score = max(0.0, z_up)
        return float(upright_score)
    except Exception:
        return 0.0


def _evaluate_door_openness_reward(states):
    """评估门开启度奖励 - 对应真实奖励函数中的door_openness奖励
    
    使用tolerance函数: bounds=(0.35, inf), sigmoid="linear", margin=0.35, value_at_margin=0
    
    Args:
        states: 状态序列
        
    Returns:
        float: 门开启度奖励得分 [0, 1]
    """
    if not states:
        return 0.0
    
    openness_scores = []
    for state in states:
        if len(state) > 15:  # 确保有门状态数据
            # 假设门的开启度在状态的特定位置 (需要根据实际环境调整)
            # 这里使用简化的方法，实际应该从环境状态中提取门的角度
            door_openness = abs(state[15]) if len(state) > 15 else 0.0
            
            # 应用tolerance函数: bounds=(0.35, inf), margin=0.35
            if door_openness >= 0.35:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=0.35，value_at_margin=0
                tolerance_score = max(0.0, door_openness / 0.35)
            
            openness_scores.append(tolerance_score)
        else:
            openness_scores.append(0.0)
    
    return np.mean(openness_scores) if openness_scores else 0.0


def _evaluate_hatch_openness_reward(states):
    """评估门闩开启度奖励 - 对应真实奖励函数中的hatch_openness奖励
    
    使用tolerance函数: bounds=(0.2, inf), sigmoid="linear", margin=0.2, value_at_margin=0
    
    Args:
        states: 状态序列
        
    Returns:
        float: 门闩开启度奖励得分 [0, 1]
    """
    if not states:
        return 0.0
    
    hatch_scores = []
    for state in states:
        if len(state) > 16:  # 确保有门闩状态数据
            # 假设门闩的开启度在状态的特定位置 (需要根据实际环境调整)
            hatch_openness = abs(state[16]) if len(state) > 16 else 0.0
            
            # 应用tolerance函数: bounds=(0.2, inf), margin=0.2
            if hatch_openness >= 0.2:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=0.2，value_at_margin=0
                tolerance_score = max(0.0, hatch_openness / 0.2)
            
            hatch_scores.append(tolerance_score)
        else:
            hatch_scores.append(0.0)
    
    return np.mean(hatch_scores) if hatch_scores else 0.0


def _evaluate_passage_completion(states, rewards):
    """评估通过完成度
    
    Args:
        states: 状态序列
        rewards: 奖励序列
        
    Returns:
        float: 通过完成度得分 [0, 1]
    """
    try:
        if not states:
            return 0.0
        
        # 基于位置变化评估通过进度
        if len(states) > 1:
            start_pos = states[0].qpos[:3] if hasattr(states[0], 'qpos') and len(states[0].qpos) > 2 else [0, 0, 0]
            end_pos = states[-1].qpos[:3] if hasattr(states[-1], 'qpos') and len(states[-1].qpos) > 2 else [0, 0, 0]
            
            # 计算前进距离 (假设门在x方向)
            forward_distance = end_pos[0] - start_pos[0]
            # 转换为完成度得分 (假设需要前进2米才算完成)
            completion = max(0.0, min(1.0, forward_distance / 2.0))
            
            # 如果有奖励信息，也考虑奖励
            if rewards:
                avg_reward = np.mean(rewards)
                reward_score = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))  # 假设奖励范围[-1, 1]
                completion = 0.7 * completion + 0.3 * reward_score
            
            return float(np.clip(completion, 0.0, 1.0))
        
        return 0.0
    except Exception:
        return 0.0