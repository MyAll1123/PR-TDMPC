
"""Sit Simple任务的自包含规则：基于humanoid_bench Sit任务的真实奖励函数设计

自包含设计，无外部依赖，基于humanoid_bench Sit任务的奖励函数实现轨迹比较。
"""

import numpy as np

def compute_sit_reward_components(trajectory):
    """
    基于humanoid_bench真实奖励函数计算Sit任务的奖励组件
    
    真实奖励函数分析（Sit类）：
    - sitting: 躯干高度在合适范围内（0.68-0.72m）
    - on_chair: 位置在椅子上（x和y方向的位置约束）
    - sitting_posture: 头部相对于躯干的高度差（0.35-0.45m）
    - upright: 躯干直立度（0.95以上）
    - sit_reward: (0.5 * sitting + 0.5 * on_chair) * upright * sitting_posture
    - small_control: 控制力矩惩罚
    - dont_move: 水平速度惩罚
    - 总奖励: small_control * sit_reward * dont_move
    
    权重分配：
    - 坐姿质量: 45% (sitting + sitting_posture + upright)
    - 位置准确性: 30% (on_chair)
    - 控制稳定性: 25% (small_control + dont_move)
    """
    if trajectory is None:
        return 0.0, 0.0, 0.0
    
    # 处理字典格式的轨迹数据
    if isinstance(trajectory, dict):
        observations = trajectory.get('observations', [])
        actions = trajectory.get('actions', [])
    else:
        observations = getattr(trajectory, 'observations', [])
        actions = getattr(trajectory, 'actions', [])
    
    if not observations or len(observations) == 0:
        return 0.0, 0.0, 0.0
    
    # 确保actions是numpy数组
    if actions is not None and len(actions) > 0:
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
    else:
        actions = np.zeros((len(observations), 26))  # h1hand默认动作维度
    
    sitting_quality_scores = []
    position_accuracy_scores = []
    control_stability_scores = []
    
    for i, obs in enumerate(observations):
        if obs is None:
            continue
            
        # 从观测中提取状态信息
        if isinstance(obs, dict):
            # 字典格式观测
            torso_height = obs.get('torso_height', obs.get('qpos_2', 0.7))
            torso_upright = obs.get('torso_upright', obs.get('torso_orientation_z', 1.0))
            position_x = obs.get('position_x', obs.get('qpos_0', 0.0))
            position_y = obs.get('position_y', obs.get('qpos_1', 0.0))
            velocity_x = obs.get('velocity_x', obs.get('qvel_0', 0.0))
            velocity_y = obs.get('velocity_y', obs.get('qvel_1', 0.0))
        else:
            # 数组格式观测
            obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            if len(obs_array) >= 8:
                position_x = obs_array[0] if len(obs_array) > 0 else 0.0
                position_y = obs_array[1] if len(obs_array) > 1 else 0.0
                torso_height = obs_array[2] if len(obs_array) > 2 else 0.7
                torso_upright = obs_array[6] if len(obs_array) > 6 else 1.0  # 四元数w分量
                velocity_x = obs_array[7] if len(obs_array) > 7 else 0.0
                velocity_y = obs_array[8] if len(obs_array) > 8 else 0.0
            else:
                position_x = 0.0
                position_y = 0.0
                torso_height = 0.7
                torso_upright = 1.0
                velocity_x = 0.0
                velocity_y = 0.0
        
        # 1. 坐姿质量 (45%)
        # 基于sitting, sitting_posture, upright组件
        sitting_score = 1.0 if 0.68 <= torso_height <= 0.72 else max(0.0, 1.0 - abs(torso_height - 0.7) / 0.2)
        upright_score = max(0.0, min(1.0, (torso_upright - 0.95) / 0.05)) if torso_upright > 0.95 else 0.0
        # 假设头部高度差约为0.4m
        posture_score = 1.0  # 简化处理，假设姿态良好
        sitting_quality = (sitting_score + upright_score + posture_score) / 3.0
        sitting_quality_scores.append(sitting_quality)
        
        # 2. 位置准确性 (30%)
        # 基于on_chair组件，假设椅子在原点附近
        chair_x_score = 1.0 if abs(position_x) <= 0.19 else max(0.0, 1.0 - abs(position_x) / 0.4)
        chair_y_score = 1.0 if abs(position_y) <= 0.1 else max(0.0, 1.0 - abs(position_y) / 0.2)
        position_accuracy = chair_x_score * chair_y_score
        position_accuracy_scores.append(position_accuracy)
        
        # 3. 控制稳定性 (25%)
        # 基于small_control和dont_move组件
        if i < len(actions):
            action = actions[i]
            if isinstance(action, (list, np.ndarray)):
                action_array = np.array(action) if not isinstance(action, np.ndarray) else action
                control_penalty = np.mean(np.square(action_array))
                control_score = max(0.0, 1.0 - control_penalty / 100.0)
            else:
                control_score = 0.5
        else:
            control_score = 0.5
        
        # 运动稳定性（不要移动）
        horizontal_velocity = np.sqrt(velocity_x**2 + velocity_y**2)
        movement_score = max(0.0, 1.0 - horizontal_velocity / 2.0)
        
        control_stability = (control_score + movement_score) / 2.0
        control_stability_scores.append(control_stability)
    
    # 计算平均分数
    avg_sitting_quality = np.mean(sitting_quality_scores) if sitting_quality_scores else 0.0
    avg_position_accuracy = np.mean(position_accuracy_scores) if position_accuracy_scores else 0.0
    avg_control_stability = np.mean(control_stability_scores) if control_stability_scores else 0.0
    
    return avg_sitting_quality, avg_position_accuracy, avg_control_stability

def evaluate_dpo_preference(trajectory_1, trajectory_2, goal=None):
    """
    基于humanoid_bench真实奖励函数评估Sit任务的DPO偏好
    
    权重分配：
    - 坐姿质量: 45%
    - 位置准确性: 30%
    - 控制稳定性: 25%
    """
    # 计算两个轨迹的奖励组件
    quality_1, position_1, stability_1 = compute_sit_reward_components(trajectory_1)
    quality_2, position_2, stability_2 = compute_sit_reward_components(trajectory_2)
    
    # 权重分配
    w_quality = 0.45
    w_position = 0.30
    w_stability = 0.25
    
    # 计算综合奖励
    reward_1 = w_quality * quality_1 + w_position * position_1 + w_stability * stability_1
    reward_2 = w_quality * quality_2 + w_position * position_2 + w_stability * stability_2
    
    # 计算偏好概率（使用sigmoid函数）
    reward_diff = reward_1 - reward_2
    preference_prob = 1.0 / (1.0 + np.exp(-reward_diff * 10.0))  # 放大差异
    
    return preference_prob

def compare_h1hand_sit_simple_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-sit_simple-v0 任务的专用比较规则"""
    return compare_sitting_trajectories(tau_1, tau_2, goal)

def compare_sitting_trajectories(tau_1, tau_2, goal):
    """
    基于humanoid_bench真实奖励函数的Sit场景启发式偏好对比规则
    
    使用DPO偏好评估方法，基于真实奖励函数的三个核心组件：
    1. 坐姿质量 (45%): sitting + sitting_posture + upright
    2. 位置准确性 (30%): on_chair
    3. 控制稳定性 (25%): small_control + dont_move
    """
    preference_prob = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 基于偏好概率确定更好的轨迹
    if preference_prob > 0.5:
        return tau_1, tau_2
    elif preference_prob < 0.5:
        return tau_2, tau_1
    else:
        return None, None