
"""Reach任务的自包含规则：基于humanoid_bench Reach任务的真实奖励函数设计

自包含设计，无外部依赖，基于humanoid_bench Reach任务的奖励函数实现轨迹比较。
"""

import numpy as np

def compute_reach_reward_components(trajectory, goal=None):
    """
    基于humanoid_bench真实奖励函数计算Reach任务的奖励组件
    
    真实奖励函数分析（Reach类）：
    - hand_dist: 手到目标距离（核心指标）
    - healthy_reward: 躯干姿态健康度 (xmat[1,-1] * 5.0)
    - motion_penalty: 运动惩罚 (qvel^2 * 0.0001)
    - reward_close: 接近奖励 (距离<1m时+5分)
    - reward_success: 成功奖励 (距离<0.05m时+10分)
    - 总奖励: healthy_reward - 0.0001 * motion_penalty + reward_close + reward_success
    
    权重分配：
    - 到达准确性: 50% (hand_dist + reward_success + reward_close)
    - 躯干稳定性: 30% (healthy_reward)
    - 运动效率: 20% (motion_penalty的反向)
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
    
    # 默认目标位置（如果未提供）
    if goal is None:
        goal = np.array([0.5, 0.5, 1.2])  # 默认目标位置
    elif not isinstance(goal, np.ndarray):
        goal = np.array(goal)
    
    reaching_accuracy_scores = []
    torso_stability_scores = []
    motion_efficiency_scores = []
    
    for i, obs in enumerate(observations):
        if obs is None:
            continue
            
        # 从观测中提取状态信息
        if isinstance(obs, dict):
            # 字典格式观测
            hand_pos_x = obs.get('hand_position_x', obs.get('left_hand_x', 0.0))
            hand_pos_y = obs.get('hand_position_y', obs.get('left_hand_y', 0.0))
            hand_pos_z = obs.get('hand_position_z', obs.get('left_hand_z', 1.0))
            torso_upright = obs.get('torso_upright', obs.get('torso_orientation_z', 1.0))
            velocities = obs.get('velocities', obs.get('qvel', [0.0] * 25))
        else:
            # 数组格式观测
            obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            if len(obs_array) >= 30:
                # 假设手部位置在观测的特定位置
                hand_pos_x = obs_array[20] if len(obs_array) > 20 else 0.0
                hand_pos_y = obs_array[21] if len(obs_array) > 21 else 0.0
                hand_pos_z = obs_array[22] if len(obs_array) > 22 else 1.0
                torso_upright = obs_array[6] if len(obs_array) > 6 else 1.0  # 四元数w分量
                velocities = obs_array[25:] if len(obs_array) > 25 else [0.0] * 25
            else:
                hand_pos_x = 0.0
                hand_pos_y = 0.0
                hand_pos_z = 1.0
                torso_upright = 1.0
                velocities = [0.0] * 25
        
        hand_position = np.array([hand_pos_x, hand_pos_y, hand_pos_z])
        
        # 1. 到达准确性 (50%)
        # 基于hand_dist, reward_success, reward_close组件
        hand_dist = np.linalg.norm(hand_position - goal)
        
        # 成功奖励（距离<0.05m）
        success_score = 1.0 if hand_dist < 0.05 else 0.0
        
        # 接近奖励（距离<1m）
        close_score = 1.0 if hand_dist < 1.0 else 0.0
        
        # 距离分数（距离越小越好）
        distance_score = max(0.0, 1.0 - hand_dist / 2.0)  # 2m内线性衰减
        
        reaching_accuracy = (success_score * 0.5 + close_score * 0.3 + distance_score * 0.2)
        reaching_accuracy_scores.append(reaching_accuracy)
        
        # 2. 躯干稳定性 (30%)
        # 基于healthy_reward组件
        stability_score = max(0.0, min(1.0, torso_upright))  # 直立度
        torso_stability_scores.append(stability_score)
        
        # 3. 运动效率 (20%)
        # 基于motion_penalty的反向
        if isinstance(velocities, (list, np.ndarray)):
            vel_array = np.array(velocities) if not isinstance(velocities, np.ndarray) else velocities
            motion_penalty = np.sum(np.square(vel_array[:25]))  # 前25个关节速度
            motion_efficiency = max(0.0, 1.0 - motion_penalty / 1000.0)  # 归一化运动惩罚
        else:
            motion_efficiency = 0.5
        motion_efficiency_scores.append(motion_efficiency)
    
    # 计算平均分数
    avg_reaching_accuracy = np.mean(reaching_accuracy_scores) if reaching_accuracy_scores else 0.0
    avg_torso_stability = np.mean(torso_stability_scores) if torso_stability_scores else 0.0
    avg_motion_efficiency = np.mean(motion_efficiency_scores) if motion_efficiency_scores else 0.0
    
    return avg_reaching_accuracy, avg_torso_stability, avg_motion_efficiency

def evaluate_dpo_preference(trajectory_1, trajectory_2, goal=None):
    """
    基于humanoid_bench真实奖励函数评估Reach任务的DPO偏好
    
    权重分配：
    - 到达准确性: 50%
    - 躯干稳定性: 30%
    - 运动效率: 20%
    """
    # 计算两个轨迹的奖励组件
    accuracy_1, stability_1, efficiency_1 = compute_reach_reward_components(trajectory_1, goal)
    accuracy_2, stability_2, efficiency_2 = compute_reach_reward_components(trajectory_2, goal)
    
    # 权重分配
    w_accuracy = 0.50
    w_stability = 0.30
    w_efficiency = 0.20
    
    # 计算综合奖励
    reward_1 = w_accuracy * accuracy_1 + w_stability * stability_1 + w_efficiency * efficiency_1
    reward_2 = w_accuracy * accuracy_2 + w_stability * stability_2 + w_efficiency * efficiency_2
    
    # 计算偏好概率（使用sigmoid函数）
    reward_diff = reward_1 - reward_2
    preference_prob = 1.0 / (1.0 + np.exp(-reward_diff * 10.0))  # 放大差异
    
    return preference_prob

def compare_h1hand_reach_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-reach-v0 任务的专用比较规则"""
    return compare_reaching_trajectories(tau_1, tau_2, goal)

def compare_reaching_trajectories(tau_1, tau_2, goal):
    """
    基于humanoid_bench真实奖励函数的Reach场景启发式偏好对比规则
    
    使用DPO偏好评估方法，基于真实奖励函数的三个核心组件：
    1. 到达准确性 (50%): hand_dist + reward_success + reward_close
    2. 躯干稳定性 (30%): healthy_reward
    3. 运动效率 (20%): motion_penalty的反向
    """
    preference_prob = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 基于偏好概率确定更好的轨迹
    if preference_prob > 0.5:
        return tau_1, tau_2
    elif preference_prob < 0.5:
        return tau_2, tau_1
    else:
        return None, None