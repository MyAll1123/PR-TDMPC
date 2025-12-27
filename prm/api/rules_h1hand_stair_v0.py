
"""Stair任务的自包含规则：基于humanoid_bench Stair任务的真实奖励函数设计

自包含设计，无外部依赖，基于humanoid_bench Stair任务的奖励函数实现轨迹比较。
"""

import numpy as np

def compute_stair_reward_components(trajectory):
    """
    基于humanoid_bench真实奖励函数计算Stair任务的奖励组件
    
    真实奖励函数分析（ClimbingUpwards类）：
    - standing: 头部相对于脚部的高度差（1.2m以上）
    - upright: 躯干直立度（0.5以上）
    - stand_reward: standing * upright
    - small_control: 控制力矩惩罚
    - move: 前向速度奖励（1m/s以上）
    - 总奖励: stand_reward * small_control * move
    
    权重分配：
    - 攀爬高度稳定性: 40% (standing + upright)
    - 前向运动效率: 35% (move)
    - 控制效率: 25% (small_control)
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
    
    climbing_stability_scores = []
    forward_motion_scores = []
    control_efficiency_scores = []
    
    for i, obs in enumerate(observations):
        if obs is None:
            continue
            
        # 从观测中提取状态信息
        if isinstance(obs, dict):
            # 字典格式观测
            torso_height = obs.get('torso_height', obs.get('qpos_2', 1.0))
            torso_upright = obs.get('torso_upright', obs.get('torso_orientation_z', 1.0))
            velocity_x = obs.get('velocity_x', obs.get('qvel_0', 0.0))
        else:
            # 数组格式观测
            obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            if len(obs_array) >= 3:
                torso_height = obs_array[2] if len(obs_array) > 2 else 1.0
                torso_upright = obs_array[6] if len(obs_array) > 6 else 1.0  # 四元数w分量
                velocity_x = obs_array[7] if len(obs_array) > 7 else 0.0  # 线速度x
            else:
                torso_height = 1.0
                torso_upright = 1.0
                velocity_x = 0.0
        
        # 1. 攀爬高度稳定性 (40%)
        # 基于standing和upright组件
        standing_score = max(0.0, min(1.0, (torso_height - 1.2) / 0.45)) if torso_height > 1.2 else 0.0
        upright_score = max(0.0, min(1.0, (torso_upright - 0.5) / 0.5)) if torso_upright > 0.5 else 0.0
        climbing_stability = standing_score * upright_score
        climbing_stability_scores.append(climbing_stability)
        
        # 2. 前向运动效率 (35%)
        # 基于move组件
        move_score = max(0.0, min(1.0, velocity_x / 1.0)) if velocity_x > 0 else 0.0
        forward_motion_scores.append(move_score)
        
        # 3. 控制效率 (25%)
        # 基于small_control组件
        if i < len(actions):
            action = actions[i]
            if isinstance(action, (list, np.ndarray)):
                action_array = np.array(action) if not isinstance(action, np.ndarray) else action
                control_penalty = np.mean(np.square(action_array))
                control_efficiency = max(0.0, 1.0 - control_penalty / 100.0)  # 归一化控制惩罚
            else:
                control_efficiency = 0.5
        else:
            control_efficiency = 0.5
        control_efficiency_scores.append(control_efficiency)
    
    # 计算平均分数
    avg_climbing_stability = np.mean(climbing_stability_scores) if climbing_stability_scores else 0.0
    avg_forward_motion = np.mean(forward_motion_scores) if forward_motion_scores else 0.0
    avg_control_efficiency = np.mean(control_efficiency_scores) if control_efficiency_scores else 0.0
    
    return avg_climbing_stability, avg_forward_motion, avg_control_efficiency

def evaluate_dpo_preference(trajectory_1, trajectory_2, goal=None):
    """
    基于humanoid_bench真实奖励函数评估Stair任务的DPO偏好
    
    权重分配：
    - 攀爬高度稳定性: 40%
    - 前向运动效率: 35%
    - 控制效率: 25%
    """
    # 计算两个轨迹的奖励组件
    climbing_1, motion_1, control_1 = compute_stair_reward_components(trajectory_1)
    climbing_2, motion_2, control_2 = compute_stair_reward_components(trajectory_2)
    
    # 权重分配
    w_climbing = 0.40
    w_motion = 0.35
    w_control = 0.25
    
    # 计算综合奖励
    reward_1 = w_climbing * climbing_1 + w_motion * motion_1 + w_control * control_1
    reward_2 = w_climbing * climbing_2 + w_motion * motion_2 + w_control * control_2
    
    # 计算偏好概率（使用sigmoid函数）
    reward_diff = reward_1 - reward_2
    preference_prob = 1.0 / (1.0 + np.exp(-reward_diff * 10.0))  # 放大差异
    
    return preference_prob

def compare_h1hand_stair_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-stair-v0 任务的专用比较规则"""
    return compare_stair_trajectories(tau_1, tau_2, goal)

def compare_stair_trajectories(tau_1, tau_2, goal):
    """
    基于humanoid_bench真实奖励函数的Stair场景启发式偏好对比规则
    
    使用DPO偏好评估方法，基于真实奖励函数的三个核心组件：
    1. 攀爬高度稳定性 (40%): standing * upright
    2. 前向运动效率 (35%): move
    3. 控制效率 (25%): small_control
    """
    preference_prob = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 基于偏好概率确定更好的轨迹
    if preference_prob > 0.5:
        return tau_1, tau_2
    elif preference_prob < 0.5:
        return tau_2, tau_1
    else:
        return None, None
