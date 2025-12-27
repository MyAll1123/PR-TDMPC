from .common_utils import tolerance

"""Run任务的自包含规则：基于humanoid_bench Run任务的真实奖励函数设计

自包含设计，无外部依赖，基于humanoid_bench Run任务的奖励函数实现轨迹比较。
"""

import numpy as np

# 基于humanoid_bench真实奖励函数的Run任务启发式规则
# Run任务的关键参数
_STAND_HEIGHT = 1.65
_RUN_SPEED = 5  # 跑步目标速度

def evaluate_dpo_preference(tau_1, tau_2, goal=None):
    """
    基于humanoid_bench真实奖励函数的Run任务DPO偏好评估
    
    Run任务奖励函数组成：
    - stand_reward = standing * upright (站立稳定性)
    - small_control (控制力效率)
    - move (高速前进运动)
    
    权重分配：
    - 站立稳定性：35% (standing + upright)
    - 高速运动：40% (move reward，跑步的核心)
    - 控制效率：25% (small_control)
    """
    
    def compute_run_reward_components(trajectory):
        """计算跑步任务的奖励组件"""
        # 处理字典格式的轨迹数据
        if isinstance(trajectory, dict):
            observations = trajectory.get('observations', [])
            actions = trajectory.get('actions', [])
        else:
            observations = getattr(trajectory, 'observations', [])
            actions = getattr(trajectory, 'actions', [])
        
        if len(observations) == 0:
            return 0.0, {'standing_stability': 0.0, 'high_speed_motion': 0.0, 'control_efficiency': 0.0}
        
        total_standing_stability = 0.0
        total_high_speed_motion = 0.0
        total_control_efficiency = 0.0
        
        for i, obs in enumerate(observations):
            if i < len(actions):
                action = np.array(actions[i]) if not isinstance(actions[i], np.ndarray) else actions[i]
            else:
                action = np.zeros(21)  # 默认动作维度
            
            # 从观测中提取状态信息
            if isinstance(obs, dict):
                head_height = obs.get('torso_height', 1.0)
                torso_upright = obs.get('torso_upright', 0.9)
                com_velocity_x = obs.get('velocity_x', 0.0)
            elif hasattr(obs, '__len__') and len(obs) >= 3:
                head_height = obs[2] if obs[2] > 0 else 1.0
                torso_upright = obs[6] if len(obs) > 6 else 0.9
                com_velocity_x = obs[26] if len(obs) > 26 else 0.0
            else:
                head_height = 1.0
                torso_upright = 0.9
                com_velocity_x = 0.0
            
            # 1. 站立稳定性 (standing * upright)
            standing = tolerance(
                head_height,
                bounds=(_STAND_HEIGHT, float("inf")),
                margin=_STAND_HEIGHT / 4,
            )
            upright = tolerance(
                torso_upright,
                bounds=(0.9, float("inf")),
                sigmoid="linear",
                margin=1.9,
                value_at_margin=0,
            )
            standing_stability = standing * upright
            
            # 2. 高速前进运动 (move reward)
            move = tolerance(
                com_velocity_x,
                bounds=(_RUN_SPEED, float("inf")),
                margin=_RUN_SPEED,
                value_at_margin=0,
                sigmoid="linear",
            )
            high_speed_motion = (5 * move + 1) / 6
            
            # 3. 控制力效率 (small_control)
            actuator_forces = np.abs(action)
            small_control = tolerance(
                actuator_forces,
                margin=10,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
            control_efficiency = (4 + small_control) / 5
            
            total_standing_stability += standing_stability
            total_high_speed_motion += high_speed_motion
            total_control_efficiency += control_efficiency
        
        num_steps = len(observations)
        if num_steps > 0:
            total_standing_stability /= num_steps
            total_high_speed_motion /= num_steps
            total_control_efficiency /= num_steps
        
        # 综合奖励计算
        total_reward = (
            0.35 * total_standing_stability +
            0.40 * total_high_speed_motion +
            0.25 * total_control_efficiency
        )
        
        return total_reward, {
            'standing_stability': total_standing_stability,
            'high_speed_motion': total_high_speed_motion,
            'control_efficiency': total_control_efficiency
        }
    
    # 计算两个轨迹的奖励
    reward_1, components_1 = compute_run_reward_components(tau_1)
    reward_2, components_2 = compute_run_reward_components(tau_2)
    
    # 计算偏好概率
    reward_diff = reward_1 - reward_2
    preference_prob = 1.0 / (1.0 + np.exp(-reward_diff))
    
    return preference_prob, {
        'tau_1_reward': reward_1,
        'tau_2_reward': reward_2,
        'tau_1_components': components_1,
        'tau_2_components': components_2,
        'reward_difference': reward_diff
    }

def compare_h1hand_run_v0_trajectories(tau_1, tau_2, goal=None):
    """h1hand-run-v0 任务的专用比较规则"""
    preference_prob, info = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 基于偏好概率判断哪个轨迹更好
    if preference_prob > 0.55:  # tau_1 明显更好
        return tau_1, tau_2
    elif preference_prob < 0.45:  # tau_2 明显更好
        return tau_2, tau_1
    else:  # 差异不明显
        return None, None