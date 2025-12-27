from .common_utils import tolerance

"""Stand任务的自包含规则：基于humanoid_bench Stand任务的真实奖励函数设计

自包含设计，无外部依赖，基于humanoid_bench Stand任务的奖励函数实现轨迹比较。
"""

import numpy as np

# 基于humanoid_bench真实奖励函数的Stand任务启发式规则
# Stand任务的关键参数
_STAND_HEIGHT = 1.65

def evaluate_dpo_preference(tau_1, tau_2, goal=None):
    """
    基于humanoid_bench真实奖励函数的Stand任务DPO偏好评估
    
    Stand任务奖励函数组成：
    - stand_reward = standing * upright (站立稳定性)
    - small_control (控制力效率)
    - dont_move (最小运动，站立时不应移动)
    
    权重分配：
    - 站立稳定性：45% (standing + upright，站立的核心)
    - 最小运动：35% (dont_move，保持静止)
    - 控制效率：20% (small_control)
    """
    
    def compute_stand_reward_components(trajectory):
        """计算站立任务的奖励组件"""
        # 处理字典格式的轨迹数据
        if isinstance(trajectory, dict):
            observations = trajectory.get('observations', [])
            actions = trajectory.get('actions', [])
        else:
            observations = getattr(trajectory, 'observations', [])
            actions = getattr(trajectory, 'actions', [])
        
        if len(observations) == 0:
            return 0.0, {'standing_stability': 0.0, 'minimal_movement': 0.0, 'control_efficiency': 0.0}
        
        total_standing_stability = 0.0
        total_minimal_movement = 0.0
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
                com_velocity_y = obs.get('velocity_y', 0.0)
            elif hasattr(obs, '__len__') and len(obs) >= 3:
                head_height = obs[2] if obs[2] > 0 else 1.0
                torso_upright = obs[6] if len(obs) > 6 else 0.9
                # 水平速度 (x, y方向)
                com_velocity_x = obs[26] if len(obs) > 26 else 0.0
                com_velocity_y = obs[27] if len(obs) > 27 else 0.0
            else:
                head_height = 1.0
                torso_upright = 0.9
                com_velocity_x = 0.0
                com_velocity_y = 0.0
            
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
            
            # 2. 最小运动 (dont_move)
            horizontal_velocity = np.array([com_velocity_x, com_velocity_y])
            dont_move = tolerance(horizontal_velocity, margin=2).mean()
            minimal_movement = dont_move
            
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
            total_minimal_movement += minimal_movement
            total_control_efficiency += control_efficiency
        
        num_steps = len(observations)
        if num_steps > 0:
            total_standing_stability /= num_steps
            total_minimal_movement /= num_steps
            total_control_efficiency /= num_steps
        
        # 综合奖励计算
        total_reward = (
            0.45 * total_standing_stability +
            0.35 * total_minimal_movement +
            0.20 * total_control_efficiency
        )
        
        return total_reward, {
            'standing_stability': total_standing_stability,
            'minimal_movement': total_minimal_movement,
            'control_efficiency': total_control_efficiency
        }
    
    # 计算两个轨迹的奖励
    reward_1, components_1 = compute_stand_reward_components(tau_1)
    reward_2, components_2 = compute_stand_reward_components(tau_2)
    
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

def compare_h1hand_stand_v0_trajectories(tau_1, tau_2, goal=None):
    """h1hand-stand-v0 任务的专用比较规则"""
    preference_prob, info = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    # 基于偏好概率判断哪个轨迹更好
    if preference_prob > 0.55:  # tau_1 明显更好
        return tau_1, tau_2
    elif preference_prob < 0.45:  # tau_2 明显更好
        return tau_2, tau_1
    else:  # 差异不明显
        return None, None