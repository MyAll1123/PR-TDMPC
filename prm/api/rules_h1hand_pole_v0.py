import numpy as np
from typing import Dict, Tuple, Union, Optional


def compare_h1hand_pole_v0_trajectories(tau_1, tau_2, goal=None):
    """h1hand-pole-v0 任务的专用比较规则"""
    preference_prob = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    if preference_prob > 0.5:
        return tau_1, tau_2  # tau_1 更好
    elif preference_prob < 0.5:
        return tau_2, tau_1  # tau_2 更好
    else:
        return None, None  # 两个轨迹相等

def compute_pole_reward_components(trajectory: Union[Dict, object], goal: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    基于humanoid_bench Pole类的真实奖励函数计算奖励组件
    
    Pole任务真实奖励函数分析：
    - standing: 基于头部高度和躯干直立度的站立奖励
    - small_control: 小控制奖励，鼓励较小的控制输入
    - move: 基于质心速度的移动奖励
    - collision_discount: 碰撞惩罚因子
    - 总奖励: standing + small_control + move * collision_discount
    
    奖励组件权重分配：
    1. 站立稳定性 (45%): 基于standing组件
    2. 移动效率 (35%): 基于move组件和collision_discount
    3. 控制效率 (20%): 基于small_control组件
    
    Args:
        trajectory: 轨迹数据，包含observations和actions
        goal: 目标位置（可选）
        
    Returns:
        Tuple[float, float, float]: (站立稳定性, 移动效率, 控制效率)
    """
    try:
        # 处理不同格式的轨迹数据
        if trajectory is None:
            return 0.0, 0.0, 0.0
            
        if isinstance(trajectory, dict):
            observations = trajectory.get('observations', [])
            actions = trajectory.get('actions', [])
        else:
            observations = getattr(trajectory, 'observations', [])
            actions = getattr(trajectory, 'actions', [])
        
        if not observations or not actions:
            return 0.0, 0.0, 0.0
        
        # 确保observations和actions是numpy数组
        if isinstance(observations, list):
            observations = np.array(observations)
        if isinstance(actions, list):
            actions = np.array(actions)
            
        # 从观测中提取关键状态信息
        # 假设观测格式：[position_x, position_y, position_z, orientation, ...]
        # 头部位置和躯干姿态
        head_positions = observations[:, :3] if observations.shape[1] > 2 else observations
        
        # 质心位置（假设在观测的特定位置）
        com_positions = observations[:, 3:6] if observations.shape[1] > 5 else observations[:, :3]
        
        # 躯干姿态（四元数或欧拉角）
        trunk_orientations = observations[:, 6:10] if observations.shape[1] > 9 else np.zeros((len(observations), 4))
        
        # 1. 站立稳定性组件
        # 头部高度奖励
        head_heights = head_positions[:, 2]  # z坐标
        avg_head_height = np.mean(head_heights)
        min_head_height = np.min(head_heights)
        
        # 站立高度奖励（头部高度应该在合理范围内）
        height_reward = np.clip(avg_head_height / 1.5, 0.0, 1.0)  # 假设理想高度1.5m
        
        # 躯干直立度奖励（基于姿态）
        if trunk_orientations.shape[1] >= 4:
            # 假设是四元数格式 [x, y, z, w]
            # 计算躯干与垂直方向的偏差
            upright_scores = []
            for quat in trunk_orientations:
                # 简化的直立度计算
                upright_score = abs(quat[3])  # w分量表示与垂直的接近程度
                upright_scores.append(upright_score)
            avg_upright = np.mean(upright_scores)
        else:
            avg_upright = 0.8  # 默认值
        
        # 综合站立稳定性
        standing_stability = 0.6 * height_reward + 0.4 * avg_upright
        standing_stability = np.clip(standing_stability, 0.0, 1.0)
        
        # 2. 移动效率组件
        # 质心速度计算
        if len(com_positions) > 1:
            com_velocities = np.diff(com_positions, axis=0)
            com_speeds = np.linalg.norm(com_velocities, axis=1)
            avg_speed = np.mean(com_speeds)
            
            # 移动奖励（适度的移动速度）
            optimal_speed = 0.5  # 假设最优速度
            speed_reward = np.exp(-abs(avg_speed - optimal_speed))
        else:
            speed_reward = 0.5
        
        # 碰撞避免（基于轨迹的平滑性）
        if len(observations) > 2:
            position_changes = np.diff(observations[:, :3], axis=0)
            smoothness = np.mean(np.linalg.norm(position_changes, axis=1))
            collision_discount = np.exp(-smoothness * 2.0)  # 平滑度越高，碰撞风险越低
        else:
            collision_discount = 1.0
        
        # 综合移动效率
        movement_efficiency = speed_reward * collision_discount
        movement_efficiency = np.clip(movement_efficiency, 0.0, 1.0)
        
        # 3. 控制效率组件
        # 基于动作的大小和平滑性
        action_magnitudes = np.linalg.norm(actions, axis=1)
        avg_action_magnitude = np.mean(action_magnitudes)
        
        # 小控制奖励（动作越小越好）
        small_control_reward = np.exp(-avg_action_magnitude)
        
        # 动作平滑性
        if len(actions) > 1:
            action_changes = np.diff(actions, axis=0)
            action_smoothness = np.mean(np.linalg.norm(action_changes, axis=1))
            smoothness_reward = np.exp(-action_smoothness)
        else:
            smoothness_reward = 1.0
        
        # 综合控制效率
        control_efficiency = 0.7 * small_control_reward + 0.3 * smoothness_reward
        control_efficiency = np.clip(control_efficiency, 0.0, 1.0)
        
        return standing_stability, movement_efficiency, control_efficiency
        
    except Exception as e:
        # 异常情况返回默认值
        return 0.0, 0.0, 0.0

def evaluate_dpo_preference(traj1: Union[Dict, object], traj2: Union[Dict, object], goal: Optional[np.ndarray] = None) -> float:
    """
    使用DPO偏好评估方法比较两个轨迹
    
    Args:
        traj1: 第一个轨迹
        traj2: 第二个轨迹
        goal: 目标位置（可选）
        
    Returns:
        float: 偏好概率，>0.5表示偏好traj1，<0.5表示偏好traj2
    """
    try:
        # 计算两个轨迹的奖励组件
        standing1, movement1, control1 = compute_pole_reward_components(traj1, goal)
        standing2, movement2, control2 = compute_pole_reward_components(traj2, goal)
        
        # 权重分配：站立稳定性45%，移动效率35%，控制效率20%
        weights = [0.45, 0.35, 0.2]
        
        # 计算加权综合奖励
        reward1 = (weights[0] * standing1 + 
                  weights[1] * movement1 + 
                  weights[2] * control1)
        
        reward2 = (weights[0] * standing2 + 
                  weights[1] * movement2 + 
                  weights[2] * control2)
        
        # 使用sigmoid函数计算偏好概率
        preference_prob = 1 / (1 + np.exp(-(reward1 - reward2)))
        
        return preference_prob
        
    except Exception as e:
        # 异常情况返回中性偏好
        return 0.5