import numpy as np
from typing import Dict, Tuple, Union, Optional


def compare_h1hand_maze_v0_trajectories(tau_1, tau_2, goal=None):
    """h1hand-maze-v0 任务的专用比较规则"""
    preference_prob = evaluate_dpo_preference(tau_1, tau_2, goal)
    
    if preference_prob > 0.5:
        return tau_1, tau_2  # tau_1 更好
    elif preference_prob < 0.5:
        return tau_2, tau_1  # tau_2 更好
    else:
        return None, None  # 两个轨迹相等

def compute_maze_reward_components(trajectory: Union[Dict, object], goal: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    基于humanoid_bench Maze类的真实奖励函数计算奖励组件
    
    Maze任务真实奖励函数分析：
    - standing: 基于头部高度和躯干直立度的站立奖励
    - small_control: 小控制奖励，鼓励较小的控制输入
    - wall_collision_discount: 墙壁碰撞惩罚
    - move: 基于质心速度的移动奖励
    - stage_transition_reward: 阶段转换奖励
    - checkpoint_approach_reward: 检查点接近奖励
    - success_subtask: 成功子任务奖励
    - 总奖励: standing + small_control + (move + stage_transition_reward + checkpoint_approach_reward + success_subtask) * wall_collision_discount
    
    奖励组件权重分配：
    1. 导航成功率 (40%): 结合checkpoint_approach_reward和success_subtask
    2. 移动效率 (35%): 基于move组件和wall_collision_discount
    3. 站立稳定性 (25%): 基于standing和small_control组件
    
    Args:
        trajectory: 轨迹数据，包含observations和actions
        goal: 目标位置（可选）
        
    Returns:
        Tuple[float, float, float]: (导航成功率, 移动效率, 站立稳定性)
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
        positions = observations[:, :3] if observations.shape[1] > 2 else observations
        
        # 质心位置（假设在观测的特定位置）
        com_positions = observations[:, 3:6] if observations.shape[1] > 5 else observations[:, :3]
        
        # 躯干姿态（四元数或欧拉角）
        trunk_orientations = observations[:, 6:10] if observations.shape[1] > 9 else np.zeros((len(observations), 4))
        
        # 1. 导航成功率组件
        # 基于位置变化和目标接近程度
        if len(positions) > 1:
            # 计算总移动距离
            position_changes = np.diff(positions, axis=0)
            total_distance = np.sum(np.linalg.norm(position_changes, axis=1))
            
            # 计算直线距离（起点到终点）
            straight_distance = np.linalg.norm(positions[-1] - positions[0])
            
            # 路径效率（直线距离/总距离，越接近1越好）
            if total_distance > 0:
                path_efficiency = min(straight_distance / total_distance, 1.0)
            else:
                path_efficiency = 0.0
            
            # 前进进度（基于x方向的移动，假设迷宫是向前的）
            forward_progress = max(0, positions[-1][0] - positions[0][0])
            progress_reward = np.tanh(forward_progress)  # 使用tanh限制在[0,1]
        else:
            path_efficiency = 0.0
            progress_reward = 0.0
        
        # 检查点接近奖励（模拟）
        # 假设检查点在特定位置，这里简化为基于前进距离
        checkpoint_reward = progress_reward
        
        # 综合导航成功率
        navigation_success = 0.4 * path_efficiency + 0.6 * checkpoint_reward
        navigation_success = np.clip(navigation_success, 0.0, 1.0)
        
        # 2. 移动效率组件
        # 质心速度计算
        if len(com_positions) > 1:
            com_velocities = np.diff(com_positions, axis=0)
            com_speeds = np.linalg.norm(com_velocities, axis=1)
            avg_speed = np.mean(com_speeds)
            
            # 移动奖励（适度的移动速度）
            optimal_speed = 0.3  # 迷宫中较慢的最优速度
            speed_reward = np.exp(-abs(avg_speed - optimal_speed) * 2.0)
        else:
            speed_reward = 0.5
        
        # 墙壁碰撞避免（基于轨迹平滑性）
        if len(positions) > 2:
            # 计算轨迹的曲率变化，急转弯可能表示碰撞
            velocity_changes = np.diff(position_changes, axis=0)
            curvature = np.mean(np.linalg.norm(velocity_changes, axis=1))
            wall_collision_discount = np.exp(-curvature * 3.0)  # 曲率越大，碰撞风险越高
        else:
            wall_collision_discount = 1.0
        
        # 综合移动效率
        movement_efficiency = speed_reward * wall_collision_discount
        movement_efficiency = np.clip(movement_efficiency, 0.0, 1.0)
        
        # 3. 站立稳定性组件
        # 头部高度奖励
        head_heights = positions[:, 2]  # z坐标
        avg_head_height = np.mean(head_heights)
        min_head_height = np.min(head_heights)
        
        # 站立高度奖励
        height_reward = np.clip(avg_head_height / 1.5, 0.0, 1.0)  # 假设理想高度1.5m
        
        # 躯干直立度奖励
        if trunk_orientations.shape[1] >= 4:
            upright_scores = []
            for quat in trunk_orientations:
                upright_score = abs(quat[3])  # w分量
                upright_scores.append(upright_score)
            avg_upright = np.mean(upright_scores)
        else:
            avg_upright = 0.8  # 默认值
        
        # 控制效率（小控制奖励）
        action_magnitudes = np.linalg.norm(actions, axis=1)
        avg_action_magnitude = np.mean(action_magnitudes)
        small_control_reward = np.exp(-avg_action_magnitude)
        
        # 综合站立稳定性
        standing_stability = 0.4 * height_reward + 0.3 * avg_upright + 0.3 * small_control_reward
        standing_stability = np.clip(standing_stability, 0.0, 1.0)
        
        return navigation_success, movement_efficiency, standing_stability
        
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
        navigation1, movement1, standing1 = compute_maze_reward_components(traj1, goal)
        navigation2, movement2, standing2 = compute_maze_reward_components(traj2, goal)
        
        # 权重分配：导航成功率40%，移动效率35%，站立稳定性25%
        weights = [0.4, 0.35, 0.25]
        
        # 计算加权综合奖励
        reward1 = (weights[0] * navigation1 + 
                  weights[1] * movement1 + 
                  weights[2] * standing1)
        
        reward2 = (weights[0] * navigation2 + 
                  weights[1] * movement2 + 
                  weights[2] * standing2)
        
        # 使用sigmoid函数计算偏好概率
        preference_prob = 1 / (1 + np.exp(-(reward1 - reward2)))
        
        return preference_prob
        
    except Exception as e:
        # 异常情况返回中性偏好
        return 0.5