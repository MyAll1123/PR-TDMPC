import numpy as np
from typing import Dict, Any, Tuple


def compute_package_reward_components(trajectory_data: Dict[str, Any]) -> Dict[str, float]:
    """
    计算package任务的各个奖励组件，基于humanoid_bench/envs/package.py的get_reward方法
    
    Args:
        trajectory_data: 包含轨迹数据的字典
        
    Returns:
        包含各个奖励组件的字典
    """
    # 获取必要的数据
    robot_positions = trajectory_data.get('robot_positions', [])
    robot_orientations = trajectory_data.get('robot_orientations', [])
    package_positions = trajectory_data.get('package_positions', [])
    destination_positions = trajectory_data.get('destination_positions', [])
    left_hand_positions = trajectory_data.get('left_hand_positions', [])
    right_hand_positions = trajectory_data.get('right_hand_positions', [])
    actuator_forces = trajectory_data.get('actuator_forces', [])
    
    if not all([robot_positions, robot_orientations, package_positions, 
                destination_positions, left_hand_positions, right_hand_positions]):
        return {
            'stand_reward': 0.0,
            'small_control': 0.0,
            'package_destination_distance': 0.0,
            'hand_package_distance': 0.0,
            'package_height': 0.0,
            'success_reward': 0.0
        }
    
    # 计算各个组件的平均值
    stand_rewards = []
    small_controls = []
    package_destination_distances = []
    hand_package_distances = []
    package_heights = []
    success_rewards = []
    
    _STAND_HEIGHT = 1.65
    
    for i in range(len(robot_positions)):
        # 站立奖励 (standing * upright)
        robot_pos = np.array(robot_positions[i])
        robot_orient = np.array(robot_orientations[i])
        
        # 计算头部高度 (假设头部在机器人位置上方)
        head_height = robot_pos[2] + 0.2  # 简化假设
        
        # 站立奖励
        standing = max(0, min(1, (head_height - _STAND_HEIGHT/4) / (_STAND_HEIGHT/4)))
        
        # 直立奖励 (基于四元数计算)
        if len(robot_orient) >= 4:
            # 从四元数计算直立度
            qw, qx, qy, qz = robot_orient[:4]
            # 计算z轴向上的程度
            upright = 2 * (qw*qw + qz*qz) - 1
            upright = max(0, min(1, (upright - 0.8) / 0.2))
        else:
            upright = 1.0
            
        stand_reward = standing * upright
        stand_rewards.append(stand_reward)
        
        # 控制力奖励
        if i < len(actuator_forces) and len(actuator_forces[i]) > 0:
            forces = np.array(actuator_forces[i])
            small_control = np.mean(np.exp(-forces**2 / 100))  # 简化的控制力惩罚
            small_control = (4 + small_control) / 5
        else:
            small_control = 0.8
        small_controls.append(small_control)
        
        # 包裹到目标距离
        if i < len(package_positions) and i < len(destination_positions):
            package_pos = np.array(package_positions[i])
            dest_pos = np.array(destination_positions[i])
            dist_package_dest = np.linalg.norm(package_pos - dest_pos)
            package_destination_distances.append(dist_package_dest)
            
            # 包裹高度奖励
            package_height = min(package_pos[2], 1.0)
            package_heights.append(package_height)
            
            # 成功奖励
            success_reward = 1000.0 if dist_package_dest < 0.1 else 0.0
            success_rewards.append(success_reward)
        else:
            package_destination_distances.append(10.0)  # 大距离作为惩罚
            package_heights.append(0.0)
            success_rewards.append(0.0)
        
        # 手到包裹距离
        if (i < len(left_hand_positions) and i < len(right_hand_positions) and 
            i < len(package_positions)):
            left_hand = np.array(left_hand_positions[i])
            right_hand = np.array(right_hand_positions[i])
            package_pos = np.array(package_positions[i])
            
            dist_left = np.linalg.norm(left_hand - package_pos)
            dist_right = np.linalg.norm(right_hand - package_pos)
            hand_package_dist = dist_left + dist_right
            hand_package_distances.append(hand_package_dist)
        else:
            hand_package_distances.append(10.0)  # 大距离作为惩罚
    
    return {
        'stand_reward': np.mean(stand_rewards) if stand_rewards else 0.0,
        'small_control': np.mean(small_controls) if small_controls else 0.0,
        'package_destination_distance': np.mean(package_destination_distances) if package_destination_distances else 10.0,
        'hand_package_distance': np.mean(hand_package_distances) if hand_package_distances else 10.0,
        'package_height': np.mean(package_heights) if package_heights else 0.0,
        'success_reward': np.mean(success_rewards) if success_rewards else 0.0
    }

def compute_trajectory_score(trajectory_data: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    计算package任务轨迹的综合得分
    
    Args:
        trajectory_data: 轨迹数据
        config: 配置参数
        
    Returns:
        轨迹综合得分
    """
    components = compute_package_reward_components(trajectory_data)
    
    # 获取权重配置
    weights = config.get('task_specific_weights', {
        'stand_reward': 1.0,
        'small_control': 1.0,
        'package_destination_penalty': 3.0,
        'hand_package_penalty': 0.1,
        'package_height': 1.0,
        'success_bonus': 1000.0
    })
    
    # 计算综合得分 (基于humanoid_bench的公式)
    score = (
        components['stand_reward'] * components['small_control'] * weights['stand_reward'] * weights['small_control']
        - components['package_destination_distance'] * weights['package_destination_penalty']
        - components['hand_package_distance'] * weights['hand_package_penalty']
        + components['package_height'] * weights['package_height']
        + components['success_reward'] * weights['success_bonus'] / 1000.0  # 归一化成功奖励
    )
    
    return score

def evaluate_package_trajectory(trajectory_data: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    评估package任务轨迹
    
    Args:
        trajectory_data: 轨迹数据
        config: 配置参数
        
    Returns:
        (得分, 详细信息字典)
    """
    components = compute_package_reward_components(trajectory_data)
    score = compute_trajectory_score(trajectory_data, config)
    
    # 判断任务完成情况
    task_completed = components['package_destination_distance'] < 0.1
    
    details = {
        'score': score,
        'task_completed': task_completed,
        'components': components,
        'metrics': {
            'standing_stability': components['stand_reward'],
            'control_efficiency': components['small_control'],
            'package_delivery_progress': max(0, 1 - components['package_destination_distance'] / 5.0),
            'manipulation_skill': max(0, 1 - components['hand_package_distance'] / 2.0),
            'package_handling': components['package_height'],
            'success_rate': 1.0 if task_completed else 0.0
        }
    }
    
    return score, details

def compare_h1hand_package_v0_trajectories(trajectory_a: Dict[str, Any], trajectory_b: Dict[str, Any], config: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    比较两个package任务轨迹，返回更好的轨迹和较差的轨迹
    
    Args:
        trajectory_a: 第一个轨迹数据
        trajectory_b: 第二个轨迹数据
        config: 配置参数
        
    Returns:
        (better_trajectory, worse_trajectory): 更好的轨迹和较差的轨迹
    """
    try:
        if config is None:
            config = {}
        
        # 计算两个轨迹的得分
        score_a = compute_trajectory_score(trajectory_a, config)
        score_b = compute_trajectory_score(trajectory_b, config)
        
        # 确保分数为标量值
        if hasattr(score_a, 'item'):
            score_a = score_a.item()
        if hasattr(score_b, 'item'):
            score_b = score_b.item()
        
        # 显式转换为float进行比较
        score_a = float(score_a)
        score_b = float(score_b)
        
        # 返回更好的轨迹和较差的轨迹
        if score_a > score_b:
            return trajectory_a, trajectory_b
        else:
            return trajectory_b, trajectory_a
            
    except Exception as e:
        # 如果比较失败，返回原始顺序
        return trajectory_a, trajectory_b

def evaluate_dpo_preference(trajectory_a: Dict[str, Any], trajectory_b: Dict[str, Any], config: Dict[str, Any] = None) -> Tuple[str, float]:
    """
    评估两个轨迹的DPO偏好
    
    Args:
        trajectory_a: 第一个轨迹数据
        trajectory_b: 第二个轨迹数据
        config: 配置参数
        
    Returns:
        (preference, confidence): 偏好结果('A'或'B')和置信度(0-1)
    """
    try:
        if config is None:
            config = {}
        
        # 计算两个轨迹的得分
        score_a = compute_trajectory_score(trajectory_a, config)
        score_b = compute_trajectory_score(trajectory_b, config)
        
        # 确保分数为标量值
        if hasattr(score_a, 'item'):
            score_a = score_a.item()
        if hasattr(score_b, 'item'):
            score_b = score_b.item()
        
        # 计算偏好和置信度
        score_diff = abs(float(score_a) - float(score_b))
        max_possible_diff = 10.0  # 假设最大可能差异
        confidence = min(score_diff / max_possible_diff, 1.0)
        
        # 确定偏好
        preference = 'A' if float(score_a) > float(score_b) else 'B'
        
        return preference, confidence
        
    except Exception as e:
        # 如果评估失败，返回默认值
        return 'A', 0.1