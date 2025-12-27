#!/usr/bin/env python3
"""
Walker Walk任务的启发式规则
基于双足机器人行走任务的步态和稳定性评估
"""

import numpy as np
from typing import Dict, Any, Tuple, List

def compare_walker_walk_trajectories(traj1: Dict[str, Any], traj2: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """
    比较两个Walker行走轨迹的质量
    
    Args:
        traj1: 第一个轨迹数据
        traj2: 第二个轨迹数据
    
    Returns:
        Tuple[int, float, Dict[str, Any]]: (偏好选择, 置信度, 详细信息)
            偏好选择: 1表示traj1更好, 2表示traj2更好, 0表示相似
    """
    
    def evaluate_walker_trajectory(traj: Dict[str, Any]) -> Dict[str, float]:
        """评估单个Walker轨迹的质量指标"""
        
        # 提取轨迹数据
        observations = traj.get('observations', [])
        actions = traj.get('actions', [])
        rewards = traj.get('rewards', [])
        
        if not observations or len(observations) == 0:
            return {
                'survival_time': 0.0,
                'forward_progress': 0.0,
                'upright_stability': 0.0,
                'gait_quality': 0.0,
                'energy_efficiency': 0.0,
                'total_reward': 0.0
            }
        
        # 1. 生存时间
        survival_time = len(observations)
        
        # 2. 前进进度和步态分析
        forward_velocities = []
        body_heights = []
        body_angles = []
        leg_positions = []
        
        for obs in observations:
            if isinstance(obs, (list, np.ndarray)) and len(obs) >= 17:  # Walker通常有17维观测
                # 前进速度
                if len(obs) > 8:
                    forward_velocities.append(obs[8])
                # 身体高度
                if len(obs) > 0:
                    body_heights.append(obs[0])
                # 身体角度
                if len(obs) > 2:
                    body_angles.append(obs[2])
                # 腿部位置（用于步态分析）
                if len(obs) > 5:
                    leg_positions.append([obs[3], obs[4], obs[5], obs[6]])  # 腿部关节角度
        
        # 前进进度
        if forward_velocities:
            avg_forward_speed = np.mean(forward_velocities)
            forward_progress = max(0.0, avg_forward_speed)  # 只考虑正向移动
        else:
            forward_progress = 0.0
        
        # 3. 直立稳定性
        if body_heights and body_angles:
            # 身体高度稳定性
            target_height = np.mean(body_heights) if body_heights else 1.0
            height_stability = 1.0 - (np.std(body_heights) / (target_height + 1e-6))
            height_stability = max(0.0, min(1.0, height_stability))
            
            # 身体角度稳定性（保持直立）
            angle_stability = 1.0 - (np.mean(np.abs(body_angles)) / (np.pi/4))  # 归一化到45度
            angle_stability = max(0.0, min(1.0, angle_stability))
            
            upright_stability = (height_stability + angle_stability) / 2.0
        else:
            upright_stability = 0.0
        
        # 4. 步态质量分析
        if len(leg_positions) > 1:
            # 分析腿部运动的周期性和协调性
            leg_movements = []
            for i in range(len(leg_positions) - 1):
                movement = np.linalg.norm(np.array(leg_positions[i+1]) - np.array(leg_positions[i]))
                leg_movements.append(movement)
            
            if leg_movements:
                # 步态的规律性（低方差表示规律的步态）
                movement_regularity = 1.0 - (np.std(leg_movements) / (np.mean(leg_movements) + 1e-6))
                movement_regularity = max(0.0, min(1.0, movement_regularity))
                
                # 步态的活跃度（适度的腿部运动）
                avg_movement = np.mean(leg_movements)
                movement_activity = min(1.0, avg_movement / 0.5)  # 归一化
                
                gait_quality = (movement_regularity + movement_activity) / 2.0
            else:
                gait_quality = 0.0
        else:
            gait_quality = 0.0
        
        # 5. 能量效率
        if actions:
            action_magnitudes = []
            for action in actions:
                if isinstance(action, (list, np.ndarray)):
                    magnitude = np.linalg.norm(action)
                    action_magnitudes.append(magnitude)
            
            if action_magnitudes:
                avg_action_magnitude = np.mean(action_magnitudes)
                # 效率 = 前进速度 / 动作幅度
                if avg_action_magnitude > 0:
                    energy_efficiency = forward_progress / (avg_action_magnitude + 1e-6)
                    energy_efficiency = min(1.0, energy_efficiency / 3.0)  # 归一化
                else:
                    energy_efficiency = 0.0
            else:
                energy_efficiency = 0.0
        else:
            energy_efficiency = 0.0
        
        # 6. 总奖励
        total_reward = sum(rewards) if rewards else 0.0
        
        return {
            'survival_time': float(survival_time),
            'forward_progress': forward_progress,
            'upright_stability': upright_stability,
            'gait_quality': gait_quality,
            'energy_efficiency': energy_efficiency,
            'total_reward': total_reward
        }
    
    # 评估两个轨迹
    metrics1 = evaluate_walker_trajectory(traj1)
    metrics2 = evaluate_walker_trajectory(traj2)
    
    # 权重配置 - 基于Walker行走任务特点
    weights = {
        'forward_progress': 0.30,     # 前进进度
        'survival_time': 0.25,        # 生存时间
        'upright_stability': 0.25,    # 直立稳定性
        'gait_quality': 0.15,         # 步态质量
        'energy_efficiency': 0.05     # 能量效率
    }
    
    # 计算加权分数
    score1 = sum(weights[key] * metrics1[key] for key in weights.keys())
    score2 = sum(weights[key] * metrics2[key] for key in weights.keys())
    
    # 确定偏好
    score_diff = abs(score1 - score2)
    confidence = min(0.95, max(0.1, score_diff * 2.0))  # 基于分数差异计算置信度
    
    if score_diff < 0.05:  # 分数相近
        preference = 0
        confidence = 0.1
    elif score1 > score2:
        preference = 1
    else:
        preference = 2
    
    # 详细信息
    details = {
        'trajectory_1_metrics': metrics1,
        'trajectory_2_metrics': metrics2,
        'trajectory_1_score': score1,
        'trajectory_2_score': score2,
        'score_difference': score_diff,
        'weights_used': weights,
        'comparison_method': 'walker_walk_heuristic'
    }
    
    return preference, confidence, details

def evaluate_dpo_preference(trajectory_pair: Tuple[Dict[str, Any], Dict[str, Any]], 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    为DPO训练评估轨迹对的偏好
    
    Args:
        trajectory_pair: 轨迹对 (chosen_trajectory, rejected_trajectory)
        context: 额外的上下文信息
    
    Returns:
        Dict[str, Any]: DPO偏好评估结果
    """
    
    chosen_traj, rejected_traj = trajectory_pair
    
    # 使用比较函数获得偏好
    preference, confidence, details = compare_walker_walk_trajectories(chosen_traj, rejected_traj)
    
    # 构建DPO格式的结果
    dpo_result = {
        'preference': preference,
        'confidence': confidence,
        'chosen_better': preference == 1,
        'rejected_better': preference == 2,
        'similar_quality': preference == 0,
        'evaluation_details': details,
        'task_type': 'walker_walk',
        'evaluation_method': 'heuristic_rules'
    }
    
    # 添加上下文信息
    if context:
        dpo_result['context'] = context
    
    # 添加质量分数
    dpo_result['chosen_score'] = details['trajectory_1_score']
    dpo_result['rejected_score'] = details['trajectory_2_score']
    dpo_result['score_margin'] = abs(details['trajectory_1_score'] - details['trajectory_2_score'])
    
    return dpo_result

# 兼容性函数
def compare_trajectories(traj1: Dict[str, Any], traj2: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """兼容性函数，调用主要的比较函数"""
    return compare_walker_walk_trajectories(traj1, traj2)