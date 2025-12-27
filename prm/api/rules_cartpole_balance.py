#!/usr/bin/env python3
"""
CartPole Balance任务的启发式规则
基于经典控制任务的平衡和稳定性评估
"""

import numpy as np
from typing import Dict, Any, Tuple, List

def compare_cartpole_balance_trajectories(traj1: Dict[str, Any], traj2: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """
    比较两个CartPole平衡轨迹的质量
    
    Args:
        traj1: 第一个轨迹数据
        traj2: 第二个轨迹数据
    
    Returns:
        Tuple[int, float, Dict[str, Any]]: (偏好选择, 置信度, 详细信息)
            偏好选择: 1表示traj1更好, 2表示traj2更好, 0表示相似
    """
    
    def evaluate_cartpole_trajectory(traj: Dict[str, Any]) -> Dict[str, float]:
        """评估单个CartPole轨迹的质量指标"""
        
        # 提取轨迹数据
        observations = traj.get('observations', [])
        actions = traj.get('actions', [])
        rewards = traj.get('rewards', [])
        
        if not observations or len(observations) == 0:
            return {
                'survival_time': 0.0,
                'balance_stability': 0.0,
                'action_smoothness': 0.0,
                'pole_angle_control': 0.0,
                'cart_position_control': 0.0,
                'total_reward': 0.0
            }
        
        # 1. 生存时间 - CartPole的核心指标
        survival_time = len(observations)
        
        # 2. 平衡稳定性 - 基于杆子角度的稳定性
        pole_angles = []
        cart_positions = []
        cart_velocities = []
        pole_velocities = []
        
        for obs in observations:
            if isinstance(obs, (list, np.ndarray)) and len(obs) >= 4:
                cart_positions.append(obs[0])  # 小车位置
                cart_velocities.append(obs[1])  # 小车速度
                pole_angles.append(obs[2])     # 杆子角度
                pole_velocities.append(obs[3]) # 杆子角速度
        
        # 平衡稳定性：杆子角度越小越好
        if pole_angles:
            angle_stability = 1.0 - (np.mean(np.abs(pole_angles)) / (np.pi/6))  # 归一化到30度
            angle_stability = max(0.0, min(1.0, angle_stability))
        else:
            angle_stability = 0.0
        
        # 3. 动作平滑度
        if len(actions) > 1:
            action_changes = np.abs(np.diff(actions))
            action_smoothness = 1.0 - (np.mean(action_changes) / 2.0)  # 动作变化范围是0-2
            action_smoothness = max(0.0, min(1.0, action_smoothness))
        else:
            action_smoothness = 1.0
        
        # 4. 杆子角度控制 - 角度变化的稳定性
        if len(pole_angles) > 1:
            angle_changes = np.abs(np.diff(pole_angles))
            pole_control = 1.0 - (np.mean(angle_changes) / (np.pi/4))  # 归一化
            pole_control = max(0.0, min(1.0, pole_control))
        else:
            pole_control = 1.0
        
        # 5. 小车位置控制 - 保持在中心附近
        if cart_positions:
            position_deviation = np.mean(np.abs(cart_positions))
            cart_control = 1.0 - (position_deviation / 2.4)  # CartPole边界通常是±2.4
            cart_control = max(0.0, min(1.0, cart_control))
        else:
            cart_control = 0.0
        
        # 6. 总奖励
        total_reward = sum(rewards) if rewards else 0.0
        
        return {
            'survival_time': float(survival_time),
            'balance_stability': angle_stability,
            'action_smoothness': action_smoothness,
            'pole_angle_control': pole_control,
            'cart_position_control': cart_control,
            'total_reward': total_reward
        }
    
    # 评估两个轨迹
    metrics1 = evaluate_cartpole_trajectory(traj1)
    metrics2 = evaluate_cartpole_trajectory(traj2)
    
    # 权重配置 - 基于CartPole任务特点
    weights = {
        'survival_time': 0.40,        # 生存时间最重要
        'balance_stability': 0.25,    # 平衡稳定性
        'pole_angle_control': 0.15,   # 杆子控制
        'cart_position_control': 0.10, # 位置控制
        'action_smoothness': 0.10     # 动作平滑度
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
        'comparison_method': 'cartpole_balance_heuristic'
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
    preference, confidence, details = compare_cartpole_balance_trajectories(chosen_traj, rejected_traj)
    
    # 构建DPO格式的结果
    dpo_result = {
        'preference': preference,
        'confidence': confidence,
        'chosen_better': preference == 1,
        'rejected_better': preference == 2,
        'similar_quality': preference == 0,
        'evaluation_details': details,
        'task_type': 'cartpole_balance',
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
    return compare_cartpole_balance_trajectories(traj1, traj2)