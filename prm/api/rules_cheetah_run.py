#!/usr/bin/env python3
"""
Cheetah Run任务的启发式规则
基于四足机器人跑步任务的速度和稳定性评估
"""

import numpy as np
from typing import Dict, Any, Tuple, List

def compare_cheetah_run_trajectories(traj1: Dict[str, Any], traj2: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """
    比较两个Cheetah跑步轨迹的质量
    
    Args:
        traj1: 第一个轨迹数据
        traj2: 第二个轨迹数据
    
    Returns:
        Tuple[int, float, Dict[str, Any]]: (偏好选择, 置信度, 详细信息)
            偏好选择: 1表示traj1更好, 2表示traj2更好, 0表示相似
    """
    
    def evaluate_cheetah_trajectory(traj: Dict[str, Any]) -> Dict[str, float]:
        """评估单个Cheetah轨迹的质量指标"""
        
        # 提取轨迹数据
        observations = traj.get('observations', [])
        actions = traj.get('actions', [])
        rewards = traj.get('rewards', [])
        
        if not observations or len(observations) == 0:
            return {
                'survival_time': 0.0,
                'forward_speed': 0.0,
                'action_smoothness': 0.0,
                'body_stability': 0.0,
                'energy_efficiency': 0.0,
                'total_reward': 0.0
            }
        
        # 1. 生存时间
        survival_time = len(observations)
        
        # 2. 前进速度分析
        forward_velocities = []
        body_heights = []
        body_orientations = []
        
        for obs in observations:
            if isinstance(obs, (list, np.ndarray)) and len(obs) >= 17:  # Cheetah通常有17维观测
                # 前进速度通常在观测的前几维
                if len(obs) > 8:
                    forward_velocities.append(obs[8])  # 前进速度
                if len(obs) > 0:
                    body_heights.append(obs[0])  # 身体高度
                if len(obs) > 2:
                    body_orientations.append(obs[2])  # 身体姿态
        
        # 平均前进速度
        if forward_velocities:
            avg_forward_speed = np.mean(forward_velocities)
            speed_consistency = 1.0 - (np.std(forward_velocities) / (abs(avg_forward_speed) + 1e-6))
            speed_consistency = max(0.0, min(1.0, speed_consistency))
        else:
            avg_forward_speed = 0.0
            speed_consistency = 0.0
        
        # 3. 动作平滑度
        if len(actions) > 1:
            action_changes = []
            for i in range(len(actions) - 1):
                if isinstance(actions[i], (list, np.ndarray)) and isinstance(actions[i+1], (list, np.ndarray)):
                    change = np.linalg.norm(np.array(actions[i+1]) - np.array(actions[i]))
                    action_changes.append(change)
            
            if action_changes:
                action_smoothness = 1.0 - (np.mean(action_changes) / 2.0)  # 归一化
                action_smoothness = max(0.0, min(1.0, action_smoothness))
            else:
                action_smoothness = 1.0
        else:
            action_smoothness = 1.0
        
        # 4. 身体稳定性
        if body_heights:
            height_stability = 1.0 - (np.std(body_heights) / (np.mean(np.abs(body_heights)) + 1e-6))
            height_stability = max(0.0, min(1.0, height_stability))
        else:
            height_stability = 0.0
        
        if body_orientations:
            orientation_stability = 1.0 - (np.std(body_orientations) / (np.pi/4))  # 归一化到45度
            orientation_stability = max(0.0, min(1.0, orientation_stability))
        else:
            orientation_stability = 0.0
        
        body_stability = (height_stability + orientation_stability) / 2.0
        
        # 5. 能量效率 - 基于动作幅度
        if actions:
            action_magnitudes = []
            for action in actions:
                if isinstance(action, (list, np.ndarray)):
                    magnitude = np.linalg.norm(action)
                    action_magnitudes.append(magnitude)
            
            if action_magnitudes:
                avg_action_magnitude = np.mean(action_magnitudes)
                # 效率 = 速度 / 动作幅度
                if avg_action_magnitude > 0:
                    energy_efficiency = abs(avg_forward_speed) / (avg_action_magnitude + 1e-6)
                    energy_efficiency = min(1.0, energy_efficiency / 5.0)  # 归一化
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
            'forward_speed': max(0.0, avg_forward_speed),  # 只考虑正向速度
            'action_smoothness': action_smoothness,
            'body_stability': body_stability,
            'energy_efficiency': energy_efficiency,
            'total_reward': total_reward
        }
    
    # 评估两个轨迹
    metrics1 = evaluate_cheetah_trajectory(traj1)
    metrics2 = evaluate_cheetah_trajectory(traj2)
    
    # 权重配置 - 基于Cheetah跑步任务特点
    weights = {
        'forward_speed': 0.35,        # 前进速度最重要
        'survival_time': 0.25,        # 生存时间
        'body_stability': 0.20,       # 身体稳定性
        'energy_efficiency': 0.10,    # 能量效率
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
        'comparison_method': 'cheetah_run_heuristic'
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
    preference, confidence, details = compare_cheetah_run_trajectories(chosen_traj, rejected_traj)
    
    # 构建DPO格式的结果
    dpo_result = {
        'preference': preference,
        'confidence': confidence,
        'chosen_better': preference == 1,
        'rejected_better': preference == 2,
        'similar_quality': preference == 0,
        'evaluation_details': details,
        'task_type': 'cheetah_run',
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
    return compare_cheetah_run_trajectories(traj1, traj2)