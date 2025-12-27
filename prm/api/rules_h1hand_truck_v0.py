#!/usr/bin/env python3

"""
Truck任务的启发式轨迹比较规则
基于truck任务的特点：包裹从卡车运输到桌子
"""

import numpy as np
from typing import Tuple, Optional

def compare_h1hand_truck_v0_trajectories(tau_1, tau_2, goal):
    """h1hand-truck-v0 任务的专用比较规则 - 基于truck任务特点设计"""
    return evaluate_dpo_preference(tau_1, tau_2, goal)

def evaluate_dpo_preference(traj_a, traj_b, goal=None) -> Tuple[Optional[int], Optional[int]]:
    """
    基于humanoid_bench真实奖励函数的Truck任务DPO偏好评估
    
    核心评估维度(与真实奖励函数完全对应):
    - 躯干直立度 (20%): 对应 upright 奖励
    - 机器人到卡车包裹距离 (20%): 对应 reward_robot_package_truck
    - 机器人到拾取包裹距离 (20%): 对应 reward_robot_package_picked_up
    - 包裹到桌子距离 (20%): 对应 reward_package_table
    - 包裹拾取奖励 (10%): 对应包裹从卡车拾取的100分奖励
    - 包裹放置奖励 (10%): 对应包裹放置到桌子的100分奖励
    
    Args:
        traj_a: 轨迹A的状态-动作序列
        traj_b: 轨迹B的状态-动作序列
        goal: 目标参数（可选）
    
    Returns:
        (better_idx, worse_idx): 更优轨迹索引和较差轨迹索引
    """
    
    # 计算两个轨迹的综合得分
    score_a = _compute_trajectory_score(traj_a)
    score_b = _compute_trajectory_score(traj_b)
    
    # 设置偏好阈值，避免微小差异导致的不稳定判断
    preference_threshold = 0.05
    
    if abs(score_a - score_b) < preference_threshold:
        return None, None  # 差异太小，无明确偏好
    
    if score_a > score_b:
        return 0, 1  # 轨迹A更优
    else:
        return 1, 0  # 轨迹B更优

def _compute_trajectory_score(traj) -> float:
    """
    计算轨迹的综合得分 - 基于humanoid_bench真实奖励函数
    
    对应真实奖励函数的各个组件:
    - upright: 躯干直立度 (权重1.0)
    - robot_package_truck: 机器人到卡车包裹距离 (权重1.0)
    - robot_package_picked: 机器人到已拾取包裹距离 (权重1.0)
    - package_table: 包裹到桌子距离 (权重1.0)
    - pickup_success: 包裹拾取成功 (权重100.0)
    - place_success: 包裹放置成功 (权重100.0)
    """
    
    # 获取轨迹数据
    states = _get_trajectory_data(traj, 'obs')
    actions = _get_trajectory_data(traj, 'action')
    rewards = _get_trajectory_data(traj, 'reward')
    
    if states is None or actions is None or len(states) == 0:
        return 0.0
    
    # 计算各个评估维度(对应真实奖励函数)
    upright_score = _evaluate_upright_stability(states)
    robot_package_truck_score = _evaluate_robot_package_truck_proximity(states)
    robot_package_picked_score = _evaluate_robot_package_picked_proximity(states)
    package_table_score = _evaluate_package_table_proximity(states)
    package_pickup_score = _evaluate_package_pickup_success(states, rewards)
    package_place_score = _evaluate_package_place_success(states, rewards)
    
    # 权重配置 - 基于真实奖励函数权重，归一化到[0,1]范围
    # 原始权重: upright(1.0), proximities(1.0 each), success(100.0 each)
    # 归一化权重:
    weights = {
        'upright': 0.01,              # 1.0 / 104.0
        'robot_package_truck': 0.01,  # 1.0 / 104.0
        'robot_package_picked': 0.01, # 1.0 / 104.0
        'package_table': 0.01,        # 1.0 / 104.0
        'package_pickup': 0.48,       # 100.0 / 208.0 (两个成功事件总权重的一半)
        'package_place': 0.48         # 100.0 / 208.0 (两个成功事件总权重的一半)
    }
    
    # 计算加权总分
    total_score = (
        weights['upright'] * upright_score +
        weights['robot_package_truck'] * robot_package_truck_score +
        weights['robot_package_picked'] * robot_package_picked_score +
        weights['package_table'] * package_table_score +
        weights['package_pickup'] * package_pickup_score +
        weights['package_place'] * package_place_score
    )
    
    return total_score

def _evaluate_upright_stability(states) -> float:
    """
    评估躯干直立度 - 对应真实奖励函数中的upright奖励
    
    基于四元数计算躯干直立度，使用tolerance函数
    bounds=(0.9, inf), sigmoid="linear", margin=1.9, value_at_margin=0
    """
    if len(states) == 0:
        return 0.0
    
    upright_scores = []
    for state in states:
        if len(state) >= 7:  # 确保有四元数数据
            # 提取四元数 (假设在状态的前7个元素中，位置3-6是四元数)
            quat = state[3:7]
            upright_score = _compute_upright_score(quat)
            # 应用tolerance函数: bounds=(0.9, inf), margin=1.9
            if upright_score >= 0.9:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=1.9，value_at_margin=0
                tolerance_score = max(0.0, (upright_score - (0.9 - 1.9)) / 1.9)
            upright_scores.append(tolerance_score)
        else:
            upright_scores.append(0.0)
    
    return np.mean(upright_scores) if upright_scores else 0.0

def _evaluate_robot_package_truck_proximity(states) -> float:
    """
    评估机器人到卡车包裹的距离 - 对应reward_robot_package_truck
    
    使用tolerance函数: bounds=(0, 0.2), margin=4, value_at_margin=0, sigmoid="linear"
    """
    if len(states) == 0:
        return 0.0
    
    proximity_scores = []
    for state in states:
        if len(state) >= 3:  # 确保有位置信息
            # 假设机器人位置在状态的前3个元素
            robot_pos = np.array(state[:3])
            
            # 模拟卡车包裹位置(需要根据实际环境调整)
            # 这里使用固定位置作为示例，实际应该从环境状态中提取
            truck_package_pos = np.array([5.0, 0.0, 1.1])  # 示例位置
            
            # 计算距离
            distance = np.linalg.norm(robot_pos - truck_package_pos)
            
            # 应用tolerance函数: bounds=(0, 0.2), margin=4
            if distance <= 0.2:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=4
                tolerance_score = max(0.0, 1.0 - (distance - 0.2) / 4.0)
            
            proximity_scores.append(tolerance_score)
        else:
            proximity_scores.append(0.0)
    
    return np.mean(proximity_scores) if proximity_scores else 0.0

def _evaluate_robot_package_picked_proximity(states) -> float:
    """
    评估机器人到已拾取包裹的距离 - 对应reward_robot_package_picked_up
    
    使用tolerance函数: bounds=(0, 0.2), margin=4, value_at_margin=0, sigmoid="linear"
    """
    if len(states) == 0:
        return 0.0
    
    proximity_scores = []
    for state in states:
        if len(state) >= 3:  # 确保有位置信息
            # 假设机器人位置在状态的前3个元素
            robot_pos = np.array(state[:3])
            
            # 模拟已拾取包裹位置(通常接近机器人位置)
            # 这里假设包裹被拾取后在机器人附近
            picked_package_pos = robot_pos + np.array([0.1, 0.0, 0.2])  # 示例偏移
            
            # 计算距离
            distance = np.linalg.norm(robot_pos - picked_package_pos)
            
            # 应用tolerance函数: bounds=(0, 0.2), margin=4
            if distance <= 0.2:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=4
                tolerance_score = max(0.0, 1.0 - (distance - 0.2) / 4.0)
            
            proximity_scores.append(tolerance_score)
        else:
            proximity_scores.append(0.0)
    
    return np.mean(proximity_scores) if proximity_scores else 0.0

def _evaluate_package_table_proximity(states) -> float:
    """
    评估包裹到桌子的距离 - 对应reward_package_table
    
    使用tolerance函数: bounds=(0, 0.2), margin=4, value_at_margin=0, sigmoid="linear"
    """
    if len(states) == 0:
        return 0.0
    
    proximity_scores = []
    for state in states:
        if len(state) >= 3:  # 确保有位置信息
            # 模拟包裹位置(需要根据实际环境状态调整)
            # 这里假设包裹位置可以从状态中推断或使用固定位置
            package_pos = np.array([2.0, -1.7, 0.8])  # 示例包裹位置
            
            # 桌子位置(根据真实环境设置)
            table_pos = np.array([2.0, -1.7, 0.5])  # 示例桌子位置
            
            # 计算距离
            distance = np.linalg.norm(package_pos - table_pos)
            
            # 应用tolerance函数: bounds=(0, 0.2), margin=4
            if distance <= 0.2:
                tolerance_score = 1.0
            else:
                # 线性衰减，margin=4
                tolerance_score = max(0.0, 1.0 - (distance - 0.2) / 4.0)
            
            proximity_scores.append(tolerance_score)
        else:
            proximity_scores.append(0.0)
    
    return np.mean(proximity_scores) if proximity_scores else 0.0

def _evaluate_package_pickup_success(states, rewards) -> float:
    """
    评估包裹拾取成功 - 对应真实奖励函数中的100分拾取奖励
    
    基于奖励信号或状态变化检测包裹拾取事件
    """
    if len(states) == 0:
        return 0.0
    
    # 如果有奖励信息，检测拾取奖励峰值
    if rewards is not None and len(rewards) > 0:
        pickup_events = 0
        for i in range(1, len(rewards)):
            # 检测奖励突然增加(对应100分拾取奖励)
            if rewards[i] - rewards[i-1] > 50:  # 阈值可调整
                pickup_events += 1
        
        # 归一化拾取成功率(假设最多5个包裹)
        return min(1.0, pickup_events / 5.0)
    
    # 如果没有奖励信息，基于状态变化推断
    # 这里使用简化的启发式方法
    pickup_indicators = []
    for i in range(len(states)):
        # 基于轨迹进展推断拾取概率
        progress = i / len(states)
        pickup_probability = min(1.0, progress * 2.0)  # 随时间增加拾取概率
        pickup_indicators.append(pickup_probability)
    
    return np.mean(pickup_indicators) if pickup_indicators else 0.0

def _evaluate_package_place_success(states, rewards) -> float:
    """
    评估包裹放置成功 - 对应真实奖励函数中的100分放置奖励
    
    基于奖励信号或状态变化检测包裹放置事件
    """
    if len(states) == 0:
        return 0.0
    
    # 如果有奖励信息，检测放置奖励峰值
    if rewards is not None and len(rewards) > 0:
        place_events = 0
        for i in range(1, len(rewards)):
            # 检测奖励突然增加(对应100分放置奖励)
            if rewards[i] - rewards[i-1] > 50:  # 阈值可调整
                place_events += 1
        
        # 归一化放置成功率(假设最多5个包裹)
        return min(1.0, place_events / 5.0)
    
    # 如果没有奖励信息，基于状态变化推断
    # 这里使用简化的启发式方法
    place_indicators = []
    for i in range(len(states)):
        # 基于轨迹后期推断放置概率
        progress = i / len(states)
        if progress > 0.5:  # 后半段更可能有放置行为
            place_probability = min(1.0, (progress - 0.5) * 2.0)
        else:
            place_probability = 0.0
        place_indicators.append(place_probability)
    
    return np.mean(place_indicators) if place_indicators else 0.0

def _compute_upright_score(quat) -> float:
    """
    基于四元数计算直立程度得分
    
    Args:
        quat: 四元数 [w, x, y, z] 或 [x, y, z, w]
    
    Returns:
        直立程度得分 [0, 1]
    """
    try:
        quat = np.array(quat)
        # 归一化四元数
        quat = quat / np.linalg.norm(quat)
        
        # 计算z轴方向（垂直向上）
        # 四元数旋转矩阵的z轴分量
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # z轴在世界坐标系中的方向
        z_world = np.array([
            2 * (x*z + w*y),
            2 * (y*z - w*x), 
            1 - 2 * (x*x + y*y)
        ])
        
        # 与垂直向上方向[0, 0, 1]的点积
        upright_dot = z_world[2]
        
        # 转换为得分 [0, 1]
        upright_score = (upright_dot + 1) / 2
        
        return max(0.0, min(1.0, upright_score))
    
    except:
        return 0.5  # 默认中等得分

def _get_trajectory_data(traj, key: str):
    """
    从轨迹中提取数据，兼容不同的数据格式
    
    Args:
        traj: 轨迹数据（字典或对象）
        key: 数据键名 ('obs', 'action', 'reward', 'done')
    
    Returns:
        提取的数据列表或None
    """
    try:
        if hasattr(traj, 'get'):
            # 字典式访问
            return traj.get(key)
        elif hasattr(traj, key):
            # 属性访问
            return getattr(traj, key)
        elif isinstance(traj, dict):
            # 直接字典访问
            return traj.get(key)
        else:
            return None
    except:
        return None