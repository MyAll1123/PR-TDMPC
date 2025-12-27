#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具函数模块
包含在多个API规则文件中重复使用的工具函数
"""

import numpy as np
from typing import Union, Optional

def tolerance(value, bounds=None, margin=1.0, value_at_margin=0.0, sigmoid="gaussian", target=None, tolerance_range=None):
    """
    兼容humanoid_bench rewards模块的tolerance函数
    
    Args:
        value: 输入值或数组
        bounds: 边界范围 (min, max) 或单个值
        margin: 容忍边界
        value_at_margin: 边界处的值
        sigmoid: 激活函数类型 ("gaussian", "linear", "quadratic")
        target: 目标值 (向后兼容)
        tolerance_range: 容忍范围 (向后兼容)
    
    Returns:
        容忍度得分
    """
    # 向后兼容旧的API
    if target is not None and tolerance_range is not None:
        if tolerance_range <= 0:
            return 1.0 if abs(value - target) < 1e-6 else 0.0
        distance = abs(value - target)
        if distance <= tolerance_range:
            return 1.0 - (distance / tolerance_range)
        else:
            return 0.0
    
    # 处理数组输入
    if hasattr(value, '__len__') and not isinstance(value, str):
        value = np.array(value)
        if bounds is None:
            # 对于数组，默认计算到零的距离
            distances = np.abs(value)
        else:
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lower, upper = bounds
                distances = np.maximum(lower - value, 0) + np.maximum(value - upper, 0)
            else:
                distances = np.abs(value - bounds)
    else:
        # 标量输入
        if bounds is None:
            distances = abs(value)
        else:
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lower, upper = bounds
                if lower <= value <= upper:
                    distances = 0
                else:
                    distances = min(abs(value - lower), abs(value - upper))
            else:
                distances = abs(value - bounds)
    
    # 应用sigmoid函数
    if sigmoid == "linear":
        result = np.maximum(0, 1 - distances / margin)
    elif sigmoid == "quadratic":
        normalized = distances / margin
        result = np.maximum(0, 1 - normalized * normalized)
    else:  # gaussian (default)
        result = np.exp(-0.5 * (distances / margin) ** 2)
    
    # 应用value_at_margin
    if value_at_margin != 0.0:
        result = result * (1 - value_at_margin) + value_at_margin
    
    return result

def exponential_decay(value: float, decay_rate: float = 0.1, 
                     offset: float = 0.0) -> float:
    """
    指数衰减函数
    
    Args:
        value: 输入值
        decay_rate: 衰减率
        offset: 偏移量
    
    Returns:
        衰减后的值
    """
    return np.exp(-decay_rate * (value - offset))

def sigmoid_activation(x: float, steepness: float = 1.0, 
                      midpoint: float = 0.0) -> float:
    """
    Sigmoid激活函数
    
    Args:
        x: 输入值
        steepness: 陡峭度参数
        midpoint: 中点位置
    
    Returns:
        Sigmoid输出 (0-1之间)
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))

def smooth_step(x: float, edge0: float = 0.0, edge1: float = 1.0) -> float:
    """
    平滑步进函数
    
    Args:
        x: 输入值
        edge0: 下边界
        edge1: 上边界
    
    Returns:
        平滑插值结果 (0-1之间)
    """
    if edge0 >= edge1:
        return 1.0 if x >= edge1 else 0.0
    
    # 将x限制在[edge0, edge1]范围内
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    
    # 使用3次Hermite插值
    return t * t * (3.0 - 2.0 * t)

def linear_interpolation(x: float, x0: float, y0: float, 
                        x1: float, y1: float) -> float:
    """
    线性插值函数
    
    Args:
        x: 插值点
        x0, y0: 第一个控制点
        x1, y1: 第二个控制点
    
    Returns:
        插值结果
    """
    if abs(x1 - x0) < 1e-6:
        return y0
    
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def gaussian_kernel(x: float, center: float = 0.0, 
                   sigma: float = 1.0) -> float:
    """
    高斯核函数
    
    Args:
        x: 输入值
        center: 中心点
        sigma: 标准差
    
    Returns:
        高斯核值
    """
    if sigma <= 0:
        return 1.0 if abs(x - center) < 1e-6 else 0.0
    
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def normalize_angle(angle: float) -> float:
    """
    将角度标准化到[-π, π]范围
    
    Args:
        angle: 输入角度（弧度）
    
    Returns:
        标准化后的角度
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 当分母为零时的默认值
    
    Returns:
        除法结果或默认值
    """
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    将值限制在指定范围内
    
    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        限制后的值
    """
    return max(min_val, min(value, max_val))

def weighted_average(values: list, weights: list) -> float:
    """
    计算加权平均值
    
    Args:
        values: 数值列表
        weights: 权重列表
    
    Returns:
        加权平均值
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight

def compute_distance_3d(pos1: Union[list, np.ndarray], 
                       pos2: Union[list, np.ndarray]) -> float:
    """
    计算3D空间中两点之间的欧几里得距离
    
    Args:
        pos1: 第一个点的坐标 [x, y, z]
        pos2: 第二个点的坐标 [x, y, z]
    
    Returns:
        两点之间的距离
    """
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    
    if pos1.shape != (3,) or pos2.shape != (3,):
        raise ValueError("位置坐标必须是3维向量")
    
    return np.linalg.norm(pos1 - pos2)

def compute_angle_between_vectors(v1: Union[list, np.ndarray], 
                                 v2: Union[list, np.ndarray]) -> float:
    """
    计算两个向量之间的夹角
    
    Args:
        v1: 第一个向量
        v2: 第二个向量
    
    Returns:
        夹角（弧度）
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # 计算向量的模长
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    # 计算余弦值
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    
    # 限制在[-1, 1]范围内，避免数值误差
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle)