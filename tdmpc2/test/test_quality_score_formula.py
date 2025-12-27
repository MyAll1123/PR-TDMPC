#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的轨迹质量评估器质量分数计算公式
新公式：质量分数 = 生存得分 × 环境奖励得分 × 状态稳定性得分 × 动作平滑性得分
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')
sys.path.append('/public/home/yaotianxiao2024/SPE/prm')

# 避免NumPy版本兼容性问题
try:
    import numpy as np
except ImportError:
    print("NumPy not available, using fallback")
    np = None

import torch
from preference_labeling_engine import PreferenceLabelingEngine

def create_test_trajectory(length=100, reward_sum=50.0):
    """创建测试轨迹数据"""
    import random
    
    # 直接使用Python列表避免NumPy兼容性问题
    obs_seq = []
    act_seq = []
    
    for i in range(length):
        # 创建观测向量 (64维)
        obs = [random.gauss(0, 1) for _ in range(64)]
        obs_seq.append(obs)
        
        # 创建动作向量 (12维)
        act = [random.gauss(0, 0.5) for _ in range(12)]
        act_seq.append(act)
    
    # 创建奖励序列
    rewards = [reward_sum / length + random.gauss(0, 0.1) for _ in range(length)]
    
    return obs_seq, act_seq, rewards

def test_quality_score_formula():
    """测试新的质量分数计算公式"""
    print("=== 测试轨迹质量评估器的新质量分数计算公式 ===")
    
    # 初始化偏好标签引擎
    config = {
        'task_name': 'h1hand-walk-v0',
        'feature_weights': {
            'survival_time': 0.3,
            'action_smoothness': 0.25,
            'state_stability': 0.25,
            'activity_score': 0.1,
            'task_progress': 0.1
        },
        'thresholds': {
            'min_survival_time': 0.1,
            'min_action_smoothness': 0.1,
            'min_state_stability': 0.1
        }
    }
    
    engine = PreferenceLabelingEngine(config)
    
    # 测试案例1：高质量轨迹（高奖励）
    print("\n--- 测试案例1：高质量轨迹（高奖励） ---")
    obs_seq1, act_seq1, rewards1 = create_test_trajectory(length=100, reward_sum=80.0)
    quality_score1, scores1 = engine.quality_evaluator.evaluate_trajectory_quality(obs_seq1, act_seq1, rewards1)
    
    print(f"轨迹长度: {len(obs_seq1)}")
    reward_sum1 = sum(rewards1) if isinstance(rewards1, list) else (np.sum(rewards1) if np is not None else sum(rewards1))
    print(f"环境奖励总和: {reward_sum1:.2f}")
    print(f"生存得分: {scores1.get('survival_time', 0):.4f}")
    print(f"环境奖励得分: {scores1.get('env_reward_score', 0):.4f}")
    print(f"状态稳定性得分: {scores1.get('state_stability', 0):.4f}")
    print(f"动作平滑性得分: {scores1.get('action_smoothness', 0):.4f}")
    print(f"最终质量分数: {quality_score1:.4f}")
    
    # 验证乘积公式（与_compute_weighted_score逻辑一致）
    survival_score = max(min(scores1.get('survival_time', 0.5), 1.0), 0.1)
    env_reward_score = max(min(scores1.get('env_reward_score', 0.5), 1.0), 0.1)
    stability_score = max(min(scores1.get('state_stability', 0.5), 1.0), 0.1)
    smoothness_score = max(min(scores1.get('action_smoothness', 0.5), 1.0), 0.1)
    expected_score1 = survival_score * env_reward_score * stability_score * smoothness_score
    expected_score1 = max(min(expected_score1, 1.0), 0.01)  # 与实际逻辑一致
    print(f"预期质量分数（乘积公式）: {expected_score1:.4f}")
    print(f"公式验证: {'✓' if abs(quality_score1 - expected_score1) < 0.001 else '✗'}")
    
    # 测试案例2：低质量轨迹（低奖励）
    print("\n--- 测试案例2：低质量轨迹（低奖励） ---")
    obs_seq2, act_seq2, rewards2 = create_test_trajectory(length=50, reward_sum=-20.0)
    quality_score2, scores2 = engine.quality_evaluator.evaluate_trajectory_quality(obs_seq2, act_seq2, rewards2)
    
    print(f"轨迹长度: {len(obs_seq2)}")
    reward_sum2 = sum(rewards2) if isinstance(rewards2, list) else (np.sum(rewards2) if np is not None else sum(rewards2))
    print(f"环境奖励总和: {reward_sum2:.2f}")
    print(f"生存得分: {scores2.get('survival_time', 0):.4f}")
    print(f"环境奖励得分: {scores2.get('env_reward_score', 0):.4f}")
    print(f"状态稳定性得分: {scores2.get('state_stability', 0):.4f}")
    print(f"动作平滑性得分: {scores2.get('action_smoothness', 0):.4f}")
    print(f"最终质量分数: {quality_score2:.4f}")
    
    # 验证乘积公式
    # 验证乘积公式（与_compute_weighted_score逻辑一致）
    survival_score2 = max(min(scores2.get('survival_time', 0.5), 1.0), 0.1)
    env_reward_score2 = max(min(scores2.get('env_reward_score', 0.5), 1.0), 0.1)
    stability_score2 = max(min(scores2.get('state_stability', 0.5), 1.0), 0.1)
    smoothness_score2 = max(min(scores2.get('action_smoothness', 0.5), 1.0), 0.1)
    expected_score2 = survival_score2 * env_reward_score2 * stability_score2 * smoothness_score2
    expected_score2 = max(min(expected_score2, 1.0), 0.01)  # 与实际逻辑一致
    print(f"预期质量分数（乘积公式）: {expected_score2:.4f}")
    print(f"公式验证: {'✓' if abs(quality_score2 - expected_score2) < 0.001 else '✗'}")
    
    # 测试案例3：无奖励轨迹
    print("\n--- 测试案例3：无奖励轨迹 ---")
    obs_seq3, act_seq3, _ = create_test_trajectory(length=75, reward_sum=0.0)
    quality_score3, scores3 = engine.quality_evaluator.evaluate_trajectory_quality(obs_seq3, act_seq3, None)
    
    print(f"轨迹长度: {len(obs_seq3)}")
    print(f"环境奖励: None")
    print(f"生存得分: {scores3.get('survival_time', 0):.4f}")
    print(f"环境奖励得分: {scores3.get('env_reward_score', 0):.4f}")
    print(f"状态稳定性得分: {scores3.get('state_stability', 0):.4f}")
    print(f"动作平滑性得分: {scores3.get('action_smoothness', 0):.4f}")
    print(f"最终质量分数: {quality_score3:.4f}")
    
    # 验证乘积公式
    # 验证乘积公式（与_compute_weighted_score逻辑一致）
    survival_score3 = max(min(scores3.get('survival_time', 0.5), 1.0), 0.1)
    env_reward_score3 = max(min(scores3.get('env_reward_score', 0.5), 1.0), 0.1)
    stability_score3 = max(min(scores3.get('state_stability', 0.5), 1.0), 0.1)
    smoothness_score3 = max(min(scores3.get('action_smoothness', 0.5), 1.0), 0.1)
    expected_score3 = survival_score3 * env_reward_score3 * stability_score3 * smoothness_score3
    expected_score3 = max(min(expected_score3, 1.0), 0.01)  # 与实际逻辑一致
    print(f"预期质量分数（乘积公式）: {expected_score3:.4f}")
    print(f"公式验证: {'✓' if abs(quality_score3 - expected_score3) < 0.001 else '✗'}")
    
    # 比较不同轨迹的质量分数
    print("\n--- 轨迹质量比较 ---")
    print(f"高质量轨迹分数: {quality_score1:.4f}")
    print(f"低质量轨迹分数: {quality_score2:.4f}")
    print(f"无奖励轨迹分数: {quality_score3:.4f}")
    
    # 验证高质量轨迹分数应该更高
    if quality_score1 > quality_score2:
        print("✓ 高质量轨迹分数 > 低质量轨迹分数")
    else:
        print("✗ 高质量轨迹分数应该更高")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_quality_score_formula()