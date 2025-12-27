#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的质量分数计算
验证环境奖励不再归一化，保持实际权重
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')

import numpy as np
import torch
from prm.preference_labeling_engine import PreferenceLabelingEngine

def create_test_trajectory(rewards, obs_length=None, action_length=None, 
                          obs_stability=0.1, action_smoothness=0.1):
    """
    创建测试轨迹数据
    
    Args:
        rewards: 奖励序列
        obs_length: 观测序列长度（默认与奖励长度相同）
        action_length: 动作序列长度（默认与奖励长度相同）
        obs_stability: 观测稳定性（数值越小越稳定）
        action_smoothness: 动作平滑性（数值越小越平滑）
    """
    if obs_length is None:
        obs_length = len(rewards)
    if action_length is None:
        action_length = len(rewards)
    
    # 创建观测序列（添加噪声模拟稳定性）
    base_obs = np.linspace(0, 1, obs_length)
    obs_noise = np.random.normal(0, obs_stability, (obs_length, 10))
    observations = base_obs.reshape(-1, 1) + obs_noise
    
    # 创建动作序列（添加噪声模拟平滑性）
    base_actions = np.sin(np.linspace(0, 2*np.pi, action_length))
    action_noise = np.random.normal(0, action_smoothness, (action_length, 5))
    actions = base_actions.reshape(-1, 1) + action_noise
    
    return {
        'observations': torch.tensor(observations, dtype=torch.float32),
        'actions': torch.tensor(actions, dtype=torch.float32),
        'rewards': rewards
    }

def test_quality_score_calculation():
    """
    测试质量分数计算
    """
    print("🧪 测试修改后的质量分数计算")
    print("=" * 50)
    
    # 创建偏好标注引擎
    engine = PreferenceLabelingEngine()
    
    # 测试案例1：高环境奖励，高执行质量
    print("\n📊 测试案例1：高环境奖励 + 高执行质量")
    high_rewards = [2.5, 2.8, 3.1, 2.9, 3.2, 2.7, 3.0, 2.6, 2.9, 3.1]
    traj1 = create_test_trajectory(
        rewards=high_rewards,
        obs_stability=0.05,  # 高稳定性
        action_smoothness=0.03  # 高平滑性
    )
    
    quality1, scores1 = engine.quality_evaluator.evaluate_trajectory_quality(
        traj1['observations'], traj1['actions'], traj1['rewards']
    )
    
    env_reward_sum1 = sum(high_rewards)
    print(f"  环境奖励总和: {env_reward_sum1:.2f}")
    print(f"  生存得分: {scores1['survival_time']:.3f}")
    print(f"  状态稳定性得分: {scores1['state_stability']:.3f}")
    print(f"  动作平滑性得分: {scores1['action_smoothness']:.3f}")
    print(f"  环境奖励得分: {scores1['env_reward_score']:.3f}")
    print(f"  最终质量分数: {quality1:.3f}")
    
    # 测试案例2：低环境奖励，低执行质量
    print("\n📊 测试案例2：低环境奖励 + 低执行质量")
    low_rewards = [0.5, 0.2, -0.1, 0.3, -0.2, 0.1]
    traj2 = create_test_trajectory(
        rewards=low_rewards,
        obs_stability=0.3,   # 低稳定性
        action_smoothness=0.4  # 低平滑性
    )
    
    quality2, scores2 = engine.quality_evaluator.evaluate_trajectory_quality(
        traj2['observations'], traj2['actions'], traj2['rewards']
    )
    
    env_reward_sum2 = sum(low_rewards)
    print(f"  环境奖励总和: {env_reward_sum2:.2f}")
    print(f"  生存得分: {scores2['survival_time']:.3f}")
    print(f"  状态稳定性得分: {scores2['state_stability']:.3f}")
    print(f"  动作平滑性得分: {scores2['action_smoothness']:.3f}")
    print(f"  环境奖励得分: {scores2['env_reward_score']:.3f}")
    print(f"  最终质量分数: {quality2:.3f}")
    
    # 测试案例3：相同环境奖励，不同执行质量
    print("\n📊 测试案例3：相同环境奖励 + 不同执行质量")
    same_rewards = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    
    # 高执行质量版本
    traj3a = create_test_trajectory(
        rewards=same_rewards,
        obs_stability=0.02,   # 很高稳定性
        action_smoothness=0.01  # 很高平滑性
    )
    
    quality3a, scores3a = engine.quality_evaluator.evaluate_trajectory_quality(
        traj3a['observations'], traj3a['actions'], traj3a['rewards']
    )
    
    # 低执行质量版本
    traj3b = create_test_trajectory(
        rewards=same_rewards,
        obs_stability=0.5,    # 低稳定性
        action_smoothness=0.6   # 低平滑性
    )
    
    quality3b, scores3b = engine.quality_evaluator.evaluate_trajectory_quality(
        traj3b['observations'], traj3b['actions'], traj3b['rewards']
    )
    
    env_reward_sum3 = sum(same_rewards)
    print(f"  环境奖励总和（相同）: {env_reward_sum3:.2f}")
    print(f"  高执行质量 - 最终分数: {quality3a:.3f}")
    print(f"    - 稳定性: {scores3a['state_stability']:.3f}, 平滑性: {scores3a['action_smoothness']:.3f}")
    print(f"  低执行质量 - 最终分数: {quality3b:.3f}")
    print(f"    - 稳定性: {scores3b['state_stability']:.3f}, 平滑性: {scores3b['action_smoothness']:.3f}")
    
    # 测试案例4：负奖励轨迹
    print("\n📊 测试案例4：负环境奖励轨迹")
    negative_rewards1 = [-1.0, -0.8, -1.2, -0.9, -1.1]
    negative_rewards2 = [-2.0, -2.5, -2.2, -2.8, -2.1]
    
    traj4a = create_test_trajectory(
        rewards=negative_rewards1,
        obs_stability=0.1,
        action_smoothness=0.1
    )
    
    traj4b = create_test_trajectory(
        rewards=negative_rewards2,
        obs_stability=0.1,
        action_smoothness=0.1
    )
    
    quality4a, scores4a = engine.quality_evaluator.evaluate_trajectory_quality(
        traj4a['observations'], traj4a['actions'], traj4a['rewards']
    )
    
    quality4b, scores4b = engine.quality_evaluator.evaluate_trajectory_quality(
        traj4b['observations'], traj4b['actions'], traj4b['rewards']
    )
    
    print(f"  较好负奖励轨迹 (总和: {sum(negative_rewards1):.2f}) - 质量分数: {quality4a:.3f}")
    print(f"  较差负奖励轨迹 (总和: {sum(negative_rewards2):.2f}) - 质量分数: {quality4b:.3f}")
    
    # 验证结果
    print("\n✅ 验证结果:")
    print(f"  1. 高质量轨迹 > 低质量轨迹: {quality1 > quality2} ({quality1:.3f} > {quality2:.3f})")
    print(f"  2. 相同奖励下，高执行质量 > 低执行质量: {quality3a > quality3b} ({quality3a:.3f} > {quality3b:.3f})")
    print(f"  3. 负奖励中，较好轨迹 > 较差轨迹: {quality4a > quality4b} ({quality4a:.3f} > {quality4b:.3f})")
    print(f"  4. 环境奖励保持原始数值: {scores1['env_reward_score'] == env_reward_sum1}")
    
    return {
        'high_quality': quality1,
        'low_quality': quality2,
        'same_reward_high_exec': quality3a,
        'same_reward_low_exec': quality3b,
        'negative_better': quality4a,
        'negative_worse': quality4b
    }

def simulate_dpo_comparison(quality_a, quality_b, beta=5.0):
    """
    模拟DPO偏好比较
    
    Args:
        quality_a: 轨迹A的质量分数
        quality_b: 轨迹B的质量分数
        beta: 温度参数
    
    Returns:
        (logit, probability)
    """
    logit = beta * (quality_a - quality_b)
    probability = 1.0 / (1.0 + np.exp(-logit))
    return logit, probability

def test_dpo_preferences():
    """
    测试DPO偏好判断
    """
    print("\n🎯 测试DPO偏好判断")
    print("=" * 50)
    
    # 获取质量分数
    results = test_quality_score_calculation()
    
    # DPO比较测试
    comparisons = [
        ('高质量 vs 低质量', results['high_quality'], results['low_quality']),
        ('相同奖励：高执行 vs 低执行', results['same_reward_high_exec'], results['same_reward_low_exec']),
        ('负奖励：较好 vs 较差', results['negative_better'], results['negative_worse'])
    ]
    
    for name, quality_a, quality_b in comparisons:
        logit, prob = simulate_dpo_comparison(quality_a, quality_b)
        print(f"\n{name}:")
        print(f"  轨迹A质量: {quality_a:.3f}")
        print(f"  轨迹B质量: {quality_b:.3f}")
        print(f"  DPO logit: {logit:.3f}")
        print(f"  P(A > B): {prob:.6f}")
        
        if prob > 0.9:
            print(f"  ✅ 强偏好A (置信度: {prob:.1%})")
        elif prob > 0.7:
            print(f"  ✅ 偏好A (置信度: {prob:.1%})")
        elif prob < 0.1:
            print(f"  ✅ 强偏好B (置信度: {1-prob:.1%})")
        elif prob < 0.3:
            print(f"  ✅ 偏好B (置信度: {1-prob:.1%})")
        else:
            print(f"  ⚠️ 偏好不明确 (置信度: {max(prob, 1-prob):.1%})")

if __name__ == "__main__":
    print("🚀 开始测试修改后的质量分数计算")
    
    try:
        test_dpo_preferences()
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()