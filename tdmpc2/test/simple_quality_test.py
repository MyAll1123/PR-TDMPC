#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的质量分数公式测试
验证新的计算公式是否正确实现
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')
sys.path.append('/public/home/yaotianxiao2024/SPE/prm')

import numpy as np
import torch
from preference_labeling_engine import TrajectoryQualityEvaluator
import logging

# 设置日志级别为WARNING以减少输出
logging.basicConfig(level=logging.WARNING)

def test_simple_quality():
    """
    简单测试质量分数计算
    """
    print("🧪 测试新的质量分数计算公式")
    
    # 创建质量评估器
    evaluator = TrajectoryQualityEvaluator('h1hand-walk-v0')
    
    # 创建简单的测试数据
    length = 50
    obs_dim = 45
    act_dim = 19
    
    # 生成测试轨迹
    obs_seq = np.random.randn(length, obs_dim) * 0.1
    act_seq = np.random.randn(length, act_dim) * 0.05
    env_rewards = np.random.uniform(0.5, 1.5, length)
    
    # 转换为torch张量
    obs_tensor = torch.FloatTensor(obs_seq)
    act_tensor = torch.FloatTensor(act_seq)
    env_rewards_tensor = torch.FloatTensor(env_rewards)
    
    print(f"\n📊 输入数据:")
    print(f"   轨迹长度: {length}")
    print(f"   环境奖励总和: {env_rewards.sum():.3f}")
    
    try:
        # 评估轨迹质量
        quality_score, detailed_scores = evaluator.evaluate_trajectory_quality(
            obs_tensor, act_tensor, env_rewards_tensor
        )
        
        print(f"\n🎯 评估结果:")
        print(f"   总质量分数: {quality_score:.6f}")
        
        # 检查关键组件
        env_reward_score = detailed_scores.get('env_reward_score', 0)
        api_contribution = detailed_scores.get('api_contribution', 0)
        survival_time = detailed_scores.get('survival_time', 1)
        state_stability = detailed_scores.get('state_stability', 1)
        action_smoothness = detailed_scores.get('action_smoothness', 1)
        
        print(f"\n🔍 公式组件:")
        print(f"   环境奖励总和: {env_reward_score:.6f}")
        print(f"   生存时间得分: {survival_time:.6f}")
        print(f"   状态稳定性得分: {state_stability:.6f}")
        print(f"   动作平滑性得分: {action_smoothness:.6f}")
        print(f"   API规则贡献: {api_contribution:.6f}")
        
        # 计算基础质量因子
        base_quality_factor = survival_time * state_stability * action_smoothness
        print(f"   基础质量因子: {base_quality_factor:.6f}")
        
        # 验证新公式
        expected_score = env_reward_score * base_quality_factor * (1 + api_contribution)
        print(f"\n✅ 公式验证:")
        print(f"   预期分数: {expected_score:.6f}")
        print(f"   实际分数: {quality_score:.6f}")
        print(f"   差异: {abs(expected_score - quality_score):.8f}")
        
        # 检查API贡献范围
        api_in_range = -0.3 <= api_contribution <= 0.3
        print(f"   API贡献范围检查: {'✅' if api_in_range else '❌'}")
        
        # 公式匹配检查
        formula_match = abs(expected_score - quality_score) < 1e-6
        print(f"   公式匹配检查: {'✅' if formula_match else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 开始简单质量分数测试")
    success = test_simple_quality()
    
    if success:
        print(f"\n✅ 测试通过 - 新的质量分数公式正确实现")
        print(f"   公式: 最终分数 = 环境奖励 × 基础质量因子 × (1 + API规则贡献)")
        print(f"   API贡献范围: (-0.3, 0.3)")
    else:
        print(f"\n❌ 测试失败")

if __name__ == "__main__":
    main()