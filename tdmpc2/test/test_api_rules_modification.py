#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API规则贡献修改
验证当前的API规则贡献计算和新的公式实现
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')
sys.path.append('/public/home/yaotianxiao2024/SPE/prm')

import numpy as np
import torch
from preference_labeling_engine import PreferenceLabelingEngine, TrajectoryQualityEvaluator

def test_current_api_rules_contribution():
    """
    测试当前API规则贡献的计算
    """
    print("\n" + "="*80)
    print("🔍 测试当前API规则贡献计算")
    print("="*80)
    
    try:
        # 创建质量评估器
        evaluator = TrajectoryQualityEvaluator('walk')
        
        # 生成测试轨迹数据
        T = 100
        obs_dim = 45
        act_dim = 17
        
        obs_seq = np.random.randn(T, obs_dim) * 0.1
        act_seq = np.random.randn(T, act_dim) * 0.1
        rewards = np.random.randn(T) * 0.5 + 1.0  # 模拟环境奖励
        
        # 评估轨迹质量
        quality_score, detailed_scores = evaluator.evaluate_trajectory_quality(
            obs_seq, act_seq, rewards
        )
        
        print(f"质量分数: {quality_score:.4f}")
        print(f"详细分数: {detailed_scores}")
        
        # 测试API规则贡献
        if hasattr(evaluator, '_apply_api_rules'):
            api_contribution = evaluator._apply_api_rules(
                obs_seq, act_seq, detailed_scores, {}
            )
            print(f"\n当前API规则贡献: {api_contribution:.4f}")
            print(f"API规则贡献范围: [{api_contribution:.4f}, {api_contribution:.4f}]")
        else:
            print("\n❌ 未找到_apply_api_rules方法")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_new_formula_simulation():
    """
    模拟新公式的计算：最终分数 = 环境奖励*基础质量因子*(1+API规则贡献)
    """
    print("\n" + "="*80)
    print("🧮 模拟新公式计算")
    print("="*80)
    
    # 模拟数据
    env_reward = 15.5  # 环境奖励
    survival_score = 0.8
    stability_score = 0.7
    smoothness_score = 0.9
    
    # 计算基础质量因子
    base_quality_factor = survival_score * stability_score * smoothness_score
    print(f"环境奖励: {env_reward:.4f}")
    print(f"基础质量因子: {base_quality_factor:.4f} (生存:{survival_score} × 稳定性:{stability_score} × 平滑性:{smoothness_score})")
    
    # 测试不同的API规则贡献值
    api_contributions = [-0.3, -0.15, 0.0, 0.15, 0.3]
    
    print("\n新公式计算结果:")
    for api_contrib in api_contributions:
        final_score = env_reward * base_quality_factor * (1 + api_contrib)
        print(f"  API贡献 {api_contrib:+.2f}: 最终分数 = {env_reward:.2f} × {base_quality_factor:.3f} × {1+api_contrib:.3f} = {final_score:.4f}")
    
    print("\n当前公式计算结果 (环境奖励 × 基础质量因子):")
    current_score = env_reward * base_quality_factor
    print(f"  当前分数 = {env_reward:.2f} × {base_quality_factor:.3f} = {current_score:.4f}")

def analyze_api_rules_range():
    """
    分析当前API规则贡献的实际范围
    """
    print("\n" + "="*80)
    print("📊 分析API规则贡献范围")
    print("="*80)
    
    print("当前实现中的API规则贡献计算:")
    print("1. compute_*_reward_components: 标准化到[-1,1]，然后×0.15 → [-0.15, 0.15]")
    print("2. evaluate_dpo_preference: (偏好分数-0.5)×0.2×置信度 → 约[-0.1, 0.1]")
    print("3. _compute_trajectory_score: 标准化到[-1,1]，然后×0.1 → [-0.1, 0.1]")
    print("4. compare_*_trajectories: 比较结果×0.05 → 约[-0.05, 0.05]")
    print("5. evaluate_*函数: 标准化到[-1,1]，然后×0.05 → [-0.05, 0.05]")
    
    print("\n理论最大范围: 所有规则叠加可能达到约[-0.45, 0.45]")
    print("实际平均范围: 由于平均化处理，通常在[-0.2, 0.2]左右")
    
    print("\n需要修改的目标范围: (-0.3, 0.3)")
    print("建议调整策略:")
    print("1. 将各个规则的权重系数适当调整")
    print("2. 在最终返回前进行范围限制")

if __name__ == "__main__":
    print("🚀 API规则贡献修改测试")
    
    # 测试当前实现
    test_current_api_rules_contribution()
    
    # 模拟新公式
    test_new_formula_simulation()
    
    # 分析范围
    analyze_api_rules_range()
    
    print("\n" + "="*80)
    print("✅ 测试完成")
    print("="*80)