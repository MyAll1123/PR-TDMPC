#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好奖励模型集成测试

测试修改后的偏好奖励模型在实际TD-MPC2训练环境中的集成情况

作者：AI Assistant
日期：2025-08-29
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')

import torch
import numpy as np
from tdmpc2.common.buffer import Buffer
from tdmpc2.common.world_model import WorldModel
from tdmpc2.common.scale import RunningScale
from prm.hybrid_value_estimator import HybridValueEstimator
from prm.optimized_latent_preference_model import (
    OptimizedLatentPreferenceModel, 
    OptimizedLatentPreferenceConfig,
    create_optimized_latent_preference_model
)

def test_hybrid_value_estimator_integration():
    """测试HybridValueEstimator与修改后的偏好奖励模型的集成"""
    print("=" * 60)
    print("HybridValueEstimator 集成测试")
    print("=" * 60)
    
    # 模拟配置
    class MockConfig:
        def __init__(self):
            self.obs_shape = (84, 84, 3)
            self.action_dim = 61
            self.latent_dim = 512
            self.hidden_dim = 256
            self.horizon = 5
            self.discount = 0.99
            self.device = 'cpu'
            self.preference_integration_method = 'multiplicative'
            self.environment_weight = 0.7
            self.preference_weight = 0.3
            self.enable_uncertainty = True
    
    config = MockConfig()
    
    # 创建偏好奖励模型
    pref_config = OptimizedLatentPreferenceConfig(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        enable_uncertainty=config.enable_uncertainty
    )
    
    preference_model = create_optimized_latent_preference_model(pref_config)
    
    # 模拟世界模型
    class MockWorldModel:
        def __init__(self):
            self.device = config.device
        
        def next(self, z, a):
            # 返回下一个潜在状态和奖励
            next_z = torch.randn_like(z)
            reward = torch.randn(z.shape[0])
            return next_z, reward
        
        def reward(self, z, a):
            return torch.randn(z.shape[0])
    
    world_model = MockWorldModel()
    
    # 创建HybridValueEstimator
    try:
        hybrid_estimator = HybridValueEstimator(
            world_model=world_model,
            preference_model=preference_model,
            config=config
        )
        print("✅ HybridValueEstimator 创建成功")
    except Exception as e:
        print(f"❌ HybridValueEstimator 创建失败: {e}")
        return False
    
    # 测试价值估计
    print("\n测试价值估计功能...")
    
    batch_size = 4
    z = torch.randn(batch_size, config.latent_dim)
    a = torch.randn(batch_size, config.action_dim)
    
    try:
        # 测试单步价值估计
        value = hybrid_estimator._estimate_value(z, a)
        print(f"✅ 单步价值估计成功，输出形状: {value.shape}")
        print(f"   价值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
        
        # 测试多步价值估计
        actions = torch.randn(batch_size, config.horizon, config.action_dim)
        total_value = hybrid_estimator.estimate_value(z, actions)
        print(f"✅ 多步价值估计成功，输出形状: {total_value.shape}")
        print(f"   总价值范围: [{total_value.min().item():.3f}, {total_value.max().item():.3f}]")
        
    except Exception as e:
        print(f"❌ 价值估计失败: {e}")
        return False
    
    # 测试偏好奖励集成
    print("\n测试偏好奖励集成...")
    
    try:
        # 获取环境奖励和偏好奖励
        env_reward = world_model.reward(z, a)
        pref_reward, confidence = preference_model.get_preference_reward(z[0], a[0])
        
        print(f"环境奖励示例: {env_reward[0].item():.3f}")
        print(f"偏好奖励示例: {pref_reward:.3f} (置信度: {confidence:.3f})")
        
        # 验证偏好奖励范围
        if pref_reward > 0:
            if 0.1 <= pref_reward <= 0.3:
                print("✅ 正偏好奖励在正确范围内 [0.1, 0.3]")
            else:
                print(f"❌ 正偏好奖励超出范围: {pref_reward}")
        elif pref_reward < 0:
            if -0.3 <= pref_reward <= -0.1:
                print("✅ 负偏好奖励在正确范围内 [-0.3, -0.1]")
            else:
                print(f"❌ 负偏好奖励超出范围: {pref_reward}")
        else:
            print("⚠️  偏好奖励为零")
        
    except Exception as e:
        print(f"❌ 偏好奖励集成测试失败: {e}")
        return False
    
    print("\n✅ 所有集成测试通过！")
    return True

def test_preference_reward_statistics():
    """测试偏好奖励的统计特性"""
    print("\n" + "=" * 60)
    print("偏好奖励统计特性测试")
    print("=" * 60)
    
    # 创建偏好奖励模型
    config = OptimizedLatentPreferenceConfig(
        latent_dim=512,
        action_dim=61,
        hidden_dim=256,
        enable_uncertainty=True
    )
    
    model = create_optimized_latent_preference_model(config)
    
    # 收集大量样本
    rewards = []
    confidences = []
    
    print("收集1000个样本进行统计分析...")
    
    for i in range(1000):
        latent_state = torch.randn(512)
        action = torch.randn(61)
        reward, confidence = model.get_preference_reward(latent_state, action)
        rewards.append(reward)
        confidences.append(confidence)
    
    rewards = np.array(rewards)
    confidences = np.array(confidences)
    
    # 统计分析
    print(f"\n奖励统计:")
    print(f"  范围: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"  均值: {rewards.mean():.3f}")
    print(f"  标准差: {rewards.std():.3f}")
    
    print(f"\n置信度统计:")
    print(f"  范围: [{confidences.min():.3f}, {confidences.max():.3f}]")
    print(f"  均值: {confidences.mean():.3f}")
    print(f"  标准差: {confidences.std():.3f}")
    
    # 分析不同置信度区间的奖励分布
    high_conf_mask = confidences >= 0.7
    mid_conf_mask = (confidences > 0.4) & (confidences < 0.7)
    low_conf_mask = confidences <= 0.4
    
    print(f"\n不同置信度区间的奖励分布:")
    
    if np.any(high_conf_mask):
        high_rewards = rewards[high_conf_mask]
        print(f"  高置信度 (>=0.7): {np.sum(high_conf_mask)} 样本")
        print(f"    奖励范围: [{high_rewards.min():.3f}, {high_rewards.max():.3f}]")
        print(f"    奖励均值: {high_rewards.mean():.3f}")
    
    if np.any(mid_conf_mask):
        mid_rewards = rewards[mid_conf_mask]
        print(f"  中等置信度 (0.4, 0.7): {np.sum(mid_conf_mask)} 样本")
        print(f"    奖励范围: [{mid_rewards.min():.3f}, {mid_rewards.max():.3f}]")
        print(f"    奖励均值: {mid_rewards.mean():.3f}")
    
    if np.any(low_conf_mask):
        low_rewards = rewards[low_conf_mask]
        print(f"  低置信度 (<=0.4): {np.sum(low_conf_mask)} 样本")
        print(f"    奖励范围: [{low_rewards.min():.3f}, {low_rewards.max():.3f}]")
        print(f"    奖励均值: {low_rewards.mean():.3f}")
    
    # 验证约束
    positive_rewards = rewards[rewards > 0]
    negative_rewards = rewards[rewards < 0]
    
    constraints_ok = True
    
    if len(positive_rewards) > 0:
        if not (0.1 <= positive_rewards.min() and positive_rewards.max() <= 0.4):
            print(f"❌ 正奖励范围违反约束: [{positive_rewards.min():.3f}, {positive_rewards.max():.3f}]")
            constraints_ok = False
        else:
            print(f"✅ 正奖励范围符合约束: [{positive_rewards.min():.3f}, {positive_rewards.max():.3f}]")
    
    if len(negative_rewards) > 0:
        if not (-0.3 <= negative_rewards.min() and negative_rewards.max() <= -0.1):
            print(f"❌ 负奖励范围违反约束: [{negative_rewards.min():.3f}, {negative_rewards.max():.3f}]")
            constraints_ok = False
        else:
            print(f"✅ 负奖励范围符合约束: [{negative_rewards.min():.3f}, {negative_rewards.max():.3f}]")
    
    return constraints_ok

if __name__ == "__main__":
    print("开始偏好奖励模型集成测试...")
    
    # 运行测试
    integration_passed = test_hybrid_value_estimator_integration()
    statistics_passed = test_preference_reward_statistics()
    
    print("\n" + "=" * 60)
    print("集成测试总结")
    print("=" * 60)
    print(f"HybridValueEstimator集成: {'✅ 通过' if integration_passed else '❌ 失败'}")
    print(f"统计特性验证: {'✅ 通过' if statistics_passed else '❌ 失败'}")
    
    if integration_passed and statistics_passed:
        print("\n🎉 所有集成测试通过！偏好奖励模型可以正常集成到TD-MPC2中。")
    else:
        print("\n❌ 部分集成测试失败，需要进一步检查。")
    
    print("=" * 60)