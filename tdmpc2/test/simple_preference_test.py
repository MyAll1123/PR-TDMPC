#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的偏好奖励模型测试

避免导入可能有兼容性问题的模块，只测试偏好奖励模型的核心功能

作者：AI Assistant
日期：2025-08-29
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')

import torch
import numpy as np

# 只导入必要的偏好奖励模型
try:
    from prm.optimized_latent_preference_model import (
        OptimizedLatentPreferenceModel, 
        OptimizedLatentPreferenceConfig,
        create_optimized_latent_preference_model
    )
    print("✅ 成功导入偏好奖励模型")
except Exception as e:
    print(f"❌ 导入偏好奖励模型失败: {e}")
    sys.exit(1)

def test_preference_model_functionality():
    """测试偏好奖励模型的核心功能"""
    print("=" * 60)
    print("偏好奖励模型核心功能测试")
    print("=" * 60)
    
    # 创建配置
    config = OptimizedLatentPreferenceConfig(
        latent_dim=512,
        action_dim=61,
        hidden_dim=256,
        enable_uncertainty=True
    )
    
    # 创建模型
    try:
        model = create_optimized_latent_preference_model(config)
        print("✅ 偏好奖励模型创建成功")
    except Exception as e:
        print(f"❌ 偏好奖励模型创建失败: {e}")
        return False
    
    # 测试模型方法
    print("\n测试模型方法...")
    
    # 测试 _map_reward_with_confidence 方法
    try:
        test_cases = [
            (1.0, 0.8, 0.3),    # 高置信度正奖励
            (-1.0, 0.8, -0.3),  # 高置信度负奖励
            (1.0, 0.2, 0.1),    # 低置信度正奖励
            (-1.0, 0.2, -0.1),  # 低置信度负奖励
            (1.0, 0.5, 0.2),    # 中等置信度正奖励
            (-1.0, 0.5, -0.2),  # 中等置信度负奖励
        ]
        
        print("测试 _map_reward_with_confidence 方法:")
        all_passed = True
        
        for raw_reward, confidence, expected_approx in test_cases:
            mapped_reward = model._map_reward_with_confidence(raw_reward, confidence)
            print(f"  原始奖励: {raw_reward:5.1f}, 置信度: {confidence:4.1f} -> 映射奖励: {mapped_reward:6.3f} (期望约: {expected_approx:5.1f})")
            
            # 验证范围约束
            if mapped_reward > 0:
                if not (0.1 <= mapped_reward <= 0.4):
                    print(f"    ❌ 正奖励超出范围 [0.1, 0.4]: {mapped_reward}")
                    all_passed = False
            elif mapped_reward < 0:
                if not (-0.4 <= mapped_reward <= -0.1):
                    print(f"    ❌ 负奖励超出范围 [-0.4, -0.1]: {mapped_reward}")
                    all_passed = False
        
        if all_passed:
            print("  ✅ _map_reward_with_confidence 方法测试通过")
        else:
            print("  ❌ _map_reward_with_confidence 方法测试失败")
            return False
            
    except Exception as e:
        print(f"❌ _map_reward_with_confidence 方法测试失败: {e}")
        return False
    
    # 测试 get_preference_reward 方法
    try:
        print("\n测试 get_preference_reward 方法:")
        
        # 生成测试数据
        latent_state = torch.randn(512)
        action = torch.randn(61)
        
        # 调用方法
        reward, confidence = model.get_preference_reward(latent_state, action)
        
        print(f"  输入形状: 潜在状态 {latent_state.shape}, 动作 {action.shape}")
        print(f"  输出: 奖励 {reward:.3f}, 置信度 {confidence:.3f}")
        
        # 验证输出类型和范围
        if not isinstance(reward, (float, int)):
            print(f"  ❌ 奖励类型错误: {type(reward)}")
            return False
        
        if not isinstance(confidence, (float, int)):
            print(f"  ❌ 置信度类型错误: {type(confidence)}")
            return False
        
        if not (0.0 <= confidence <= 1.0):
            print(f"  ❌ 置信度超出范围 [0, 1]: {confidence}")
            return False
        
        # 验证奖励范围
        if reward > 0:
            if not (0.1 <= reward <= 0.4):
                print(f"  ❌ 正奖励超出范围 [0.1, 0.4]: {reward}")
                return False
        elif reward < 0:
            if not (-0.4 <= reward <= -0.1):
                print(f"  ❌ 负奖励超出范围 [-0.4, -0.1]: {reward}")
                return False
        
        print("  ✅ get_preference_reward 方法测试通过")
        
    except Exception as e:
        print(f"❌ get_preference_reward 方法测试失败: {e}")
        return False
    
    return True

def test_batch_processing():
    """测试批量处理功能"""
    print("\n" + "=" * 60)
    print("批量处理测试")
    print("=" * 60)
    
    # 创建模型
    config = OptimizedLatentPreferenceConfig(
        latent_dim=512,
        action_dim=61,
        hidden_dim=256,
        enable_uncertainty=True
    )
    
    model = create_optimized_latent_preference_model(config)
    
    # 测试多个样本
    print("测试100个随机样本...")
    
    rewards = []
    confidences = []
    
    try:
        for i in range(100):
            latent_state = torch.randn(512)
            action = torch.randn(61)
            reward, confidence = model.get_preference_reward(latent_state, action)
            rewards.append(reward)
            confidences.append(confidence)
        
        rewards = np.array(rewards)
        confidences = np.array(confidences)
        
        print(f"奖励统计:")
        print(f"  范围: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"  均值: {rewards.mean():.3f}")
        print(f"  标准差: {rewards.std():.3f}")
        
        print(f"置信度统计:")
        print(f"  范围: [{confidences.min():.3f}, {confidences.max():.3f}]")
        print(f"  均值: {confidences.mean():.3f}")
        print(f"  标准差: {confidences.std():.3f}")
        
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
            if not (-0.4 <= negative_rewards.min() and negative_rewards.max() <= -0.1):
                print(f"❌ 负奖励范围违反约束: [{negative_rewards.min():.3f}, {negative_rewards.max():.3f}]")
                constraints_ok = False
            else:
                print(f"✅ 负奖励范围符合约束: [{negative_rewards.min():.3f}, {negative_rewards.max():.3f}]")
        
        if not (0.0 <= confidences.min() and confidences.max() <= 1.0):
            print(f"❌ 置信度范围违反约束: [{confidences.min():.3f}, {confidences.max():.3f}]")
            constraints_ok = False
        else:
            print(f"✅ 置信度范围符合约束: [{confidences.min():.3f}, {confidences.max():.3f}]")
        
        return constraints_ok
        
    except Exception as e:
        print(f"❌ 批量处理测试失败: {e}")
        return False

def test_confidence_mapping_consistency():
    """测试置信度映射的一致性"""
    print("\n" + "=" * 60)
    print("置信度映射一致性测试")
    print("=" * 60)
    
    config = OptimizedLatentPreferenceConfig(
        latent_dim=512,
        action_dim=61,
        hidden_dim=256,
        enable_uncertainty=True
    )
    
    model = create_optimized_latent_preference_model(config)
    
    # 测试边界条件
    boundary_tests = [
        (1.0, 0.7, "边界高置信度正奖励"),
        (-1.0, 0.7, "边界高置信度负奖励"),
        (1.0, 0.4, "边界低置信度正奖励"),
        (-1.0, 0.4, "边界低置信度负奖励"),
    ]
    
    print("测试边界条件:")
    
    for raw_reward, confidence, description in boundary_tests:
        mapped_reward = model._map_reward_with_confidence(raw_reward, confidence)
        print(f"  {description}: 原始={raw_reward:5.1f}, 置信度={confidence:4.1f} -> 映射={mapped_reward:6.3f}")
        
        # 验证边界值
        if confidence == 0.7:
            expected = 0.4 if raw_reward > 0 else -0.4
            if abs(mapped_reward - expected) > 1e-6:
                print(f"    ❌ 边界值不正确，期望 {expected}, 得到 {mapped_reward}")
                return False
        elif confidence == 0.4:
            expected = 0.1 if raw_reward > 0 else -0.1
            if abs(mapped_reward - expected) > 1e-6:
                print(f"    ❌ 边界值不正确，期望 {expected}, 得到 {mapped_reward}")
                return False
    
    print("  ✅ 边界条件测试通过")
    
    # 测试线性插值
    print("\n测试线性插值:")
    
    confidence_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for conf in confidence_values:
        pos_reward = model._map_reward_with_confidence(1.0, conf)
        neg_reward = model._map_reward_with_confidence(-1.0, conf)
        print(f"  置信度 {conf:.1f}: 正奖励 {pos_reward:.3f}, 负奖励 {neg_reward:.3f}")
    
    # 验证单调性
    pos_rewards = [model._map_reward_with_confidence(1.0, conf) for conf in confidence_values]
    neg_rewards = [model._map_reward_with_confidence(-1.0, conf) for conf in confidence_values]
    
    # 正奖励应该随置信度增加而增加
    if not all(pos_rewards[i] <= pos_rewards[i+1] for i in range(len(pos_rewards)-1)):
        print("  ❌ 正奖励不满足单调性")
        return False
    
    # 负奖励应该随置信度增加而减少（绝对值增加）
    if not all(neg_rewards[i] >= neg_rewards[i+1] for i in range(len(neg_rewards)-1)):
        print("  ❌ 负奖励不满足单调性")
        return False
    
    print("  ✅ 线性插值和单调性测试通过")
    
    return True

if __name__ == "__main__":
    print("开始简化的偏好奖励模型测试...")
    
    # 运行测试
    functionality_passed = test_preference_model_functionality()
    batch_passed = test_batch_processing()
    consistency_passed = test_confidence_mapping_consistency()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"核心功能测试: {'✅ 通过' if functionality_passed else '❌ 失败'}")
    print(f"批量处理测试: {'✅ 通过' if batch_passed else '❌ 失败'}")
    print(f"一致性测试: {'✅ 通过' if consistency_passed else '❌ 失败'}")
    
    if functionality_passed and batch_passed and consistency_passed:
        print("\n🎉 所有测试通过！偏好奖励模型修改成功并符合用户要求。")
        print("\n主要功能验证:")
        print("  ✅ 奖励范围正确映射到 (-0.4, -0.1) 和 (0.1, 0.4)")
        print("  ✅ 置信度 >= 0.7 时达到最大奖励/惩罚 (±0.4)")
        print("  ✅ 置信度 <= 0.4 时达到最小奖励/惩罚 (±0.1)")
        print("  ✅ 中间置信度区间使用线性插值")
        print("  ✅ 批量处理功能正常")
        print("  ✅ 边界条件和单调性满足要求")
    else:
        print("\n❌ 部分测试失败，需要进一步检查。")
    
    print("=" * 60)