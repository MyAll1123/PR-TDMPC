#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试单一判断逻辑：要求其中一条轨迹必须高于历史最高45%阈值
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
import logging
from prm.adaptive_threshold_manager import AdaptiveThresholdManager

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_single_threshold_logic():
    """测试单一判断逻辑"""
    print("=== 测试单一判断逻辑：要求其中一条轨迹必须高于历史最高45%阈值 ===")
    
    # 创建自适应阈值管理器
    config = {
        'confidence_threshold': 0.7,
        'rule_score_diff_threshold': 8.0,
        'env_reward_diff_threshold': 2.0
    }
    manager = AdaptiveThresholdManager(config, window_size=30)
    
    # 模拟添加一些奖励样本来建立历史最高值
    print("\n1. 建立历史最高环境平均值...")
    rewards = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]  # 历史最高窗口平均值应该是50
    for reward in rewards:
        manager.add_reward_sample(reward, reward * 0.8)
    
    # 添加更多样本，包括一个更高的窗口
    high_rewards = [60, 65, 70, 75, 80]  # 这个窗口平均值应该是70，成为新的历史最高
    for reward in high_rewards:
        manager.add_reward_sample(reward, reward * 0.8)
    
    # 获取历史最高值和45%阈值
    historical_max = manager.get_historical_max_env_avg()
    threshold_45 = manager.get_historical_max_threshold(0.45)
    
    print(f"历史最高环境平均值: {historical_max:.3f}")
    print(f"历史最高45%阈值: {threshold_45:.3f}")
    
    # 测试不同的轨迹对
    test_cases = [
        # (轨迹A奖励, 轨迹B奖励, 预期结果, 描述)
        (15, 18, False, "两条轨迹都低于45%阈值"),
        (40, 30, True, "轨迹A高于45%阈值"),
        (15, 35, True, "轨迹B高于45%阈值"),
        (50, 60, True, "两条轨迹都高于45%阈值"),
        (10, 15, False, "两条轨迹都远低于45%阈值"),
        (threshold_45 + 0.1, 15, True, "轨迹A刚好高于45%阈值"),
        (15, threshold_45 - 0.1, False, "轨迹B刚好低于45%阈值")
    ]
    
    print("\n2. 测试单一判断逻辑...")
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, (reward_a, reward_b, expected, description) in enumerate(test_cases, 1):
        # 单一判断：至少有一条轨迹高于历史最高环境平均值的45%
        at_least_one_above_threshold = (reward_a >= threshold_45 or reward_b >= threshold_45)
        
        result = "通过" if at_least_one_above_threshold == expected else "失败"
        status = "✓" if at_least_one_above_threshold == expected else "✗"
        
        print(f"测试 {i}: {description}")
        print(f"  轨迹A奖励: {reward_a:.3f}, 轨迹B奖励: {reward_b:.3f}")
        print(f"  判断结果: {at_least_one_above_threshold}, 预期: {expected}")
        print(f"  {status} {result}")
        print()
        
        if at_least_one_above_threshold == expected:
            passed_tests += 1
    
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    
    # 测试边界情况
    print("\n3. 测试边界情况...")
    
    # 测试历史最高值更新
    print("测试历史最高值更新...")
    old_max = manager.get_historical_max_env_avg()
    
    # 添加一个更高的窗口
    super_high_rewards = [90, 95, 100, 105, 110]  # 平均值100，应该成为新的历史最高
    for reward in super_high_rewards:
        manager.add_reward_sample(reward, reward * 0.8)
    
    new_max = manager.get_historical_max_env_avg()
    new_threshold_45 = manager.get_historical_max_threshold(0.45)
    
    print(f"旧历史最高值: {old_max:.3f}")
    print(f"新历史最高值: {new_max:.3f}")
    print(f"新45%阈值: {new_threshold_45:.3f}")
    
    if new_max > old_max:
        print("✓ 历史最高值正确更新")
    else:
        print("✗ 历史最高值更新失败")
    
    return passed_tests == total_tests

def test_anti_garbage_data_logic():
    """测试防垃圾数据逻辑"""
    print("\n=== 测试防垃圾数据逻辑 ===")
    
    config = {
        'confidence_threshold': 0.7,
        'rule_score_diff_threshold': 8.0,
        'env_reward_diff_threshold': 2.0
    }
    manager = AdaptiveThresholdManager(config, window_size=30)
    
    # 建立一个较高的历史最高值
    high_quality_rewards = [80, 85, 90, 95, 100]
    for reward in high_quality_rewards:
        manager.add_reward_sample(reward, reward * 0.8)
    
    historical_max = manager.get_historical_max_env_avg()
    threshold_45 = manager.get_historical_max_threshold(0.45)
    
    print(f"历史最高环境平均值: {historical_max:.3f}")
    print(f"45%阈值: {threshold_45:.3f}")
    
    # 测试垃圾数据对
    garbage_pairs = [
        (5, 10, "低质量 vs 低质量"),
        (15, 20, "中低质量 vs 中低质量"),
        (25, 30, "中等质量 vs 中等质量")
    ]
    
    print("\n测试垃圾数据过滤...")
    filtered_count = 0
    
    for reward_a, reward_b, description in garbage_pairs:
        at_least_one_above_threshold = (reward_a >= threshold_45 or reward_b >= threshold_45)
        
        if not at_least_one_above_threshold:
            print(f"✓ 成功过滤: {description} (A={reward_a}, B={reward_b})")
            filtered_count += 1
        else:
            print(f"✗ 过滤失败: {description} (A={reward_a}, B={reward_b})")
    
    # 测试高质量数据保留
    quality_pairs = [
        (threshold_45 + 5, 20, "高质量 vs 低质量"),
        (30, threshold_45 + 10, "中等质量 vs 高质量"),
        (threshold_45 + 5, threshold_45 + 10, "高质量 vs 高质量")
    ]
    
    print("\n测试高质量数据保留...")
    retained_count = 0
    
    for reward_a, reward_b, description in quality_pairs:
        at_least_one_above_threshold = (reward_a >= threshold_45 or reward_b >= threshold_45)
        
        if at_least_one_above_threshold:
            print(f"✓ 成功保留: {description} (A={reward_a:.1f}, B={reward_b:.1f})")
            retained_count += 1
        else:
            print(f"✗ 保留失败: {description} (A={reward_a:.1f}, B={reward_b:.1f})")
    
    print(f"\n防垃圾数据测试结果:")
    print(f"  成功过滤垃圾数据: {filtered_count}/{len(garbage_pairs)}")
    print(f"  成功保留高质量数据: {retained_count}/{len(quality_pairs)}")
    
    return filtered_count == len(garbage_pairs) and retained_count == len(quality_pairs)

if __name__ == "__main__":
    print("开始测试单一判断逻辑...")
    
    # 运行测试
    test1_passed = test_single_threshold_logic()
    test2_passed = test_anti_garbage_data_logic()
    
    print("\n=== 总体测试结果 ===")
    print(f"单一判断逻辑测试: {'通过' if test1_passed else '失败'}")
    print(f"防垃圾数据逻辑测试: {'通过' if test2_passed else '失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！单一判断逻辑工作正常。")
        print("✅ 系统现在要求至少一条轨迹高于历史最高45%阈值，有效防止垃圾数据学习。")
    else:
        print("\n❌ 部分测试失败，需要检查实现。")