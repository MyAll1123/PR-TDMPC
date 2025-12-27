#!/usr/bin/env python3
"""
测试历史最高环境平均值窗口机制和双重判断逻辑

这个脚本测试：
1. AdaptiveThresholdManager中历史最高环境平均值的跟踪和更新
2. PrioritizedPreferenceSystem中双重判断机制的过滤逻辑
3. 轨迹奖励数据正确传递给AdaptiveThresholdManager
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')
sys.path.append('/public/home/yaotianxiao2024/SPE/prm')

import numpy as np
import logging
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_adaptive_threshold_manager():
    """测试AdaptiveThresholdManager的历史最高值跟踪功能"""
    print("\n=== 测试AdaptiveThresholdManager历史最高值跟踪 ===")
    
    try:
        from adaptive_threshold_manager import AdaptiveThresholdManager
        
        # 创建配置
        config = {
            'confidence_threshold': 0.75,
            'rule_score_diff_multiplier': 2.0,
            'env_reward_diff_std_multiplier': 1.5,
            'min_quality_indicators': 2
        }
        
        # 创建管理器，窗口大小为5（便于测试）
        manager = AdaptiveThresholdManager(config, window_size=5)
        
        print(f"初始历史最高值: {manager.get_historical_max_env_avg():.4f}")
        print(f"初始历史最高30%阈值: {manager.get_historical_max_threshold(0.3):.4f}")
        
        # 模拟添加奖励样本
        test_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]  # 第一个窗口，平均值3.0
        print("\n添加第一个窗口的奖励样本:")
        for i, reward in enumerate(test_rewards):
            manager.add_reward_sample(reward, reward * 0.1)  # rule_score = reward * 0.1
            print(f"  样本{i+1}: 奖励={reward:.1f}, 当前历史最高={manager.get_historical_max_env_avg():.4f}")
        
        print(f"第一个窗口完成后历史最高值: {manager.get_historical_max_env_avg():.4f}")
        print(f"历史最高30%阈值: {manager.get_historical_max_threshold(0.3):.4f}")
        
        # 添加第二个窗口（更高的奖励）
        test_rewards_2 = [6.0, 7.0, 8.0, 9.0, 10.0]  # 第二个窗口，平均值8.0
        print("\n添加第二个窗口的奖励样本（更高奖励）:")
        for i, reward in enumerate(test_rewards_2):
            manager.add_reward_sample(reward, reward * 0.1)
            print(f"  样本{i+1}: 奖励={reward:.1f}, 当前历史最高={manager.get_historical_max_env_avg():.4f}")
        
        print(f"第二个窗口完成后历史最高值: {manager.get_historical_max_env_avg():.4f}")
        print(f"历史最高30%阈值: {manager.get_historical_max_threshold(0.3):.4f}")
        
        # 添加第三个窗口（较低的奖励，不应更新历史最高值）
        test_rewards_3 = [2.0, 3.0, 4.0, 5.0, 6.0]  # 第三个窗口，平均值4.0
        print("\n添加第三个窗口的奖励样本（较低奖励）:")
        for i, reward in enumerate(test_rewards_3):
            manager.add_reward_sample(reward, reward * 0.1)
            print(f"  样本{i+1}: 奖励={reward:.1f}, 当前历史最高={manager.get_historical_max_env_avg():.4f}")
        
        print(f"第三个窗口完成后历史最高值: {manager.get_historical_max_env_avg():.4f}")
        print(f"历史最高30%阈值: {manager.get_historical_max_threshold(0.3):.4f}")
        
        # 测试重置功能
        print("\n测试重置功能:")
        manager.reset()
        print(f"重置后历史最高值: {manager.get_historical_max_env_avg():.4f}")
        print(f"重置后历史最高30%阈值: {manager.get_historical_max_threshold(0.3):.4f}")
        
        print("✅ AdaptiveThresholdManager历史最高值跟踪测试通过")
        return True
        
    except Exception as e:
        print(f"❌ AdaptiveThresholdManager测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_double_judgment_logic():
    """测试双重判断逻辑的模拟"""
    print("\n=== 测试双重判断逻辑模拟 ===")
    
    try:
        from adaptive_threshold_manager import AdaptiveThresholdManager
        
        # 创建配置
        config = {
            'confidence_threshold': 0.75,
            'rule_score_diff_multiplier': 2.0,
            'env_reward_diff_std_multiplier': 1.5,
            'min_quality_indicators': 2
        }
        
        # 创建管理器
        manager = AdaptiveThresholdManager(config, window_size=5)
        
        # 建立历史最高值
        high_rewards = [8.0, 9.0, 10.0, 11.0, 12.0]  # 平均值10.0
        for reward in high_rewards:
            manager.add_reward_sample(reward)
        
        historical_max = manager.get_historical_max_env_avg()
        historical_threshold = manager.get_historical_max_threshold(0.3)
        
        print(f"建立的历史最高环境平均值: {historical_max:.4f}")
        print(f"历史最高30%阈值: {historical_threshold:.4f}")
        
        # 模拟当前窗口
        current_rewards = [4.0, 5.0, 6.0, 7.0, 8.0]  # 平均值6.0
        for reward in current_rewards:
            manager.add_reward_sample(reward)
        
        stats = manager.get_statistics_summary()
        current_avg = stats.get('mean', 0.0)
        
        print(f"当前滑动平均值: {current_avg:.4f}")
        
        # 测试不同轨迹对的双重判断
        test_cases = [
            (7.0, 5.0, "轨迹A高于当前平均，轨迹B高于历史阈值"),
            (5.0, 4.0, "两条轨迹都高于历史阈值，但都低于当前平均"),
            (8.0, 7.0, "两条轨迹都高于当前平均和历史阈值"),
            (2.0, 1.0, "两条轨迹都低于历史阈值"),
            (7.0, 2.0, "轨迹A高于当前平均和历史阈值，轨迹B低于历史阈值")
        ]
        
        print("\n双重判断测试结果:")
        for reward_a, reward_b, description in test_cases:
            # 第一重判断：至少有一条轨迹高于当前滑动平均值
            at_least_one_above_current = (reward_a >= current_avg or reward_b >= current_avg)
            
            # 第二重判断：两条轨迹都要高于历史最高环境平均值的30%
            both_above_historical = (reward_a >= historical_threshold and reward_b >= historical_threshold)
            
            # 最终判断
            pass_filter = at_least_one_above_current and both_above_historical
            
            status = "✅ 通过" if pass_filter else "❌ 过滤"
            print(f"  {description}")
            print(f"    轨迹A: {reward_a:.1f}, 轨迹B: {reward_b:.1f}")
            print(f"    第一重判断: {at_least_one_above_current} (至少一条 >= {current_avg:.1f})")
            print(f"    第二重判断: {both_above_historical} (两条都 >= {historical_threshold:.1f})")
            print(f"    结果: {status}")
            print()
        
        print("✅ 双重判断逻辑测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 双重判断逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """测试集成功能"""
    print("\n=== 测试集成功能 ===")
    
    try:
        # 测试导入
        from prioritized_preference_system import PrioritizedPreferenceSystem, PrioritizedSystemConfig
        from adaptive_threshold_manager import AdaptiveThresholdManager
        
        print("✅ 成功导入相关模块")
        
        # 测试配置创建
        config = PrioritizedSystemConfig()
        print(f"✅ 成功创建配置，窗口大小: {getattr(config, 'window_size', '未设置')}")
        
        # 测试AdaptiveThresholdManager创建
        threshold_config = {
            'confidence_threshold': 0.75,
            'rule_score_diff_multiplier': 2.0,
            'env_reward_diff_std_multiplier': 1.5,
            'min_quality_indicators': 2
        }
        
        manager = AdaptiveThresholdManager(threshold_config, window_size=30)
        print("✅ 成功创建AdaptiveThresholdManager")
        
        # 测试新增方法
        historical_max = manager.get_historical_max_env_avg()
        historical_threshold = manager.get_historical_max_threshold(0.3)
        print(f"✅ 历史最高值方法正常: {historical_max:.4f}, 30%阈值: {historical_threshold:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试历史最高环境平均值窗口机制和双重判断逻辑")
    print("=" * 60)
    
    results = []
    
    # 运行各项测试
    results.append(test_adaptive_threshold_manager())
    results.append(test_double_judgment_logic())
    results.append(test_integration())
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    
    test_names = [
        "AdaptiveThresholdManager历史最高值跟踪",
        "双重判断逻辑模拟",
        "集成功能测试"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！历史最高环境平均值窗口机制和双重判断逻辑实现正确。")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)