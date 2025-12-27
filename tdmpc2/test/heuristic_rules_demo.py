#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启发式规则作用演示脚本

本脚本演示启发式规则在偏好学习系统中的具体作用和计算过程
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')

# 确保prm.api模块可以被正确导入
sys.path.append('/public/home/yaotianxiao2024/SPE/prm')

import numpy as np
from prm.preference_labeling_engine import PreferenceLabelingEngine
from prm.trajectory_metrics import TrajectoryQualityEvaluator
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_trajectory(length=100, reward_level='high', quality_level='high'):
    """
    创建演示用的轨迹数据
    
    Args:
        length: 轨迹长度
        reward_level: 奖励水平 ('high', 'medium', 'low')
        quality_level: 执行质量 ('high', 'medium', 'low')
    """
    
    # 根据奖励水平设置环境奖励
    if reward_level == 'high':
        base_reward = 2.0
        reward_noise = 0.3
    elif reward_level == 'medium':
        base_reward = 1.0
        reward_noise = 0.2
    else:  # low
        base_reward = 0.2
        reward_noise = 0.1
    
    # 根据质量水平设置动作和状态特征
    if quality_level == 'high':
        action_noise = 0.1
        state_noise = 0.05
        survival_factor = 1.0
    elif quality_level == 'medium':
        action_noise = 0.3
        state_noise = 0.15
        survival_factor = 0.8
    else:  # low
        action_noise = 0.8
        state_noise = 0.4
        survival_factor = 0.6
    
    # 调整实际长度
    actual_length = int(length * survival_factor)
    
    # 生成轨迹数据
    observations = []
    actions = []
    rewards = []
    
    for i in range(actual_length):
        # 生成观测（模拟机器人状态）
        obs = np.random.randn(37) * state_noise  # H1机器人通常有37维状态
        obs[2] = 1.0 + np.random.randn() * 0.1  # 头部高度
        observations.append(obs)
        
        # 生成动作
        action = np.random.randn(19) * action_noise  # H1机器人通常有19维动作
        actions.append(action)
        
        # 生成奖励
        reward = base_reward + np.random.randn() * reward_noise
        rewards.append(reward)
    
    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'obs': observations,  # 兼容性别名
        'action': actions,    # 兼容性别名
        'reward': rewards     # 兼容性别名
    }

def demonstrate_api_rules_loading():
    """
    演示API规则的加载过程
    """
    print("\n" + "="*80)
    print("🔧 API规则加载演示")
    print("="*80)
    
    # 创建偏好标注引擎
    engine = PreferenceLabelingEngine()
    
    # 检查API规则加载情况
    if hasattr(engine, 'api_rules') and engine.api_rules:
        print(f"✅ 成功加载 {len(engine.api_rules)} 个API规则函数:")
        for rule_name in engine.api_rules.keys():
            print(f"   - {rule_name}")
    else:
        print("❌ 未加载任何API规则")
    
    return engine

def demonstrate_trajectory_quality_evaluation(engine):
    """
    演示轨迹质量评估中启发式规则的作用
    """
    print("\n" + "="*80)
    print("📊 轨迹质量评估中的启发式规则作用")
    print("="*80)
    
    # 创建不同类型的轨迹
    trajectories = {
        "高奖励高质量": create_demo_trajectory(150, 'high', 'high'),
        "高奖励低质量": create_demo_trajectory(150, 'high', 'low'),
        "低奖励高质量": create_demo_trajectory(150, 'low', 'high'),
        "低奖励低质量": create_demo_trajectory(150, 'low', 'low')
    }
    
    print("\n🧪 轨迹质量评估结果:")
    
    for traj_name, traj_data in trajectories.items():
        try:
            # 计算环境奖励总和
            env_reward_sum = sum(traj_data['rewards'])
            
            # 使用质量评估器评估轨迹
            quality_score, detailed_scores = engine.quality_evaluator.evaluate_trajectory_quality(
                traj_data['observations'], 
                traj_data['actions'], 
                traj_data['rewards']
            )
            
            print(f"\n📈 {traj_name}:")
            print(f"   环境奖励总和: {env_reward_sum:.2f}")
            print(f"   质量分数: {quality_score:.4f}")
            print(f"   详细分数: {detailed_scores}")
            
            # 分析启发式规则的贡献
            if hasattr(engine.quality_evaluator, '_apply_api_rules'):
                # 尝试获取API规则的贡献
                try:
                    api_bonus = engine.quality_evaluator._apply_api_rules(
                        np.array(traj_data['observations']),
                        np.array(traj_data['actions']),
                        detailed_scores,
                        {}
                    )
                    print(f"   API规则贡献: {api_bonus:.4f}")
                except Exception as e:
                    print(f"   API规则贡献: 无法计算 ({e})")
            
        except Exception as e:
            print(f"❌ {traj_name} 评估失败: {e}")

def demonstrate_preference_calculation(engine):
    """
    演示偏好计算中启发式规则的作用
    """
    print("\n" + "="*80)
    print("🎯 偏好计算中的启发式规则作用")
    print("="*80)
    
    # 创建对比轨迹对
    traj_a = create_demo_trajectory(120, 'high', 'high')  # 高奖励高质量
    traj_b = create_demo_trajectory(100, 'medium', 'low')  # 中等奖励低质量
    
    print("\n🔄 轨迹对比分析:")
    
    # 计算环境奖励
    reward_a = sum(traj_a['rewards'])
    reward_b = sum(traj_b['rewards'])
    
    print(f"轨迹A - 环境奖励: {reward_a:.2f}, 长度: {len(traj_a['observations'])}")
    print(f"轨迹B - 环境奖励: {reward_b:.2f}, 长度: {len(traj_b['observations'])}")
    
    try:
        # 使用偏好标注引擎计算偏好
        preference_result = engine.generate_preference_label(
            np.array(traj_a['observations']),
            np.array(traj_a['actions']),
            np.array(traj_b['observations']),
            np.array(traj_b['actions']),
            trajectory_a_data=traj_a,
            trajectory_b_data=traj_b
        )
        
        if preference_result:
            preference_score, confidence, label_type = preference_result
            print(f"\n📊 偏好计算结果:")
            print(f"   偏好分数: {preference_score:.4f} (>0.5偏好A, <0.5偏好B)")
            print(f"   置信度: {confidence:.4f}")
            print(f"   标签类型: {label_type}")
            
            # 解释结果
            if preference_score > 0.6:
                print(f"   ✅ 强偏好轨迹A (置信度: {confidence*100:.1f}%)")
            elif preference_score < 0.4:
                print(f"   ✅ 强偏好轨迹B (置信度: {confidence*100:.1f}%)")
            else:
                print(f"   ⚖️ 偏好不明确 (置信度: {confidence*100:.1f}%)")
        else:
            print("❌ 偏好计算失败")
            
    except Exception as e:
        print(f"❌ 偏好计算出错: {e}")

def demonstrate_heuristic_vs_environment_reward():
    """
    演示启发式评估与环境奖励的对比
    """
    print("\n" + "="*80)
    print("⚖️ 启发式评估 vs 环境奖励对比")
    print("="*80)
    
    # 创建问题场景：高环境奖励但动作激进的轨迹
    print("\n🧪 问题场景分析:")
    
    scenarios = [
        {
            "name": "高奖励激进动作",
            "description": "任务成功但动作变化大",
            "reward_level": "high",
            "quality_level": "low",
            "expected": "应该获得正向偏好"
        },
        {
            "name": "低奖励平滑动作", 
            "description": "任务失败但动作平滑",
            "reward_level": "low",
            "quality_level": "high",
            "expected": "应该获得负向偏好"
        },
        {
            "name": "高奖励高质量",
            "description": "理想情况",
            "reward_level": "high",
            "quality_level": "high",
            "expected": "应该获得强正向偏好"
        }
    ]
    
    engine = PreferenceLabelingEngine()
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']} - {scenario['description']}")
        
        # 创建轨迹
        traj = create_demo_trajectory(
            120, 
            scenario['reward_level'], 
            scenario['quality_level']
        )
        
        # 计算环境奖励
        env_reward = sum(traj['rewards'])
        
        try:
            # 计算质量分数（包含启发式规则）
            quality_score, detailed_scores = engine.quality_evaluator.evaluate_trajectory_quality(
                traj['observations'],
                traj['actions'],
                traj['rewards']
            )
            
            print(f"   🌍 环境奖励总和: {env_reward:.2f}")
            print(f"   🧠 质量分数: {quality_score:.4f}")
            print(f"   📝 {scenario['expected']}")
            
            # 分析一致性
            env_normalized = env_reward / 100.0  # 简单归一化
            if (env_normalized > 0.5 and quality_score > env_normalized * 0.8) or \
               (env_normalized <= 0.5 and quality_score <= env_normalized * 1.2):
                print(f"   ✅ 启发式评估与环境奖励基本一致")
            else:
                print(f"   ⚠️ 启发式评估与环境奖励存在差异")
                
        except Exception as e:
            print(f"   ❌ 评估失败: {e}")

def main():
    """
    主演示函数
    """
    print("🚀 启发式规则作用演示")
    print("本演示将展示启发式规则在偏好学习系统中的具体作用和计算过程")
    
    try:
        # 1. 演示API规则加载
        engine = demonstrate_api_rules_loading()
        
        # 2. 演示轨迹质量评估
        demonstrate_trajectory_quality_evaluation(engine)
        
        # 3. 演示偏好计算
        demonstrate_preference_calculation(engine)
        
        # 4. 演示启发式评估与环境奖励对比
        demonstrate_heuristic_vs_environment_reward()
        
        print("\n" + "="*80)
        print("🎉 演示完成！")
        print("="*80)
        print("\n📚 总结:")
        print("1. 启发式规则通过API文件为特定任务提供专用评估逻辑")
        print("2. 在轨迹质量评估中，启发式规则作为额外的评估维度参与计算")
        print("3. 在偏好计算中，启发式规则通过质量评估间接影响偏好判断")
        print("4. 当前系统已优化，确保启发式评估与环境奖励的一致性")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()