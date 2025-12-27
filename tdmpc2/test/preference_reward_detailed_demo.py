#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好奖励正负判断的详细演示

本脚本通过具体的代码示例，详细演示偏好奖励模型如何判断正负奖励，
以及这个判断过程的每一个步骤。
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')

import torch
import numpy as np
from typing import Dict, List, Tuple

# 导入相关模块
from prm.optimized_latent_preference_model import OptimizedLatentPreferenceModel, OptimizedLatentPreferenceConfig

def demonstrate_reward_mapping_logic():
    """
    演示奖励映射逻辑的详细过程
    """
    print("="*80)
    print("偏好奖励正负判断的详细过程演示")
    print("="*80)
    
    print("\n1. _map_reward_with_confidence 方法的核心逻辑：")
    print("   这是决定偏好奖励正负性的关键方法")
    
    # 模拟不同的原始奖励值和置信度
    test_cases = [
        (2.5, 0.8),    # 正数，高置信度
        (-1.8, 0.9),   # 负数，高置信度
        (0.5, 0.5),    # 正数，中等置信度
        (-0.3, 0.3),   # 负数，低置信度
        (0.0, 0.6),    # 零值，中等置信度
        (3.2, 0.2),    # 正数，低置信度
        (-2.1, 0.75),  # 负数，高置信度
    ]
    
    print("\n2. 测试不同情况下的奖励映射：")
    print("   格式：原始奖励 -> 判断正负 -> 置信度映射 -> 最终奖励")
    print("   " + "-"*70)
    
    for i, (raw_reward, confidence) in enumerate(test_cases, 1):
        # 模拟 _map_reward_with_confidence 的逻辑
        is_positive = raw_reward >= 0
        
        # 根据置信度计算奖励强度
        if confidence >= 0.7:
            reward_magnitude = 0.4  # 高置信度：最大奖励/惩罚
        elif confidence <= 0.4:
            reward_magnitude = 0.1  # 低置信度：最小奖励/惩罚
        else:
            # 中等置信度：线性插值
            alpha = (confidence - 0.4) / (0.7 - 0.4)
            reward_magnitude = 0.1 + alpha * (0.4 - 0.1)
        
        # 应用正负号
        final_reward = reward_magnitude if is_positive else -reward_magnitude
        
        print(f"   案例{i}: {raw_reward:+6.2f} -> {'正数' if is_positive else '负数'} -> "
              f"置信度{confidence:.2f} -> {final_reward:+6.3f}")
        
        # 详细解释这个案例
        if i <= 3:  # 只详细解释前3个案例
            print(f"          详细过程：")
            print(f"          - 原始奖励 {raw_reward:+.2f} {'≥' if is_positive else '<'} 0，判断为{'正向' if is_positive else '负向'}偏好")
            if confidence >= 0.7:
                print(f"          - 置信度 {confidence:.2f} ≥ 0.7，使用最大强度 0.4")
            elif confidence <= 0.4:
                print(f"          - 置信度 {confidence:.2f} ≤ 0.4，使用最小强度 0.1")
            else:
                alpha = (confidence - 0.4) / (0.7 - 0.4)
                print(f"          - 置信度 {confidence:.2f} 在中等范围，插值系数 α = {alpha:.3f}")
                print(f"          - 强度 = 0.1 + {alpha:.3f} × (0.4 - 0.1) = {reward_magnitude:.3f}")
            print(f"          - 最终奖励 = {'+'if is_positive else '-'}{reward_magnitude:.3f} = {final_reward:+.3f}")
            print()

def demonstrate_trajectory_preference_counting():
    """
    演示轨迹中正负偏好的统计过程
    """
    print("\n" + "="*80)
    print("轨迹中正负偏好统计的详细过程")
    print("="*80)
    
    # 模拟一条轨迹的偏好奖励序列
    trajectory_rewards = [
        0.35,   # 正向偏好 +1
        -0.25,  # 负向偏好 +1
        0.12,   # 正向偏好 +1
        0.28,   # 正向偏好 +1
        -0.15,  # 负向偏好 +1
        0.40,   # 正向偏好 +1
        -0.32,  # 负向偏好 +1
        0.18,   # 正向偏好 +1
        -0.11,  # 负向偏好 +1
        0.22,   # 正向偏好 +1
    ]
    
    print(f"\n模拟轨迹长度：{len(trajectory_rewards)} 步")
    print("\n逐步统计过程：")
    print("步骤 | 偏好奖励 | 正负判断 | 正向累计 | 负向累计")
    print("-" * 50)
    
    positive_count = 0
    negative_count = 0
    
    for step, reward in enumerate(trajectory_rewards, 1):
        if reward > 0:
            positive_count += 1
            judgment = "正向 +1"
        elif reward < 0:
            negative_count += 1
            judgment = "负向 +1"
        else:
            judgment = "零值 +0"
        
        print(f"{step:4d} | {reward:+8.3f} | {judgment:8s} | {positive_count:8d} | {negative_count:8d}")
    
    print("-" * 50)
    print(f"最终统计：正向偏好 = {positive_count}，负向偏好 = {negative_count}")
    
    # 计算修正因子
    trajectory_length = len(trajectory_rewards)
    average_confidence = 0.75  # 假设平均置信度
    
    correction_factor = (positive_count - negative_count) / trajectory_length * average_confidence
    
    print(f"\n修正因子计算：")
    print(f"修正因子 = ({positive_count} - {negative_count}) / {trajectory_length} × {average_confidence}")
    print(f"修正因子 = {positive_count - negative_count} / {trajectory_length} × {average_confidence}")
    print(f"修正因子 = {(positive_count - negative_count) / trajectory_length:.3f} × {average_confidence}")
    print(f"修正因子 = {correction_factor:.4f}")
    
    # 应用边界约束
    if correction_factor > 0:
        bounded_factor = max(0.1, min(0.4, correction_factor))
        print(f"\n正向修正因子边界约束：[0.1, 0.4]")
    else:
        bounded_factor = max(-0.4, min(-0.1, correction_factor))
        print(f"\n负向修正因子边界约束：[-0.4, -0.1]")
    
    print(f"约束后修正因子 = {bounded_factor:.4f}")
    
    # 最终奖励融合
    env_reward = 15.8  # 假设环境奖励
    fused_reward = env_reward * (1 + bounded_factor)
    
    print(f"\n最终奖励融合：")
    print(f"融合奖励 = 环境奖励 × (1 + 修正因子)")
    print(f"融合奖励 = {env_reward} × (1 + {bounded_factor:.4f})")
    print(f"融合奖励 = {env_reward} × {1 + bounded_factor:.4f}")
    print(f"融合奖励 = {fused_reward:.2f}")
    
    if bounded_factor > 0:
        print(f"\n结果解释：正向偏好多于负向偏好，环境奖励被增强了 {bounded_factor*100:.1f}%")
    else:
        print(f"\n结果解释：负向偏好多于正向偏好，环境奖励被削弱了 {abs(bounded_factor)*100:.1f}%")

def demonstrate_model_forward_process():
    """
    演示模型前向传播过程中的奖励生成
    """
    print("\n" + "="*80)
    print("模型前向传播中的奖励生成过程")
    print("="*80)
    
    # 创建模型
    config = OptimizedLatentPreferenceConfig(
        latent_dim=512,
        action_dim=61,
        hidden_dim=256
    )
    model = OptimizedLatentPreferenceModel(config)
    model.eval()
    
    print("\n1. 模型结构概览：")
    print(f"   - 潜空间维度：{config.latent_dim}")
    print(f"   - 动作维度：{config.action_dim}")
    print(f"   - 隐藏层维度：{config.hidden_dim}")
    print(f"   - Transformer层数：{config.n_transformer_layers}")
    print(f"   - 注意力头数：{config.n_attention_heads}")
    
    print("\n2. 单步奖励生成过程演示：")
    
    # 生成测试数据
    latent_state = torch.randn(512)
    action = torch.randn(61)
    
    print(f"   输入：")
    print(f"   - 潜空间状态形状：{latent_state.shape}")
    print(f"   - 动作形状：{action.shape}")
    
    with torch.no_grad():
        # 手动模拟 get_preference_reward 的过程
        print(f"\n   处理步骤：")
        
        # 1. 添加批次和序列维度
        latent_batch = latent_state.unsqueeze(0).unsqueeze(0)
        action_batch = action.unsqueeze(0).unsqueeze(0)
        print(f"   1. 扩展维度后：")
        print(f"      - 潜空间批次形状：{latent_batch.shape}")
        print(f"      - 动作批次形状：{action_batch.shape}")
        
        # 2. 前向传播
        score, confidence = model.forward(latent_batch, action_batch, return_confidence=True)
        raw_score = float(score.item())
        confidence_value = float(confidence.item())
        
        print(f"   2. 模型前向传播：")
        print(f"      - 原始分数：{raw_score:.6f}")
        print(f"      - 置信度：{confidence_value:.6f}")
        
        # 3. 标准化（简化演示，不更新统计）
        normalized_score = raw_score * 0.05  # 简化的标准化
        print(f"   3. 标准化处理：")
        print(f"      - 标准化分数：{normalized_score:.6f}")
        
        # 4. 映射到最终范围
        is_positive = normalized_score >= 0
        
        if confidence_value >= 0.7:
            reward_magnitude = 0.4
        elif confidence_value <= 0.4:
            reward_magnitude = 0.1
        else:
            alpha = (confidence_value - 0.4) / (0.7 - 0.4)
            reward_magnitude = 0.1 + alpha * (0.4 - 0.1)
        
        final_reward = reward_magnitude if is_positive else -reward_magnitude
        
        print(f"   4. 最终映射：")
        print(f"      - 正负判断：{'正数' if is_positive else '负数'} (基于标准化分数 {normalized_score:+.6f})")
        print(f"      - 置信度等级：{confidence_value:.3f} -> ", end="")
        if confidence_value >= 0.7:
            print("高置信度 (≥0.7) -> 最大强度 0.4")
        elif confidence_value <= 0.4:
            print("低置信度 (≤0.4) -> 最小强度 0.1")
        else:
            print(f"中等置信度 -> 插值强度 {reward_magnitude:.3f}")
        print(f"      - 最终偏好奖励：{final_reward:+.3f}")
        
        # 5. 统计意义
        print(f"\n   5. 统计意义：")
        if final_reward > 0:
            print(f"      - 这一步会被计入：正向偏好 +1")
            print(f"      - 含义：该(状态,动作)对有利于轨迹质量")
        else:
            print(f"      - 这一步会被计入：负向偏好 +1")
            print(f"      - 含义：该(状态,动作)对不利于轨迹质量")

def main():
    """
    主函数
    """
    print("偏好奖励正负判断的详细演示")
    print("深入解析每个步骤的具体实现")
    
    # 演示奖励映射逻辑
    demonstrate_reward_mapping_logic()
    
    # 演示轨迹偏好统计
    demonstrate_trajectory_preference_counting()
    
    # 演示模型前向传播过程
    demonstrate_model_forward_process()
    
    print("\n" + "="*80)
    print("核心要点总结")
    print("="*80)
    print("\n1. 正负判断的根本依据：")
    print("   - 模型输出的原始分数的正负性")
    print("   - 原始分数 ≥ 0 → 正向偏好")
    print("   - 原始分数 < 0 → 负向偏好")
    
    print("\n2. 奖励强度的决定因素：")
    print("   - 置信度决定奖励的绝对值大小")
    print("   - 高置信度 → 强奖励/惩罚 (±0.4)")
    print("   - 低置信度 → 弱奖励/惩罚 (±0.1)")
    
    print("\n3. 轨迹级别的统计：")
    print("   - 遍历轨迹中每一步的偏好奖励")
    print("   - 正数奖励 → 正向偏好计数 +1")
    print("   - 负数奖励 → 负向偏好计数 +1")
    print("   - 最终通过 (正向-负向)/轨迹长度 计算修正因子")
    
    print("\n4. 从轨迹级别到单步级别的桥梁：")
    print("   - 训练时：模型学习轨迹级别的偏好")
    print("   - 推理时：模型为单步输出偏好分数")
    print("   - 单步分数反映该步骤对整体轨迹质量的贡献")
    
    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)

if __name__ == "__main__":
    main()