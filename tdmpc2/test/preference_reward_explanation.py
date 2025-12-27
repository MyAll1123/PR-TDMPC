#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好奖励模型工作原理详细解释

本脚本详细解释了偏好奖励模型如何从轨迹级别的判断转换为单步奖励，
以及正负偏好奖励的判断依据。
"""

import sys
import os
sys.path.append('/public/home/yaotianxiao2024/SPE')

import torch
import numpy as np
from typing import Dict, List, Tuple

# 导入相关模块
from prm.optimized_latent_preference_model import OptimizedLatentPreferenceModel, OptimizedLatentPreferenceConfig
from prm.preference_labeling_engine import PreferenceLabelingEngine, LabelType
from prm.prioritized_preference_system import PrioritizedPreferenceSystem

def explain_preference_reward_system():
    """
    详细解释偏好奖励系统的工作原理
    """
    print("="*80)
    print("偏好奖励模型工作原理详细解释")
    print("="*80)
    
    print("\n1. 偏好奖励模型的本质：")
    print("   - 偏好奖励模型确实是一个轨迹级别的判断器")
    print("   - 它通过比较两条完整轨迹的质量来学习偏好")
    print("   - 但在推理时，它可以为单个(状态,动作)对输出偏好奖励")
    
    print("\n2. 训练阶段 - 轨迹级别的偏好学习：")
    print("   a) 轨迹对生成：")
    print("      - 系统收集多条轨迹，每条轨迹包含：观测序列、动作序列、环境奖励")
    print("      - 根据轨迹的总环境奖励对轨迹进行排序")
    print("      - 选择奖励差异显著的轨迹对作为训练数据")
    
    print("   b) 偏好标签生成：")
    print("      - 比较两条轨迹的总环境奖励：reward_A vs reward_B")
    print("      - 如果 reward_A > reward_B，则轨迹A被标记为'偏好'，轨迹B为'非偏好'")
    print("      - 偏好标签是二元的：1表示偏好轨迹A，0表示偏好轨迹B")
    
    print("   c) 模型训练：")
    print("      - 模型学习预测：给定轨迹A和B，哪个更好")
    print("      - 使用Bradley-Terry损失函数进行训练")
    print("      - 损失函数：-log(sigmoid(score_A - score_B))")
    
    print("\n3. 推理阶段 - 单步奖励生成：")
    print("   a) 输入：单个(潜空间状态, 动作)对")
    print("   b) 模型处理：")
    print("      - 将单步数据扩展为序列格式 [1, 1, dim]")
    print("      - 通过Transformer网络处理")
    print("      - 输出原始偏好分数")
    
    print("   c) 分数处理：")
    print("      - 原始分数经过标准化处理")
    print("      - 根据置信度映射到指定范围")
    print("      - 最终输出范围：(-0.4, -0.1) 和 (0.1, 0.4)")
    
    print("\n4. 正负偏好奖励的判断依据：")
    print("   a) 模型输出的原始分数可能是任意实数")
    print("   b) 通过_map_reward_with_confidence方法处理：")
    print("      - 如果原始分数 >= 0：映射到正值范围 [0.1, 0.4]")
    print("      - 如果原始分数 < 0：映射到负值范围 [-0.4, -0.1]")
    print("   c) 置信度影响奖励幅度：")
    print("      - 高置信度(>=0.7)：最大奖励/惩罚 (±0.4)")
    print("      - 低置信度(<=0.4)：最小奖励/惩罚 (±0.1)")
    print("      - 中等置信度：线性插值")
    
    print("\n5. 关键理解：")
    print("   - 偏好奖励模型在训练时确实是轨迹级别的判断器")
    print("   - 但它学到的是'什么样的(状态,动作)对更可能出现在好轨迹中'")
    print("   - 因此可以为单步(状态,动作)对输出偏好奖励")
    print("   - 正负奖励反映了该步骤对轨迹质量的贡献")

def demonstrate_preference_reward_generation():
    """
    演示偏好奖励生成过程
    """
    print("\n" + "="*80)
    print("偏好奖励生成过程演示")
    print("="*80)
    
    # 创建模型配置
    config = OptimizedLatentPreferenceConfig(
        latent_dim=512,
        action_dim=61,
        hidden_dim=256
    )
    
    # 创建模型
    model = OptimizedLatentPreferenceModel(config)
    model.eval()
    
    print("\n1. 模拟单步数据：")
    # 模拟潜空间状态和动作
    latent_state = torch.randn(512)  # 潜空间状态
    action = torch.randn(61)  # 动作
    
    print(f"   潜空间状态维度: {latent_state.shape}")
    print(f"   动作维度: {action.shape}")
    
    print("\n2. 获取偏好奖励：")
    with torch.no_grad():
        preference_reward, confidence = model.get_preference_reward(latent_state, action)
    
    print(f"   原始偏好奖励: {preference_reward:.4f}")
    print(f"   置信度: {confidence:.4f}")
    
    # 解释奖励含义
    if preference_reward > 0:
        print(f"   → 正向偏好奖励：该(状态,动作)对被认为有利于轨迹质量")
        print(f"   → 在轨迹统计中会被计为：正向偏好+1")
    else:
        print(f"   → 负向偏好奖励：该(状态,动作)对被认为不利于轨迹质量")
        print(f"   → 在轨迹统计中会被计为：负向偏好+1")
    
    print("\n3. 多个样本演示：")
    positive_count = 0
    negative_count = 0
    
    for i in range(10):
        test_latent = torch.randn(512)
        test_action = torch.randn(61)
        
        with torch.no_grad():
            reward, conf = model.get_preference_reward(test_latent, test_action)
        
        if reward > 0:
            positive_count += 1
            sign = "+"
        else:
            negative_count += 1
            sign = "-"
        
        print(f"   样本{i+1}: 奖励={reward:+.4f}, 置信度={conf:.3f} [{sign}]")
    
    print(f"\n   统计结果：")
    print(f"   - 正向偏好步数: {positive_count}")
    print(f"   - 负向偏好步数: {negative_count}")
    print(f"   - 如果这是一条完整轨迹，修正因子 = ({positive_count}-{negative_count})/10 * 平均置信度")

def explain_trajectory_level_correction():
    """
    解释轨迹级别修正因子的计算
    """
    print("\n" + "="*80)
    print("轨迹级别修正因子详细解释")
    print("="*80)
    
    print("\n1. 修正因子的计算公式：")
    print("   修正因子 = (正向偏好个数 - 负向偏好个数) / 总轨迹长度 * 置信度")
    
    print("\n2. 正向和负向偏好的统计：")
    print("   - 遍历轨迹中的每一步")
    print("   - 获取该步的偏好奖励")
    print("   - 如果偏好奖励 > 0：正向偏好个数 += 1")
    print("   - 如果偏好奖励 < 0：负向偏好个数 += 1")
    print("   - 如果偏好奖励 = 0：不计入统计（实际很少发生）")
    
    print("\n3. 示例计算：")
    print("   假设一条长度为50的轨迹：")
    print("   - 30步获得正向偏好奖励 (>0)")
    print("   - 20步获得负向偏好奖励 (<0)")
    print("   - 平均置信度为0.8")
    print("   ")
    print("   修正因子 = (30 - 20) / 50 * 0.8 = 10/50 * 0.8 = 0.16")
    print("   ")
    print("   最终融合奖励 = 环境奖励 × (1 + 0.16) = 环境奖励 × 1.16")
    
    print("\n4. 修正因子的边界约束：")
    print("   - 正向修正因子范围：[0.1, 0.4]")
    print("   - 负向修正因子范围：[-0.4, -0.1]")
    print("   - 如果计算结果超出范围，会被截断到边界值")
    
    print("\n5. 物理意义：")
    print("   - 正向修正因子：轨迹中好的步骤多于坏的步骤，增强环境奖励")
    print("   - 负向修正因子：轨迹中坏的步骤多于好的步骤，削弱环境奖励")
    print("   - 修正因子接近0：轨迹中好坏步骤基本平衡，对环境奖励影响很小")

def main():
    """
    主函数
    """
    print("偏好奖励系统详细解释")
    print("回答用户问题：正负偏好奖励的判断依据，以及偏好奖励模型是否是轨迹级别的判断器")
    
    # 详细解释系统工作原理
    explain_preference_reward_system()
    
    # 演示偏好奖励生成
    demonstrate_preference_reward_generation()
    
    # 解释轨迹级别修正
    explain_trajectory_level_correction()
    
    print("\n" + "="*80)
    print("总结回答用户问题：")
    print("="*80)
    print("\n问题1：正向和负向偏好是通过那一步的偏好奖励是正数还是负数判断的")
    print("答案：是的！正是通过每一步的偏好奖励的正负性来判断：")
    print("      - 偏好奖励 > 0 → 正向偏好 +1")
    print("      - 偏好奖励 < 0 → 负向偏好 +1")
    
    print("\n问题2：偏好奖励模型不是轨迹级别的判断器吗？")
    print("答案：偏好奖励模型既是轨迹级别的判断器，也能输出单步奖励：")
    print("      - 训练时：确实是轨迹级别的判断器，通过比较完整轨迹学习偏好")
    print("      - 推理时：可以为单个(状态,动作)对输出偏好奖励")
    print("      - 单步奖励反映该步骤对整体轨迹质量的贡献")
    
    print("\n问题3：怎么得出是正数还是负数的？")
    print("答案：通过以下步骤：")
    print("      1. 模型输出原始偏好分数（可能是任意实数）")
    print("      2. 判断原始分数的正负性")
    print("      3. 根据置信度映射到指定范围：")
    print("         - 正数 → [0.1, 0.4]")
    print("         - 负数 → [-0.4, -0.1]")
    print("      4. 最终输出的正负性决定了偏好统计")
    
    print("\n" + "="*80)
    print("解释完成！")
    print("="*80)

if __name__ == "__main__":
    main()