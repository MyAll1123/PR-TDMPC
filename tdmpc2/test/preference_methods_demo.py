#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO偏好方法演示脚本

本脚本演示QUALITY_BASED和HYBRID_DPO_QUALITY方法的具体计算过程，
展示如何依靠DPO和这两种方法产生偏好对。
"""

import numpy as np
import torch
import math
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum

class LabelType(Enum):
    """偏好标签类型"""
    QUALITY_BASED = "quality_based"
    HYBRID_DPO_QUALITY = "hybrid_dpo_quality"
    DPO_BINARY = "dpo_binary"

@dataclass
class TrajectoryData:
    """轨迹数据结构"""
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    name: str
    
    def get_reward_sum(self) -> float:
        """获取奖励总和"""
        return float(np.sum(self.rewards))
    
    def get_length(self) -> int:
        """获取轨迹长度"""
        return len(self.rewards)

class QualityEvaluator:
    """质量评估器"""
    
    def __init__(self):
        self.survival_weight = 1.0
        self.stability_weight = 0.85
        self.smoothness_weight = 0.90
    
    def calculate_survival_score(self, trajectory_length: int, max_length: int = 100) -> float:
        """计算生存得分"""
        return min(trajectory_length / max_length, 1.0)
    
    def calculate_stability_score(self, obs: np.ndarray) -> float:
        """计算状态稳定性得分"""
        if len(obs) < 2:
            return 0.3
        
        # 计算状态变化的方差
        state_changes = np.diff(obs, axis=0)
        variance = np.mean(np.var(state_changes, axis=0))
        
        # 将方差转换为稳定性得分 [0.3, 1.0]
        stability = max(0.3, 1.0 - min(variance / 10.0, 0.7))
        return stability
    
    def calculate_smoothness_score(self, actions: np.ndarray) -> float:
        """计算动作平滑性得分"""
        if len(actions) < 2:
            return 0.3
        
        # 计算动作变化的平均绝对差
        action_changes = np.diff(actions, axis=0)
        smoothness_metric = np.mean(np.abs(action_changes))
        
        # 将变化转换为平滑性得分 [0.3, 1.0]
        smoothness = max(0.3, 1.0 - min(smoothness_metric / 5.0, 0.7))
        return smoothness
    
    def evaluate_trajectory_quality(self, trajectory: TrajectoryData) -> Tuple[float, Dict[str, float]]:
        """评估轨迹质量"""
        # 计算各项得分
        survival_score = self.calculate_survival_score(trajectory.get_length())
        stability_score = self.calculate_stability_score(trajectory.obs)
        smoothness_score = self.calculate_smoothness_score(trajectory.actions)
        
        # 计算基础质量因子
        base_quality_factor = survival_score * stability_score * smoothness_score
        
        # 计算最终质量分数
        reward_sum = trajectory.get_reward_sum()
        quality_score = reward_sum * base_quality_factor
        
        feature_scores = {
            'survival_score': survival_score,
            'stability_score': stability_score,
            'smoothness_score': smoothness_score,
            'base_quality_factor': base_quality_factor,
            'reward_sum': reward_sum,
            'quality_score': quality_score
        }
        
        return quality_score, feature_scores

class DPOEvaluator:
    """DPO评估器"""
    
    def __init__(self, beta: float = 5.0, label_smoothing: float = 0.0):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.quality_evaluator = QualityEvaluator()
    
    def _heuristic_reward_estimate(self, trajectory: TrajectoryData) -> float:
        """启发式奖励估计（基于质量评估器）"""
        quality_score, _ = self.quality_evaluator.evaluate_trajectory_quality(trajectory)
        return float(np.clip(quality_score, 0.0, 100.0))  # 限制在合理范围内
    
    def _compute_confidence(self, reward_a: float, reward_b: float, preference_logit: float) -> float:
        """计算置信度"""
        reward_diff = abs(reward_a - reward_b)
        logit_magnitude = abs(preference_logit)
        
        # 基于奖励差异和logit大小计算置信度
        confidence = min(0.5 + 0.1 * reward_diff + 0.05 * logit_magnitude, 0.95)
        return confidence
    
    def evaluate_dpo_preference(self, trajectory_a: TrajectoryData, 
                               trajectory_b: TrajectoryData) -> Tuple[float, float]:
        """使用DPO方法评估轨迹偏好"""
        # 1. 计算轨迹奖励（使用启发式估计）
        reward_a = self._heuristic_reward_estimate(trajectory_a)
        reward_b = self._heuristic_reward_estimate(trajectory_b)
        
        # 2. DPO偏好概率计算
        reward_diff = float(reward_a) - float(reward_b)
        preference_logit = float(self.beta * reward_diff)
        
        # 3. 计算置信度
        confidence = self._compute_confidence(reward_a, reward_b, preference_logit)
        
        return preference_logit, confidence

class PreferenceMethodsDemo:
    """偏好方法演示类"""
    
    def __init__(self):
        self.quality_evaluator = QualityEvaluator()
        self.dpo_evaluator = DPOEvaluator(beta=5.0)
    
    def _calculate_quality_based_score(self, quality_a: float, quality_b: float) -> Tuple[float, float]:
        """QUALITY_BASED方法：基于质量分数计算偏好分数和置信度"""
        quality_diff = quality_a - quality_b
        abs_diff = abs(quality_diff)
        
        print(f"    质量差异: {quality_diff:.6f}")
        print(f"    绝对差异: {abs_diff:.6f}")
        
        # 1. 不确定性判断（极严格）
        uncertainty_range = 0.01  # 极小的不确定性阈值
        min_uncertainty_threshold = 0.1
        
        if abs_diff < uncertainty_range:
            print(f"    -> 差异小于不确定性阈值({uncertainty_range})，标记为不确定")
            return 0.5, min_uncertainty_threshold  # 不确定
        
        # 2. 计算偏好分数（高敏感度）
        sigmoid_input = quality_diff * 10.0  # 高敏感度乘数
        print(f"    Sigmoid输入: {quality_diff} * 10.0 = {sigmoid_input:.3f}")
        
        preference_score = torch.sigmoid(torch.tensor(sigmoid_input)).item()
        print(f"    原始偏好分数: {preference_score:.6f}")
        
        # 3. 应用标签平滑（几乎不使用）
        smoothing = 0.01 * 0.1  # 极小的标签平滑
        if preference_score > 0.5:
            preference_score = preference_score * (1 - smoothing) + 0.5 * smoothing
        else:
            preference_score = preference_score * (1 - smoothing) + 0.5 * smoothing
        print(f"    标签平滑后: {preference_score:.6f} (平滑系数: {smoothing})")
        
        # 4. 计算置信度（高置信度）
        confidence = min(abs_diff * 10.0 + 0.5, 0.95)
        print(f"    置信度: min({abs_diff:.6f} * 10.0 + 0.5, 0.95) = {confidence:.6f}")
        
        return preference_score, confidence
    
    def _calculate_hybrid_dpo_quality_score(self, trajectory_a: TrajectoryData, 
                                          trajectory_b: TrajectoryData,
                                          quality_a: float, quality_b: float) -> Tuple[float, float]:
        """HYBRID_DPO_QUALITY方法：结合DPO评估和质量评估"""
        print("    === DPO评估部分 ===")
        # 1. 计算DPO分数
        dpo_logit, dpo_conf = self.dpo_evaluator.evaluate_dpo_preference(trajectory_a, trajectory_b)
        dpo_score = torch.sigmoid(torch.tensor(dpo_logit)).item()
        
        print(f"    DPO logit: {dpo_logit:.6f}")
        print(f"    DPO分数: sigmoid({dpo_logit:.6f}) = {dpo_score:.6f}")
        print(f"    DPO置信度: {dpo_conf:.6f}")
        
        print("    === 质量评估部分 ===")
        # 2. 计算质量分数
        quality_score, quality_conf = self._calculate_quality_based_score(quality_a, quality_b)
        
        print(f"    质量分数: {quality_score:.6f}")
        print(f"    质量置信度: {quality_conf:.6f}")
        
        print("    === 混合计算 ===")
        # 3. 加权组合（DPO主导）
        dpo_weight = 0.8
        quality_weight = 0.2
        
        combined_score = dpo_weight * dpo_score + quality_weight * quality_score
        combined_conf = (dpo_conf + quality_conf) / 2  # 平均置信度
        
        print(f"    组合分数: {dpo_weight} * {dpo_score:.6f} + {quality_weight} * {quality_score:.6f} = {combined_score:.6f}")
        print(f"    组合置信度: ({dpo_conf:.6f} + {quality_conf:.6f}) / 2 = {combined_conf:.6f}")
        
        return combined_score, combined_conf
    
    def create_sample_trajectories(self) -> List[TrajectoryData]:
        """创建示例轨迹数据"""
        trajectories = []
        
        # 轨迹A：高质量轨迹
        obs_a = np.array([
            [1.0, 0.5, 0.2],
            [1.1, 0.52, 0.21],
            [1.15, 0.54, 0.22],
            [1.2, 0.56, 0.23],
            [1.25, 0.58, 0.24],
            [1.3, 0.6, 0.25],
            [1.35, 0.62, 0.26],
            [1.4, 0.64, 0.27],
            [1.45, 0.66, 0.28],
            [1.5, 0.68, 0.29]
        ])
        actions_a = np.array([
            [0.1, 0.05],
            [0.11, 0.052],
            [0.12, 0.054],
            [0.13, 0.056],
            [0.14, 0.058],
            [0.15, 0.06],
            [0.16, 0.062],
            [0.17, 0.064],
            [0.18, 0.066]
        ])
        rewards_a = np.array([2.5, 2.8, 3.1, 2.9, 3.2, 2.7, 3.0, 2.6, 2.9, 3.1])
        
        trajectory_a = TrajectoryData(obs_a, actions_a, rewards_a, "高质量轨迹A")
        trajectories.append(trajectory_a)
        
        # 轨迹B：低质量轨迹
        obs_b = np.array([
            [0.5, 0.2, 0.1],
            [0.3, 0.4, 0.15],
            [0.8, 0.1, 0.05],
            [0.2, 0.6, 0.2],
            [0.9, 0.05, 0.02],
            [0.1, 0.7, 0.25]
        ])
        actions_b = np.array([
            [0.2, 0.1],
            [-0.1, 0.3],
            [0.4, -0.2],
            [-0.3, 0.5],
            [0.6, -0.4]
        ])
        rewards_b = np.array([0.5, 0.2, -0.1, 0.3, -0.2, 0.1])
        
        trajectory_b = TrajectoryData(obs_b, actions_b, rewards_b, "低质量轨迹B")
        trajectories.append(trajectory_b)
        
        # 轨迹C：中等质量轨迹
        obs_c = np.array([
            [0.8, 0.4, 0.15],
            [0.85, 0.42, 0.16],
            [0.9, 0.44, 0.17],
            [0.95, 0.46, 0.18],
            [1.0, 0.48, 0.19],
            [1.05, 0.5, 0.2],
            [1.1, 0.52, 0.21],
            [1.15, 0.54, 0.22]
        ])
        actions_c = np.array([
            [0.08, 0.04],
            [0.09, 0.042],
            [0.1, 0.044],
            [0.11, 0.046],
            [0.12, 0.048],
            [0.13, 0.05],
            [0.14, 0.052]
        ])
        rewards_c = np.array([1.5, 1.8, 2.0, 1.9, 2.1, 1.7, 2.2, 1.6])
        
        trajectory_c = TrajectoryData(obs_c, actions_c, rewards_c, "中等质量轨迹C")
        trajectories.append(trajectory_c)
        
        return trajectories
    
    def demonstrate_preference_calculation(self):
        """演示偏好计算过程"""
        print("🎯 DPO偏好方法演示")
        print("=" * 80)
        
        # 创建示例轨迹
        trajectories = self.create_sample_trajectories()
        
        # 评估每条轨迹的质量
        print("\n📊 轨迹质量评估")
        print("-" * 50)
        
        trajectory_qualities = []
        for i, traj in enumerate(trajectories):
            quality_score, feature_scores = self.quality_evaluator.evaluate_trajectory_quality(traj)
            trajectory_qualities.append(quality_score)
            
            print(f"\n{traj.name}:")
            print(f"  奖励总和: {feature_scores['reward_sum']:.3f}")
            print(f"  生存得分: {feature_scores['survival_score']:.3f}")
            print(f"  稳定性得分: {feature_scores['stability_score']:.3f}")
            print(f"  平滑性得分: {feature_scores['smoothness_score']:.3f}")
            print(f"  基础质量因子: {feature_scores['base_quality_factor']:.3f}")
            print(f"  最终质量分数: {quality_score:.3f}")
        
        # 进行偏好比较
        comparisons = [
            (0, 1, "高质量 vs 低质量"),
            (0, 2, "高质量 vs 中等质量"),
            (2, 1, "中等质量 vs 低质量")
        ]
        
        for idx_a, idx_b, comparison_name in comparisons:
            traj_a = trajectories[idx_a]
            traj_b = trajectories[idx_b]
            quality_a = trajectory_qualities[idx_a]
            quality_b = trajectory_qualities[idx_b]
            
            print(f"\n\n🔍 偏好比较: {comparison_name}")
            print("=" * 60)
            print(f"轨迹A ({traj_a.name}): 质量分数 = {quality_a:.6f}")
            print(f"轨迹B ({traj_b.name}): 质量分数 = {quality_b:.6f}")
            
            # QUALITY_BASED方法
            print("\n🎯 QUALITY_BASED方法计算:")
            print("-" * 40)
            quality_pref_score, quality_conf = self._calculate_quality_based_score(quality_a, quality_b)
            
            print(f"\n  结果: 偏好分数 = {quality_pref_score:.6f}, 置信度 = {quality_conf:.6f}")
            
            if quality_pref_score > 0.7:
                print(f"  ✅ 强偏好轨迹A (置信度: {quality_conf:.1%})")
            elif quality_pref_score > 0.55:
                print(f"  ✅ 偏好轨迹A (置信度: {quality_conf:.1%})")
            elif quality_pref_score < 0.3:
                print(f"  ✅ 强偏好轨迹B (置信度: {quality_conf:.1%})")
            elif quality_pref_score < 0.45:
                print(f"  ✅ 偏好轨迹B (置信度: {quality_conf:.1%})")
            else:
                print(f"  ⚠️ 偏好不明确 (置信度: {quality_conf:.1%})")
            
            # HYBRID_DPO_QUALITY方法
            print("\n🎯 HYBRID_DPO_QUALITY方法计算:")
            print("-" * 40)
            hybrid_pref_score, hybrid_conf = self._calculate_hybrid_dpo_quality_score(
                traj_a, traj_b, quality_a, quality_b
            )
            
            print(f"\n  结果: 偏好分数 = {hybrid_pref_score:.6f}, 置信度 = {hybrid_conf:.6f}")
            
            if hybrid_pref_score > 0.7:
                print(f"  ✅ 强偏好轨迹A (置信度: {hybrid_conf:.1%})")
            elif hybrid_pref_score > 0.55:
                print(f"  ✅ 偏好轨迹A (置信度: {hybrid_conf:.1%})")
            elif hybrid_pref_score < 0.3:
                print(f"  ✅ 强偏好轨迹B (置信度: {hybrid_conf:.1%})")
            elif hybrid_pref_score < 0.45:
                print(f"  ✅ 偏好轨迹B (置信度: {hybrid_conf:.1%})")
            else:
                print(f"  ⚠️ 偏好不明确 (置信度: {hybrid_conf:.1%})")
            
            # 方法比较
            print("\n📈 方法比较:")
            print("-" * 20)
            print(f"  QUALITY_BASED:     偏好分数={quality_pref_score:.6f}, 置信度={quality_conf:.6f}")
            print(f"  HYBRID_DPO_QUALITY: 偏好分数={hybrid_pref_score:.6f}, 置信度={hybrid_conf:.6f}")
            
            score_diff = abs(quality_pref_score - hybrid_pref_score)
            conf_diff = abs(quality_conf - hybrid_conf)
            print(f"  分数差异: {score_diff:.6f}")
            print(f"  置信度差异: {conf_diff:.6f}")
    
    def demonstrate_preference_pair_generation(self):
        """演示偏好对生成过程"""
        print("\n\n🔄 偏好对生成演示")
        print("=" * 80)
        
        trajectories = self.create_sample_trajectories()
        
        # 计算所有轨迹的质量分数
        scored_trajectories = []
        for traj in trajectories:
            quality_score, _ = self.quality_evaluator.evaluate_trajectory_quality(traj)
            scored_trajectories.append((traj, quality_score))
        
        # 按质量分数排序
        scored_trajectories.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 轨迹质量排序:")
        for i, (traj, score) in enumerate(scored_trajectories):
            print(f"  {i+1}. {traj.name}: {score:.6f}")
        
        # 生成偏好对
        print("\n🎯 生成偏好对:")
        preference_pairs = []
        
        # 策略1: 高质量 vs 低质量（强对比）
        for i in range(len(scored_trajectories)):
            for j in range(i+1, len(scored_trajectories)):
                traj_better, score_better = scored_trajectories[i]
                traj_worse, score_worse = scored_trajectories[j]
                
                # 计算质量差异
                quality_diff = score_better - score_worse
                
                # 只保留有意义的对比（差异足够大）
                if quality_diff > 1.0:  # 阈值可调
                    preference_pairs.append({
                        'trajectory_a': traj_better,
                        'trajectory_b': traj_worse,
                        'quality_a': score_better,
                        'quality_b': score_worse,
                        'quality_diff': quality_diff,
                        'expected_preference': 'A',
                        'pair_type': 'strong_contrast'
                    })
        
        print(f"\n生成了 {len(preference_pairs)} 个偏好对:")
        
        for i, pair in enumerate(preference_pairs):
            print(f"\n偏好对 {i+1}:")
            print(f"  轨迹A: {pair['trajectory_a'].name} (质量: {pair['quality_a']:.6f})")
            print(f"  轨迹B: {pair['trajectory_b'].name} (质量: {pair['quality_b']:.6f})")
            print(f"  质量差异: {pair['quality_diff']:.6f}")
            print(f"  预期偏好: {pair['expected_preference']}")
            print(f"  对比类型: {pair['pair_type']}")
            
            # 使用两种方法计算偏好标签
            quality_score, quality_conf = self._calculate_quality_based_score(
                pair['quality_a'], pair['quality_b']
            )
            
            hybrid_score, hybrid_conf = self._calculate_hybrid_dpo_quality_score(
                pair['trajectory_a'], pair['trajectory_b'], 
                pair['quality_a'], pair['quality_b']
            )
            
            print(f"  QUALITY_BASED标签: {quality_score:.6f} (置信度: {quality_conf:.6f})")
            print(f"  HYBRID_DPO_QUALITY标签: {hybrid_score:.6f} (置信度: {hybrid_conf:.6f})")

def main():
    """主函数"""
    demo = PreferenceMethodsDemo()
    
    # 演示偏好计算
    demo.demonstrate_preference_calculation()
    
    # 演示偏好对生成
    demo.demonstrate_preference_pair_generation()
    
    print("\n\n🎉 演示完成！")
    print("=" * 80)
    print("\n总结:")
    print("1. QUALITY_BASED方法直接基于质量分数差异，敏感度高，置信度高")
    print("2. HYBRID_DPO_QUALITY方法结合DPO理论和质量评估，更稳定")
    print("3. 两种方法都能有效生成高质量的偏好对用于训练")
    print("4. DPO改造后的系统能够自动化生成大量有学习价值的偏好数据")

if __name__ == "__main__":
    main()