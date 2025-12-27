#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复偏好标签生成引擎的核心问题

基于问题分析，实施最高优先级的解决方案：直接使用环境奖励替代启发式估计
"""

import os
import sys
import shutil
from pathlib import Path

# 添加项目路径
project_root = Path("/public/home/yaotianxiao2024/SPE/tdmpc2")
sys.path.append(str(project_root))
sys.path.append(str(project_root / "prm"))

class PreferenceLabelingEngineFixer:
    """偏好标签生成引擎修复器"""
    
    def __init__(self):
        self.project_root = Path("/public/home/yaotianxiao2024/SPE/tdmpc2")
        self.prm_dir = self.project_root / "prm"
        self.engine_file = self.prm_dir / "preference_labeling_engine.py"
        
    def backup_original_file(self):
        """备份原始文件"""
        backup_file = self.engine_file.with_suffix(".py.backup")
        if not backup_file.exists():
            shutil.copy2(self.engine_file, backup_file)
            print(f"✅ 已备份原始文件到: {backup_file}")
        else:
            print(f"📁 备份文件已存在: {backup_file}")
    
    def read_original_file(self):
        """读取原始文件内容"""
        with open(self.engine_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_fixed_heuristic_method(self):
        """创建修复后的启发式方法"""
        return '''
    def _heuristic_reward_estimate(self, obs_seq: List, act_seq: List, 
                                 trajectory_data: Optional[Dict] = None) -> float:
        """
        修复后的启发式奖励估计 - 直接使用环境奖励
        
        Args:
            obs_seq: 观测序列
            act_seq: 动作序列  
            trajectory_data: 轨迹数据，包含环境奖励信息
            
        Returns:
            float: 奖励估计值
        """
        try:
            # 优先使用轨迹数据中的环境奖励
            if trajectory_data and 'env_rewards' in trajectory_data:
                env_rewards = trajectory_data['env_rewards']
                if isinstance(env_rewards, (list, tuple)) and len(env_rewards) > 0:
                    # 使用累积环境奖励
                    total_env_reward = sum(env_rewards)
                    # 归一化到合理范围 [0, 1]
                    normalized_reward = max(0.0, min(1.0, total_env_reward / 100.0))
                    return normalized_reward
            
            # 如果轨迹数据中有总奖励信息
            if trajectory_data and 'total_reward' in trajectory_data:
                total_reward = trajectory_data['total_reward']
                normalized_reward = max(0.0, min(1.0, total_reward / 100.0))
                return normalized_reward
            
            # 如果轨迹数据中有累积奖励信息
            if trajectory_data and 'cumulative_reward' in trajectory_data:
                cumulative_reward = trajectory_data['cumulative_reward']
                normalized_reward = max(0.0, min(1.0, cumulative_reward / 100.0))
                return normalized_reward
            
            # 备用方案：使用改进的启发式估计（但权重调整为更符合任务目标）
            return self._improved_heuristic_estimate(obs_seq, act_seq)
            
        except Exception as e:
            self.logger.warning(f"环境奖励获取失败，使用备用启发式估计: {e}")
            return self._improved_heuristic_estimate(obs_seq, act_seq)
    
    def _improved_heuristic_estimate(self, obs_seq: List, act_seq: List) -> float:
        """
        改进的启发式估计 - 更符合任务目标
        
        Args:
            obs_seq: 观测序列
            act_seq: 动作序列
            
        Returns:
            float: 改进的启发式奖励估计
        """
        try:
            # 1. 任务完成度评估（基于轨迹长度，但不过度惩罚短轨迹）
            survival_score = min(len(obs_seq) / 150.0, 1.0)  # 降低长度要求
            
            # 2. 降低对动作平滑性的过度要求
            if len(act_seq) > 1:
                act_array = np.array(act_seq)
                action_variance = np.mean(np.var(act_array, axis=0))
                # 使用更宽松的平滑性评估
                smoothness_score = np.exp(-action_variance * 0.1)  # 降低惩罚系数
            else:
                smoothness_score = 0.7  # 提高默认分数
            
            # 3. 状态稳定性（降低权重）
            if len(obs_seq) > 1:
                obs_array = np.array(obs_seq)
                obs_variance = np.mean(np.var(obs_array, axis=0))
                stability_score = np.exp(-obs_variance * 0.05)  # 进一步降低惩罚
            else:
                stability_score = 0.7  # 提高默认分数
            
            # 4. 调整权重：更重视任务完成度，降低对"美观度"的要求
            heuristic_reward = (
                0.6 * survival_score +      # 提高生存/完成度权重
                0.2 * smoothness_score +    # 降低平滑性权重  
                0.2 * stability_score       # 降低稳定性权重
            )
            
            # 5. 确保结果在合理范围内
            return max(0.1, min(0.9, heuristic_reward))
            
        except Exception as e:
            self.logger.warning(f"改进启发式估计失败: {e}")
            return 0.5  # 返回中性值
'''
    
    def create_environment_reward_aware_methods(self):
        """创建环境奖励感知的方法"""
        return '''
    def _extract_env_reward_from_trajectory(self, trajectory_data: Dict) -> Optional[float]:
        """
        从轨迹数据中提取环境奖励
        
        Args:
            trajectory_data: 轨迹数据字典
            
        Returns:
            Optional[float]: 环境奖励，如果无法提取则返回None
        """
        try:
            # 尝试多种可能的环境奖励字段名
            reward_fields = [
                'env_rewards', 'environment_rewards', 'rewards',
                'total_reward', 'cumulative_reward', 'episode_reward',
                'env_reward', 'reward_sum'
            ]
            
            for field in reward_fields:
                if field in trajectory_data:
                    reward_data = trajectory_data[field]
                    
                    if isinstance(reward_data, (list, tuple)):
                        # 如果是序列，计算总和
                        return sum(reward_data)
                    elif isinstance(reward_data, (int, float)):
                        # 如果是单个数值
                        return float(reward_data)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"提取环境奖励失败: {e}")
            return None
    
    def _validate_preference_with_env_reward(self, 
                                           trajectory_a_data: Dict,
                                           trajectory_b_data: Dict,
                                           preference_score: float,
                                           confidence: float) -> Tuple[float, float, bool]:
        """
        使用环境奖励验证偏好标签的一致性
        
        Args:
            trajectory_a_data: 轨迹A的数据
            trajectory_b_data: 轨迹B的数据
            preference_score: 原始偏好分数
            confidence: 原始置信度
            
        Returns:
            Tuple[float, float, bool]: (调整后的偏好分数, 调整后的置信度, 是否一致)
        """
        try:
            # 提取环境奖励
            env_reward_a = self._extract_env_reward_from_trajectory(trajectory_a_data)
            env_reward_b = self._extract_env_reward_from_trajectory(trajectory_b_data)
            
            if env_reward_a is None or env_reward_b is None:
                # 无法获取环境奖励，返回原始值
                return preference_score, confidence, True
            
            # 基于环境奖励计算期望的偏好
            env_reward_diff = env_reward_a - env_reward_b
            
            # 如果环境奖励差异很小，降低置信度
            if abs(env_reward_diff) < 5.0:  # 阈值可调
                confidence = min(confidence, 0.6)
            
            # 检查偏好方向是否一致
            env_prefers_a = env_reward_diff > 0
            model_prefers_a = preference_score > 0.5
            
            is_consistent = env_prefers_a == model_prefers_a
            
            if not is_consistent:
                # 如果不一致，根据环境奖励调整偏好分数
                if abs(env_reward_diff) > 10.0:  # 环境奖励差异较大时
                    # 强制使用环境奖励的偏好方向
                    adjusted_score = 0.7 if env_prefers_a else 0.3
                    adjusted_confidence = 0.8
                    self.logger.warning(
                        f"偏好不一致已修正: 环境奖励差异={env_reward_diff:.2f}, "
                        f"原始偏好={preference_score:.3f} -> 调整后={adjusted_score:.3f}"
                    )
                    return adjusted_score, adjusted_confidence, False
                else:
                    # 环境奖励差异较小时，降低置信度但保持原偏好
                    return preference_score, min(confidence, 0.4), False
            
            return preference_score, confidence, True
            
        except Exception as e:
            self.logger.warning(f"偏好验证失败: {e}")
            return preference_score, confidence, True
'''
    
    def apply_fixes(self):
        """应用修复"""
        print("🔧 开始修复偏好标签生成引擎...")
        
        # 1. 备份原始文件
        self.backup_original_file()
        
        # 2. 读取原始内容
        original_content = self.read_original_file()
        
        # 3. 替换启发式奖励估计方法
        print("📝 修复启发式奖励估计方法...")
        
        # 查找并替换_heuristic_reward_estimate方法
        import re
        
        # 匹配原始的_heuristic_reward_estimate方法
        pattern = r'(\s+def _heuristic_reward_estimate\([^}]+?\n\s+except[^}]+?return [^\n]+)'
        
        if re.search(pattern, original_content, re.DOTALL):
            # 替换现有方法
            fixed_content = re.sub(
                pattern,
                self.create_fixed_heuristic_method(),
                original_content,
                flags=re.DOTALL
            )
        else:
            # 如果找不到，在类的末尾添加新方法
            # 查找类的结束位置
            class_end_pattern = r'(class PreferenceLabelingEngine[^}]+)(\n\n|$)'
            fixed_content = re.sub(
                class_end_pattern,
                r'\1' + self.create_fixed_heuristic_method() + r'\2',
                original_content,
                flags=re.DOTALL
            )
        
        # 4. 添加环境奖励感知方法
        print("📝 添加环境奖励感知方法...")
        
        # 在类的末尾添加新方法
        class_end_pattern = r'(class PreferenceLabelingEngine[^}]+)(\n\n|$)'
        fixed_content = re.sub(
            class_end_pattern,
            r'\1' + self.create_environment_reward_aware_methods() + r'\2',
            fixed_content,
            flags=re.DOTALL
        )
        
        # 5. 写入修复后的内容
        with open(self.engine_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ 修复完成！已更新文件: {self.engine_file}")
        
        return True
    
    def create_validation_script(self):
        """创建验证脚本"""
        validation_script = self.project_root / "test" / "validate_preference_fix.py"
        
        validation_content = '''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证偏好标签生成修复效果
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path("/public/home/yaotianxiao2024/SPE/tdmpc2")
sys.path.append(str(project_root))
sys.path.append(str(project_root / "prm"))

try:
    from preference_labeling_engine import PreferenceLabelingEngine
    print("✅ 成功导入修复后的PreferenceLabelingEngine")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_heuristic_reward_with_env_data():
    """测试带环境奖励数据的启发式估计"""
    print("\n🧪 测试环境奖励感知的启发式估计...")
    
    engine = PreferenceLabelingEngine()
    
    # 测试用例1：高环境奖励轨迹
    obs_seq_1 = [np.random.randn(10) for _ in range(100)]
    act_seq_1 = [np.random.randn(5) for _ in range(100)]
    trajectory_data_1 = {
        'env_rewards': [1.0] * 100,  # 总奖励100
        'total_reward': 100.0
    }
    
    # 测试用例2：低环境奖励轨迹
    obs_seq_2 = [np.random.randn(10) for _ in range(50)]
    act_seq_2 = [np.random.randn(5) for _ in range(50)]
    trajectory_data_2 = {
        'env_rewards': [0.1] * 50,  # 总奖励5
        'total_reward': 5.0
    }
    
    try:
        reward_1 = engine._heuristic_reward_estimate(obs_seq_1, act_seq_1, trajectory_data_1)
        reward_2 = engine._heuristic_reward_estimate(obs_seq_2, act_seq_2, trajectory_data_2)
        
        print(f"高环境奖励轨迹 (总奖励100): 启发式奖励 = {reward_1:.4f}")
        print(f"低环境奖励轨迹 (总奖励5): 启发式奖励 = {reward_2:.4f}")
        
        if reward_1 > reward_2:
            print("✅ 修复成功：高环境奖励轨迹获得更高的启发式奖励")
        else:
            print("❌ 修复失败：启发式奖励与环境奖励不一致")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def test_preference_validation():
    """测试偏好验证功能"""
    print("\n🧪 测试偏好验证功能...")
    
    engine = PreferenceLabelingEngine()
    
    # 模拟轨迹数据
    trajectory_a = {'env_rewards': [2.0] * 50, 'total_reward': 100.0}
    trajectory_b = {'env_rewards': [0.1] * 50, 'total_reward': 5.0}
    
    # 模拟错误的偏好分数（偏好低奖励轨迹）
    wrong_preference = 0.3  # 错误地偏好B
    confidence = 0.8
    
    try:
        if hasattr(engine, '_validate_preference_with_env_reward'):
            adjusted_score, adjusted_confidence, is_consistent = engine._validate_preference_with_env_reward(
                trajectory_a, trajectory_b, wrong_preference, confidence
            )
            
            print(f"原始偏好分数: {wrong_preference:.3f}")
            print(f"调整后偏好分数: {adjusted_score:.3f}")
            print(f"原始置信度: {confidence:.3f}")
            print(f"调整后置信度: {adjusted_confidence:.3f}")
            print(f"是否一致: {is_consistent}")
            
            if not is_consistent and adjusted_score > 0.5:
                print("✅ 偏好验证成功：错误偏好已被修正")
            else:
                print("❌ 偏好验证失败")
        else:
            print("⚠️ 偏好验证方法未找到，可能需要手动添加")
            
    except Exception as e:
        print(f"❌ 偏好验证测试失败: {e}")

if __name__ == "__main__":
    print("🚀 开始验证偏好标签生成修复效果...")
    
    test_heuristic_reward_with_env_data()
    test_preference_validation()
    
    print("\n✅ 验证完成！")
'''
        
        with open(validation_script, 'w', encoding='utf-8') as f:
            f.write(validation_content)
        
        print(f"📝 已创建验证脚本: {validation_script}")
        return validation_script
    
    def run_complete_fix(self):
        """运行完整修复流程"""
        print("🚀 开始偏好标签生成引擎完整修复流程...")
        
        try:
            # 1. 应用修复
            success = self.apply_fixes()
            
            if not success:
                print("❌ 修复失败")
                return False
            
            # 2. 创建验证脚本
            validation_script = self.create_validation_script()
            
            print("\n" + "="*60)
            print("📋 修复总结")
            print("="*60)
            print("\n✅ 已完成的修复:")
            print("  1. 修改_heuristic_reward_estimate方法直接使用环境奖励")
            print("  2. 添加改进的启发式估计作为备用方案")
            print("  3. 降低对动作平滑性的过度要求")
            print("  4. 提高任务完成度的权重")
            print("  5. 添加环境奖励提取和验证方法")
            
            print("\n🎯 修复效果:")
            print("  • 偏好标签将优先基于真实的环境奖励生成")
            print("  • 高环境奖励的轨迹将获得正向偏好")
            print("  • 低环境奖励的轨迹将获得负向偏好")
            print("  • 减少了启发式规则与任务目标的矛盾")
            
            print("\n📝 下一步建议:")
            print(f"  1. 运行验证脚本: python {validation_script}")
            print("  2. 清理现有的偏好数据缓冲区")
            print("  3. 重新启动训练以使用修复后的偏好标签")
            print("  4. 监控训练日志中的偏好统计变化")
            
            return True
            
        except Exception as e:
            print(f"❌ 修复过程中出现错误: {e}")
            return False

if __name__ == "__main__":
    fixer = PreferenceLabelingEngineFixer()
    fixer.run_complete_fix()