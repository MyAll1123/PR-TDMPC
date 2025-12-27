#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好标签质量验证模块
提供标签质量评估、一致性检查和异常检测功能
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from prm.preference_labeling_engine import PreferenceLabel, LabelType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"          # 基础验证
    STANDARD = "standard"    # 标准验证
    COMPREHENSIVE = "comprehensive"  # 全面验证

class QualityIssue(Enum):
    """质量问题类型"""
    LOW_CONFIDENCE = "low_confidence"        # 低置信度
    INCONSISTENT_LABELS = "inconsistent_labels"  # 标签不一致
    EXTREME_VALUES = "extreme_values"        # 极端值
    INSUFFICIENT_DIVERSITY = "insufficient_diversity"  # 多样性不足
    BIAS_DETECTED = "bias_detected"          # 检测到偏差
    OUTLIER_DETECTED = "outlier_detected"    # 检测到异常值

@dataclass
class ValidationResult:
    """验证结果"""
    overall_quality_score: float  # 总体质量分数 [0, 1]
    issues_detected: List[QualityIssue]  # 检测到的问题
    detailed_metrics: Dict[str, float]  # 详细指标
    recommendations: List[str]  # 改进建议
    validation_level: ValidationLevel
    total_labels: int
    valid_labels: int
    
class LabelQualityValidator:
    """标签质量验证器"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.quality_thresholds = self._load_quality_thresholds()
        
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """加载质量阈值"""
        thresholds = {
            'min_confidence': 0.3,      # 最小置信度
            'min_diversity': 0.2,       # 最小多样性
            'max_bias_ratio': 0.8,      # 最大偏差比例
            'outlier_threshold': 2.0,   # 异常值阈值（标准差倍数）
            'consistency_threshold': 0.7, # 一致性阈值
            'min_valid_ratio': 0.8      # 最小有效标签比例
        }
        
        # 根据验证级别调整阈值
        if self.validation_level == ValidationLevel.BASIC:
            thresholds['min_confidence'] = 0.2
            thresholds['min_valid_ratio'] = 0.6
        elif self.validation_level == ValidationLevel.COMPREHENSIVE:
            thresholds['min_confidence'] = 0.5
            thresholds['min_valid_ratio'] = 0.9
            
        return thresholds
    
    def validate_labels(self, labels: List[PreferenceLabel]) -> ValidationResult:
        """验证标签质量"""
        if not labels:
            return ValidationResult(
                overall_quality_score=0.0,
                issues_detected=[QualityIssue.INSUFFICIENT_DIVERSITY],
                detailed_metrics={},
                recommendations=["需要提供标签数据"],
                validation_level=self.validation_level,
                total_labels=0,
                valid_labels=0
            )
        
        # 基础统计
        total_labels = len(labels)
        valid_labels = sum(1 for label in labels if label.is_valid)
        
        # 执行各项验证
        issues_detected = []
        detailed_metrics = {}
        recommendations = []
        
        # 1. 基础验证
        basic_metrics = self._basic_validation(labels)
        detailed_metrics.update(basic_metrics)
        
        if basic_metrics['valid_ratio'] < self.quality_thresholds['min_valid_ratio']:
            issues_detected.append(QualityIssue.INCONSISTENT_LABELS)
            recommendations.append(f"有效标签比例过低 ({basic_metrics['valid_ratio']:.2f})，建议检查数据质量")
        
        # 2. 置信度验证
        confidence_metrics = self._confidence_validation(labels)
        detailed_metrics.update(confidence_metrics)
        
        if confidence_metrics['avg_confidence'] < self.quality_thresholds['min_confidence']:
            issues_detected.append(QualityIssue.LOW_CONFIDENCE)
            recommendations.append(f"平均置信度过低 ({confidence_metrics['avg_confidence']:.2f})，建议优化标签生成策略")
        
        # 3. 多样性验证
        diversity_metrics = self._diversity_validation(labels)
        detailed_metrics.update(diversity_metrics)
        
        if diversity_metrics['label_diversity'] < self.quality_thresholds['min_diversity']:
            issues_detected.append(QualityIssue.INSUFFICIENT_DIVERSITY)
            recommendations.append("标签多样性不足，建议增加不同类型的偏好对")
        
        # 4. 偏差检测
        bias_metrics = self._bias_detection(labels)
        detailed_metrics.update(bias_metrics)
        
        if bias_metrics['bias_ratio'] > self.quality_thresholds['max_bias_ratio']:
            issues_detected.append(QualityIssue.BIAS_DETECTED)
            recommendations.append(f"检测到标签偏差 ({bias_metrics['bias_ratio']:.2f})，建议平衡数据分布")
        
        # 5. 异常值检测（仅在标准和全面验证中执行）
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            outlier_metrics = self._outlier_detection(labels)
            detailed_metrics.update(outlier_metrics)
            
            if outlier_metrics['outlier_ratio'] > 0.1:  # 超过10%的异常值
                issues_detected.append(QualityIssue.OUTLIER_DETECTED)
                recommendations.append(f"检测到异常标签 ({outlier_metrics['outlier_ratio']:.2f})，建议检查数据生成过程")
        
        # 6. 一致性检查（仅在全面验证中执行）
        if self.validation_level == ValidationLevel.COMPREHENSIVE:
            consistency_metrics = self._consistency_validation(labels)
            detailed_metrics.update(consistency_metrics)
            
            if consistency_metrics['consistency_score'] < self.quality_thresholds['consistency_threshold']:
                issues_detected.append(QualityIssue.INCONSISTENT_LABELS)
                recommendations.append(f"标签一致性较低 ({consistency_metrics['consistency_score']:.2f})，建议检查标签生成逻辑")
        
        # 计算总体质量分数
        overall_quality_score = self._calculate_overall_quality(detailed_metrics, issues_detected)
        
        return ValidationResult(
            overall_quality_score=overall_quality_score,
            issues_detected=issues_detected,
            detailed_metrics=detailed_metrics,
            recommendations=recommendations,
            validation_level=self.validation_level,
            total_labels=total_labels,
            valid_labels=valid_labels
        )
    
    def _basic_validation(self, labels: List[PreferenceLabel]) -> Dict[str, float]:
        """基础验证"""
        total_labels = len(labels)
        valid_labels = sum(1 for label in labels if label.is_valid)
        
        # 计算偏好分数分布
        preference_scores = [label.preference_score for label in labels if label.is_valid]
        
        metrics = {
            'total_labels': total_labels,
            'valid_labels': valid_labels,
            'valid_ratio': valid_labels / max(total_labels, 1),
            'avg_preference_score': np.mean(preference_scores) if preference_scores else 0.0,
            'std_preference_score': np.std(preference_scores) if preference_scores else 0.0
        }
        
        return metrics
    
    def _confidence_validation(self, labels: List[PreferenceLabel]) -> Dict[str, float]:
        """置信度验证"""
        valid_labels = [label for label in labels if label.is_valid]
        
        if not valid_labels:
            return {'avg_confidence': 0.0, 'min_confidence': 0.0, 'max_confidence': 0.0}
        
        confidences = [label.metadata.confidence for label in valid_labels]
        
        metrics = {
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'std_confidence': np.std(confidences),
            'low_confidence_ratio': sum(1 for c in confidences if c < self.quality_thresholds['min_confidence']) / len(confidences)
        }
        
        return metrics
    
    def _diversity_validation(self, labels: List[PreferenceLabel]) -> Dict[str, float]:
        """多样性验证"""
        valid_labels = [label for label in labels if label.is_valid]
        
        if not valid_labels:
            return {'label_diversity': 0.0, 'type_diversity': 0.0}
        
        # 标签值多样性（基于偏好分数的分布）
        preference_scores = [label.preference_score for label in valid_labels]
        
        # 使用熵来衡量多样性
        hist, _ = np.histogram(preference_scores, bins=10, range=(0, 1))
        hist = hist / np.sum(hist)  # 归一化
        hist = hist[hist > 0]  # 移除零值
        label_diversity = -np.sum(hist * np.log(hist)) / np.log(10)  # 归一化熵
        
        # 标签类型多样性
        label_types = [label.metadata.label_type.value for label in valid_labels]
        type_counts = Counter(label_types)
        type_probs = np.array(list(type_counts.values())) / len(label_types)
        type_diversity = -np.sum(type_probs * np.log(type_probs)) / np.log(len(type_counts))
        
        metrics = {
            'label_diversity': label_diversity,
            'type_diversity': type_diversity,
            'unique_types': len(type_counts)
        }
        
        return metrics
    
    def _bias_detection(self, labels: List[PreferenceLabel]) -> Dict[str, float]:
        """偏差检测"""
        valid_labels = [label for label in labels if label.is_valid]
        
        if not valid_labels:
            return {'bias_ratio': 0.0, 'prefer_a_ratio': 0.5, 'prefer_b_ratio': 0.5}
        
        preference_scores = [label.preference_score for label in valid_labels]
        
        # 计算偏好分布
        prefer_a = sum(1 for score in preference_scores if score > 0.6)
        prefer_b = sum(1 for score in preference_scores if score < 0.4)
        neutral = len(preference_scores) - prefer_a - prefer_b
        
        prefer_a_ratio = prefer_a / len(preference_scores)
        prefer_b_ratio = prefer_b / len(preference_scores)
        neutral_ratio = neutral / len(preference_scores)
        
        # 计算偏差比例（偏离均匀分布的程度）
        expected_ratio = 1.0 / 3.0  # 期望每类占1/3
        bias_ratio = max(abs(prefer_a_ratio - expected_ratio), 
                        abs(prefer_b_ratio - expected_ratio),
                        abs(neutral_ratio - expected_ratio)) / expected_ratio
        
        metrics = {
            'bias_ratio': bias_ratio,
            'prefer_a_ratio': prefer_a_ratio,
            'prefer_b_ratio': prefer_b_ratio,
            'neutral_ratio': neutral_ratio
        }
        
        return metrics
    
    def _outlier_detection(self, labels: List[PreferenceLabel]) -> Dict[str, float]:
        """异常值检测"""
        valid_labels = [label for label in labels if label.is_valid]
        
        if len(valid_labels) < 3:
            return {'outlier_ratio': 0.0, 'outlier_count': 0}
        
        # 基于偏好分数检测异常值
        preference_scores = np.array([label.preference_score for label in valid_labels])
        
        # 使用Z-score方法
        z_scores = np.abs(stats.zscore(preference_scores))
        outliers = z_scores > self.quality_thresholds['outlier_threshold']
        
        # 基于置信度检测异常值
        confidences = np.array([label.metadata.confidence for label in valid_labels])
        confidence_z_scores = np.abs(stats.zscore(confidences))
        confidence_outliers = confidence_z_scores > self.quality_thresholds['outlier_threshold']
        
        # 合并异常值
        combined_outliers = outliers | confidence_outliers
        
        metrics = {
            'outlier_ratio': np.sum(combined_outliers) / len(valid_labels),
            'outlier_count': int(np.sum(combined_outliers)),
            'preference_outliers': int(np.sum(outliers)),
            'confidence_outliers': int(np.sum(confidence_outliers))
        }
        
        return metrics
    
    def _consistency_validation(self, labels: List[PreferenceLabel]) -> Dict[str, float]:
        """一致性验证"""
        valid_labels = [label for label in labels if label.is_valid]
        
        if len(valid_labels) < 2:
            return {'consistency_score': 1.0}
        
        # 基于质量分数差异和偏好分数的一致性
        consistency_scores = []
        
        for label in valid_labels:
            quality_diff = label.metadata.score_difference
            preference_score = label.preference_score
            
            # 检查质量差异和偏好分数是否一致
            if quality_diff > 0.1:  # 有明显质量差异
                if abs(preference_score - 0.5) > 0.1:  # 偏好分数也应该偏离中性
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)
            else:  # 质量差异很小
                if abs(preference_score - 0.5) < 0.1:  # 偏好分数应该接近中性
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.5)
        
        metrics = {
            'consistency_score': np.mean(consistency_scores) if consistency_scores else 1.0,
            'consistent_labels': sum(consistency_scores),
            'inconsistent_labels': len(consistency_scores) - sum(consistency_scores)
        }
        
        return metrics
    
    def _calculate_overall_quality(self, metrics: Dict[str, float], issues: List[QualityIssue]) -> float:
        """计算总体质量分数"""
        base_score = 1.0
        
        # 根据各项指标调整分数
        if 'valid_ratio' in metrics:
            base_score *= metrics['valid_ratio']
        
        if 'avg_confidence' in metrics:
            base_score *= min(metrics['avg_confidence'] * 2, 1.0)  # 置信度权重
        
        if 'label_diversity' in metrics:
            base_score *= min(metrics['label_diversity'] * 2, 1.0)  # 多样性权重
        
        # 根据问题数量降低分数
        issue_penalty = len(issues) * 0.1
        base_score = max(base_score - issue_penalty, 0.0)
        
        return min(base_score, 1.0)
    
    def generate_quality_report(self, validation_result: ValidationResult, save_path: str = None) -> str:
        """生成质量报告"""
        report = []
        report.append("=" * 60)
        report.append("偏好标签质量验证报告")
        report.append("=" * 60)
        report.append(f"验证级别: {validation_result.validation_level.value}")
        report.append(f"总体质量分数: {validation_result.overall_quality_score:.3f}")
        report.append(f"总标签数: {validation_result.total_labels}")
        report.append(f"有效标签数: {validation_result.valid_labels}")
        report.append("")
        
        # 检测到的问题
        if validation_result.issues_detected:
            report.append("检测到的问题:")
            for issue in validation_result.issues_detected:
                report.append(f"  - {issue.value}")
        else:
            report.append("未检测到质量问题")
        report.append("")
        
        # 详细指标
        report.append("详细指标:")
        for key, value in validation_result.detailed_metrics.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")
        
        # 改进建议
        if validation_result.recommendations:
            report.append("改进建议:")
            for i, rec in enumerate(validation_result.recommendations, 1):
                report.append(f"  {i}. {rec}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"质量报告已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return report_text
    
    def visualize_label_distribution(self, labels: List[PreferenceLabel], save_path: str = None):
        """可视化标签分布"""
        valid_labels = [label for label in labels if label.is_valid]
        
        if not valid_labels:
            logger.warning("没有有效标签可供可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('偏好标签分布分析', fontsize=16)
        
        # 1. 偏好分数分布
        preference_scores = [label.preference_score for label in valid_labels]
        axes[0, 0].hist(preference_scores, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('偏好分数分布')
        axes[0, 0].set_xlabel('偏好分数')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='中性线')
        axes[0, 0].legend()
        
        # 2. 置信度分布
        confidences = [label.metadata.confidence for label in valid_labels]
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('置信度分布')
        axes[0, 1].set_xlabel('置信度')
        axes[0, 1].set_ylabel('频次')
        
        # 3. 标签类型分布
        label_types = [label.metadata.label_type.value for label in valid_labels]
        type_counts = Counter(label_types)
        axes[1, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('标签类型分布')
        
        # 4. 质量分数差异分布
        score_diffs = [label.metadata.score_difference for label in valid_labels]
        axes[1, 1].hist(score_diffs, bins=20, alpha=0.7, color='orange')
        axes[1, 1].set_title('质量分数差异分布')
        axes[1, 1].set_xlabel('分数差异')
        axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图表已保存到: {save_path}")
        
        plt.show()

# 工具函数
def validate_preference_labels(labels: List[PreferenceLabel], 
                             validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """验证偏好标签的便捷函数"""
    validator = LabelQualityValidator(validation_level)
    return validator.validate_labels(labels)

def generate_validation_report(labels: List[PreferenceLabel], 
                             report_path: str = None,
                             validation_level: ValidationLevel = ValidationLevel.STANDARD) -> str:
    """生成验证报告的便捷函数"""
    validator = LabelQualityValidator(validation_level)
    result = validator.validate_labels(labels)
    return validator.generate_quality_report(result, report_path)

if __name__ == "__main__":
    # 测试代码
    from prm.preference_labeling_engine import create_preference_labeling_engine, LabelType
    
    print("测试标签质量验证器...")
    
    # 创建测试标签
    engine = create_preference_labeling_engine("test_task")
    
    # 生成一些测试标签
    test_labels = []
    for i in range(50):
        obs_a = np.random.randn(30, 10)
        act_a = np.random.randn(30, 5)
        obs_b = np.random.randn(25, 10)
        act_b = np.random.randn(25, 5)
        
        label = engine.generate_preference_labels(obs_a, act_a, obs_b, act_b)
        test_labels.append(label)
    
    # 验证标签质量
    validator = LabelQualityValidator(ValidationLevel.COMPREHENSIVE)
    result = validator.validate_labels(test_labels)
    
    # 生成报告
    report = validator.generate_quality_report(result)
    print(report)
    
    print("测试完成！")