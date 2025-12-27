import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

class AdaptiveThresholdManager:
    """
    自适应阈值管理器 - 根据环境奖励分布动态调整阈值参数
    解决固定阈值无法适应不同环境奖励范围的问题
    """
    
    def __init__(self, config: Dict, window_size: int = 30):
        self.config = config
        self.window_size = window_size
        
        # 奖励统计缓存
        self.reward_history = deque(maxlen=window_size)
        self.rule_score_history = deque(maxlen=window_size)
        
        # 统计信息缓存
        self._stats_cache = {}
        self._cache_valid = False
        
        self.logger = logging.getLogger(__name__)
        
        # 历史最高环境平均值跟踪
        self.historical_max_env_avg = float('-inf')  # 历史最高环境平均值
        self.current_window_rewards = deque(maxlen=window_size)  # 当前窗口奖励
        
    def add_reward_sample(self, env_reward: float, rule_score: Optional[float] = None):
        """添加奖励样本到统计缓存"""
        self.reward_history.append(env_reward)
        if rule_score is not None:
            self.rule_score_history.append(rule_score)
        
        # 更新当前窗口奖励
        self.current_window_rewards.append(env_reward)
        
        # 检查并更新历史最高环境平均值
        # 当窗口有足够样本时（至少5个）就开始计算和更新
        if len(self.current_window_rewards) >= min(5, self.window_size):
            current_window_avg = np.mean(list(self.current_window_rewards))
            if current_window_avg > self.historical_max_env_avg:
                self.historical_max_env_avg = current_window_avg
                self.logger.info(f"Updated historical max env avg: {self.historical_max_env_avg:.4f} (window size: {len(self.current_window_rewards)})")
        
        # 如果是第一个样本，直接设为历史最高
        elif len(self.current_window_rewards) == 1 and self.historical_max_env_avg == float('-inf'):
            self.historical_max_env_avg = env_reward
            self.logger.info(f"Initialized historical max env avg: {self.historical_max_env_avg:.4f}")
        
        # 清除缓存
        self._cache_valid = False
    
    def _compute_stats(self) -> Dict:
        """计算当前奖励分布统计信息"""
        if self._cache_valid and self._stats_cache:
            return self._stats_cache
        
        if len(self.reward_history) < 10:  # 最少需要10个样本
            # 使用默认统计值
            stats = {
                'reward_mean': 0.0,
                'reward_std': 1.0,
                'reward_abs_mean': 1.0,
                'rule_score_mean': 0.0,
                'rule_score_std': 1.0,
                'rule_score_abs_mean': 1.0
            }
        else:
            rewards = np.array(list(self.reward_history))
            
            stats = {
                'reward_mean': np.mean(rewards),
                'reward_std': max(np.std(rewards), 1e-6),  # 避免除零
                'reward_abs_mean': max(np.mean(np.abs(rewards)), 1e-6),
            }
            
            if len(self.rule_score_history) >= 10:
                rule_scores = np.array(list(self.rule_score_history))
                stats.update({
                    'rule_score_mean': np.mean(rule_scores),
                    'rule_score_std': max(np.std(rule_scores), 1e-6),
                    'rule_score_abs_mean': max(np.mean(np.abs(rule_scores)), 1e-6),
                })
            else:
                stats.update({
                    'rule_score_mean': 0.0,
                    'rule_score_std': 1.0,
                    'rule_score_abs_mean': 1.0,
                })
        
        self._stats_cache = stats
        self._cache_valid = True
        return stats
    
    def get_adaptive_threshold(self, threshold_type: str, **kwargs) -> float:
        """获取自适应阈值
        
        Args:
            threshold_type: 阈值类型
                - 'rule_score_diff': 规则得分差异阈值
                - 'env_reward_diff_std': 环境奖励差异（标准差倍数）
                - 'weak_signal': 弱信号阈值
                - 'strong_signal': 强信号阈值
                - 'stability': 稳定性阈值
                - 'significance': 显著性阈值
        """
        stats = self._compute_stats()
        
        if threshold_type == 'rule_score_diff':
            # 规则得分差异 = 倍数 * 环境平均奖励绝对值 + 最小阈值
            multiplier = self.config.get('rule_score_diff_multiplier', 0.5)
            min_threshold = self.config.get('rule_score_diff_min_threshold', 1.0)
            return multiplier * stats['reward_abs_mean'] + min_threshold
            
        elif threshold_type == 'env_reward_diff_std':
            # 环境奖励差异 = 标准差倍数 * 环境奖励标准差，设置最小值防止训练初期阈值过低
            multiplier = self.config.get('env_reward_diff_std_multiplier', 1.5)
            min_threshold = 0.1  # 设置最小阈值
            return max(min_threshold, multiplier * stats['reward_std'])
            
        elif threshold_type == 'weak_signal':
            # 弱信号阈值 = 标准差倍数 * 环境奖励标准差，设置最小值
            multiplier = self.config.get('weak_signal_std_multiplier', 0.1)
            min_threshold = 0.01  # 设置最小阈值
            return max(min_threshold, multiplier * stats['reward_std'])
            
        elif threshold_type == 'strong_signal':
            # 强信号阈值 = 标准差倍数 * 环境奖励标准差，设置最小值
            multiplier = self.config.get('strong_signal_std_multiplier', 1.0)
            min_threshold = 0.1  # 设置最小阈值
            return max(min_threshold, multiplier * stats['reward_std'])
            
        elif threshold_type == 'stability':
            # 稳定性阈值 = 标准差倍数 * 环境奖励标准差，设置最小值
            multiplier = self.config.get('stability_std_multiplier', 0.5)
            min_threshold = 0.05  # 设置最小阈值
            return max(min_threshold, multiplier * stats['reward_std'])
            
        elif threshold_type == 'significance':
            # 显著性阈值 = 均值倍数 * 环境平均奖励绝对值
            multiplier = self.config.get('significance_mean_multiplier', 0.1)
            return multiplier * stats['reward_abs_mean']
            
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """获取质量检测相关的所有自适应阈值"""
        return {
            'rule_score_diff': self.get_adaptive_threshold('rule_score_diff'),
            'env_reward_diff_std': self.get_adaptive_threshold('env_reward_diff_std'),
            'weak_signal': self.get_adaptive_threshold('weak_signal'),
            'strong_signal': self.get_adaptive_threshold('strong_signal'),
            'stability': self.get_adaptive_threshold('stability'),
            'significance': self.get_adaptive_threshold('significance'),
        }
    
    def is_high_quality_experience(self, 
                                 env_reward: float,
                                 rule_score: Optional[float] = None,
                                 confidence: Optional[float] = None) -> Tuple[bool, Dict[str, bool]]:
        """判断是否为高质量经验
        
        Returns:
            Tuple[bool, Dict[str, bool]]: (是否高质量, 各指标详情)
        """
        indicators = {}
        
        # 置信度检查（固定阈值）
        if confidence is not None:
            confidence_threshold = self.config.get('confidence_threshold', 0.75)
            indicators['confidence'] = confidence >= confidence_threshold
        
        # 规则得分差异检查
        if rule_score is not None:
            rule_threshold = self.get_adaptive_threshold('rule_score_diff')
            indicators['rule_score_diff'] = abs(rule_score) >= rule_threshold
        
        # 环境奖励差异检查
        stats = self._compute_stats()
        reward_diff_threshold = self.get_adaptive_threshold('env_reward_diff_std')
        reward_diff = abs(env_reward - stats['reward_mean'])
        indicators['env_reward_diff'] = reward_diff >= reward_diff_threshold
        
        # 百分位数检查
        percentile_threshold = self.config.get('env_reward_diff_percentile_threshold', 75.0)
        if len(self.reward_history) >= 10:
            percentile_value = np.percentile(list(self.reward_history), percentile_threshold)
            indicators['percentile'] = env_reward >= percentile_value
        else:
            indicators['percentile'] = True  # 样本不足时默认通过
        
        # 计算满足的指标数量
        satisfied_indicators = sum(indicators.values())
        min_indicators = self.config.get('min_quality_indicators', 2)
        
        is_high_quality = satisfied_indicators >= min_indicators
        
        return is_high_quality, indicators
    
    def get_statistics_summary(self) -> Dict:
        """获取统计信息摘要"""
        stats = self._compute_stats()
        return {
            'sample_count': len(self.reward_history),
            'mean': stats['reward_mean'],
            'std': stats['reward_std'],
            'abs_mean': stats['reward_abs_mean'],
            'current_thresholds': self.get_quality_thresholds()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.reward_history.clear()
        self.rule_score_history.clear()
        self.current_window_rewards.clear()
        self._stats_cache.clear()
        self._cache_valid = False
        self.historical_max_env_avg = float('-inf')
        self.logger.info("Adaptive threshold statistics reset")
    
    def reset(self):
        """重置统计信息"""
        self.reward_history.clear()
        self.rule_score_history.clear()
        self.current_window_rewards.clear()
        self._stats_cache.clear()
        self._cache_valid = False
        self.historical_max_env_avg = float('-inf')
    
    def get_historical_max_env_avg(self) -> float:
        """获取历史最高环境平均值"""
        return self.historical_max_env_avg if self.historical_max_env_avg != float('-inf') else 0.0
    
    def get_historical_max_threshold(self, percentage: float = 0.3) -> float:
        """获取历史最高环境平均值的百分比阈值
        
        Args:
            percentage: 百分比，默认30%
            
        Returns:
            历史最高环境平均值的百分比阈值
        """
        historical_max = self.get_historical_max_env_avg()
        return historical_max * percentage
    
    def get_confidence_threshold(self) -> float:
        """获取置信度阈值（固定值）"""
        return self.config.get('confidence_threshold', 0.75)
    
    def get_rule_score_diff_threshold(self) -> float:
        """获取规则得分差异阈值"""
        return self.get_adaptive_threshold('rule_score_diff')
    
    def get_env_reward_diff_threshold(self) -> float:
        """获取环境奖励差异阈值"""
        return self.get_adaptive_threshold('env_reward_diff_std')
    
    def get_weak_signal_threshold(self) -> float:
        """获取弱信号阈值"""
        return self.get_adaptive_threshold('weak_signal')
    
    def get_strong_signal_threshold(self) -> float:
        """获取强信号阈值"""
        return self.get_adaptive_threshold('strong_signal')
    
    def get_stability_threshold(self) -> float:
        """获取稳定性阈值"""
        return self.get_adaptive_threshold('stability')
    
    def get_significance_threshold(self) -> float:
        """获取显著性阈值"""
        return self.get_adaptive_threshold('significance')
    
    def check_quality_thresholds(self, quality_indicators: Dict[str, float]) -> bool:
        """检查质量指标是否满足阈值要求"""
        passed_indicators = 0
        
        # 检查是否为早期训练阶段
        reward_samples_count = len(self.reward_history)
        stats = self.get_statistics_summary()
        env_mean_reward = stats.get('mean', 0.0)
        is_early_training = (reward_samples_count < 50 or env_mean_reward < 5.0)
        
        # 早期训练阶段使用更宽松的标准
        if is_early_training:
            # 检查置信度阈值（降低标准）
            if 'confidence' in quality_indicators:
                confidence_threshold = max(0.5, self.get_confidence_threshold() * 0.7)  # 降低30%
                if quality_indicators['confidence'] >= confidence_threshold:
                    passed_indicators += 1
            
            # 检查规则得分差异阈值（降低标准）
            if 'rule_score_diff' in quality_indicators:
                rule_score_diff_threshold = self.get_rule_score_diff_threshold() * 0.5  # 降低50%
                if quality_indicators['rule_score_diff'] >= rule_score_diff_threshold:
                    passed_indicators += 1
            
            # 检查环境奖励差异阈值（降低标准）
            if 'env_reward_diff' in quality_indicators:
                env_reward_diff_threshold = max(0.1, self.get_env_reward_diff_threshold() * 0.3)  # 降低70%
                if quality_indicators['env_reward_diff'] >= env_reward_diff_threshold:
                    passed_indicators += 1
            
            # 早期训练阶段只需要满足1个指标即可
            min_indicators = 1
        else:
            # 正常训练阶段使用标准阈值
            # 检查置信度
            if 'confidence' in quality_indicators:
                if quality_indicators['confidence'] >= self.get_confidence_threshold():
                    passed_indicators += 1
            
            # 检查规则得分差异
            if 'rule_score_diff' in quality_indicators:
                if quality_indicators['rule_score_diff'] >= self.get_rule_score_diff_threshold():
                    passed_indicators += 1
            
            # 检查环境奖励差异
            if 'env_reward_diff' in quality_indicators:
                if quality_indicators['env_reward_diff'] >= self.get_env_reward_diff_threshold():
                    passed_indicators += 1
            
            # 正常训练阶段需要满足配置的最小指标数量
            min_indicators = self.config.get('min_quality_indicators', 2)
        
        result = passed_indicators >= min_indicators
        return result
    
    def get_stats(self) -> Dict:
        """获取当前统计信息"""
        stats = self._compute_stats()
        return {
            'sample_count': len(self.reward_history),
            'mean': stats['reward_mean'],
            'std': stats['reward_std'],
            'rule_score_mean': stats.get('rule_score_mean', 0.0),
            'rule_score_std': stats.get('rule_score_std', 1.0)
        }
    
    def get_historical_max_env_avg(self) -> float:
        """获取历史最高环境平均值"""
        return self.historical_max_env_avg if self.historical_max_env_avg != float('-inf') else 0.0
    
    def get_historical_max_threshold(self, percentage: float) -> float:
        """获取历史最高环境平均值的百分比阈值"""
        historical_max = self.get_historical_max_env_avg()
        return historical_max * percentage