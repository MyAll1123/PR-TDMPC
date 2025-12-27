
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
from collections import deque
import hashlib

# 导入依赖
from grpo.reward_model import PreferenceRewardModel
from prm.unified_preference_system import UnifiedTrajectory

@dataclass
class ValueEstimationConfig:
    """价值估计配置"""
    batch_size: int = 32
    horizon: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HybridValueConfig:
    """混合价值估计配置"""
    # 基础权重
    base_mpc_weight: float = 0.7
    base_preference_weight: float = 0.3
    
    # 自适应调整
    enable_adaptive_weighting: bool = True
    enable_confidence_weighting: bool = True  # 启用基于置信度的权重调整
    quality_sensitivity: float = 0.2
    uncertainty_sensitivity: float = 0.15
    stage_sensitivity: float = 0.1
    
    # 置信度权重调整参数
    min_preference_weight: float = 0.05  # 最小偏好权重
    max_preference_weight: float = 0.45  # 最大偏好权重
    confidence_sensitivity: float = 0.8   # 置信度敏感性
    confidence_threshold_low: float = 0.3  # 低置信度阈值
    confidence_threshold_high: float = 0.8 # 高置信度阈值
    
    # 偏好计算
    preference_horizon: int = 10  # 偏好评估的时间窗口
    preference_batch_size: int = 32
    enable_preference_caching: bool = True
    
    # 不确定性量化
    enable_uncertainty_penalty: bool = True
    uncertainty_penalty_scale: float = 0.1
    confidence_threshold: float = 0.8
    
    # 性能优化
    enable_parallel_computation: bool = True
    max_workers: int = 4
    computation_timeout: float = 0.1  # 秒
    
    # 新增：价值校准参数
    enable_value_calibration: bool = True
    calibration_window: int = 100  # 校准窗口大小
    max_value_drift: float = 2.0   # 最大价值漂移
    value_decay_factor: float = 0.99  # 价值衰减因子
    
    # 新增：缓存优化参数
    enable_value_caching: bool = True
    cache_similarity_threshold: float = 0.95  # 缓存相似度阈值
    max_cache_size: int = 10000  # 最大缓存大小
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'HybridValueConfig':
        """从配置字典创建配置对象"""
        # 过滤掉不相关的配置项，特别是 checkpoint 等训练相关配置
        excluded_keys = {'checkpoint', 'eval_episodes', 'eval_freq', 'steps', 'batch_size', 
                        'reward_coef', 'value_coef', 'consistency_coef', 'lr', 'data_dir',
                        'task', 'obs', 'exp_name', 'save_freq', 'buffer_size'}
        
        # 安全地获取配置值，避免访问缺失的必需值
        filtered_config = {}
        for k in dir(cls):
            if (not k.startswith('_') and 
                hasattr(cls, k) and 
                k not in excluded_keys and
                k in config_dict):
                try:
                    filtered_config[k] = config_dict[k]
                except Exception:
                    # 跳过无法访问的配置项
                    continue
        
        return cls(**filtered_config)

class ValueCache:
    """价值估计缓存系统"""
    
    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache = {}  # key: hash, value: (z, actions, value, timestamp)
        self.access_count = {}  # 访问计数，用于LRU
        self.total_requests = 0
        self.cache_hits = 0
    
    def _compute_hash(self, z: torch.Tensor, actions: torch.Tensor) -> str:
        """计算状态-动作对的哈希值"""
        try:
            # 安全地将tensor转换为CPU，避免CUDA初始化问题
            if z.is_cuda:
                z_data = z.detach().cpu().numpy().tobytes()
            else:
                z_data = z.detach().numpy().tobytes()
                
            if actions.is_cuda:
                actions_data = actions.detach().cpu().numpy().tobytes()
            else:
                actions_data = actions.detach().numpy().tobytes()
                
            z_hash = hashlib.md5(z_data).hexdigest()[:8]
            actions_hash = hashlib.md5(actions_data).hexdigest()[:8]
        except Exception as e:
            # 如果tensor操作失败，使用tensor的形状和设备信息作为备用键
            z_hash = hashlib.md5(f"shape_{z.shape}_device_{z.device}".encode()).hexdigest()[:8]
            actions_hash = hashlib.md5(f"shape_{actions.shape}_device_{actions.device}".encode()).hexdigest()[:8]
        return f"{z_hash}_{actions_hash}"
    
    def _compute_similarity(self, z1: torch.Tensor, a1: torch.Tensor, 
                          z2: torch.Tensor, a2: torch.Tensor) -> float:
        """计算两个状态-动作对的相似度"""
        z_sim = torch.cosine_similarity(z1.flatten(), z2.flatten(), dim=0).item()
        a_sim = torch.cosine_similarity(a1.flatten(), a2.flatten(), dim=0).item()
        return (z_sim + a_sim) / 2.0
    
    def get(self, z: torch.Tensor, actions: torch.Tensor) -> Optional[torch.Tensor]:
        """从缓存获取价值估计"""
        self.total_requests += 1
        cache_key = self._compute_hash(z, actions)
        
        if cache_key in self.cache:
            cached_z, cached_actions, cached_value, timestamp = self.cache[cache_key]
            similarity = self._compute_similarity(z, actions, cached_z, cached_actions)
            
            if similarity >= self.similarity_threshold:
                self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                self.cache_hits += 1
                return cached_value
        
        return None
    
    def put(self, z: torch.Tensor, actions: torch.Tensor, value: torch.Tensor):
        """将价值估计存入缓存"""
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        cache_key = self._compute_hash(z, actions)
        self.cache[cache_key] = (z.clone(), actions.clone(), value.clone(), time.time())
        self.access_count[cache_key] = 1
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        return self.cache_hits / max(1, self.total_requests)

class AdaptiveWeightController:
    """自适应权重控制器：根据上下文动态调整融合权重"""
    
    def __init__(self, config: HybridValueConfig):
        self.config = config
        self.quality_history = deque(maxlen=50)
        self.uncertainty_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)
        # 新增：环境奖励历史
        self.env_reward_history = deque(maxlen=50)
        self.performance_tracker = deque(maxlen=50)  # 性能跟踪
    
    def get_weights(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """获取自适应权重"""
        if not self.config.enable_adaptive_weighting:
            return {
                'mpc': self.config.base_mpc_weight,
                'preference': self.config.base_preference_weight
            }
        
        if context is None:
            return {
                'mpc': self.config.base_mpc_weight,
                'preference': self.config.base_preference_weight
            }
        
        # 基于置信度的权重调整（如果启用）
        if self.config.enable_confidence_weighting and 'confidence' in context:
            confidence = context['confidence']
            pref_weight = self._compute_confidence_weight(confidence)
            
            # 新增：基于环境奖励表现的权重调整
            if 'env_reward' in context:
                pref_weight = self._adjust_weight_by_performance(pref_weight, context['env_reward'])
            
            return {
                'preference': pref_weight,
                'mpc': 1.0 - pref_weight
            }
        
        # 传统的多因子权重调整
        quality_factor = self._compute_quality_factor(context)
        uncertainty_factor = self._compute_uncertainty_factor(context)
        stage_factor = self._compute_stage_factor(context)
        
        # 综合调整因子
        total_factor = quality_factor * uncertainty_factor * stage_factor
        
        # 计算最终权重
        adjusted_pref_weight = self.config.base_preference_weight * total_factor
        adjusted_pref_weight = max(0.1, min(0.9, adjusted_pref_weight))
        
        return {
            'preference': adjusted_pref_weight,
            'mpc': 1.0 - adjusted_pref_weight
        }
    
    def _compute_quality_factor(self, context: Dict[str, Any]) -> float:
        """计算质量调整因子"""
        quality = context.get('trajectory_quality', 0.5)
        self.quality_history.append(quality)
        
        # 如果质量高，增加偏好权重
        avg_quality = np.mean(self.quality_history) if self.quality_history else 0.5
        quality_factor = 1.0 + self.config.quality_sensitivity * (quality - avg_quality)
        return max(0.5, min(1.5, quality_factor))
    
    def _compute_uncertainty_factor(self, context: Dict[str, Any]) -> float:
        """计算不确定性调整因子"""
        uncertainty = context.get('uncertainty', 0.5)
        self.uncertainty_history.append(uncertainty)
        
        # 如果不确定性高，增加偏好权重
        avg_uncertainty = np.mean(self.uncertainty_history) if self.uncertainty_history else 0.5
        uncertainty_factor = 1.0 + self.config.uncertainty_sensitivity * (uncertainty - avg_uncertainty)
        return max(0.5, min(1.5, uncertainty_factor))
    
    def _compute_stage_factor(self, context: Dict[str, Any]) -> float:
        """计算训练阶段调整因子"""
        stage = context.get('training_stage', 'mid')
        
        if stage == 'early':
            return 1.0 - self.config.stage_sensitivity  # 早期降低偏好权重
        elif stage == 'late':
            return 1.0 + self.config.stage_sensitivity  # 后期增加偏好权重
        else:
            return 1.0  # 中期保持不变
    
    def _compute_confidence_weight(self, confidence: float) -> float:
        """基于置信度计算偏好权重
        
        Args:
            confidence: 置信度值 (0.0 - 1.0)
            
        Returns:
            偏好权重 (0.05 - 0.45)
        """
        self.confidence_history.append(confidence)
        
        # 使用分段线性函数映射置信度到权重
        if confidence <= self.config.confidence_threshold_low:
            # 低置信度：使用最小权重
            weight = self.config.min_preference_weight
        elif confidence >= self.config.confidence_threshold_high:
            # 高置信度：使用最大权重
            weight = self.config.max_preference_weight
        else:
            # 中等置信度：线性插值
            confidence_range = self.config.confidence_threshold_high - self.config.confidence_threshold_low
            weight_range = self.config.max_preference_weight - self.config.min_preference_weight
            
            normalized_confidence = (confidence - self.config.confidence_threshold_low) / confidence_range
            weight = self.config.min_preference_weight + normalized_confidence * weight_range
        
        # 应用置信度敏感性调整
        if len(self.confidence_history) > 10:
            avg_confidence = np.mean(list(self.confidence_history)[-10:])
            confidence_deviation = confidence - avg_confidence
            sensitivity_adjustment = self.config.confidence_sensitivity * confidence_deviation * 0.1
            weight = weight + sensitivity_adjustment
        
        # 确保权重在合理范围内
        return max(self.config.min_preference_weight, 
                  min(self.config.max_preference_weight, weight))
    
    def _adjust_weight_by_performance(self, base_weight: float, env_reward: float) -> float:
        """基于环境奖励表现调整权重"""
        self.env_reward_history.append(env_reward)
        
        if len(self.env_reward_history) < 10:
            return base_weight
        
        # 计算环境奖励趋势
        recent_rewards = list(self.env_reward_history)[-10:]
        early_rewards = list(self.env_reward_history)[-20:-10] if len(self.env_reward_history) >= 20 else recent_rewards
        
        recent_avg = np.mean(recent_rewards)
        early_avg = np.mean(early_rewards)
        
        # 如果环境奖励在下降或停滞，降低偏好权重
        if recent_avg <= early_avg * 1.01:  # 几乎没有改善
            adjustment = -0.1  # 降低偏好权重
        elif recent_avg > early_avg * 1.1:  # 显著改善
            adjustment = 0.05   # 略微增加偏好权重
        else:
            adjustment = 0.0    # 保持不变
        
        adjusted_weight = base_weight + adjustment
        return max(self.config.min_preference_weight, 
                  min(self.config.max_preference_weight, adjusted_weight))

class ImprovedUncertaintyEstimator:
    """改进的不确定性量化器"""
    
    def __init__(self, config: HybridValueConfig):
        self.config = config
        # 使用滑动窗口替代无限增长的历史
        self.value_history = deque(maxlen=30)  # 减小窗口大小
        self.calibration_history = deque(maxlen=config.calibration_window)
        self.last_reset_time = time.time()
        
    def estimate(self, z: torch.Tensor, actions: torch.Tensor, 
                values: Optional[torch.Tensor] = None) -> float:
        """估计不确定性"""
        if values is not None:
            # 基于值的方差估计不确定性
            uncertainty = torch.std(values).item()
        else:
            # 基于历史数据估计不确定性
            if len(self.value_history) > 10:
                # 使用最近的数据计算不确定性
                recent_values = list(self.value_history)[-30:]  # 只使用最近30个值
                uncertainty = np.std(recent_values)
            else:
                uncertainty = 0.5  # 默认不确定性
        
        return min(1.0, max(0.0, uncertainty))
    
    def apply_uncertainty_penalty(self, value: torch.Tensor, uncertainty: float) -> torch.Tensor:
        """应用不确定性惩罚"""
        if not self.config.enable_uncertainty_penalty:
            return value
        
        penalty = self.config.uncertainty_penalty_scale * uncertainty
        return value - penalty
    
    def update_history(self, value: float):
        """更新价值历史（带衰减）"""
        # 应用价值衰减
        decayed_value = value * self.config.value_decay_factor
        self.value_history.append(decayed_value)
        
        # 定期重置历史（防止长期漂移）
        current_time = time.time()
        if current_time - self.last_reset_time > 3600:  # 每小时重置一次
            self._reset_history()
            self.last_reset_time = current_time
    
    def _reset_history(self):
        """重置价值历史"""
        if len(self.value_history) > 30:
            # 保留最近的一部分数据
            recent_values = list(self.value_history)[-15:]
            self.value_history.clear()
            self.value_history.extend(recent_values)
            logger.info("[改进不确定性估计器] 价值历史已重置")
    
    def calibrate_with_env_reward(self, hybrid_value: float, env_reward: float):
        """使用环境奖励校准价值估计"""
        if not self.config.enable_value_calibration:
            return
        
        self.calibration_history.append((hybrid_value, env_reward))
        
        if len(self.calibration_history) >= self.config.calibration_window:
            # 检查价值漂移
            hybrid_values = [h for h, e in self.calibration_history]
            env_rewards = [e for h, e in self.calibration_history]
            
            hybrid_trend = np.polyfit(range(len(hybrid_values)), hybrid_values, 1)[0]
            env_trend = np.polyfit(range(len(env_rewards)), env_rewards, 1)[0]
            
            # 如果混合价值趋势与环境奖励趋势差异过大，发出警告
            if abs(hybrid_trend - env_trend) > self.config.max_value_drift:
                logger.warning(f"[价值校准] 检测到价值漂移: 混合价值趋势={hybrid_trend:.4f}, 环境奖励趋势={env_trend:.4f}")

class HybridValueEstimator:
    """改进的混合价值估计器：融合MPC原生价值和偏好奖励"""
    
    def __init__(self, 
                 mpc_agent,
                 preference_model: Optional[PreferenceRewardModel] = None,
                 config: Optional[HybridValueConfig] = None):
        """
        初始化混合价值估计器
        
        Args:
            mpc_agent: TD-MPC2智能体
            preference_model: 偏好奖励模型
            config: 混合价值估计配置
        """
        self.mpc_agent = mpc_agent
        self.preference_model = preference_model
        self.config = config or HybridValueConfig()
        
        # 初始化组件
        self.weight_controller = AdaptiveWeightController(self.config)
        self.uncertainty_estimator = ImprovedUncertaintyEstimator(self.config)
        
        # 初始化缓存系统
        if self.config.enable_value_caching:
            self.value_cache = ValueCache(
                max_size=self.config.max_cache_size,
                similarity_threshold=self.config.cache_similarity_threshold
            )
        else:
            self.value_cache = None
        
        # 初始化偏好价值估计器（如果有偏好模型）
        if self.preference_model is not None:
            value_config = ValueEstimationConfig()
            self.preference_estimator = self._create_simple_preference_estimator(value_config)
        else:
            self.preference_estimator = None
        
        # 传统缓存系统（保持兼容性）
        self.preference_cache = {} if self.config.enable_preference_caching else None
        
        # 统计信息
        self.stats = {
            'total_estimates': 0,
            'hybrid_estimates': 0,
            'mpc_estimates': 0,
            'preference_estimates': 0,
            'cache_hits': 0,
            'average_computation_time': 0.0,
            'calibration_warnings': 0
        }
        
        # 滑动窗口平均（替代全局累计平均）
        self.sliding_window_values = deque(maxlen=30)
        
        logger.info(f"[改进混合价值估计器] 初始化完成")
        logger.info(f"  - 价值缓存: {'启用' if self.config.enable_value_caching else '禁用'}")
        logger.info(f"  - 价值校准: {'启用' if self.config.enable_value_calibration else '禁用'}")
        logger.info(f"  - 置信度权重调整: {'启用' if self.config.enable_confidence_weighting else '禁用'}")
    
    def estimate_hybrid_value(self, 
                            z: torch.Tensor, 
                            actions: torch.Tensor, 
                            task: int,
                            context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """计算混合价值
        
        Args:
            z: 潜在状态
            actions: 动作序列
            task: 任务ID
            context: 上下文信息
            
        Returns:
            混合价值估计
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            if self.value_cache is not None:
                cached_value = self.value_cache.get(z, actions)
                if cached_value is not None:
                    self.stats['cache_hits'] += 1
                    return cached_value
            
            # 1. MPC原生价值估计
            mpc_value = self.mpc_agent._estimate_value(z, actions, task)
            self.stats['mpc_estimates'] += 1
            
            # 2. 偏好价值估计和置信度计算
            confidence = 0.5  # 默认置信度
            if self.preference_estimator is not None:
                pref_value = self._compute_preference_value(z, actions, task)
                self.stats['preference_estimates'] += 1
                
                # 计算置信度
                if hasattr(self.preference_model, 'get_confidence'):
                    try:
                        confidence = self.preference_model.get_confidence(z, actions)
                        if isinstance(confidence, torch.Tensor):
                            confidence = confidence.mean().item()
                    except Exception as e:
                        logger.warning(f"[混合价值估计器] 置信度计算失败: {e}")
                        confidence = 0.5
                else:
                    # 基于偏好价值的稳定性估计置信度
                    if pref_value.numel() > 1:
                        pref_std = torch.std(pref_value).item()
                        # 标准差越小，置信度越高
                        confidence = max(0.1, min(0.95, 1.0 - pref_std))
                    else:
                        confidence = 0.5
            else:
                pref_value = torch.zeros_like(mpc_value)
            
            # 3. 自适应权重计算（传递更多上下文信息）
            if context is None:
                context = {}
            context['confidence'] = confidence
            
            # 添加环境奖励信息用于权重调整
            if 'env_reward' not in context and len(self.weight_controller.env_reward_history) > 0:
                context['env_reward'] = self.weight_controller.env_reward_history[-1]
            
            weights = self.weight_controller.get_weights(context)
            
            # 4. 融合价值
            # 确保MPC价值和偏好价值都是一维张量
            if mpc_value.dim() > 1:
                mpc_value = mpc_value.squeeze()
            if pref_value.dim() > 1:
                pref_value = pref_value.squeeze()
            
            # 确保两个价值张量都是一维且长度匹配
            if mpc_value.dim() != 1:
                mpc_value = mpc_value.flatten()
            if pref_value.dim() != 1:
                pref_value = pref_value.flatten()
            
            # 确保长度匹配
            if mpc_value.shape[0] != pref_value.shape[0]:
                min_len = min(mpc_value.shape[0], pref_value.shape[0])
                mpc_value = mpc_value[:min_len]
                pref_value = pref_value[:min_len]
            
            # 修改价值函数计算方式：集成奖励 = 环境权重 × (1 + 偏好奖励)
            # 偏好奖励范围限制在(-0.3, 0.3)
            # 首先将偏好价值标准化到(-0.3, 0.3)范围
            if pref_value.numel() > 0:
                # 将偏好价值标准化到(-0.3, 0.3)范围
                pref_value_normalized = torch.tanh(pref_value) * 0.3
            else:
                pref_value_normalized = torch.zeros_like(mpc_value)
            
            # 新的价值函数计算：集成奖励 = 环境权重 × (1 + 偏好奖励)
            hybrid_value = weights['mpc'] * (1.0 + pref_value_normalized)
            
            # 5. 不确定性调整
            if self.config.enable_uncertainty_penalty:
                uncertainty = self.uncertainty_estimator.estimate(z, actions, hybrid_value)
                hybrid_value = self.uncertainty_estimator.apply_uncertainty_penalty(
                    hybrid_value, uncertainty
                )
            
            # 6. 价值校准（新增）
            if self.config.enable_value_calibration and 'env_reward' in context:
                self.uncertainty_estimator.calibrate_with_env_reward(
                    hybrid_value.mean().item(), context['env_reward']
                )
            
            # 更新统计信息
            self.stats['total_estimates'] += 1
            self.stats['hybrid_estimates'] += 1
            computation_time = time.time() - start_time
            self.stats['average_computation_time'] = (
                (self.stats['average_computation_time'] * (self.stats['total_estimates'] - 1) + 
                 computation_time) / self.stats['total_estimates']
            )
            
            # 更新滑动窗口平均
            current_value = hybrid_value.mean().item()
            self.sliding_window_values.append(current_value)
            
            # 精简日志输出（每1000次估计记录一次）
            if self.stats['total_estimates'] % 1000 == 0:
                sliding_avg = np.mean(self.sliding_window_values) if self.sliding_window_values else 0.0
                cache_hit_rate = self.value_cache.get_hit_rate() if self.value_cache else 0.0
                logger.info(f"[改进混合价值估计器] 第{self.stats['total_estimates']}次估计 - 混合价值: {current_value:.4f}, 滑动平均: {sliding_avg:.4f}, 缓存命中率: {cache_hit_rate:.2%}")
            
            # 更新不确定性历史（使用改进的方法）
            self.uncertainty_estimator.update_history(current_value)
            
            # 存入缓存
            if self.value_cache is not None:
                self.value_cache.put(z, actions, hybrid_value)
            
            return hybrid_value
            
        except Exception as e:
            logger.error(f"[混合价值估计器] 价值估计失败: {e}")
            # 回退到MPC原生估计
            return self.mpc_agent._estimate_value(z, actions, task)
    
    def _compute_preference_value(self, 
                                z: torch.Tensor, 
                                actions: torch.Tensor, 
                                task: int) -> torch.Tensor:
        """计算偏好价值"""
        if self.preference_estimator is None:
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        
        try:
            # 使用偏好估计器计算价值
            with torch.no_grad():
                preference_value = self.preference_estimator.estimate_value(
                    z, actions, task
                )
            return preference_value
        except Exception as e:
            logger.warning(f"[混合价值估计器] 偏好价值计算失败: {e}")
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    
    def _create_simple_preference_estimator(self, config: ValueEstimationConfig):
        """创建简单的偏好价值估计器"""
        class SimplePreferenceEstimator:
            def __init__(self, preference_model, config):
                self.preference_model = preference_model
                self.config = config
            
            def estimate_value(self, z, actions, task):
                # 简化的偏好价值估计
                batch_size = z.shape[0]
                horizon = min(actions.shape[1] if actions.dim() > 1 else 1, self.config.horizon)
                
                total_value = torch.zeros(batch_size, device=z.device)
                
                for t in range(horizon):
                    if actions.dim() > 1 and t < actions.shape[1]:
                        action = actions[:, t]
                    else:
                        action = actions
                    
                    # 获取偏好奖励
                    try:
                        pref_reward, _ = self.preference_model.get_preference_reward(
                            z, action
                        )
                        if isinstance(pref_reward, torch.Tensor):
                            total_value += pref_reward.squeeze()
                        else:
                            total_value += torch.tensor(pref_reward, device=z.device)
                    except Exception:
                        # 如果偏好奖励计算失败，使用零值
                        pass
                
                return total_value
        
        return SimplePreferenceEstimator(self.preference_model, config)
    
    def get_sliding_window_average(self) -> float:
        """获取滑动窗口平均值"""
        return np.mean(self.sliding_window_values) if self.sliding_window_values else 0.0
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'total_estimates': 0,
            'hybrid_estimates': 0,
            'mpc_estimates': 0,
            'preference_estimates': 0,
            'cache_hits': 0,
            'average_computation_time': 0.0,
            'calibration_warnings': 0
        }
        self.sliding_window_values.clear()
        if self.value_cache is not None:
            self.value_cache.cache.clear()
            self.value_cache.access_count.clear()
            self.value_cache.total_requests = 0
            self.value_cache.cache_hits = 0
        logger.info("[改进混合价值估计器] 统计信息已重置")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['sliding_window_average'] = self.get_sliding_window_average()
        if self.value_cache is not None:
            stats['cache_hit_rate'] = self.value_cache.get_hit_rate()
            stats['cache_size'] = len(self.value_cache.cache)
        return stats
    
    def clear_cache(self):
        """清除缓存"""
        if self.preference_cache is not None:
            self.preference_cache.clear()
        if self.value_cache is not None:
            self.value_cache.cache.clear()
            self.value_cache.access_count.clear()
        logger.info("[改进混合价值估计器] 缓存已清除")
    
    def update_config(self, new_config: HybridValueConfig):
        """更新配置"""
        self.config = new_config
        self.weight_controller = AdaptiveWeightController(new_config)
        self.uncertainty_estimator = ImprovedUncertaintyEstimator(new_config)
        logger.info("[改进混合价值估计器] 配置已更新")
