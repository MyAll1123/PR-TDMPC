#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的偏好模型训练器 (Optimized Preference Trainer)

改进点：
1. 更高效的训练策略
2. 数据质量过滤
3. 动态学习率调整
4. 早停机制
5. 模型集成支持

作者：AI Assistant
日期：2025-01-11
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import time
import os
from collections import deque, defaultdict
import json

# 延迟导入以避免循环依赖
from .optimized_latent_preference_model import OptimizedLatentPreferenceModel, OptimizedLatentPreferenceConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedTrainingConfig:
    """优化的训练配置"""
    # 基础训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    
    # 学习率调度
    scheduler_type: str = "cosine_warm"  # "cosine_warm", "reduce_on_plateau", "none"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 早停机制
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # 数据质量过滤
    enable_data_filtering: bool = True
    min_score_diff: float = 0.1  # 最小分数差异
    max_uncertainty: float = 0.8  # 最大不确定性阈值
    
    # 训练频率控制
    train_every_n_episodes: int = 5  # 每N个episode训练一次
    min_data_size: int = 100  # 最小数据量
    
    # 模型集成
    enable_ensemble: bool = False
    n_ensemble_models: int = 3
    
    # 保存和日志
    save_dir: str = "./checkpoints"
    log_interval: int = 10
    save_interval: int = 50
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DataQualityFilter:
    """数据质量过滤器"""
    
    def __init__(self, config: OptimizedTrainingConfig):
        self.config = config
        self.stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'filter_reasons': defaultdict(int)
        }
    
    def filter_preference_data(self, 
                             chosen_data: List[Dict],
                             rejected_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """过滤偏好数据
        
        Args:
            chosen_data: 偏好轨迹数据
            rejected_data: 非偏好轨迹数据
            
        Returns:
            过滤后的数据
        """
        if not self.config.enable_data_filtering:
            return chosen_data, rejected_data
        
        filtered_chosen = []
        filtered_rejected = []
        
        for chosen, rejected in zip(chosen_data, rejected_data):
            self.stats['total_samples'] += 1
            
            # 检查分数差异
            chosen_score = chosen.get('reward', 0.0)
            rejected_score = rejected.get('reward', 0.0)
            score_diff = chosen_score - rejected_score
            
            if score_diff < self.config.min_score_diff:
                self.stats['filtered_samples'] += 1
                self.stats['filter_reasons']['low_score_diff'] += 1
                continue
            
            # 检查不确定性（如果有）
            chosen_uncertainty = chosen.get('uncertainty', 0.0)
            rejected_uncertainty = rejected.get('uncertainty', 0.0)
            max_uncertainty = max(chosen_uncertainty, rejected_uncertainty)
            
            if max_uncertainty > self.config.max_uncertainty:
                self.stats['filtered_samples'] += 1
                self.stats['filter_reasons']['high_uncertainty'] += 1
                continue
            
            # 检查轨迹长度
            chosen_len = len(chosen.get('latent_states', []))
            rejected_len = len(rejected.get('latent_states', []))
            
            if chosen_len < 5 or rejected_len < 5:  # 最小轨迹长度
                self.stats['filtered_samples'] += 1
                self.stats['filter_reasons']['short_trajectory'] += 1
                continue
            
            # 通过所有过滤条件
            filtered_chosen.append(chosen)
            filtered_rejected.append(rejected)
        
        filter_rate = self.stats['filtered_samples'] / max(self.stats['total_samples'], 1)
        if self.stats['total_samples'] % 100 == 0:  # 每100个样本打印一次
            logger.info(f"[数据过滤] 过滤率: {filter_rate:.2%}, 原因: {dict(self.stats['filter_reasons'])}")
        
        return filtered_chosen, filtered_rejected
    
    def get_stats(self) -> Dict[str, Any]:
        """获取过滤统计信息"""
        return dict(self.stats)

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

class OptimizedPreferenceTrainer:
    """优化的偏好模型训练器"""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        
        # 延迟导入以避免循环依赖
        from .optimized_latent_preference_model import OptimizedLatentPreferenceModel
        
        # 延迟创建模型
        if training_config.enable_ensemble:
            self.models = []
            for i in range(training_config.n_ensemble_models):
                model = OptimizedLatentPreferenceModel(model_config)
                model.to(training_config.device)
                self.models.append(model)
            # logger.info(f"[优化训练器] 创建了{len(self.models)}个集成模型")
        else:
            self.model = OptimizedLatentPreferenceModel(model_config)
            self.model.to(training_config.device)
            self.models = [self.model]
        
        # 创建优化器和调度器
        self.optimizers = []
        self.schedulers = []
        
        for model in self.models:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
            self.optimizers.append(optimizer)
            
            # 学习率调度器
            if training_config.scheduler_type == "cosine_warm":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=training_config.warmup_epochs,
                    eta_min=training_config.min_lr
                )
            elif training_config.scheduler_type == "reduce_on_plateau":
                scheduler = ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=5,
                    min_lr=training_config.min_lr
                )
            else:
                scheduler = None
            
            self.schedulers.append(scheduler)
        
        # 数据质量过滤器
        self.data_filter = DataQualityFilter(training_config)
        
        # 早停机制
        if training_config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=training_config.patience,
                min_delta=training_config.min_delta
            )
        else:
            self.early_stopping = None
        
        # 训练统计
        self.training_stats = {
            'epoch': 0,
            'total_episodes': 0,
            'training_losses': deque(maxlen=1000),
            'validation_losses': deque(maxlen=1000),
            'learning_rates': deque(maxlen=1000),
            'training_times': deque(maxlen=100)
        }
        
        # 创建保存目录
        os.makedirs(training_config.save_dir, exist_ok=True)
        
        logger.info(f"[优化训练器] 初始化完成")
        logger.info(f"  - 模型数量: {len(self.models)}")
        logger.info(f"  - 学习率: {training_config.learning_rate}")
        logger.info(f"  - 批次大小: {training_config.batch_size}")
        logger.info(f"  - 设备: {training_config.device}")
    
    def _create_preference_model(self):
        """创建偏好模型"""
        # 延迟导入
        from .optimized_latent_preference_model import OptimizedLatentPreferenceModel, OptimizedLatentPreferenceConfig
        
        config = OptimizedLatentPreferenceConfig(
            latent_dim=self.model_config.get('latent_dim', 50),
            action_dim=self.model_config.get('action_dim', 6),
            hidden_dim=self.model_config.get('hidden_dim', 256),
            num_layers=self.model_config.get('num_layers', 3),
            dropout=self.model_config.get('dropout', 0.1),
            activation=self.model_config.get('activation', 'relu'),
            device=str(self.training_config.device)
        )
        
        model = OptimizedLatentPreferenceModel(config)
        model.to(self.training_config.device)
        return model
    
    def should_train(self, episode: int, data_size: int) -> bool:
        """判断是否应该进行训练
        
        Args:
            episode: 当前episode
            data_size: 当前数据量
            
        Returns:
            是否应该训练
        """
        # 检查数据量
        if data_size < self.training_config.min_data_size:
            return False
        
        # 检查训练频率
        if episode % self.training_config.train_every_n_episodes != 0:
            return False
        
        return True
    
    def train_step(self, 
                   chosen_data: List[Dict],
                   rejected_data: List[Dict]) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            chosen_data: 偏好轨迹数据
            rejected_data: 非偏好轨迹数据
            
        Returns:
            训练损失字典
        """
        start_time = time.time()
        
        # 记录原始数据量
        original_chosen_count = len(chosen_data)
        original_rejected_count = len(rejected_data)
        
        # 数据质量过滤
        chosen_data, rejected_data = self.data_filter.filter_preference_data(
            chosen_data, rejected_data
        )
        
        filtered_chosen_count = len(chosen_data)
        filtered_rejected_count = len(rejected_data)
        
        # 详细日志：数据过滤情况
        logger.info(f"[潜空间偏好奖励模型] 数据过滤: 偏好轨迹 {original_chosen_count}->{filtered_chosen_count}, 非偏好轨迹 {original_rejected_count}->{filtered_rejected_count}")
        
        if len(chosen_data) == 0:
            logger.warning("[潜空间偏好奖励模型] 过滤后没有有效数据")
            return {'total_loss': 0.0}
        
        # 计算并输出轨迹质量分数
        self._log_trajectory_quality_scores(chosen_data, rejected_data)
        
        # 准备批次数据
        batch_losses = []
        detailed_loss_info = []
        
        for model_idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            model.train()
            optimizer.zero_grad()
            
            # 随机采样批次
            batch_size = min(self.training_config.batch_size, len(chosen_data))
            indices = np.random.choice(len(chosen_data), batch_size, replace=False)
            
            batch_chosen = [chosen_data[i] for i in indices]
            batch_rejected = [rejected_data[i] for i in indices]
            
            # 输出偏好对信息
            self._log_preference_pair_info(batch_chosen, batch_rejected, indices)
            
            # 转换为张量
            chosen_latents, chosen_actions = self._prepare_batch_data(batch_chosen)
            rejected_latents, rejected_actions = self._prepare_batch_data(batch_rejected)
            
            # 计算损失
            loss_dict = model.compute_preference_loss(
                chosen_latents, chosen_actions,
                rejected_latents, rejected_actions
            )
            
            # 反向传播
            loss = loss_dict['total_loss']
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录详细损失信息
            model_losses = {
                f'model_{model_idx}_' + k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()
            }
            batch_losses.append(model_losses)
            detailed_loss_info.append({
                'model_idx': model_idx,
                'total_loss': loss.item(),
                'batch_size': batch_size,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # 更新学习率
        for scheduler in self.schedulers:
            if scheduler is not None and self.training_config.scheduler_type != "reduce_on_plateau":
                scheduler.step()
        
        # 合并损失
        combined_losses = {}
        for key in batch_losses[0].keys():
            if 'model_' not in key:
                combined_losses[key] = np.mean([losses[key] for losses in batch_losses])
        
        # 计算平均损失
        avg_total_loss = np.mean([losses[f'model_{i}_total_loss'] for i, losses in enumerate(batch_losses)])
        combined_losses['total_loss'] = avg_total_loss
        
        # 详细日志：训练损失
        training_time = time.time() - start_time
        logger.info(f"[潜空间偏好奖励模型] 训练完成 - 总损失: {avg_total_loss:.6f}, 用时: {training_time:.3f}s")
        
        # 输出每个模型的详细损失
        for info in detailed_loss_info:
            logger.info(f"  模型{info['model_idx']}: 损失={info['total_loss']:.6f}, 批次大小={info['batch_size']}, 学习率={info['learning_rate']:.2e}")
        
        # 更新统计信息
        self.training_stats['training_losses'].append(avg_total_loss)
        self.training_stats['training_times'].append(training_time)
        self.training_stats['learning_rates'].append(self.optimizers[0].param_groups[0]['lr'])
        
        # 更新模型训练状态
        for model in self.models:
            if hasattr(model, 'update_training_state'):
                model.update_training_state()  # 不传参数，让方法自动累加1轮
        
        return combined_losses
    
    def _prepare_batch_data(self, batch_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备批次数据
        
        Args:
            batch_data: 批次数据列表
            
        Returns:
            (潜空间状态张量, 动作张量)
        """
        max_len = max(len(data['latent_states']) for data in batch_data)
        batch_size = len(batch_data)
        
        # 初始化张量
        latent_states = torch.zeros(
            batch_size, max_len, self.model_config.latent_dim,
            device=self.training_config.device
        )
        actions = torch.zeros(
            batch_size, max_len, self.model_config.action_dim,
            device=self.training_config.device
        )
        
        # 填充数据
        for i, data in enumerate(batch_data):
            seq_len = len(data['latent_states'])
            latent_states[i, :seq_len] = torch.tensor(
                data['latent_states'], device=self.training_config.device
            )
            actions[i, :seq_len] = torch.tensor(
                data['actions'], device=self.training_config.device
            )
        
        return latent_states, actions
    
    def _log_trajectory_quality_scores(self, chosen_data: List[Dict], rejected_data: List[Dict]):
        """输出轨迹质量分数等详细信息"""
        try:
            # 计算偏好轨迹的质量统计
            chosen_rewards = [data.get('reward', 0.0) for data in chosen_data]
            chosen_uncertainties = [data.get('uncertainty', 0.0) for data in chosen_data]
            chosen_lengths = [len(data.get('latent_states', [])) for data in chosen_data]
            
            # 计算非偏好轨迹的质量统计
            rejected_rewards = [data.get('reward', 0.0) for data in rejected_data]
            rejected_uncertainties = [data.get('uncertainty', 0.0) for data in rejected_data]
            rejected_lengths = [len(data.get('latent_states', [])) for data in rejected_data]
            
            # 输出偏好轨迹质量统计
            if chosen_rewards:
                logger.info(f"[轨迹质量] 偏好轨迹 ({len(chosen_data)}条):")
                logger.info(f"  奖励: 均值={np.mean(chosen_rewards):.4f}, 标准差={np.std(chosen_rewards):.4f}, 范围=[{np.min(chosen_rewards):.4f}, {np.max(chosen_rewards):.4f}]")
                logger.info(f"  不确定性: 均值={np.mean(chosen_uncertainties):.4f}, 标准差={np.std(chosen_uncertainties):.4f}")
                logger.info(f"  轨迹长度: 均值={np.mean(chosen_lengths):.1f}, 范围=[{np.min(chosen_lengths)}, {np.max(chosen_lengths)}]")
            
            # 输出非偏好轨迹质量统计
            if rejected_rewards:
                logger.info(f"[轨迹质量] 非偏好轨迹 ({len(rejected_data)}条):")
                logger.info(f"  奖励: 均值={np.mean(rejected_rewards):.4f}, 标准差={np.std(rejected_rewards):.4f}, 范围=[{np.min(rejected_rewards):.4f}, {np.max(rejected_rewards):.4f}]")
                logger.info(f"  不确定性: 均值={np.mean(rejected_uncertainties):.4f}, 标准差={np.std(rejected_uncertainties):.4f}")
                logger.info(f"  轨迹长度: 均值={np.mean(rejected_lengths):.1f}, 范围=[{np.min(rejected_lengths)}, {np.max(rejected_lengths)}]")
            
            # 输出质量对比
            if chosen_rewards and rejected_rewards:
                reward_diff = np.mean(chosen_rewards) - np.mean(rejected_rewards)
                uncertainty_diff = np.mean(chosen_uncertainties) - np.mean(rejected_uncertainties)
                logger.info(f"[轨迹质量] 偏好差异: 奖励差={reward_diff:.4f}, 不确定性差={uncertainty_diff:.4f}")
                
        except Exception as e:
            logger.warning(f"[轨迹质量] 计算质量分数时出错: {e}")
    
    def _log_preference_pair_info(self, batch_chosen: List[Dict], batch_rejected: List[Dict], indices: np.ndarray):
        """输出训练用的偏好对信息，包括优先级等详细信息"""
        try:
            # 限制输出数量，避免日志过多
            max_pairs_to_log = min(3, len(batch_chosen))
            
            logger.info(f"[偏好对信息] 本批次训练偏好对 ({len(batch_chosen)}对，显示前{max_pairs_to_log}对):")
            
            for i in range(max_pairs_to_log):
                chosen = batch_chosen[i]
                rejected = batch_rejected[i]
                pair_idx = indices[i]
                
                # 提取偏好对信息
                chosen_reward = chosen.get('reward', 0.0)
                rejected_reward = rejected.get('reward', 0.0)
                chosen_uncertainty = chosen.get('uncertainty', 0.0)
                rejected_uncertainty = rejected.get('uncertainty', 0.0)
                chosen_length = len(chosen.get('latent_states', []))
                rejected_length = len(rejected.get('latent_states', []))
                
                # 计算偏好强度和置信度
                preference_strength = abs(chosen_reward - rejected_reward)
                avg_uncertainty = (chosen_uncertainty + rejected_uncertainty) / 2
                confidence = max(0.0, 1.0 - avg_uncertainty)
                
                # 模拟优先级计算（基于偏好强度和置信度）
                priority = preference_strength * confidence + 0.1  # 基础优先级
                
                logger.info(f"  对{i+1} (索引{pair_idx}): 偏好={chosen_reward:.4f} vs {rejected_reward:.4f}, 强度={preference_strength:.4f}")
                logger.info(f"    置信度={confidence:.4f}, 优先级={priority:.4f}, 长度={chosen_length} vs {rejected_length}")
                
                # 输出额外的元数据信息
                if 'metadata' in chosen:
                    metadata = chosen['metadata']
                    if isinstance(metadata, dict):
                        quality_info = []
                        if 'quality_score_a' in metadata:
                            quality_info.append(f"质量A={metadata['quality_score_a']:.3f}")
                        if 'quality_score_b' in metadata:
                            quality_info.append(f"质量B={metadata['quality_score_b']:.3f}")
                        if 'generation_method' in metadata:
                            quality_info.append(f"方法={metadata['generation_method']}")
                        if quality_info:
                            logger.info(f"    元数据: {', '.join(quality_info)}")
                            
        except Exception as e:
            logger.warning(f"[偏好对信息] 输出偏好对信息时出错: {e}")
    
    def validate(self, val_chosen_data: List[Dict], val_rejected_data: List[Dict]) -> float:
        """验证模型
        
        Args:
            val_chosen_data: 验证集偏好数据
            val_rejected_data: 验证集非偏好数据
            
        Returns:
            验证损失
        """
        if len(val_chosen_data) == 0:
            return 0.0
        
        val_losses = []
        
        for model in self.models:
            model.eval()
            
            with torch.no_grad():
                # 准备验证数据
                chosen_latents, chosen_actions = self._prepare_batch_data(val_chosen_data)
                rejected_latents, rejected_actions = self._prepare_batch_data(val_rejected_data)
                
                # 计算验证损失
                loss_dict = model.compute_preference_loss(
                    chosen_latents, chosen_actions,
                    rejected_latents, rejected_actions
                )
                
                val_losses.append(loss_dict['total_loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        self.training_stats['validation_losses'].append(avg_val_loss)
        
        # 更新学习率（如果使用ReduceLROnPlateau）
        for scheduler in self.schedulers:
            if scheduler is not None and self.training_config.scheduler_type == "reduce_on_plateau":
                scheduler.step(avg_val_loss)
        
        return avg_val_loss
    
    def get_preference_reward(self, latent_state: torch.Tensor, action: torch.Tensor) -> Tuple[float, float]:
        """获取偏好奖励（集成预测）
        
        Args:
            latent_state: 潜空间状态
            action: 动作
            
        Returns:
            (平均偏好奖励, 平均置信度)
        """
        # 延迟初始化模型（如果需要）
        if not hasattr(self, 'models') or len(self.models) == 0:
            self.models = [self._create_preference_model()]
            
        rewards = []
        confidences = []
        
        for model in self.models:
            reward, confidence = model.get_preference_reward(latent_state, action)
            rewards.append(reward)
            confidences.append(confidence)
        
        # 如果是集成模型，计算平均值和不确定性
        if len(self.models) > 1:
            avg_reward = np.mean(rewards)
            avg_confidence = np.mean(confidences)
            # 考虑模型间的不一致性作为不确定性的一部分
            reward_std = np.std(rewards)
            adjusted_confidence = avg_confidence * (1.0 - min(reward_std, 0.5))
            return avg_reward, adjusted_confidence
        else:
            return rewards[0], confidences[0]
    
    def save_checkpoint(self, epoch: int, additional_info: Optional[Dict] = None):
        """保存检查点
        
        Args:
            epoch: 当前epoch
            additional_info: 额外信息
        """
        checkpoint = {
            'epoch': epoch,
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'training_stats': dict(self.training_stats),
            'data_filter_stats': self.data_filter.get_stats()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # 保存模型状态
        if len(self.models) == 1:
            checkpoint['model_state_dict'] = self.models[0].state_dict()
            checkpoint['optimizer_state_dict'] = self.optimizers[0].state_dict()
        else:
            checkpoint['model_state_dicts'] = [model.state_dict() for model in self.models]
            checkpoint['optimizer_state_dicts'] = [opt.state_dict() for opt in self.optimizers]
        
        # 保存文件
        save_path = os.path.join(
            self.training_config.save_dir, 
            f'optimized_preference_model_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, save_path)
        
        # 保存最新模型
        latest_path = os.path.join(self.training_config.save_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
        
        logger.info(f"[优化训练器] 保存检查点: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.training_config.device)
        
        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            self.models[0].load_state_dict(checkpoint['model_state_dict'])
            self.optimizers[0].load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                model.load_state_dict(checkpoint['model_state_dicts'][i])
                optimizer.load_state_dict(checkpoint['optimizer_state_dicts'][i])
        
        # 加载训练统计
        if 'training_stats' in checkpoint:
            self.training_stats.update(checkpoint['training_stats'])
        
        logger.info(f"[优化训练器] 加载检查点: {checkpoint_path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = dict(self.training_stats)
        stats['data_filter_stats'] = self.data_filter.get_stats()
        
        # 计算一些有用的统计量
        if len(self.training_stats['training_losses']) > 0:
            stats['avg_training_loss'] = np.mean(list(self.training_stats['training_losses']))
            stats['recent_training_loss'] = np.mean(list(self.training_stats['training_losses'])[-10:])
        
        if len(self.training_stats['validation_losses']) > 0:
            stats['avg_validation_loss'] = np.mean(list(self.training_stats['validation_losses']))
            stats['recent_validation_loss'] = np.mean(list(self.training_stats['validation_losses'])[-10:])
        
        if len(self.training_stats['training_times']) > 0:
            stats['avg_training_time'] = np.mean(list(self.training_stats['training_times']))
        
        return stats

# 工厂函数
def create_optimized_preference_trainer(
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None
) -> 'OptimizedPreferenceTrainer':
    """创建优化的偏好训练器
    
    Args:
        model_config: 模型配置
        training_config: 训练配置
        
    Returns:
        优化的训练器
    """
    if model_config is None:
        model_config = OptimizedLatentPreferenceConfig()
    
    if training_config is None:
        training_config = OptimizedTrainingConfig()
    
    trainer = OptimizedPreferenceTrainer(model_config, training_config)
    
    # logger.info("[优化训练器] 创建成功")
    return trainer

if __name__ == "__main__":
    # 测试代码
    print("优化偏好训练器测试")
    
    # 创建配置
    model_config = OptimizedLatentPreferenceConfig()
    training_config = OptimizedTrainingConfig(
        batch_size=32,
        enable_ensemble=False
    )
    
    # 创建训练器
    trainer = create_optimized_preference_trainer(model_config, training_config)
    
    # 模拟训练数据
    def create_mock_data(n_samples: int) -> Tuple[List[Dict], List[Dict]]:
        chosen_data = []
        rejected_data = []
        
        for _ in range(n_samples):
            seq_len = np.random.randint(10, 30)
            
            chosen = {
                'latent_states': np.random.randn(seq_len, model_config.latent_dim).tolist(),
                'actions': np.random.randn(seq_len, model_config.action_dim).tolist(),
                'reward': np.random.uniform(0.5, 1.0),
                'uncertainty': np.random.uniform(0.0, 0.3)
            }
            
            rejected = {
                'latent_states': np.random.randn(seq_len, model_config.latent_dim).tolist(),
                'actions': np.random.randn(seq_len, model_config.action_dim).tolist(),
                'reward': np.random.uniform(0.0, 0.5),
                'uncertainty': np.random.uniform(0.0, 0.3)
            }
            
            chosen_data.append(chosen)
            rejected_data.append(rejected)
        
        return chosen_data, rejected_data
    
    # 创建模拟数据
    chosen_data, rejected_data = create_mock_data(100)
    
    # 测试训练步骤
    loss_dict = trainer.train_step(chosen_data, rejected_data)
    print(f"训练损失: {loss_dict}")
    
    # 测试偏好奖励获取
    latent_state = torch.randn(model_config.latent_dim)
    action = torch.randn(model_config.action_dim)
    reward, confidence = trainer.get_preference_reward(latent_state, action)
    print(f"偏好奖励: {reward:.4f}, 置信度: {confidence:.4f}")
    
    # 获取训练统计
    stats = trainer.get_training_stats()
    print(f"训练统计: {stats}")
    
    print("\n优化偏好训练器测试完成！")