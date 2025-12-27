import torch
import torch.nn as nn
import numpy as np
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# 导入现有模块
from grpo.reward_model import PreferenceRewardModel
from prm.unified_preference_system import UnifiedTrajectory
from prm.trajectory_metrics import TrajectoryQualityEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryGenerationConfig:
    """轨迹生成配置"""
    total_trajectories: int = 256  # 总轨迹数量
    elite_trajectories: int = 64   # 精英轨迹数量
    max_trajectory_length: int = 1000  # 最大轨迹长度
    min_trajectory_length: int = 10    # 最小轨迹长度
    
    # 生成策略参数
    exploration_noise_std: float = 0.1  # 探索噪声标准差
    temperature: float = 1.0            # 采样温度
    diversity_weight: float = 0.2       # 多样性权重
    
    # 精英选择参数
    selection_method: str = "reward_based"  # 选择方法: reward_based, quality_based, hybrid
    quality_threshold: float = 0.3      # 质量阈值
    reward_weight: float = 0.7          # 奖励权重（混合选择时）
    quality_weight: float = 0.3         # 质量权重（混合选择时）
    
    # 并行处理参数
    max_workers: int = 4                # 最大工作线程数
    batch_size: int = 32                # 批处理大小
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'TrajectoryGenerationConfig':
        """从配置字典创建配置实例
        
        Args:
            config_dict: 包含elite_trajectory_system配置的字典
            
        Returns:
            TrajectoryGenerationConfig实例
        """
        # 获取精英轨迹系统配置部分
        elite_config = config_dict.get('elite_trajectory_system', {})
        
        # 创建配置实例，使用配置文件中的值覆盖默认值
        return cls(
            # 轨迹生成配置
            total_trajectories=elite_config.get('total_trajectories', 256),
            elite_trajectories=elite_config.get('elite_trajectories', 64),
            max_trajectory_length=elite_config.get('max_trajectory_length', 1000),
            min_trajectory_length=elite_config.get('min_trajectory_length', 10),
            
            # 生成策略参数
            exploration_noise_std=elite_config.get('exploration_noise_std', 0.1),
            temperature=elite_config.get('temperature', 1.0),
            diversity_weight=elite_config.get('diversity_weight', 0.2),
            
            # 精英选择参数
            selection_method=elite_config.get('selection_method', "reward_based"),
            quality_threshold=elite_config.get('quality_threshold', 0.3),
            reward_weight=elite_config.get('reward_weight', 0.7),
            quality_weight=elite_config.get('quality_weight', 0.3),
            
            # 并行处理参数
            max_workers=elite_config.get('max_workers', 4),
            batch_size=elite_config.get('batch_size', 32),
            
            # 设备配置
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

class TrajectoryGenerator:
    """轨迹生成器
    
    基于现有偏好奖励模型生成高质量轨迹序列，并选出精英序列
    """
    
    def __init__(self, 
                 reward_model: PreferenceRewardModel,
                 env,
                 agent,
                 config: TrajectoryGenerationConfig = None,
                 task_name: str = "default"):
        """
        初始化轨迹生成器
        
        Args:
            reward_model: 训练好的偏好奖励模型
            env: 环境实例
            agent: 智能体实例
            config: 生成配置
            task_name: 任务名称
        """
        self.reward_model = reward_model
        self.env = env
        self.agent = agent
        self.config = config or TrajectoryGenerationConfig()
        self.task_name = task_name
        
        # 设置设备
        self.device = torch.device(self.config.device)
        self.reward_model.to(self.device)
        self.reward_model.eval()
        
        # 初始化质量评估器
        self.quality_evaluator = TrajectoryQualityEvaluator(task_name=task_name)
        
        # 统计信息
        self.stats = {
            'total_generated': 0,
            'successful_trajectories': 0,
            'elite_trajectories': 0,
            'generation_time': 0.0,
            'selection_time': 0.0,
            'average_reward': 0.0,
            'average_quality': 0.0
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        logger.info(f"[轨迹生成器] 初始化完成 - 任务: {task_name}, 设备: {self.device}")
    
    def generate_trajectories(self) -> Tuple[List[UnifiedTrajectory], List[UnifiedTrajectory]]:
        """生成轨迹并选出精英序列
        
        Returns:
            Tuple[所有轨迹列表, 精英轨迹列表]
        """
        logger.info(f"[轨迹生成器] 开始生成 {self.config.total_trajectories} 条轨迹...")
        start_time = time.time()
        
        # 生成所有轨迹
        all_trajectories = self._generate_all_trajectories()
        
        generation_time = time.time() - start_time
        self.stats['generation_time'] = generation_time
        
        logger.info(f"[轨迹生成器] 轨迹生成完成，耗时: {generation_time:.2f}s")
        logger.info(f"[轨迹生成器] 成功生成 {len(all_trajectories)} 条轨迹")
        
        # 选择精英轨迹
        start_time = time.time()
        elite_trajectories = self._select_elite_trajectories(all_trajectories)
        selection_time = time.time() - start_time
        self.stats['selection_time'] = selection_time
        
        logger.info(f"[轨迹生成器] 精英选择完成，耗时: {selection_time:.2f}s")
        logger.info(f"[轨迹生成器] 选出 {len(elite_trajectories)} 条精英轨迹")
        
        # 更新统计信息
        self._update_stats(all_trajectories, elite_trajectories)
        
        return all_trajectories, elite_trajectories
    
    def _generate_all_trajectories(self) -> List[UnifiedTrajectory]:
        """生成所有轨迹"""
        trajectories = []
        
        # 使用多线程并行生成
        if self.config.max_workers > 1:
            trajectories = self._generate_trajectories_parallel()
        else:
            trajectories = self._generate_trajectories_sequential()
        
        # 过滤有效轨迹
        valid_trajectories = [t for t in trajectories if self._is_valid_trajectory(t)]
        
        logger.info(f"[轨迹生成器] 有效轨迹: {len(valid_trajectories)}/{len(trajectories)}")
        
        return valid_trajectories
    
    def _generate_trajectories_parallel(self) -> List[UnifiedTrajectory]:
        """并行生成轨迹"""
        trajectories = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交任务
            futures = []
            batch_size = max(1, self.config.total_trajectories // self.config.max_workers)
            
            for i in range(0, self.config.total_trajectories, batch_size):
                end_idx = min(i + batch_size, self.config.total_trajectories)
                future = executor.submit(self._generate_trajectory_batch, i, end_idx)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    batch_trajectories = future.result()
                    trajectories.extend(batch_trajectories)
                except Exception as e:
                    logger.error(f"[轨迹生成器] 批次生成失败: {e}")
        
        return trajectories
    
    def _generate_trajectories_sequential(self) -> List[UnifiedTrajectory]:
        """顺序生成轨迹"""
        return self._generate_trajectory_batch(0, self.config.total_trajectories)
    
    def _generate_trajectory_batch(self, start_idx: int, end_idx: int) -> List[UnifiedTrajectory]:
        """生成一批轨迹"""
        trajectories = []
        
        for i in range(start_idx, end_idx):
            try:
                trajectory = self._generate_single_trajectory(i)
                if trajectory is not None:
                    trajectories.append(trajectory)
                    
                    with self.lock:
                        self.stats['total_generated'] += 1
                        
                    if (i + 1) % 50 == 0:
                        logger.info(f"[轨迹生成器] 已生成 {i + 1}/{self.config.total_trajectories} 条轨迹")
                        
            except Exception as e:
                logger.warning(f"[轨迹生成器] 生成轨迹 {i} 失败: {e}")
        
        return trajectories
    
    def _generate_single_trajectory(self, trajectory_idx: int) -> Optional[UnifiedTrajectory]:
        """生成单条轨迹"""
        try:
            # 重置环境
            obs, _ = self.env.reset()
            
            # 轨迹数据
            obs_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []
            
            done = False
            steps = 0
            total_env_reward = 0.0
            
            while not done and steps < self.config.max_trajectory_length:
                # 智能体选择动作（添加探索噪声）
                action = self._get_action_with_exploration(obs, steps)
                
                # 执行动作
                next_obs, env_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 记录数据
                obs_sequence.append(obs.copy())
                action_sequence.append(action.copy())
                reward_sequence.append(env_reward)
                done_sequence.append(done)
                
                total_env_reward += env_reward
                obs = next_obs
                steps += 1
            
            # 检查轨迹长度
            if steps < self.config.min_trajectory_length:
                return None
            
            # 转换为numpy数组
            obs_array = np.array(obs_sequence)
            action_array = np.array(action_sequence)
            reward_array = np.array(reward_sequence)
            done_array = np.array(done_sequence, dtype=bool)
            
            # 计算偏好奖励
            preference_reward = self._compute_preference_reward(obs_array, action_array)
            
            # 计算质量分数
            quality_score, quality_features = self.quality_evaluator.evaluate_trajectory_quality(
                obs_array, action_array, reward_array
            )
            
            # 创建统一轨迹对象
            trajectory = UnifiedTrajectory(
                trajectory_id=f"gen_{trajectory_idx}_{int(time.time() * 1000000) % 1000000}",
                obs_sequence=obs_array,
                action_sequence=action_array,
                reward_sequence=reward_array,
                done_sequence=done_array,
                episode_idx=trajectory_idx,
                step_range=(0, steps),
                total_reward=total_env_reward,
                length=steps,
                quality_score=quality_score,
                dpo_reward_estimate=preference_reward,
                preference_features=quality_features
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"[轨迹生成器] 生成轨迹 {trajectory_idx} 时出错: {e}")
            return None
    
    def _get_action_with_exploration(self, obs: np.ndarray, step: int) -> np.ndarray:
        """获取带探索噪声的动作"""
        # 基础动作
        action = self.agent.act(obs, t0=(step == 0), eval_mode=False)
        
        # 添加探索噪声
        if self.config.exploration_noise_std > 0:
            noise = np.random.normal(0, self.config.exploration_noise_std, action.shape)
            action = action + noise
            
            # 确保动作在有效范围内
            if hasattr(self.env, 'action_space'):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        return action
    
    def _compute_preference_reward(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> float:
        """计算轨迹的偏好奖励"""
        try:
            with torch.no_grad():
                # 转换为张量
                obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)  # [1, T, obs_dim]
                action_tensor = torch.FloatTensor(action_seq).unsqueeze(0).to(self.device)  # [1, T, action_dim]
                
                # 计算偏好奖励
                reward = self.reward_model(obs_tensor, action_tensor)
                return float(reward.item())
                
        except Exception as e:
            logger.warning(f"[轨迹生成器] 计算偏好奖励失败: {e}")
            return 0.0
    
    def _is_valid_trajectory(self, trajectory: UnifiedTrajectory) -> bool:
        """检查轨迹是否有效"""
        if trajectory is None:
            return False
        
        # 检查长度
        if trajectory.length < self.config.min_trajectory_length:
            return False
        
        # 检查数据完整性
        if (len(trajectory.obs_sequence) != trajectory.length or
            len(trajectory.action_sequence) != trajectory.length or
            len(trajectory.reward_sequence) != trajectory.length):
            return False
        
        # 检查数据有效性
        if (np.any(np.isnan(trajectory.obs_sequence)) or
            np.any(np.isnan(trajectory.action_sequence)) or
            np.any(np.isnan(trajectory.reward_sequence))):
            return False
        
        return True
    
    def _select_elite_trajectories(self, trajectories: List[UnifiedTrajectory]) -> List[UnifiedTrajectory]:
        """选择精英轨迹"""
        if len(trajectories) <= self.config.elite_trajectories:
            logger.warning(f"[轨迹生成器] 轨迹数量不足，返回所有轨迹")
            return trajectories
        
        # 根据选择方法进行排序
        if self.config.selection_method == "reward_based":
            sorted_trajectories = self._sort_by_reward(trajectories)
        elif self.config.selection_method == "quality_based":
            sorted_trajectories = self._sort_by_quality(trajectories)
        elif self.config.selection_method == "hybrid":
            sorted_trajectories = self._sort_by_hybrid_score(trajectories)
        else:
            raise ValueError(f"未知的选择方法: {self.config.selection_method}")
        
        # 选择前N个
        elite_trajectories = sorted_trajectories[:self.config.elite_trajectories]
        
        logger.info(f"[轨迹生成器] 精英选择方法: {self.config.selection_method}")
        logger.info(f"[轨迹生成器] 精英轨迹平均奖励: {np.mean([t.dpo_reward_estimate for t in elite_trajectories]):.4f}")
        logger.info(f"[轨迹生成器] 精英轨迹平均质量: {np.mean([t.quality_score for t in elite_trajectories]):.4f}")
        
        return elite_trajectories
    
    def _sort_by_reward(self, trajectories: List[UnifiedTrajectory]) -> List[UnifiedTrajectory]:
        """按偏好奖励排序"""
        return sorted(trajectories, key=lambda t: t.dpo_reward_estimate, reverse=True)
    
    def _sort_by_quality(self, trajectories: List[UnifiedTrajectory]) -> List[UnifiedTrajectory]:
        """按质量分数排序"""
        return sorted(trajectories, key=lambda t: t.quality_score, reverse=True)
    
    def _sort_by_hybrid_score(self, trajectories: List[UnifiedTrajectory]) -> List[UnifiedTrajectory]:
        """按混合分数排序"""
        # 计算混合分数
        for trajectory in trajectories:
            reward_score = trajectory.dpo_reward_estimate
            quality_score = trajectory.quality_score
            
            # 归一化分数（简单的min-max归一化）
            reward_scores = [t.dpo_reward_estimate for t in trajectories]
            quality_scores = [t.quality_score for t in trajectories]
            
            reward_min, reward_max = min(reward_scores), max(reward_scores)
            quality_min, quality_max = min(quality_scores), max(quality_scores)
            
            if reward_max > reward_min:
                norm_reward = (reward_score - reward_min) / (reward_max - reward_min)
            else:
                norm_reward = 0.5
            
            if quality_max > quality_min:
                norm_quality = (quality_score - quality_min) / (quality_max - quality_min)
            else:
                norm_quality = 0.5
            
            # 计算混合分数
            hybrid_score = (self.config.reward_weight * norm_reward + 
                          self.config.quality_weight * norm_quality)
            
            # 临时存储混合分数
            trajectory._hybrid_score = hybrid_score
        
        # 按混合分数排序
        sorted_trajectories = sorted(trajectories, key=lambda t: t._hybrid_score, reverse=True)
        
        # 清理临时属性
        for trajectory in trajectories:
            if hasattr(trajectory, '_hybrid_score'):
                delattr(trajectory, '_hybrid_score')
        
        return sorted_trajectories
    
    def _update_stats(self, all_trajectories: List[UnifiedTrajectory], elite_trajectories: List[UnifiedTrajectory]):
        """更新统计信息"""
        self.stats['successful_trajectories'] = len(all_trajectories)
        self.stats['elite_trajectories'] = len(elite_trajectories)
        
        if all_trajectories:
            self.stats['average_reward'] = np.mean([t.dpo_reward_estimate for t in all_trajectories])
            self.stats['average_quality'] = np.mean([t.quality_score for t in all_trajectories])
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def save_trajectories(self, trajectories: List[UnifiedTrajectory], save_path: str):
        """保存轨迹到文件"""
        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 转换为可序列化的格式
            trajectory_data = []
            for traj in trajectories:
                data = {
                    'trajectory_id': traj.trajectory_id,
                    'obs_sequence': traj.obs_sequence.tolist(),
                    'action_sequence': traj.action_sequence.tolist(),
                    'reward_sequence': traj.reward_sequence.tolist(),
                    'done_sequence': traj.done_sequence.tolist(),
                    'episode_idx': traj.episode_idx,
                    'step_range': traj.step_range,
                    'total_reward': traj.total_reward,
                    'length': traj.length,
                    'quality_score': traj.quality_score,
                    'dpo_reward_estimate': traj.dpo_reward_estimate,
                    'preference_features': traj.preference_features
                }
                trajectory_data.append(data)
            
            # 保存到文件
            import json
            with open(save_path, 'w') as f:
                json.dump({
                    'task_name': self.task_name,
                    'config': {
                        'total_trajectories': self.config.total_trajectories,
                        'elite_trajectories': self.config.elite_trajectories,
                        'selection_method': self.config.selection_method
                    },
                    'stats': self.stats,
                    'trajectories': trajectory_data
                }, f, indent=2)
            
            logger.info(f"[轨迹生成器] 轨迹已保存到: {save_path}")
            
        except Exception as e:
            logger.error(f"[轨迹生成器] 保存轨迹失败: {e}")
    
    def load_trajectories(self, load_path: str) -> List[UnifiedTrajectory]:
        """从文件加载轨迹"""
        try:
            import json
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            trajectories = []
            for traj_data in data['trajectories']:
                trajectory = UnifiedTrajectory(
                    trajectory_id=traj_data['trajectory_id'],
                    obs_sequence=np.array(traj_data['obs_sequence']),
                    action_sequence=np.array(traj_data['action_sequence']),
                    reward_sequence=np.array(traj_data['reward_sequence']),
                    done_sequence=np.array(traj_data['done_sequence'], dtype=bool),
                    episode_idx=traj_data['episode_idx'],
                    step_range=tuple(traj_data['step_range']),
                    total_reward=traj_data['total_reward'],
                    length=traj_data['length'],
                    quality_score=traj_data['quality_score'],
                    dpo_reward_estimate=traj_data['dpo_reward_estimate'],
                    preference_features=traj_data['preference_features']
                )
                trajectories.append(trajectory)
            
            logger.info(f"[轨迹生成器] 从 {load_path} 加载了 {len(trajectories)} 条轨迹")
            return trajectories
            
        except Exception as e:
            logger.error(f"[轨迹生成器] 加载轨迹失败: {e}")
            return []


def create_trajectory_generator(reward_model: PreferenceRewardModel,
                              env,
                              agent,
                              task_name: str = "default",
                              **config_kwargs) -> TrajectoryGenerator:
    """创建轨迹生成器的工厂函数
    
    Args:
        reward_model: 偏好奖励模型
        env: 环境
        agent: 智能体
        task_name: 任务名称
        **config_kwargs: 配置参数
    
    Returns:
        TrajectoryGenerator实例
    """
    config = TrajectoryGenerationConfig(**config_kwargs)
    return TrajectoryGenerator(reward_model, env, agent, config, task_name)


if __name__ == "__main__":
    # 测试代码
    print("轨迹生成器模块测试")
    
    # 创建模拟的奖励模型
    class MockRewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, obs, actions):
            # 简单的模拟奖励计算
            combined = torch.cat([obs.mean(dim=-1, keepdim=True), actions.mean(dim=-1, keepdim=True)], dim=-1)
            return self.linear(combined.mean(dim=1))
    
    # 创建模拟环境和智能体
    class MockEnv:
        def __init__(self):
            self.action_space = type('ActionSpace', (), {'low': np.array([-1]), 'high': np.array([1])})()
        
        def reset(self):
            return np.random.randn(5), {}
        
        def step(self, action):
            obs = np.random.randn(5)
            reward = np.random.randn()
            done = np.random.random() < 0.1
            return obs, reward, done, False, {}
    
    class MockAgent:
        def act(self, obs, t0=False, eval_mode=False):
            return np.random.randn(1)
    
    # 测试轨迹生成器
    reward_model = MockRewardModel()
    env = MockEnv()
    agent = MockAgent()
    
    config = TrajectoryGenerationConfig(
        total_trajectories=10,
        elite_trajectories=3,
        max_trajectory_length=1000
    )
    
    generator = TrajectoryGenerator(reward_model, env, agent, config, "test_task")
    
    print("开始生成轨迹...")
    all_trajectories, elite_trajectories = generator.generate_trajectories()
    
    print(f"生成轨迹数量: {len(all_trajectories)}")
    print(f"精英轨迹数量: {len(elite_trajectories)}")
    print(f"统计信息: {generator.get_stats()}")
    
    print("轨迹生成器测试完成")