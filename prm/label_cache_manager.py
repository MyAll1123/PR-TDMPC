#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏好标签缓存管理模块
提供标签缓存、批处理优化和性能监控功能
"""

import os
import pickle
import hashlib
import time
import threading
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict
import numpy as np
import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from prm.preference_labeling_engine import PreferenceLabel, LabelType, LabelMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """缓存统计信息"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_generation_time: float = 0.0
    total_cache_time: float = 0.0
    cache_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def avg_generation_time(self) -> float:
        """平均生成时间"""
        if self.cache_misses == 0:
            return 0.0
        return self.total_generation_time / self.cache_misses
    
    @property
    def avg_cache_time(self) -> float:
        """平均缓存访问时间"""
        if self.cache_hits == 0:
            return 0.0
        return self.total_cache_time / self.cache_hits

class LabelCacheManager:
    """标签缓存管理器"""
    
    def __init__(self, 
                 cache_dir: str = "./cache/labels",
                 max_cache_size: int = 10000,
                 enable_disk_cache: bool = True,
                 enable_memory_cache: bool = True,
                 cache_ttl: int = 3600 * 24):  # 24小时TTL
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        self.cache_ttl = cache_ttl
        
        # 内存缓存（LRU）
        self.memory_cache: OrderedDict[str, Tuple[PreferenceLabel, float]] = OrderedDict()
        
        # 缓存统计
        self.stats = CacheStats()
        
        # 线程锁
        self.cache_lock = threading.RLock()
        
        # 批处理配置
        self.batch_size = 32
        self.max_workers = 4
        
        logger.info(f"标签缓存管理器初始化完成，缓存目录: {self.cache_dir}")
    
    def _generate_cache_key(self, obs_a: np.ndarray, act_a: np.ndarray, 
                           obs_b: np.ndarray, act_b: np.ndarray, 
                           label_type) -> str:
        """生成缓存键"""
        # 使用轨迹数据的哈希值作为缓存键
        label_type_str = label_type.value if hasattr(label_type, 'value') else str(label_type)
        data_str = f"{obs_a.shape}_{act_a.shape}_{obs_b.shape}_{act_b.shape}_{label_type_str}"
        
        # 添加数据内容的哈希（采样部分数据以提高效率）
        sample_data = np.concatenate([
            obs_a.flatten()[:100],
            act_a.flatten()[:50],
            obs_b.flatten()[:100],
            act_b.flatten()[:50]
        ])
        
        data_hash = hashlib.md5(sample_data.tobytes()).hexdigest()[:16]
        cache_key = f"{data_str}_{data_hash}"
        
        return cache_key
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """检查缓存是否有效"""
        return time.time() - timestamp < self.cache_ttl
    
    def _get_from_memory_cache(self, cache_key: str) -> Optional[PreferenceLabel]:
        """从内存缓存获取标签"""
        if not self.enable_memory_cache:
            return None
        
        with self.cache_lock:
            if cache_key in self.memory_cache:
                label, timestamp = self.memory_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    # 移动到末尾（LRU更新）
                    self.memory_cache.move_to_end(cache_key)
                    return label
                else:
                    # 缓存过期，删除
                    del self.memory_cache[cache_key]
        
        return None
    
    def _put_to_memory_cache(self, cache_key: str, label: PreferenceLabel):
        """将标签放入内存缓存"""
        if not self.enable_memory_cache:
            return
        
        with self.cache_lock:
            # 检查缓存大小限制
            while len(self.memory_cache) >= self.max_cache_size:
                # 删除最旧的条目
                self.memory_cache.popitem(last=False)
            
            self.memory_cache[cache_key] = (label, time.time())
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """获取磁盘缓存文件路径"""
        # 使用两级目录结构避免单个目录文件过多
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.pkl"
    
    def _get_from_disk_cache(self, cache_key: str) -> Optional[PreferenceLabel]:
        """从磁盘缓存获取标签"""
        if not self.enable_disk_cache:
            return None
        
        cache_path = self._get_disk_cache_path(cache_key)
        
        try:
            if cache_path.exists():
                # 检查文件修改时间
                if self._is_cache_valid(cache_path.stat().st_mtime):
                    with open(cache_path, 'rb') as f:
                        label = pickle.load(f)
                    return label
                else:
                    # 缓存过期，删除文件
                    cache_path.unlink()
        except Exception as e:
            logger.warning(f"读取磁盘缓存失败: {e}")
            if cache_path.exists():
                cache_path.unlink()
        
        return None
    
    def _put_to_disk_cache(self, cache_key: str, label: PreferenceLabel):
        """将标签保存到磁盘缓存"""
        if not self.enable_disk_cache:
            return
        
        cache_path = self._get_disk_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(label, f)
        except Exception as e:
            logger.warning(f"保存磁盘缓存失败: {e}")
    
    def get_cached_label(self, obs_a: np.ndarray, act_a: np.ndarray,
                        obs_b: np.ndarray, act_b: np.ndarray,
                        label_type) -> Optional[PreferenceLabel]:
        """获取缓存的标签"""
        start_time = time.time()
        
        cache_key = self._generate_cache_key(obs_a, act_a, obs_b, act_b, label_type)
        
        # 首先尝试内存缓存
        label = self._get_from_memory_cache(cache_key)
        if label is not None:
            self.stats.cache_hits += 1
            self.stats.total_cache_time += time.time() - start_time
            return label
        
        # 然后尝试磁盘缓存
        label = self._get_from_disk_cache(cache_key)
        if label is not None:
            # 将标签放入内存缓存
            self._put_to_memory_cache(cache_key, label)
            self.stats.cache_hits += 1
            self.stats.total_cache_time += time.time() - start_time
            return label
        
        self.stats.cache_misses += 1
        return None
    
    def cache_label(self, obs_a: np.ndarray, act_a: np.ndarray,
                   obs_b: np.ndarray, act_b: np.ndarray,
                   label_type, label: PreferenceLabel):
        """缓存标签"""
        cache_key = self._generate_cache_key(obs_a, act_a, obs_b, act_b, label_type)
        
        # 同时保存到内存和磁盘缓存
        self._put_to_memory_cache(cache_key, label)
        self._put_to_disk_cache(cache_key, label)
        
        self.stats.cache_size = len(self.memory_cache)
    
    def batch_get_cached_labels(self, batch_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, any]]) -> List[Optional[PreferenceLabel]]:
        """批量获取缓存标签"""
        results = []
        
        for obs_a, act_a, obs_b, act_b, label_type in batch_data:
            self.stats.total_requests += 1
            label = self.get_cached_label(obs_a, act_a, obs_b, act_b, label_type)
            results.append(label)
        
        return results
    
    def batch_cache_labels(self, batch_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, any]], 
                          labels: List[PreferenceLabel]):
        """批量缓存标签"""
        if len(batch_data) != len(labels):
            raise ValueError("批量数据和标签数量不匹配")
        
        for (obs_a, act_a, obs_b, act_b, label_type), label in zip(batch_data, labels):
            self.cache_label(obs_a, act_a, obs_b, act_b, label_type, label)
    
    def clear_cache(self, clear_disk: bool = True, clear_memory: bool = True):
        """清空缓存"""
        if clear_memory:
            with self.cache_lock:
                self.memory_cache.clear()
        
        if clear_disk and self.enable_disk_cache:
            try:
                import shutil
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"清空磁盘缓存失败: {e}")
        
        # 重置统计信息
        self.stats = CacheStats()
        logger.info("缓存已清空")
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        
        # 清理内存缓存
        with self.cache_lock:
            expired_keys = []
            for key, (_, timestamp) in self.memory_cache.items():
                if current_time - timestamp > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
        
        # 清理磁盘缓存
        if self.enable_disk_cache:
            try:
                for cache_file in self.cache_dir.rglob("*.pkl"):
                    if current_time - cache_file.stat().st_mtime > self.cache_ttl:
                        cache_file.unlink()
            except Exception as e:
                logger.warning(f"清理磁盘缓存失败: {e}")
        
        logger.info("过期缓存清理完成")
    
    def get_cache_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        self.stats.cache_size = len(self.memory_cache)
        return self.stats
    
    def print_cache_stats(self):
        """打印缓存统计信息"""
        stats = self.get_cache_stats()
        
        print("=" * 50)
        print("标签缓存统计信息")
        print("=" * 50)
        print(f"总请求数: {stats.total_requests}")
        print(f"缓存命中: {stats.cache_hits}")
        print(f"缓存未命中: {stats.cache_misses}")
        print(f"命中率: {stats.hit_rate:.2%}")
        print(f"当前缓存大小: {stats.cache_size}")
        print(f"平均生成时间: {stats.avg_generation_time:.4f}s")
        print(f"平均缓存访问时间: {stats.avg_cache_time:.4f}s")
        
        if stats.cache_hits > 0 and stats.cache_misses > 0:
            speedup = stats.avg_generation_time / stats.avg_cache_time
            print(f"缓存加速比: {speedup:.2f}x")
        
        print("=" * 50)
    
    def export_cache_stats(self, file_path: str):
        """导出缓存统计信息"""
        stats = self.get_cache_stats()
        stats_dict = asdict(stats)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"缓存统计信息已导出到: {file_path}")
        except Exception as e:
            logger.error(f"导出统计信息失败: {e}")

class BatchLabelProcessor:
    """批量标签处理器"""
    
    def __init__(self, cache_manager: LabelCacheManager, 
                 labeling_engine, batch_size: int = 32, max_workers: int = 4):
        self.cache_manager = cache_manager
        self.labeling_engine = labeling_engine
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batch(self, batch_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelType]]) -> List[PreferenceLabel]:
        """处理批量数据"""
        start_time = time.time()
        
        # 首先尝试从缓存获取
        cached_labels = self.cache_manager.batch_get_cached_labels(batch_data)
        
        # 找出需要生成的标签
        to_generate = []
        to_generate_indices = []
        
        for i, (cached_label, data) in enumerate(zip(cached_labels, batch_data)):
            if cached_label is None:
                to_generate.append(data)
                to_generate_indices.append(i)
        
        # 生成缺失的标签
        generated_labels = []
        if to_generate:
            generation_start = time.time()
            
            # 使用多线程并行生成
            if len(to_generate) > 1 and self.max_workers > 1:
                generated_labels = self._parallel_generate_labels(to_generate)
            else:
                generated_labels = self._sequential_generate_labels(to_generate)
            
            generation_time = time.time() - generation_start
            self.cache_manager.stats.total_generation_time += generation_time
            
            # 缓存新生成的标签
            self.cache_manager.batch_cache_labels(to_generate, generated_labels)
        
        # 合并结果
        results = cached_labels.copy()
        for i, label in zip(to_generate_indices, generated_labels):
            results[i] = label
        
        total_time = time.time() - start_time
        logger.debug(f"批量处理完成，耗时: {total_time:.4f}s，批量大小: {len(batch_data)}")
        
        return results
    
    def _sequential_generate_labels(self, batch_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelType]]) -> List[PreferenceLabel]:
        """顺序生成标签"""
        labels = []
        for obs_a, act_a, obs_b, act_b, label_type in batch_data:
            # 使用统一的generate_preference_labels方法
            label = self.labeling_engine.generate_preference_labels(
                obs_a, act_a, obs_b, act_b, label_type=label_type, batch_mode=False
            )
            labels.append(label)
        return labels
    
    def _parallel_generate_labels(self, batch_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelType]]) -> List[PreferenceLabel]:
        """并行生成标签"""
        labels = [None] * len(batch_data)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_index = {}
            for i, (obs_a, act_a, obs_b, act_b, label_type) in enumerate(batch_data):
                # 使用统一的generate_preference_labels方法
                future = executor.submit(
                    self.labeling_engine.generate_preference_labels,
                    obs_a, act_a, obs_b, act_b, label_type, False
                )
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    label = future.result()
                    labels[index] = label
                except Exception as e:
                    logger.error(f"并行生成标签失败 (索引 {index}): {e}")
                    # 创建一个默认标签
                    from prm.preference_labeling_engine import LabelMetadata
                    labels[index] = PreferenceLabel(
                        preference_score=0.5,
                        is_valid=False,
                        metadata=LabelMetadata(
                            label_type=batch_data[index][4],
                            confidence=0.0,
                            quality_score_a=0.0,
                            quality_score_b=0.0,
                            score_difference=0.0,
                            generation_time=0.0,
                            features_used=[],
                            additional_info={"error": str(e)}
                        )
                    )
        
        return labels
    
    def process_large_dataset(self, dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelType]]) -> List[PreferenceLabel]:
        """处理大型数据集"""
        logger.info(f"开始处理大型数据集，总数据量: {len(dataset)}")
        
        all_labels = []
        
        # 分批处理
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            batch_labels = self.process_batch(batch)
            all_labels.extend(batch_labels)
            
            # 打印进度
            if (i // self.batch_size + 1) % 10 == 0:
                progress = (i + len(batch)) / len(dataset) * 100
                logger.info(f"处理进度: {progress:.1f}% ({i + len(batch)}/{len(dataset)})")
        
        logger.info(f"大型数据集处理完成，总标签数: {len(all_labels)}")
        return all_labels

# 工具函数
def create_label_cache_manager(cache_dir: str = "./cache/labels", 
                              max_cache_size: int = 10000,
                              enable_disk_cache: bool = True) -> LabelCacheManager:
    """创建标签缓存管理器"""
    return LabelCacheManager(
        cache_dir=cache_dir,
        max_cache_size=max_cache_size,
        enable_disk_cache=enable_disk_cache
    )

def create_batch_processor(cache_manager: LabelCacheManager, 
                          labeling_engine,
                          batch_size: int = 32,
                          max_workers: int = 4) -> BatchLabelProcessor:
    """创建批量标签处理器"""
    return BatchLabelProcessor(
        cache_manager=cache_manager,
        labeling_engine=labeling_engine,
        batch_size=batch_size,
        max_workers=max_workers
    )

if __name__ == "__main__":
    # 测试代码
    from prm.preference_labeling_engine import create_preference_labeling_engine, LabelType
    
    print("测试标签缓存管理器...")
    
    # 创建缓存管理器和标签引擎
    cache_manager = create_label_cache_manager()
    labeling_engine = create_preference_labeling_engine("test_task")
    batch_processor = create_batch_processor(cache_manager, labeling_engine)
    
    # 生成测试数据
    test_data = []
    for i in range(100):
        obs_a = np.random.randn(30, 10)
        act_a = np.random.randn(30, 5)
        obs_b = np.random.randn(25, 10)
        act_b = np.random.randn(25, 5)
        label_type = LabelType.RULE_BASED if i % 2 == 0 else LabelType.HEURISTIC_BASED
        
        test_data.append((obs_a, act_a, obs_b, act_b, label_type))
    
    # 第一次处理（无缓存）
    print("第一次处理（无缓存）...")
    start_time = time.time()
    labels1 = batch_processor.process_large_dataset(test_data)
    time1 = time.time() - start_time
    print(f"第一次处理耗时: {time1:.4f}s")
    
    # 第二次处理（有缓存）
    print("第二次处理（有缓存）...")
    start_time = time.time()
    labels2 = batch_processor.process_large_dataset(test_data)
    time2 = time.time() - start_time
    print(f"第二次处理耗时: {time2:.4f}s")
    
    # 打印缓存统计
    cache_manager.print_cache_stats()
    
    print(f"缓存加速比: {time1/time2:.2f}x")
    print("测试完成！")