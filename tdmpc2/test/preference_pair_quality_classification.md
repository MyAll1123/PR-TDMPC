# TD-MPC2 偏好对质量分类详解

## 概述

在TD-MPC2的优先级经验回放系统中，偏好对被分为三种类型：**高质量对**、**中等质量对**和**探索性对**。这些分类不仅影响偏好对的生成策略，还直接影响其在经验回放池中的优先级权重。

## 偏好对类型判断标准

### 1. 高质量对 (high_quality)

**判断逻辑**：
- 选择**前1/3高奖励轨迹** vs **后1/3低奖励轨迹**
- 这种对比能够提供最明确的偏好信号
- 代表"好 vs 坏"的强对比

**代码实现**：
```python
if pair_type == "high_quality":
    # 高质量对：选择高奖励轨迹 vs 低奖励轨迹
    high_idx = np.random.choice(len(traj_rewards) // 3)  # 前1/3
    low_idx = np.random.choice(range(2 * len(traj_rewards) // 3, len(traj_rewards)))  # 后1/3
    traj_a = traj_rewards[high_idx][1]
    traj_b = traj_rewards[low_idx][1]
```

**优先级权重**：2.0（从原来的1.5提升）

### 2. 中等质量对 (medium_quality)

**判断逻辑**：
- 选择**中间1/4到3/4奖励相近的轨迹**
- 这种对比提供更细粒度的偏好判断
- 代表"好 vs 较好"或"一般 vs 较一般"的微妙对比

**代码实现**：
```python
elif pair_type == "medium_quality":
    # 中等质量对：选择相近奖励的轨迹
    mid_start = len(traj_rewards) // 4
    mid_end = 3 * len(traj_rewards) // 4
    indices = np.random.choice(range(mid_start, mid_end), size=2, replace=False)
    traj_a = traj_rewards[indices[0]][1]
    traj_b = traj_rewards[indices[1]][1]
```

**优先级权重**：1.5（从原来的1.2提升）

### 3. 探索性对 (exploration)

**判断逻辑**：
- **随机选择**任意两条轨迹
- 提供多样化的对比样本
- 帮助模型学习各种可能的偏好模式

**代码实现**：
```python
else:  # exploration
    # 探索性对：随机选择
    indices = np.random.choice(len(traj_rewards), size=2, replace=False)
    traj_a = traj_rewards[indices[0]][1]
    traj_b = traj_rewards[indices[1]][1]
```

**优先级权重**：1.0（默认权重）

## 优先级权重应用机制

### 权重设置位置

在 `prioritized_experience_replay.py` 的 `PriorityCalculator.calculate_confidence_priority` 方法中：

```python
# 4. 偏好对类型加权 - 提高权重差异
pair_type_weight = 1.0
if 'pair_type' in preference_pair.metadata:
    pair_type = preference_pair.metadata['pair_type']
    if pair_type == 'high_quality':
        pair_type_weight = 2.0  # 从1.5提升到2.0
    elif pair_type == 'medium_quality':
        pair_type_weight = 1.5  # 从1.2提升到1.5
    # exploration类型保持默认权重1.0
```

### 权重应用方式

权重通过乘法应用到置信度优先级计算中：

```python
# 5. 综合计算置信度优先级
confidence_priority = (
    0.6 * base_confidence +      # 基础置信度
    0.25 * score_diff_priority + # 规则得分差异
    0.1 * env_reward_diff_priority + # 环境奖励差异
    0.05  # 基础优先级
) * pair_type_weight  # 应用类型权重
```

## 质量过滤机制

### 多层过滤策略

系统还实施了多层质量过滤机制，确保只保留有价值的偏好对：

#### 1. 平均奖励过滤
```python
# 只保留至少有一条轨迹高于平均分的偏好对
if reward_a < env_mean_reward and reward_b < env_mean_reward:
    continue  # 抛弃"坏 vs 更坏"的无意义对比
```

#### 2. 训练阶段自适应过滤

**早期训练阶段**（轨迹数量 < 30 或平均奖励 < 5.0）：
- 放宽过滤条件
- 只要两条轨迹有明显差异就保留
- 最小差异阈值：`max(0.05, env_mean_reward * 0.05)`

**训练后期**：
- 特别保留低质量vs高质量的对比对
- 高质量阈值：`env_mean_reward * 0.8`
- 低质量阈值：`env_mean_reward * 0.4`
- 有价值对比：一个轨迹高质量，另一个低质量

#### 3. 奖励差异阈值
```python
# 非早期训练阶段的差异阈值
reward_diff_threshold = max(1.0, env_mean_reward * 0.1)
if reward_diff < reward_diff_threshold:
    continue  # 跳过差异不够显著的偏好对
```

## 生成比例配置

在 `PrioritizedSystemConfig` 中，偏好对生成方法的默认配置：

```python
preference_pair_generation: Dict[str, Any] = field(default_factory=lambda: {
    'pairs_per_generation': 20,
    'generation_methods': {
        'dpo': {
            'enabled': True,
            'ratio': 0.4  # 40%
        },
        'quality': {
            'enabled': True,
            'ratio': 0.4  # 40%
        },
        'hybrid': {
            'enabled': True,
            'ratio': 0.2  # 20%
        }
    }
})
```

## 总结

### 判断标准总结

1. **高质量对**：前1/3高奖励 vs 后1/3低奖励，权重2.0
2. **中等质量对**：中间1/4-3/4相近奖励，权重1.5
3. **探索性对**：随机选择，权重1.0

### 关键特点

- **基于奖励排序**：所有判断都基于轨迹的总奖励排序
- **动态阈值**：使用自适应阈值管理器动态调整过滤条件
- **训练阶段感知**：早期和后期训练采用不同的过滤策略
- **质量保证**：多层过滤机制确保偏好对的学习价值
- **权重递减**：高质量对 > 中等质量对 > 探索性对，体现重要性差异

这种分层的偏好对生成和优先级设置机制，确保了偏好模型能够从最有价值的对比中学习，同时保持足够的多样性来避免过拟合。