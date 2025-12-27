# DPO偏好方法详细解释

## 概述

本文档详细解释了DPO改造完成后，`QUALITY_BASED` 和 `HYBRID_DPO_QUALITY` 方法的计算方式，以及如何依靠DPO和这两种方法产生偏好对。

## 1. DPO核心原理

### 1.1 基本公式
```
P(轨迹A > 轨迹B) = σ(β * (R(轨迹A) - R(轨迹B)))
```

其中：
- `σ` 是 sigmoid 函数
- `β` 是温度参数（默认0.1-1.0）
- `R(轨迹)` 是轨迹的质量分数

### 1.2 质量分数计算（修改后）
```
质量分数 = 环境奖励总和 × 基础质量因子 × (1 + API规则贡献)
基础质量因子 = 生存得分 × 状态稳定性得分 × 动作平滑性得分
```

## 2. DPOPreferenceEvaluator类实现

### 2.1 核心方法：evaluate_dpo_preference

```python
def evaluate_dpo_preference(self, 
                           trajectory_a: Dict[str, np.ndarray],
                           trajectory_b: Dict[str, np.ndarray],
                           reward_model: Optional[Callable] = None) -> Tuple[float, float]:
    """
    使用DPO方法评估轨迹偏好
    
    Returns:
        (preference_logit, confidence)
    """
    # 1. 计算轨迹奖励
    if reward_model is not None:
        reward_a = self._compute_trajectory_reward(trajectory_a, reward_model)
        reward_b = self._compute_trajectory_reward(trajectory_b, reward_model)
    else:
        # 使用启发式奖励估计（基于质量评估器）
        reward_a = self._heuristic_reward_estimate(trajectory_a)
        reward_b = self._heuristic_reward_estimate(trajectory_b)
    
    # 2. DPO偏好概率计算
    reward_diff = float(reward_a) - float(reward_b)
    preference_logit = float(self.beta * reward_diff)
    
    # 3. 应用标签平滑（如果启用）
    if self.label_smoothing > 0:
        preference_logit = self._apply_label_smoothing(preference_logit)
    
    # 4. 计算置信度
    confidence = self._compute_confidence(reward_a, reward_b, preference_logit)
    
    return preference_logit, confidence
```

### 2.2 启发式奖励估计

当没有奖励模型时，使用 `TrajectoryQualityEvaluator` 进行质量评估：

```python
def _heuristic_reward_estimate(self, trajectory: Dict[str, np.ndarray]) -> float:
    """
    启发式奖励估计（基于质量评估器）
    """
    obs_seq = trajectory.get('obs', [])
    act_seq = trajectory.get('action', [])
    
    # 使用质量评估器计算综合质量分数
    quality_score, feature_scores = self.quality_evaluator.evaluate_trajectory_quality(
        obs_array, act_array
    )
    
    # 应用任务特定的启发式规则
    heuristic_bonus = self._apply_task_specific_heuristics(obs_array, act_array, feature_scores)
    
    # 综合启发式奖励
    heuristic_reward = quality_score + heuristic_bonus
    
    return float(np.clip(heuristic_reward, 0.0, 1.0))
```

## 3. QUALITY_BASED方法

### 3.1 计算流程

`QUALITY_BASED` 方法纯粹基于轨迹质量分数进行偏好判断：

```python
def _calculate_quality_based_score(self, quality_a: float, quality_b: float, 
                                  label_type: LabelType) -> Tuple[float, float]:
    """
    基于质量分数计算偏好分数和置信度
    """
    quality_diff = quality_a - quality_b
    abs_diff = abs(quality_diff)
    
    # 1. 不确定性判断（极严格）
    uncertainty_range = 0.01  # 极小的不确定性阈值
    min_uncertainty_threshold = 0.1
    
    if abs_diff < uncertainty_range:
        return 0.5, min_uncertainty_threshold  # 不确定
    
    # 2. 计算偏好分数（高敏感度）
    sigmoid_input = quality_diff * 10.0  # 高敏感度乘数
    preference_score = torch.sigmoid(torch.tensor(sigmoid_input)).item()
    
    # 3. 应用标签平滑（几乎不使用）
    smoothing = 0.01 * 0.1  # 极小的标签平滑
    if preference_score > 0.5:
        preference_score = preference_score * (1 - smoothing) + 0.5 * smoothing
    else:
        preference_score = preference_score * (1 - smoothing) + 0.5 * smoothing
    
    # 4. 计算置信度（高置信度）
    confidence = min(abs_diff * 10.0 + 0.5, 0.95)
    
    return preference_score, confidence
```

### 3.2 特点

1. **高敏感度**：`sigmoid_input = quality_diff * 10.0`，微小差异也能产生明显偏好
2. **极严格的不确定性阈值**：只有差异小于0.01才标记为不确定
3. **高置信度**：`confidence = min(abs_diff * 10.0 + 0.5, 0.95)`
4. **几乎不使用标签平滑**：`smoothing = 0.01 * 0.1`

## 4. HYBRID_DPO_QUALITY方法

### 4.1 计算流程

`HYBRID_DPO_QUALITY` 方法结合DPO评估和质量评估：

```python
elif label_type == LabelType.HYBRID_DPO_QUALITY:
    # DPO与质量混合：DPO为主，质量为辅
    trajectory_a = {'obs': obs_a, 'action': act_a}
    trajectory_b = {'obs': obs_b, 'action': act_b}
    
    # 1. 计算DPO分数
    dpo_logit, dpo_conf = self.dpo_evaluator.evaluate_dpo_preference(
        trajectory_a, trajectory_b, self.reward_model
    )
    dpo_score = torch.sigmoid(torch.tensor(dpo_logit)).item()
    
    # 2. 计算质量分数
    quality_score, quality_conf = self._calculate_quality_based_score(
        quality_a, quality_b, LabelType.QUALITY_BASED
    )
    
    # 3. 加权组合（DPO主导）
    combined_score = 0.8 * dpo_score + 0.2 * quality_score
    combined_conf = (dpo_conf + quality_conf) / 2  # 平均置信度
    
    return combined_score, combined_conf
```

### 4.2 权重分配

- **DPO权重**：80%（主导作用）
- **质量权重**：20%（辅助调节）
- **置信度**：两者平均值

### 4.3 优势

1. **结合两种方法的优势**：DPO的理论基础 + 质量评估的直观性
2. **更稳定的偏好判断**：避免单一方法的偏差
3. **适应性更强**：在不同场景下都能提供合理的偏好判断

## 5. 偏好对生成流程

### 5.1 PrioritizedPreferenceSystem中的生成

```python
def _generate_pairs_by_method(self, method_type: str, trajectories: List[Dict], 
                             config: Dict[str, Any]) -> List[Tuple]:
    """
    根据方法类型生成偏好对
    """
    if method_type == "QUALITY_BASED":
        return self._generate_quality_based_pairs(trajectories, config)
    elif method_type == "HYBRID_DPO_QUALITY":
        return self._generate_hybrid_dpo_quality_pairs(trajectories, config)
    # ... 其他方法
```

### 5.2 分层偏好对生成

```python
def _generate_stratified_pairs(self, trajectories: List[Dict], 
                              method_config: Dict[str, Any]) -> List[Tuple]:
    """
    生成分层偏好对
    """
    # 1. 按质量分数排序轨迹
    scored_trajectories = []
    for traj in trajectories:
        quality_score = self._calculate_trajectory_quality(traj)
        scored_trajectories.append((traj, quality_score))
    
    scored_trajectories.sort(key=lambda x: x[1], reverse=True)
    
    # 2. 分层生成偏好对
    pairs = []
    
    # 高质量对比（前25% vs 中间25%）
    high_quality = scored_trajectories[:len(scored_trajectories)//4]
    mid_quality = scored_trajectories[len(scored_trajectories)//4:len(scored_trajectories)//2]
    
    for high_traj, _ in high_quality[:5]:  # 限制数量
        for mid_traj, _ in mid_quality[:3]:
            pairs.append((high_traj, mid_traj))
    
    # 中等质量对比（中间25% vs 后25%）
    low_quality = scored_trajectories[3*len(scored_trajectories)//4:]
    
    for mid_traj, _ in mid_quality[:3]:
        for low_traj, _ in low_quality[:3]:
            pairs.append((mid_traj, low_traj))
    
    return pairs
```

### 5.3 偏好标签生成

```python
def generate_preference_label(self, obs_a: np.ndarray, act_a: np.ndarray,
                             obs_b: np.ndarray, act_b: np.ndarray,
                             label_type: LabelType = LabelType.HYBRID_DPO_QUALITY) -> PreferenceLabel:
    """
    生成偏好标签
    """
    start_time = time.time()
    
    # 1. 计算轨迹质量
    quality_a, features_a = self._get_trajectory_quality(obs_a, act_a)
    quality_b, features_b = self._get_trajectory_quality(obs_b, act_b)
    
    # 2. 根据标签类型计算偏好分数
    preference_score, confidence = self._calculate_preference_score(
        quality_a, quality_b, label_type, obs_a, act_a, obs_b, act_b
    )
    
    # 3. 创建偏好标签
    metadata = LabelMetadata(
        label_type=label_type,
        confidence=confidence,
        quality_score_a=quality_a,
        quality_score_b=quality_b,
        score_difference=abs(quality_a - quality_b),
        generation_time=time.time() - start_time,
        features_used=list(features_a.keys()),
        additional_info={
            'method': 'unified_preference_system',
            'features_a': features_a,
            'features_b': features_b
        }
    )
    
    return PreferenceLabel(
        preference_score=preference_score,
        metadata=metadata,
        is_valid=True
    )
```

## 6. 实际应用示例

### 6.1 H1机器人行走任务

假设有两条轨迹：

**轨迹A（高质量）**：
- 环境奖励总和：28.8
- 基础质量因子：0.765
- 质量分数：22.03

**轨迹B（低质量）**：
- 环境奖励总和：0.8
- 基础质量因子：0.0945
- 质量分数：0.076

### 6.2 QUALITY_BASED方法计算

```python
quality_diff = 22.03 - 0.076 = 21.954
sigmoid_input = 21.954 * 10.0 = 219.54
preference_score = sigmoid(219.54) ≈ 1.0
confidence = min(21.954 * 10.0 + 0.5, 0.95) = 0.95
```

### 6.3 HYBRID_DPO_QUALITY方法计算

```python
# DPO计算
dpo_logit = 0.1 * (22.03 - 0.076) = 2.1954
dpo_score = sigmoid(2.1954) ≈ 0.9
dpo_conf = 0.85

# 质量计算
quality_score = 1.0
quality_conf = 0.95

# 组合
combined_score = 0.8 * 0.9 + 0.2 * 1.0 = 0.92
combined_conf = (0.85 + 0.95) / 2 = 0.9
```

## 7. 优势与特点

### 7.1 QUALITY_BASED方法

**优势**：
- 直接基于轨迹质量，直观易懂
- 高敏感度，能捕捉微小差异
- 计算效率高

**特点**：
- 极严格的不确定性判断
- 高置信度输出
- 几乎不使用标签平滑

### 7.2 HYBRID_DPO_QUALITY方法

**优势**：
- 结合DPO理论基础和质量评估直观性
- 更稳定的偏好判断
- 适应性强，在不同场景下都能工作

**特点**：
- DPO主导（80%权重）
- 质量辅助（20%权重）
- 平均置信度

## 8. 总结

DPO改造完成后，系统通过以下方式产生偏好对：

1. **轨迹收集**：从环境中收集多条轨迹
2. **质量评估**：使用修改后的质量分数公式评估每条轨迹
3. **分层采样**：按质量分数分层，生成有学习价值的偏好对
4. **偏好标签生成**：
   - `QUALITY_BASED`：纯质量比较，高敏感度
   - `HYBRID_DPO_QUALITY`：DPO+质量混合，更稳定
5. **置信度评估**：为每个偏好标签计算置信度
6. **偏好对输出**：生成用于训练的偏好对数据

这种设计既保持了DPO的理论优势，又结合了质量评估的实用性，为强化学习提供了高质量的偏好信号。