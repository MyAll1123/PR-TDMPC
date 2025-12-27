# TD-MPC2轨迹生成完整流程示例

## 概述
本文档通过具体例子详细说明TD-MPC2中一条轨迹从诞生到完成的完整过程，包括每个决策步骤中各模块的计算细节。

## 示例场景设置
- **任务**: H1Hand机器人平衡任务 (h1hand-balance_hard-v0)
- **轨迹长度**: 67步 (从日志可见)
- **状态维度**: 观测空间维度
- **动作维度**: 61维连续动作
- **规划视野**: horizon = 3步

---

## 第一阶段：轨迹初始化

### 1.1 环境重置
```python
# 环境初始化
obs_0 = env.reset()  # 形状: [obs_dim]
task = 0  # 任务ID
t = 0    # 时间步
```

**输出**:
- `obs_0`: 初始观测，包含机器人关节角度、角速度等信息
- 轨迹ID: 321 (从日志示例)

### 1.2 状态编码
```python
# 将观测编码到潜空间
z_0 = model.encode(obs_0, task)  # 形状: [latent_dim=512]
```

**模块调用**:
- **编码器**: `world_model.py` 中的编码网络
- **输入**: 原始观测 + 任务ID
- **输出**: 512维潜在状态表示

---

## 第二阶段：动作规划与选择 (每个时间步)

### 2.1 第1步决策 (t=0)

#### 2.1.1 策略轨迹采样
```python
# 在plan()方法中
if cfg.num_pi_trajs > 0:
    # 使用策略网络采样多条轨迹
    pi_actions = torch.empty(horizon, num_pi_trajs, action_dim)
    for i in range(num_pi_trajs):
        z_temp = z_0.clone()
        for t in range(horizon):  # horizon=3
            # 策略网络预测动作
            action, _ = model.pi(z_temp, task)
            pi_actions[t, i] = action
            # 预测下一状态
            z_temp = model.next(z_temp, action, task)
```

**模块调用**:
- **策略网络**: `world_model.py` 中的 `pi` 网络
- **动力学模型**: `world_model.py` 中的 `next` 网络
- **输出**: 多条3步动作序列候选

#### 2.1.2 随机轨迹采样
```python
# 随机采样轨迹作为探索
if cfg.num_samples > 0:
    actions = torch.randn(horizon, num_samples, action_dim)
    # 应用动作边界约束
    actions = torch.clamp(actions, -1, 1)
```

#### 2.1.3 价值评估
```python
# 对每条候选轨迹进行价值评估
for i, action_sequence in enumerate(all_candidates):
    value = _estimate_value(z_0, action_sequence, task)
    values.append(value)
```

**`_estimate_value` 详细计算**:
```python
def _estimate_value(z, actions, task):
    G, discount = 0, 1
    z_current = z.clone()
    
    # 步骤1: 前向展开3步
    for t in range(3):  # horizon=3
        # 1.1 预测即时奖励
        reward_logits = model.reward(z_current, actions[t], task)
        reward = math.two_hot_inv(reward_logits, cfg)  # 解码奖励
        
        # 1.2 预测下一状态
        z_current = model.next(z_current, actions[t], task)
        
        # 1.3 累积折扣奖励
        G += discount * reward
        discount *= 0.99  # 折扣因子
    
    # 步骤2: 终端状态价值估计
    # 2.1 策略网络预测终端动作
    terminal_action, _ = model.pi(z_current, task)
    
    # 2.2 Q网络估计终端价值
    terminal_value = model.Q(z_current, terminal_action, task, return_type="avg")
    
    # 步骤3: 总价值 = 累积奖励 + 折扣终端价值
    total_value = G + discount * terminal_value
    return total_value
```

**模块调用详情**:
1. **奖励预测器**: 预测每步的即时奖励
2. **动力学模型**: 预测状态转移
3. **策略网络**: 预测终端动作
4. **Q网络集成**: 估计终端状态价值

#### 2.1.4 最优动作选择
```python
# 选择价值最高的轨迹的第一个动作
best_idx = torch.argmax(values)
best_action = all_candidates[best_idx][0]  # 只执行第一步
```

### 2.2 偏好信息集成 (如果启用)

#### 2.2.1 当前系统 (训练时融合)
```python
# 在训练阶段，update()方法中
if preference_module is not None and preference_module.is_enabled():
    # 计算融合奖励用于训练
    for t in range(sample_z.shape[0]):
        reward_details = integrator.compute_integrated_reward(
            latent_state=sample_z[t],
            action=sample_action[t],
            environment_reward=sample_reward[t].item()
        )
        # 更新训练数据中的奖励
        integrated_reward[t, b] = reward_details['integrated_reward']
```

**特点**: 偏好信息在训练时融合到奖励中，推理时直接使用训练好的模型

#### 2.2.2 实时混合 (理论设计)
```python
# 如果实现实时混合，_estimate_value会变成:
def _estimate_value_hybrid(z, actions, task):
    # 1. 计算MPC价值 (原有逻辑)
    mpc_value = _estimate_value_original(z, actions, task)
    
    # 2. 实时计算偏好价值
    pref_value = preference_estimator.get_preference_reward(
        latent_states=z_sequence,  # 整个轨迹的状态序列
        actions=actions,
        task=task
    )
    
    # 3. 动态权重调整
    confidence = preference_estimator.get_confidence()
    weights = weight_controller.get_adaptive_weights(confidence)
    
    # 4. 混合计算
    pref_normalized = torch.tanh(pref_value) * 0.3
    hybrid_value = weights['mpc'] * (1.0 + pref_normalized)
    
    return hybrid_value
```

**特点**: 每次价值评估都实时计算偏好，计算开销更大但更灵活

### 2.3 动作执行
```python
# 执行选定的动作
obs_1, reward_1, done, info = env.step(best_action)
```

---

## 第三阶段：后续步骤 (t=1, 2, ..., 66)

### 3.1 状态更新
```python
# 编码新观测
z_1 = model.encode(obs_1, task)
t += 1
```

### 3.2 重复规划过程
每个时间步都重复2.1的完整规划过程：
1. 从当前状态 `z_t` 开始
2. 采样多条3步候选轨迹
3. 评估每条轨迹的价值
4. 选择最优动作执行
5. 更新状态，继续下一步

### 3.3 轨迹终止条件
```python
if done or t >= max_episode_length:
    break  # 轨迹结束
```

---

## 第四阶段：训练数据收集与模型更新

### 4.1 经验存储
```python
# 将轨迹数据存入回放缓冲区
buffer.add(obs_sequence, action_sequence, reward_sequence, task)
```

### 4.2 模型更新 (批次训练)
```python
# 从缓冲区采样批次数据
obs, action, reward, task = buffer.sample()

# 如果启用偏好模块，计算融合奖励
if preference_module is not None:
    integrated_reward = compute_integrated_rewards(
        obs, action, reward, preference_module
    )
else:
    integrated_reward = reward

# 使用融合奖励训练各个网络
loss_dict = update_networks(obs, action, integrated_reward, task)
```

---

## 关键差异对比

### 当前系统的完整流程
```
训练阶段:
环境奖励 + 偏好奖励 → 融合奖励 → 训练所有网络

推理阶段:
观测 → 编码 → 规划(纯MPC) → 动作选择 → 执行
```

### 实时混合的完整流程
```
训练阶段:
环境奖励 → 训练MPC网络
偏好数据 → 训练偏好网络

推理阶段:
观测 → 编码 → 规划(MPC+偏好实时混合) → 动作选择 → 执行
```

---

## 具体数值示例

### 示例轨迹片段
```
时间步 t=0:
- 观测: [关节角度: 0.1, 0.2, ..., 角速度: 0.05, ...]
- 编码状态 z_0: [512维向量]
- 候选动作1: [0.3, -0.1, 0.5, ...] → 价值: 2.3
- 候选动作2: [0.1, 0.2, -0.3, ...] → 价值: 1.8
- 候选动作3: [-0.2, 0.4, 0.1, ...] → 价值: 2.7 ← 最优
- 执行动作: [-0.2, 0.4, 0.1, ...]
- 获得奖励: 0.15

时间步 t=1:
- 新观测: [更新后的关节状态]
- 编码状态 z_1: [新的512维向量]
- 重复规划过程...

...

时间步 t=66:
- 任务完成或达到最大步数
- 轨迹长度: 67步
- 总奖励: 累积所有步骤的奖励
```

---

## 总结

**当前系统特点**:
- 偏好信息在训练时"烘焙"到模型参数中
- 推理时只需运行MPC模型，效率高
- 偏好变化需要重新训练

**实时混合特点**:
- 偏好信息在每次决策时实时计算和混合
- 推理时需要同时运行MPC和偏好模型，开销大
- 可以实时响应偏好变化

两种方式的核心区别在于**偏好信息的使用时机**：一个是"训练时融合"，一个是"推理时混合"。