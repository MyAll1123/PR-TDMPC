
import numpy as np

def survival_time(trajectory, goal=None):
    """返回轨迹的步数（即存活时间）"""
    return len(getattr(trajectory, "states", []))

def distance_to_goal(trajectory, goal):
    last_state = trajectory.states[-1]
    if isinstance(last_state, dict):
        pos = last_state["obs"][:3]
    else:
        pos = last_state[:3]
    
    if isinstance(goal, (list, tuple, np.ndarray)) and len(goal) == 3:
        return float(np.linalg.norm(np.array(pos) - np.array(goal)))
    else:
        # 对于walk任务，应该奖励前向移动距离
        return -float(pos[0])  # 负的x坐标，越远越好

def is_task_successful(trajectory, goal):
    last_state = trajectory.states[-1]
    # 1. 优先用 success 字段（如 info["success"]）
    if isinstance(last_state, dict) and "success" in last_state:
        return 1.0 if last_state["success"] else 0.1
    # 2. 如果 goal 是三维坐标，按距离判断
    if isinstance(goal, (list, tuple, np.ndarray)) and len(goal) == 3:
        if isinstance(last_state, dict):
            pos = last_state["obs"][:3]
        else:
            pos = last_state[:3]
        return 1.0 if float(np.linalg.norm(np.array(pos) - np.array(goal))) < 0.1 else 0.1
    # 3. 其他任务可自定义（如存活时间大于阈值等）
    # return survival_time(trajectory) > 某阈值
    return 0.1  # 失败轨迹返回0.1而不是0

def efficiency(trajectory, goal=None):
    # 示例：效率=reward/steps
    return np.sum(getattr(trajectory, "rewards", [])) / max(1, survival_time(trajectory))

def safety_score(trajectory, goal=None):
    # 示例：无碰撞为1，有碰撞为0
    return 1.0  # 你可以根据实际情况实现

def energy_usage(trajectory, goal=None):
    """
    计算轨迹的能耗：所有动作模长之和
    """
    actions = np.array(getattr(trajectory, "actions", []))
    if actions.size == 0:
        return float("inf")
    return float(np.sum(np.linalg.norm(actions, axis=-1)))

def action_diversity(trajectory, goal=None):
    """
    计算轨迹动作的多样性（方差），越大代表信息量越大
    """
    actions = np.array(getattr(trajectory, "actions", []))
    if actions.size == 0:
        return 0.0
    # 计算每个动作维度的方差，取均值
    return float(np.mean(np.var(actions, axis=0)))

def action_entropy(trajectory, goal=None, bins=10):
    """
    计算轨迹动作的熵，越大代表信息量越大
    """
    actions = np.array(getattr(trajectory, "actions", []))
    if actions.size == 0:
        return 0.0
    # 对每个动作维度分箱，计算熵，最后取均值
    entropies = []
    for i in range(actions.shape[1]):
        hist, _ = np.histogram(actions[:, i], bins=bins, density=True)
        hist = hist[hist > 0]
        entropies.append(-np.sum(hist * np.log(hist)))
    return float(np.mean(entropies))

def violates_joint_limits(trajectory, joint_limit=1.0):
    """
    检查轨迹是否有任何状态的关节超限。
    假设每个 state 是 dict，且有 'joint' 字段（为一维数组）。
    joint_limit: 关节绝对值上限（可根据你的机器人实际调整）。
    返回：True（有超限）/False（无超限）
    """
    for state in getattr(trajectory, "states", []):
        if isinstance(state, dict) and "joint" in state:
            joint = np.array(state["joint"])
            if np.any(np.abs(joint) > joint_limit):
                return True
    return False

# 可选：批量处理
def batch_metrics(trajectories, goal):
    results = []
    for traj in trajectories:
        results.append({
            "success": is_task_successful(traj, goal),
            "survival_time": survival_time(traj, goal),
            "distance_to_goal": distance_to_goal(traj, goal),
            "safety_score": safety_score(traj, goal),
            "efficiency": efficiency(traj, goal),
            "energy_usage": energy_usage(traj, goal),
            "action_diversity": action_diversity(traj, goal),
            "action_entropy": action_entropy(traj, goal),
        })
    return results


def gait_quality(trajectory, goal=None):
    """
    评估步态质量：基于质心高度变化的平滑性
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 提取质心高度（z坐标）
    heights = []
    for state in states:
        if isinstance(state, dict):
            heights.append(state["obs"][2])  # z坐标
        else:
            heights.append(state[2])
    
    heights = np.array(heights)
    # 计算高度变化的标准差，越小表示步态越稳定
    height_std = np.std(heights)
    # 转换为奖励：标准差越小，质量越高
    return float(1.0 / (1.0 + height_std))

def forward_progress(trajectory, goal=None):
    """
    计算前向进展：起点到终点的x方向距离
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 2:
        return 0.0
    
    start_state = states[0]
    end_state = states[-1]
    
    if isinstance(start_state, dict):
        start_x = start_state["obs"][0]
        end_x = end_state["obs"][0]
    else:
        start_x = start_state[0]
        end_x = end_state[0]
    
    return float(end_x - start_x)

def balance_stability(trajectory, goal=None):
    """
    评估平衡稳定性：基于躯干姿态变化
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 假设四元数在obs的3:7位置
    orientations = []
    for state in states:
        if isinstance(state, dict):
            quat = state["obs"][3:7]  # 四元数
        else:
            quat = state[3:7]
        # 计算z轴分量（直立度）
        orientations.append(quat[3])  # w分量，接近1表示直立
    
    orientations = np.array(orientations)
    # 计算直立度的标准差
    orientation_std = np.std(orientations)
    return float(1.0 / (1.0 + orientation_std))

# ===== STAIR 任务专用指标函数 =====
def height_gain(trajectory, goal=None):
    """计算高度增益：起点到终点的z方向距离"""
    states = getattr(trajectory, "states", [])
    if len(states) < 2:
        return 0.0
    
    start_state = states[0]
    end_state = states[-1]
    
    if isinstance(start_state, dict):
        start_z = start_state["obs"][2]
        end_z = end_state["obs"][2]
    else:
        start_z = start_state[2]
        end_z = end_state[2]
    
    return float(end_z - start_z)

def climbing_efficiency(trajectory, goal=None):
    """计算攀爬效率：高度增益/能耗"""
    height = height_gain(trajectory, goal)
    energy = energy_usage(trajectory, goal)
    if energy == 0 or energy == float("inf"):
        return 0.0
    return float(height / energy)

def step_coordination(trajectory, goal=None):
    """评估步伐协调性：基于足部接触模式"""
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 假设足部接触信息在obs的某个位置，这里用简化计算
    # 基于质心高度变化的周期性
    heights = []
    for state in states:
        if isinstance(state, dict):
            heights.append(state["obs"][2])
        else:
            heights.append(state[2])
    
    heights = np.array(heights)
    # 计算高度变化的周期性（通过自相关）
    if len(heights) > 20:
        autocorr = np.correlate(heights, heights, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        # 寻找第一个局部最大值（周期性指标）
        peaks = []
        for i in range(1, min(len(autocorr)-1, 20)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(autocorr[i])
        if peaks:
            return float(max(peaks) / autocorr[0])  # 归一化
    
    return 0.5  # 默认值

# ===== STAND 任务专用指标函数 =====
def standing_height_stability(trajectory, goal=None):
    """评估站立高度稳定性"""
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 提取质心高度（z坐标）
    heights = []
    for state in states:
        if isinstance(state, dict):
            heights.append(state["obs"][2])
        else:
            heights.append(state[2])
    
    heights = np.array(heights)
    # 计算高度的标准差，越小越稳定
    height_std = np.std(heights)
    # 同时考虑平均高度，站立应该保持一定高度
    avg_height = np.mean(heights)
    
    # 理想站立高度约为1.0米（可根据实际调整）
    ideal_height = 1.0
    height_penalty = abs(avg_height - ideal_height)
    
    # 综合评分：高度稳定性 + 高度适宜性
    stability_score = 1.0 / (1.0 + height_std)
    height_score = 1.0 / (1.0 + height_penalty)
    
    return float(0.7 * stability_score + 0.3 * height_score)

def minimal_movement(trajectory, goal=None):
    """评估最小运动：站立任务应该尽量减少不必要的移动"""
    states = getattr(trajectory, "states", [])
    if len(states) < 2:
        return 1.0
    
    # 计算位置变化
    total_movement = 0.0
    for i in range(1, len(states)):
        if isinstance(states[i], dict):
            pos1 = np.array(states[i-1]["obs"][:3])
            pos2 = np.array(states[i]["obs"][:3])
        else:
            pos1 = np.array(states[i-1][:3])
            pos2 = np.array(states[i][:3])
        
        # 只考虑x,y方向的移动，z方向的小幅变化是正常的
        movement = np.linalg.norm(pos2[:2] - pos1[:2])
        total_movement += movement
    
    # 转换为奖励：移动越少越好
    return float(1.0 / (1.0 + total_movement))

# ===== 增强版指标计算函数 =====
def enhanced_gait_quality(trajectory, goal=None):
    """增强版步态质量评估：基于观测序列的详细分析"""
    try:
        states = getattr(trajectory, "states", [])
        if len(states) < 10:
            return gait_quality(trajectory, goal)
        
        # 提取观测序列
        observations = []
        for state in states:
            if isinstance(state, dict) and "obs" in state:
                observations.append(np.array(state["obs"]))
            elif hasattr(state, "obs"):
                observations.append(np.array(state.obs))
            else:
                return gait_quality(trajectory, goal)
        
        observations = np.array(observations)
        
        # 质心位置 (x, y, z)
        com_positions = observations[:, :3]
        
        # 质心高度变化的平滑性
        heights = com_positions[:, 2]
        height_smoothness = 1.0 / (1.0 + np.std(heights))
        
        # 步态周期性分析
        if len(heights) > 20:
            # 使用FFT分析周期性
            fft = np.fft.fft(heights - np.mean(heights))
            power_spectrum = np.abs(fft[:len(fft)//2])
            # 寻找主频率
            if len(power_spectrum) > 5:
                dominant_freq_power = np.max(power_spectrum[1:])  # 排除DC分量
                total_power = np.sum(power_spectrum[1:])
                periodicity = dominant_freq_power / total_power if total_power > 0 else 0
            else:
                periodicity = 0.5
        else:
            periodicity = 0.5
        
        # 综合评分
        quality_score = 0.6 * height_smoothness + 0.4 * periodicity
        return float(np.clip(quality_score, 0.0, 1.0))
        
    except Exception:
        return gait_quality(trajectory, goal)

def enhanced_forward_progress(trajectory, goal=None):
    """增强版前向进展：考虑路径效率和速度一致性"""
    try:
        states = getattr(trajectory, "states", [])
        if len(states) < 10:
            return forward_progress(trajectory, goal)
        
        # 提取观测序列
        observations = []
        for state in states:
            if isinstance(state, dict) and "obs" in state:
                observations.append(np.array(state["obs"]))
            elif hasattr(state, "obs"):
                observations.append(np.array(state.obs))
            else:
                return forward_progress(trajectory, goal)
        
        observations = np.array(observations)
        
        # 质心位置
        com_positions = observations[:, :3]
        
        # 总前向距离
        total_forward = com_positions[-1, 0] - com_positions[0, 0]
        
        # 路径效率：直线距离 vs 实际路径长度
        path_length = 0.0
        for i in range(1, len(com_positions)):
            path_length += np.linalg.norm(com_positions[i] - com_positions[i-1])
        
        straight_distance = abs(total_forward)
        path_efficiency = straight_distance / path_length if path_length > 0 else 0
        
        # 速度一致性
        velocities = []
        for i in range(1, len(com_positions)):
            vel = np.linalg.norm(com_positions[i] - com_positions[i-1])
            velocities.append(vel)
        
        velocity_consistency = 1.0 / (1.0 + np.std(velocities)) if velocities else 0
        
        # 综合评分：前向距离 + 路径效率 + 速度一致性
        enhanced_progress = total_forward * (0.5 + 0.3 * path_efficiency + 0.2 * velocity_consistency)
        return float(enhanced_progress)
        
    except Exception:
        return forward_progress(trajectory, goal)

def enhanced_balance_stability(trajectory, goal=None):
    """增强版平衡稳定性：综合姿态、角速度和质心稳定性"""
    try:
        states = getattr(trajectory, "states", [])
        if len(states) < 10:
            return balance_stability(trajectory, goal)
        
        # 提取观测序列
        observations = []
        for state in states:
            if isinstance(state, dict) and "obs" in state:
                observations.append(np.array(state["obs"]))
            elif hasattr(state, "obs"):
                observations.append(np.array(state.obs))
            else:
                return balance_stability(trajectory, goal)
        
        observations = np.array(observations)
        
        # 质心位置和姿态
        com_positions = observations[:, :3]
        orientations = observations[:, 3:7]  # 四元数
        
        # 姿态稳定性：四元数变化的平滑性
        orientation_changes = []
        for i in range(1, len(orientations)):
            # 计算四元数之间的角度差
            q1, q2 = orientations[i-1], orientations[i]
            dot_product = np.abs(np.dot(q1, q2))
            angle_change = 2 * np.arccos(np.clip(dot_product, 0, 1))
            orientation_changes.append(angle_change)
        
        orientation_stability = 1.0 / (1.0 + np.std(orientation_changes)) if orientation_changes else 0
        
        # 质心稳定性
        com_stability = 1.0 / (1.0 + np.std(com_positions[:, :2], axis=0).mean())
        
        # 垂直稳定性
        vertical_stability = 1.0 / (1.0 + np.std(com_positions[:, 2]))
        
        # 综合评分
        total_stability = 0.4 * orientation_stability + 0.3 * com_stability + 0.3 * vertical_stability
        return float(np.clip(total_stability, 0.0, 1.0))
    except Exception as e:
        # 如果出现错误，回退到基础版本
        return balance_stability(trajectory, goal)


# ==================== API规则集成：新增任务特定指标函数 ====================

# Basketball任务指标函数
def basketball_standing_stability(trajectory, goal=None):
    """篮球任务的站立稳定性评估 - 基于standing * upright奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        head_heights = []
        upright_scores = []
        
        for state in states:
            if isinstance(state, dict):
                # 从字典格式状态中提取信息
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
                head_height = obs[2] if len(obs) > 2 else 1.0
                if len(obs) >= 7:
                    quat = obs[3:7]
                    qw, qx, qy, qz = quat
                    upright = 2 * (qw * qz + qx * qy)
                else:
                    upright = 1.0
            else:
                # 从数组格式状态中提取信息
                state_array = np.array(state) if not isinstance(state, np.ndarray) else state
                head_height = state_array[2] if len(state_array) > 2 else 1.0
                if len(state_array) >= 7:
                    quat = state_array[3:7]
                    qw, qx, qy, qz = quat
                    upright = 2 * (qw * qz + qx * qy)
                else:
                    upright = 1.0
            
            head_heights.append(head_height)
            upright_scores.append(upright)
        
        # 计算头部高度稳定性
        target_height = 1.65
        height_stability = 1.0 - np.mean(np.abs(np.array(head_heights) - target_height)) / target_height
        height_stability = max(0.0, height_stability)
        
        # 计算直立度稳定性
        upright_stability = np.mean(upright_scores)
        
        return float(0.6 * height_stability + 0.4 * upright_stability)
    except Exception as e:
        print(f"Error in basketball_standing_stability: {e}")
        return 0.0

def basketball_control_efficiency(trajectory, goal=None):
    """篮球任务的控制效率评估 - 基于small_control奖励"""
    try:
        actions = getattr(trajectory, 'actions', [])
        if not actions:
            return 1.0
        
        actions_array = np.array(actions)
        if len(actions_array.shape) == 1:
            actions_array = actions_array.reshape(-1, 1)
        
        # 计算控制力幅度
        control_magnitudes = np.sum(actions_array**2, axis=1)
        
        # 使用tolerance函数的近似实现
        control_margin = 10.0
        small_control_scores = np.exp(-control_magnitudes / control_margin)
        
        return float(np.mean(small_control_scores))
    except Exception as e:
        print(f"Error in basketball_control_efficiency: {e}")
        return 0.0

def basketball_hand_proximity(trajectory, goal=None):
    """篮球任务的手部接近度评估 - 基于reward_hand_proximity奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        hand_ball_distances = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            # 估算手部和篮球位置（基于观测维度）
            if len(obs) >= 30:
                hand_pos = obs[20:23] if len(obs) > 22 else [0, 0, 1]
                ball_pos = obs[23:26] if len(obs) > 25 else [0.5, 0.5, 1.2]
            else:
                hand_pos = [0, 0, 1]
                ball_pos = [0.5, 0.5, 1.2]
            
            distance = np.linalg.norm(np.array(hand_pos) - np.array(ball_pos))
            hand_ball_distances.append(distance)
        
        # 距离越小越好，使用指数衰减
        avg_distance = np.mean(hand_ball_distances)
        proximity_score = np.exp(-avg_distance / 0.5)  # 0.5米为衰减常数
        
        return float(proximity_score)
    except Exception as e:
        print(f"Error in basketball_hand_proximity: {e}")
        return 0.0

def basketball_ball_success(trajectory, goal=None):
    """篮球任务的投篮成功度评估 - 基于reward_ball_success奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        ball_hoop_distances = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            # 估算篮球和篮筐位置
            if len(obs) >= 30:
                ball_pos = obs[23:26] if len(obs) > 25 else [0.5, 0.5, 1.2]
                hoop_pos = [3.0, 0.0, 3.05]  # 固定篮筐位置
            else:
                ball_pos = [0.5, 0.5, 1.2]
                hoop_pos = [3.0, 0.0, 3.05]
            
            distance = np.linalg.norm(np.array(ball_pos) - np.array(hoop_pos))
            ball_hoop_distances.append(distance)
        
        # 计算最小距离（最接近篮筐的时刻）
        min_distance = np.min(ball_hoop_distances)
        
        # 成功奖励：距离小于0.2米为成功
        if min_distance < 0.2:
            success_score = 1.0
        else:
            # 距离越小越好，使用指数衰减
            success_score = np.exp(-min_distance / 1.0)
        
        return float(success_score)
    except Exception as e:
        print(f"Error in basketball_ball_success: {e}")
        return 0.0

# Balance Hard任务指标函数
def balance_hard_small_control(trajectory, goal=None):
    """平衡困难任务的控制力度最小化评估 - 基于small_control奖励"""
    try:
        actions = getattr(trajectory, 'actions', [])
        if not actions:
            return 1.0
        
        actions_array = np.array(actions)
        if len(actions_array.shape) == 1:
            actions_array = actions_array.reshape(-1, 1)
        
        # 基于动作幅度计算控制效率
        action_forces = np.sum(actions_array**2, axis=1)
        
        # 使用tolerance函数的近似实现
        control_margin = 10.0
        small_control_raw = np.exp(-action_forces / control_margin)
        small_control_score = np.mean(small_control_raw)
        
        # 应用 (4 + small_control) / 5 的变换
        small_control_score = (4 + small_control_score) / 5
        
        return float(small_control_score)
    except Exception as e:
        print(f"Error in balance_hard_small_control: {e}")
        return 0.0

def balance_hard_standing(trajectory, goal=None):
    """平衡困难任务的站立评估 - 基于standing奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        _STAND_HEIGHT = 1.65
        head_heights = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
                head_height = obs[2] if len(obs) > 2 else 1.0
            else:
                state_array = np.array(state) if not isinstance(state, np.ndarray) else state
                head_height = state_array[2] if len(state_array) > 2 else 1.0
            
            head_heights.append(head_height)
        
        # 基于真实奖励函数: bounds=(_STAND_HEIGHT + 0.37, inf), margin=_STAND_HEIGHT/4
        target_height = _STAND_HEIGHT + 0.37  # 1.65 + 0.37 = 2.02
        margin = _STAND_HEIGHT / 4  # 0.4125
        
        # tolerance函数的近似实现
        height_deviations = np.maximum(0, target_height - np.array(head_heights))
        standing_scores = np.exp(-height_deviations / margin)
        
        return float(np.mean(standing_scores))
    except Exception as e:
        print(f"Error in balance_hard_standing: {e}")
        return 0.0

def balance_hard_upright(trajectory, goal=None):
    """平衡困难任务的直立度评估 - 基于upright奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        upright_scores = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            if len(obs) >= 7:
                # 假设四元数在状态的3-6位置
                quat = obs[3:7]
                qw, qx, qy, qz = quat
                upright = 2 * (qw * qz + qx * qy)
            else:
                upright = 1.0
            
            upright_scores.append(upright)
        
        # 基于真实奖励函数: bounds=(0.9, inf), margin=0.5
        target_upright = 0.9
        margin = 0.5
        
        # tolerance函数的近似实现
        upright_deviations = np.maximum(0, target_upright - np.array(upright_scores))
        upright_rewards = np.exp(-upright_deviations / margin)
        
        return float(np.mean(upright_rewards))
    except Exception as e:
        print(f"Error in balance_hard_upright: {e}")
        return 0.0

def balance_hard_dont_move(trajectory, goal=None):
    """平衡困难任务的静止评估 - 基于dont_move奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 1.0
        
        horizontal_velocities = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            # 估算水平速度（通常在观测的后半部分）
            if len(obs) >= 25:
                velocity = obs[22:25] if len(obs) > 24 else [0, 0, 0]
                horizontal_vel = np.linalg.norm(velocity[:2])
            else:
                horizontal_vel = 0.0
            
            horizontal_velocities.append(horizontal_vel)
        
        # 基于真实奖励函数: bounds=(0, 2), margin=5, sigmoid='linear'
        max_vel = 2.0
        margin = 5.0
        
        # tolerance函数的线性实现
        avg_horizontal_vel = np.mean(horizontal_velocities)
        if avg_horizontal_vel <= max_vel:
            dont_move_score = 1.0
        else:
            dont_move_score = max(0.0, 1.0 - (avg_horizontal_vel - max_vel) / margin)
        
        return float(dont_move_score)
    except Exception as e:
        print(f"Error in balance_hard_dont_move: {e}")
        return 0.0

# Crawl任务指标函数
def crawl_crawling_posture(trajectory, goal=None):
    """爬行任务的爬行姿态评估 - 基于crawling + crawling_head奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        crawling_scores = []
        crawling_head_scores = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
                torso_height = obs[2] if len(obs) > 2 else 0.5
                head_height = torso_height + 0.2  # 估算头部高度
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
                torso_height = obs[2] if len(obs) > 2 else 0.5
                head_height = torso_height + 0.2
            
            # crawling奖励：躯干高度应在合适范围内
            target_crawl_height = 0.4
            height_margin = 0.2
            crawling_deviation = abs(torso_height - target_crawl_height)
            crawling_score = max(0.0, 1.0 - crawling_deviation / height_margin)
            crawling_scores.append(crawling_score)
            
            # crawling_head奖励：头部高度稳定性
            target_head_height = 0.6
            head_deviation = abs(head_height - target_head_height)
            crawling_head_score = max(0.0, 1.0 - head_deviation / height_margin)
            crawling_head_scores.append(crawling_head_score)
        
        # 取两个分数的最小值（如真实奖励函数中的min(crawling, crawling_head)）
        combined_scores = [min(c, ch) for c, ch in zip(crawling_scores, crawling_head_scores)]
        
        return float(np.mean(combined_scores))
    except Exception as e:
        print(f"Error in crawl_crawling_posture: {e}")
        return 0.0

def crawl_forward_motion(trajectory, goal=None):
    """爬行任务的前进运动评估 - 基于move奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        positions = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
                x_pos = obs[0] if len(obs) > 0 else 0.0
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
                x_pos = obs[0] if len(obs) > 0 else 0.0
            
            positions.append(x_pos)
        
        # 计算前进距离
        if len(positions) > 1:
            forward_distance = positions[-1] - positions[0]
            # 计算平均前进速度
            time_steps = len(positions)
            avg_forward_speed = forward_distance / max(1, time_steps)
            
            # 目标前进速度
            target_speed = 1.0
            speed_score = min(1.0, max(0.0, avg_forward_speed / target_speed))
        else:
            speed_score = 0.0
        
        return float(speed_score)
    except Exception as e:
        print(f"Error in crawl_forward_motion: {e}")
        return 0.0

def crawl_control_efficiency(trajectory, goal=None):
    """爬行任务的控制效率评估 - 基于small_control奖励"""
    try:
        actions = getattr(trajectory, 'actions', [])
        if not actions:
            return 1.0
        
        actions_array = np.array(actions)
        if len(actions_array.shape) == 1:
            actions_array = actions_array.reshape(-1, 1)
        
        # 计算控制力幅度
        control_magnitudes = np.sum(actions_array**2, axis=1)
        
        # 使用tolerance函数的近似实现
        control_margin = 10.0
        small_control_scores = np.exp(-control_magnitudes / control_margin)
        
        return float(np.mean(small_control_scores))
    except Exception as e:
        print(f"Error in crawl_control_efficiency: {e}")
        return 0.0

def crawl_orientation_stability(trajectory, goal=None):
    """爬行任务的方向稳定性评估 - 基于reward_xquat奖励"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 1.0
        
        orientation_scores = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            if len(obs) >= 7:
                quat = obs[3:7]
                qw, qx, qy, qz = quat
                # x轴方向的稳定性（爬行方向）
                x_orientation = 1 - 2 * (qy**2 + qz**2)
            else:
                x_orientation = 1.0
            
            # 方向稳定性分数
            orientation_score = max(0.0, x_orientation)
            orientation_scores.append(orientation_score)
        
        return float(np.mean(orientation_scores))
    except Exception as e:
        print(f"Error in crawl_orientation_stability: {e}")
        return 0.0

# Reach任务增强指标函数
def reach_accuracy_enhanced(trajectory, goal=None):
    """到达任务的准确性评估 - 基于hand_dist + reward_success + reward_close"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        # 默认目标位置
        if goal is None:
            goal = np.array([0.5, 0.5, 1.2])
        elif not isinstance(goal, np.ndarray):
            goal = np.array(goal)
        
        hand_distances = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            # 估算手部位置
            if len(obs) >= 23:
                hand_pos = obs[20:23]
            else:
                hand_pos = np.array([0, 0, 1])
            
            # 计算手到目标的距离
            distance = np.linalg.norm(hand_pos - goal)
            hand_distances.append(distance)
        
        # 计算最小距离（最接近目标的时刻）
        min_distance = np.min(hand_distances)
        
        # 成功奖励（距离<0.05m）
        success_score = 1.0 if min_distance < 0.05 else 0.0
        
        # 接近奖励（距离<1m）
        close_score = 1.0 if min_distance < 1.0 else 0.0
        
        # 距离分数（距离越小越好）
        distance_score = max(0.0, 1.0 - min_distance / 2.0)  # 2m内线性衰减
        
        # 综合准确性分数
        reaching_accuracy = (success_score * 0.5 + close_score * 0.3 + distance_score * 0.2)
        
        return float(reaching_accuracy)
    except Exception as e:
        print(f"Error in reach_accuracy_enhanced: {e}")
        return 0.0

def reach_torso_stability_enhanced(trajectory, goal=None):
    """到达任务的躯干稳定性评估 - 基于healthy_reward"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 0.0
        
        torso_stability_scores = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            if len(obs) >= 7:
                # 从四元数计算直立度
                quat = obs[3:7]
                qw, qx, qy, qz = quat
                torso_upright = 2 * (qw * qz + qx * qy)  # z轴分量
            else:
                torso_upright = 1.0
            
            # 躯干稳定性分数（基于xmat[1,-1] * 5.0的近似）
            stability_score = max(0.0, min(1.0, torso_upright * 5.0))
            torso_stability_scores.append(stability_score)
        
        return float(np.mean(torso_stability_scores))
    except Exception as e:
        print(f"Error in reach_torso_stability_enhanced: {e}")
        return 0.0

def reach_motion_efficiency_enhanced(trajectory, goal=None):
    """到达任务的运动效率评估 - 基于motion_penalty的反向"""
    try:
        states = getattr(trajectory, 'states', [])
        if not states:
            return 1.0
        
        motion_penalties = []
        
        for state in states:
            if isinstance(state, dict):
                obs = state.get('obs', state)
                obs = np.array(obs) if not isinstance(obs, np.ndarray) else obs
            else:
                obs = np.array(state) if not isinstance(state, np.ndarray) else state
            
            # 估算速度信息（通常在观测的后半部分）
            if len(obs) >= 50:
                velocities = obs[25:50] if len(obs) > 49 else [0.0] * 25
            else:
                velocities = [0.0] * 25
            
            # 计算运动惩罚（qvel^2 * 0.0001）
            if isinstance(velocities, (list, np.ndarray)):
                velocities_array = np.array(velocities)
                motion_penalty = np.sum(velocities_array**2) * 0.0001
            else:
                motion_penalty = 0.0
            
            motion_penalties.append(motion_penalty)
        
        # 运动效率是运动惩罚的反向
        avg_motion_penalty = np.mean(motion_penalties)
        motion_efficiency = 1.0 / (1.0 + avg_motion_penalty)
        
        return float(motion_efficiency)
    except Exception as e:
        print(f"Error in reach_motion_efficiency_enhanced: {e}")
        return 0.0
    except Exception as e:
        print(f"Error in enhanced_balance_stability: {e}")
        return balance_stability(trajectory, goal)


# TrajectoryQualityEvaluator类已移至preference_labeling_engine.py中
# 为保持向后兼容性，这里提供一个导入别名
try:
    from .preference_labeling_engine import TrajectoryQualityEvaluator
except ImportError:
    # 如果导入失败，提供一个简化的兼容性类
    class TrajectoryQualityEvaluator:
        """兼容性类 - 实际实现在preference_labeling_engine.py中"""
        def __init__(self, task_name=None, weights=None):
            import warnings
            warnings.warn(
                "TrajectoryQualityEvaluator已移至preference_labeling_engine.py，"
                "请直接从该模块导入以获得完整功能",
                DeprecationWarning
            )
            self.task_name = task_name
            self.weights = weights
    
        def evaluate_trajectory_quality(self, obs_seq, act_seq, rewards=None):
            """兼容性方法 - 请使用preference_labeling_engine中的完整实现"""
            return 0.5, {}


class TrajectoryMetrics:
    """轨迹指标计算器：为DPO标签系统提供轨迹质量评估功能
    
    该类封装了所有轨迹评估函数，支持任务感知的指标计算，
    为preference_data_engine.py中的DPO标签增强提供核心评估能力。
    """
    
    def __init__(self):
        """初始化轨迹指标计算器"""
        # 通用指标函数映射
        self.general_metrics = {
            'survival_time': survival_time,
            'distance_to_goal': distance_to_goal,
            'is_task_successful': is_task_successful,
            'efficiency': efficiency,
            'safety_score': safety_score,
            'energy_usage': energy_usage,
            'action_diversity': action_diversity,
            'action_entropy': action_entropy,
        }
        
        # 任务特定指标函数映射
        self.task_specific_metrics = {
            'walk': {
                'gait_quality': gait_quality,
                'enhanced_gait_quality': enhanced_gait_quality,
                'forward_progress': forward_progress,
                'enhanced_forward_progress': enhanced_forward_progress,
                'balance_stability': balance_stability,
                'enhanced_balance_stability': enhanced_balance_stability,
            },
            'stand': {
                'standing_height_stability': standing_height_stability,
                'minimal_movement': minimal_movement,
                'balance_stability': balance_stability,
                'enhanced_balance_stability': enhanced_balance_stability,
            },
            'hurdle': {
                'hurdle_forward_progress': hurdle_forward_progress,
                'hurdle_obstacle_collision': hurdle_obstacle_collision,
                'hurdle_running_efficiency': hurdle_running_efficiency,
                'hurdle_speed_consistency': hurdle_speed_consistency,
                'hurdle_balance_stability': hurdle_balance_stability,
                'hurdle_obstacle_clearance': hurdle_obstacle_clearance,
            },
            'climb': {
                'height_gain': height_gain,
                'climbing_efficiency': climbing_efficiency,
                'step_coordination': step_coordination,
            },
            'maze': {
                'maze_standing_stability': maze_standing_stability,
                'forward_progress': forward_progress,
                'balance_stability': balance_stability,
            },
            'reach': {
                'hand_to_goal_distance': hand_to_goal_distance,
                'reach_efficiency': reach_efficiency,
                'reach_success_rate': reach_success_rate,
                'reach_smoothness': reach_smoothness,
                'reach_stability': reach_stability,
            }
        }
    
    def compute_metrics(self, trajectories, task_name=None, requested_metrics=None, goal=None):
        """批量计算轨迹指标
        
        Args:
            trajectories: 轨迹列表或单个轨迹
            task_name: 任务名称，用于选择任务特定指标
            requested_metrics: 请求的指标列表，如果为None则计算所有相关指标
            goal: 目标参数，用于需要目标信息的指标计算
            
        Returns:
            dict: 指标名称到值的映射，或轨迹列表对应的指标字典列表
        """
        # 处理单个轨迹的情况
        if not isinstance(trajectories, list):
            trajectories = [trajectories]
        
        results = []
        for trajectory in trajectories:
            trajectory_metrics = self._compute_single_trajectory_metrics(
                trajectory, task_name, requested_metrics, goal
            )
            results.append(trajectory_metrics)
        
        # 如果输入是单个轨迹，返回单个结果
        return results[0] if len(results) == 1 else results
    
    def compute_single_metric(self, trajectory, metric_name, task_name=None, goal=None):
        """计算单个指标
        
        Args:
            trajectory: 轨迹数据
            metric_name: 指标名称
            task_name: 任务名称
            goal: 目标参数，用于需要目标信息的指标计算
            
        Returns:
            float: 指标值
        """
        # 首先检查通用指标
        if metric_name in self.general_metrics:
            metric_func = self.general_metrics[metric_name]
            return self._safe_compute_metric(metric_func, trajectory, goal)
        
        # 检查任务特定指标
        if task_name and task_name in self.task_specific_metrics:
            task_metrics = self.task_specific_metrics[task_name]
            if metric_name in task_metrics:
                metric_func = task_metrics[metric_name]
                return self._safe_compute_metric(metric_func, trajectory, goal)
        
        # 如果找不到指标，返回默认值
        print(f"Warning: Unknown metric '{metric_name}' for task '{task_name}'")
        return 0.0
    
    def _compute_single_trajectory_metrics(self, trajectory, task_name=None, requested_metrics=None, goal=None):
        """计算单个轨迹的所有相关指标"""
        metrics = {}
        
        # 计算通用指标
        for metric_name, metric_func in self.general_metrics.items():
            if requested_metrics is None or metric_name in requested_metrics:
                metrics[metric_name] = self._safe_compute_metric(metric_func, trajectory, goal)
        
        # 计算任务特定指标
        if task_name and task_name in self.task_specific_metrics:
            task_metrics = self.task_specific_metrics[task_name]
            for metric_name, metric_func in task_metrics.items():
                if requested_metrics is None or metric_name in requested_metrics:
                    metrics[metric_name] = self._safe_compute_metric(metric_func, trajectory, goal)
        
        return metrics
    
    def _safe_compute_metric(self, metric_func, trajectory, goal=None):
        """安全地计算指标，包含错误处理"""
        try:
            # 转换轨迹格式以确保兼容性
            formatted_trajectory = self._format_trajectory(trajectory)
            result = metric_func(formatted_trajectory, goal)
            
            # 确保结果是数值类型
            if isinstance(result, (int, float)):
                return float(np.clip(result, -1e6, 1e6))  # 防止极值
            else:
                return 0.0
        except Exception as e:
            print(f"Error computing metric {metric_func.__name__}: {e}")
            return 0.0
    
    def _format_trajectory(self, trajectory):
        """格式化轨迹数据以确保与指标函数的兼容性"""
        # 如果轨迹已经是对象格式，直接返回
        if hasattr(trajectory, 'states') or hasattr(trajectory, 'actions'):
            return trajectory
        
        # 如果是字典格式，转换为对象格式
        if isinstance(trajectory, dict):
            class TrajectoryWrapper:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
                    # 确保 'obs' 键映射到 'states' 属性
                    if 'obs' in data and not hasattr(self, 'states'):
                        self.states = data['obs']
                    # 确保 'action' 键映射到 'actions' 属性
                    if 'action' in data and not hasattr(self, 'actions'):
                        self.actions = data['action']
            
            return TrajectoryWrapper(trajectory)
        
        # 其他情况直接返回
        return trajectory
    
    def get_available_metrics(self, task_name=None):
        """获取可用的指标列表
        
        Args:
            task_name: 任务名称，如果提供则包含任务特定指标
            
        Returns:
            list: 可用指标名称列表
        """
        metrics = list(self.general_metrics.keys())
        
        if task_name and task_name in self.task_specific_metrics:
            metrics.extend(list(self.task_specific_metrics[task_name].keys()))
        
        return sorted(metrics)


# 删除重复的TrajectoryMetrics类定义

# ==================== Hurdle Task Metrics ====================

def hurdle_forward_progress(trajectory, goal=None):
    """计算hurdle任务的前向进展：基于x轴位移的进展程度
    
    基于Hurdle类的奖励函数实现：
    - 使用_RUN_SPEED作为移动速度基准
    - 前向进展是hurdle任务的核心目标
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 2:
        return 0.0
    
    # 提取初始和最终位置
    initial_state = states[0]
    final_state = states[-1]
    
    # 获取观测数据
    if isinstance(initial_state, dict):
        initial_obs = initial_state["obs"]
    else:
        initial_obs = initial_state
    
    if isinstance(final_state, dict):
        final_obs = final_state["obs"]
    else:
        final_obs = final_state
    
    initial_obs = np.array(initial_obs)
    final_obs = np.array(final_obs)
    
    # 提取x坐标（通常是观测的第一个元素）
    initial_x = initial_obs[0] if len(initial_obs) > 0 else 0.0
    final_x = final_obs[0] if len(final_obs) > 0 else 0.0
    
    # 计算前向进展（x轴正方向）
    forward_progress = final_x - initial_x
    
    # 归一化：基于轨迹长度和期望速度
    trajectory_length = len(states)
    expected_distance = trajectory_length * 0.02  # 假设每步0.02m的期望进展
    
    if expected_distance > 0:
        normalized_progress = forward_progress / expected_distance
    else:
        normalized_progress = 0.0
    
    return float(max(0.0, normalized_progress))

def hurdle_obstacle_collision(trajectory, goal=None):
    """检测hurdle任务中的障碍物碰撞：基于Hurdle类的碰撞检测机制
    
    基于Hurdle类的奖励函数实现：
    - 检测与墙壁/障碍物的碰撞
    - 碰撞会降低总奖励
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return 1.0  # 无状态时认为无碰撞
    
    collision_count = 0
    total_checks = 0
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 检查是否有碰撞指示器
        # 在humanoid_bench中，碰撞通常通过接触力或特定观测维度表示
        # 这里使用启发式方法：检查身体高度和姿态异常
        
        if len(obs) >= 3:
            # 检查身体高度（z坐标）
            body_height = obs[2] if len(obs) > 2 else 1.0
            
            # 检查是否跌倒（高度过低）
            if body_height < 0.5:  # 正常站立高度应该在1.0左右
                collision_count += 1
            
            # 检查姿态异常（如果有四元数信息）
            if len(obs) >= 7:
                quaternion = obs[3:7]
                w, x, y, z = quaternion
                
                # 检查是否严重倾斜
                z_up_component = 2 * (x * z + w * y)
                if abs(z_up_component) > 0.8:  # 严重倾斜
                    collision_count += 1
        
        total_checks += 1
    
    # 计算无碰撞率
    if total_checks > 0:
        collision_rate = collision_count / total_checks
        no_collision_rate = 1.0 - collision_rate
    else:
        no_collision_rate = 1.0
    
    return float(max(0.0, no_collision_rate))

def hurdle_running_efficiency(trajectory, goal=None):
    """计算hurdle任务的跑步效率：基于能量消耗和速度的比值
    
    基于Hurdle类的奖励函数实现：
    - 包含小控制力奖励（small_control）
    - 效率 = 前向进展 / 能量消耗
    """
    states = getattr(trajectory, "states", [])
    actions = getattr(trajectory, "actions", [])
    
    if len(states) < 2 or len(actions) < 1:
        return 0.0
    
    # 计算前向进展
    forward_progress = hurdle_forward_progress(trajectory, goal)
    
    # 计算能量消耗（基于动作的平方和）
    total_energy = 0.0
    action_count = 0
    
    for action in actions:
        if isinstance(action, dict):
            action_vec = np.array(action["action"])
        else:
            action_vec = np.array(action)
        
        # 计算动作的平方和（对应控制力的平方）
        energy = np.square(action_vec).sum()
        total_energy += energy
        action_count += 1
    
    # 计算平均能量消耗
    if action_count > 0:
        avg_energy = total_energy / action_count
        
        # 效率 = 进展 / (1 + 归一化能量消耗)
        # 归一化能量消耗，避免除零
        normalized_energy = avg_energy * 0.001  # 缩放因子
        efficiency = forward_progress / (1.0 + normalized_energy)
    else:
        efficiency = 0.0
    
    return float(max(0.0, efficiency))

def hurdle_speed_consistency(trajectory, goal=None):
    """评估hurdle任务中的速度一致性：稳定的跑步速度
    
    基于Hurdle类的移动奖励实现：
    - 使用_RUN_SPEED作为目标速度
    - 速度一致性有助于稳定通过障碍物
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 3:
        return 0.0
    
    # 计算每个时间步的速度
    speeds = []
    
    for i in range(1, len(states)):
        prev_state = states[i-1]
        curr_state = states[i]
        
        # 获取位置信息
        if isinstance(prev_state, dict):
            prev_obs = prev_state["obs"]
        else:
            prev_obs = prev_state
        
        if isinstance(curr_state, dict):
            curr_obs = curr_state["obs"]
        else:
            curr_obs = curr_state
        
        prev_obs = np.array(prev_obs)
        curr_obs = np.array(curr_obs)
        
        # 计算位置变化（前3维通常是x,y,z坐标）
        if len(prev_obs) >= 3 and len(curr_obs) >= 3:
            prev_pos = prev_obs[:3]
            curr_pos = curr_obs[:3]
            
            # 计算速度（主要关注x方向）
            velocity = curr_pos - prev_pos
            speed = np.linalg.norm(velocity)
            speeds.append(speed)
    
    if len(speeds) < 2:
        return 0.0
    
    # 计算速度的一致性
    speeds = np.array(speeds)
    mean_speed = np.mean(speeds)
    speed_variance = np.var(speeds)
    
    # 一致性评分：方差越小越一致
    if mean_speed > 0:
        consistency_score = 1.0 / (1.0 + speed_variance / (mean_speed ** 2) * 10.0)
    else:
        consistency_score = 0.0
    
    return float(np.clip(consistency_score, 0.0, 1.0))

def hurdle_balance_stability(trajectory, goal=None):
    """评估hurdle任务中的平衡稳定性：基于躯干姿态的稳定性
    
    基于Hurdle类的直立奖励实现：
    - 包含站立奖励和直立奖励
    - 在跨越障碍物时保持平衡至关重要
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    # 计算平衡稳定性指标
    stability_scores = []
    height_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        if len(obs) >= 7:
            # 提取身体高度（z坐标）
            body_height = obs[2] if len(obs) > 2 else 1.0
            
            # 高度稳定性：接近正常站立高度
            height_stability = 1.0 - abs(body_height - 1.0) / 2.0
            height_scores.append(max(0.0, height_stability))
            
            # 提取四元数姿态
            quaternion = obs[3:7]
            w, x, y, z = quaternion
            
            # 计算躯干直立程度
            z_up_component = 2 * (x * z + w * y)
            upright_score = 1.0 - abs(z_up_component)
            stability_scores.append(max(0.0, upright_score))
        else:
            # 如果观测维度不足，使用默认值
            height_scores.append(0.5)
            stability_scores.append(0.5)
    
    # 综合评估稳定性
    if stability_scores and height_scores:
        avg_stability = np.mean(stability_scores)
        avg_height_stability = np.mean(height_scores)
        
        # 考虑稳定性的一致性
        stability_variance = np.var(stability_scores) if len(stability_scores) > 1 else 0.0
        consistency_bonus = 1.0 / (1.0 + stability_variance * 5.0)
        
        # 综合评分
        total_stability = 0.4 * avg_stability + 0.4 * avg_height_stability + 0.2 * consistency_bonus
    else:
        total_stability = 0.0
    
    return float(np.clip(total_stability, 0.0, 1.0))

def hurdle_obstacle_clearance(trajectory, goal=None):
    """评估hurdle任务中的障碍物通过能力：基于高度变化和前向进展
    
    障碍物通过需要：
    1. 适当的高度变化（跳跃或抬腿）
    2. 持续的前向进展
    3. 稳定的着陆
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 分析高度变化模式
    heights = []
    x_positions = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        if len(obs) >= 3:
            x_pos = obs[0]
            height = obs[2]
            
            x_positions.append(x_pos)
            heights.append(height)
    
    if len(heights) < 10:
        return 0.0
    
    heights = np.array(heights)
    x_positions = np.array(x_positions)
    
    # 检测高度变化模式（寻找跳跃或抬腿动作）
    height_changes = np.diff(heights)
    
    # 寻找显著的高度增加（可能的跳跃）
    significant_height_increases = np.sum(height_changes > 0.1)
    
    # 检查前向进展的连续性
    x_progress = x_positions[-1] - x_positions[0]
    
    # 检查高度变化的合理性（不应该有剧烈下降）
    severe_drops = np.sum(height_changes < -0.3)
    
    # 综合评分
    clearance_score = 0.0
    
    # 前向进展奖励
    if x_progress > 0:
        clearance_score += min(1.0, x_progress / 10.0) * 0.5
    
    # 高度变化奖励（适度的高度变化是好的）
    if significant_height_increases > 0:
        clearance_score += min(1.0, significant_height_increases / 5.0) * 0.3
    
    # 稳定性惩罚（严重下降是不好的）
    if severe_drops > 0:
        clearance_score -= min(0.5, severe_drops / 3.0)
    
    # 高度一致性奖励（最终高度应该接近初始高度）
    height_consistency = 1.0 - abs(heights[-1] - heights[0]) / 2.0
    clearance_score += max(0.0, height_consistency) * 0.2
    
    return float(np.clip(clearance_score, 0.0, 1.0))

# ===== MAZE 任务专用指标函数 =====
def maze_standing_stability(trajectory, goal=None):
    """评估maze任务中的站立稳定性：基于实际奖励函数的stand_reward
    
    基于maze.py中的实际实现：
    - standing = tolerance(head_height, bounds=(_STAND_HEIGHT, inf), margin=_STAND_HEIGHT/4)
    - upright = tolerance(torso_upright, bounds=(0.9, inf), sigmoid="linear", margin=1.9)
    - stand_reward = standing * upright
    - _STAND_HEIGHT = 1.65
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    # 提取头部高度和躯干姿态信息
    standing_scores = []
    upright_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 头部高度（假设在质心高度基础上加上头部偏移）
        if len(obs) >= 3:
            # 质心高度 + 头部偏移（约0.2m）
            head_height = obs[2] + 0.2
            
            # 站立奖励：基于头部高度的tolerance函数
            # tolerance(head_height, bounds=(1.65, inf), margin=1.65/4)
            stand_height = 1.65
            margin = stand_height / 4
            if head_height >= stand_height:
                standing_score = 1.0
            elif head_height >= stand_height - margin:
                # 线性插值
                standing_score = (head_height - (stand_height - margin)) / margin
            else:
                standing_score = 0.0
            
            standing_scores.append(standing_score)
        
        # 躯干直立度（基于四元数）
        if len(obs) >= 7:
            quaternion = obs[3:7]
            w, x, y, z = quaternion
            
            # 计算躯干直立度（Z轴向上分量）
            # torso_upright类似于Z轴的cos值
            torso_upright = 1.0 - 2.0 * (x*x + y*y)  # 四元数到旋转矩阵的Z轴分量
            
            # upright奖励：tolerance(torso_upright, bounds=(0.9, inf), margin=1.9)
            if torso_upright >= 0.9:
                upright_score = 1.0
            elif torso_upright >= 0.9 - 1.9:  # 下界为-1.0
                upright_score = (torso_upright - (0.9 - 1.9)) / 1.9
            else:
                upright_score = 0.0
            
            upright_scores.append(max(0.0, upright_score))
    
    # 计算平均站立稳定性
    if standing_scores and upright_scores:
        avg_standing = np.mean(standing_scores)
        avg_upright = np.mean(upright_scores)
        # stand_reward = standing * upright
        stability = avg_standing * avg_upright
    else:
        stability = 0.0
    
    return float(np.clip(stability, 0.0, 1.0))

def maze_control_efficiency(trajectory, goal=None):
    """评估maze任务中的控制效率：基于实际奖励函数的small_control
    
    基于maze.py中的实际实现：
    - small_control = tolerance(actuator_forces, margin=10, sigmoid="quadratic").mean()
    - small_control = (4 + small_control) / 5
    """
    actions = getattr(trajectory, "actions", [])
    if len(actions) < 2:
        return 1.0
    
    # 计算执行器力的控制效率
    control_scores = []
    
    for action in actions:
        if isinstance(action, dict):
            act = np.array(action["action"])
        else:
            act = np.array(action)
        
        # 计算执行器力的大小（假设动作直接对应执行器力）
        actuator_forces = np.abs(act)
        
        # tolerance函数：margin=10, sigmoid="quadratic"
        # 对于quadratic sigmoid: score = 1 / (1 + (x/margin)^2)
        margin = 10.0
        force_scores = 1.0 / (1.0 + np.square(actuator_forces / margin))
        
        # 平均控制分数
        avg_control = np.mean(force_scores)
        
        # 应用maze.py中的变换：(4 + small_control) / 5
        adjusted_control = (4 + avg_control) / 5
        control_scores.append(adjusted_control)
    
    # 计算平均控制效率
    if control_scores:
        efficiency = np.mean(control_scores)
    else:
        efficiency = 1.0
    
    return float(np.clip(efficiency, 0.0, 1.0))

def maze_movement_efficiency(trajectory, goal=None):
    """评估maze任务中的移动效率：基于实际奖励函数的move奖励
    
    基于maze.py中的实际实现：
    - move = tolerance(com_velocity[0] - move_direction[0] * _MOVE_SPEED, margin=1) *
             tolerance(com_velocity[1] - move_direction[1] * _MOVE_SPEED, margin=1)
    - move = (5 * move + 1) / 6
    - _MOVE_SPEED = 2.0
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 3:
        return 0.0
    
    # 计算质心速度序列
    velocities = []
    
    for i in range(1, len(states)):
        if isinstance(states[i], dict):
            pos1 = np.array(states[i-1]["obs"][:3])
            pos2 = np.array(states[i]["obs"][:3])
        else:
            pos1 = np.array(states[i-1][:3])
            pos2 = np.array(states[i][:3])
        
        # 计算速度（前向和侧向）
        dt = 0.002  # maze.py中的timestep
        velocity = (pos2 - pos1) / dt
        velocities.append(velocity[:2])  # 只考虑x,y方向
    
    if not velocities:
        return 0.0
    
    # 计算移动奖励（简化版本，假设主要是前向移动）
    move_speed = 2.0
    move_scores = []
    
    for vel in velocities:
        # 假设主要移动方向是前向（x方向）
        target_vel_x = move_speed  # 前向移动
        target_vel_y = 0.0         # 无侧向移动
        
        # tolerance函数：margin=1, sigmoid="linear"
        # 对于linear sigmoid: score = max(0, 1 - |error|/margin)
        margin = 1.0
        
        error_x = abs(vel[0] - target_vel_x)
        error_y = abs(vel[1] - target_vel_y)
        
        score_x = max(0.0, 1.0 - error_x / margin)
        score_y = max(0.0, 1.0 - error_y / margin)
        
        # move = score_x * score_y
        move_score = score_x * score_y
        
        # 应用maze.py中的变换：(5 * move + 1) / 6
        adjusted_score = (5 * move_score + 1) / 6
        move_scores.append(adjusted_score)
    
    # 计算平均移动效率
    avg_efficiency = np.mean(move_scores)
    return float(np.clip(avg_efficiency, 0.0, 1.0))

def maze_checkpoint_progress(trajectory, goal=None):
    """评估maze任务中的检查点进展：基于实际奖励函数的checkpoint_proximity_reward
    
    基于maze.py中的实际实现：
    - checkpoint_proximity = norm(checkpoints[maze_stage][:2] - imu_pos[:2])
    - checkpoint_proximity_reward = tolerance(checkpoint_proximity, margin=1)
    - checkpoints = [[0,0,1], [3,0,1], [3,6,1], [6,6,1], [6,6,1]]
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return 0.1
    
    # 定义检查点（与maze.py中的Maze类一致）
    checkpoints = [
        np.array([0, 0, 1]),
        np.array([3, 0, 1]),
        np.array([3, 6, 1]),
        np.array([6, 6, 1]),
        np.array([6, 6, 1]),
    ]
    
    # 计算轨迹中每个状态到最近检查点的距离
    proximity_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 机器人位置（假设是质心位置）
        if len(obs) >= 3:
            robot_pos = obs[:2]  # x, y坐标
            
            # 计算到所有检查点的距离，找到最小值
            min_distance = float('inf')
            for checkpoint in checkpoints:
                distance = np.linalg.norm(robot_pos - checkpoint[:2])
                min_distance = min(min_distance, distance)
            
            # tolerance函数：margin=1
            # 对于默认sigmoid: score = exp(-distance/margin)
            margin = 1.0
            proximity_score = np.exp(-min_distance / margin)
            proximity_scores.append(proximity_score)
    
    # 计算平均检查点接近度
    if proximity_scores:
        avg_proximity = np.mean(proximity_scores)
        
        # 额外奖励：如果轨迹显示出向检查点移动的趋势
        if len(proximity_scores) > 1:
            # 检查是否有改善趋势（距离减小）
            improvement_trend = 0.0
            for i in range(1, len(proximity_scores)):
                if proximity_scores[i] > proximity_scores[i-1]:
                    improvement_trend += 1
            
            trend_bonus = improvement_trend / (len(proximity_scores) - 1)
            total_progress = 0.8 * avg_proximity + 0.2 * trend_bonus
        else:
            total_progress = avg_proximity
    else:
        total_progress = 0.0
    
    return float(np.clip(total_progress, 0.0, 1.0))

def maze_wall_collision_avoidance(trajectory, goal=None):
    """评估maze任务中的墙壁碰撞避免：基于实际奖励函数的wall_collision_discount
    
    基于maze.py中的实际实现：
    - wall_collision_discount = 1 if no collision else 0.1
    - 检测与block_collision_xx的碰撞
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return 1.0  # 没有状态，假设无碰撞
    
    # 简化的碰撞检测：基于位置是否接近墙壁
    # 由于无法直接访问碰撞信息，使用位置启发式
    collision_penalty = 0.0
    total_checks = 0
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 机器人位置
        if len(obs) >= 3:
            robot_pos = obs[:2]  # x, y坐标
            
            # 检查是否接近迷宫边界（简化的墙壁检测）
            # 假设迷宫边界在[0,7] x [0,7]范围内
            boundary_margin = 0.3  # 接近边界的阈值
            
            # 检查是否太接近边界
            too_close_to_boundary = (
                robot_pos[0] < boundary_margin or robot_pos[0] > 7 - boundary_margin or
                robot_pos[1] < boundary_margin or robot_pos[1] > 7 - boundary_margin
            )
            
            # 检查是否在已知的墙壁区域（基于迷宫布局）
            # 这里使用简化的墙壁位置检测
            in_wall_area = False
            
            # 垂直墙壁区域（简化）
            if (1.5 < robot_pos[0] < 2.5 and 0 < robot_pos[1] < 5) or \
               (4.5 < robot_pos[0] < 5.5 and 2 < robot_pos[1] < 7):
                in_wall_area = True
            
            # 水平墙壁区域（简化）
            if (0 < robot_pos[0] < 3 and 2.5 < robot_pos[1] < 3.5) or \
               (3 < robot_pos[0] < 7 and 4.5 < robot_pos[1] < 5.5):
                in_wall_area = True
            
            if too_close_to_boundary or in_wall_area:
                collision_penalty += 1.0
            
            total_checks += 1
    
    # 计算碰撞避免分数
    if total_checks > 0:
        collision_rate = collision_penalty / total_checks
        # 模拟maze.py中的wall_collision_discount逻辑
        if collision_rate > 0.1:  # 如果碰撞率超过10%
            avoidance_score = 0.1  # 对应maze.py中的0.1折扣
        else:
            avoidance_score = 1.0 - collision_rate * 0.9  # 线性惩罚
    else:
        avoidance_score = 1.0
    
    return float(np.clip(avoidance_score, 0.0, 1.0))

def maze_navigation_efficiency(trajectory, goal=None):
    """评估maze任务的导航效率：路径长度vs直线距离的比值
    
    高效的导航应该：
    1. 路径相对较短
    2. 避免不必要的绕行
    3. 朝着目标方向移动
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 2:
        return 0.0
    
    # 提取位置序列
    positions = []
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        if len(obs) >= 2:
            positions.append(obs[:2])  # x, y坐标
    
    if len(positions) < 2:
        return 0.0
    
    positions = np.array(positions)
    
    # 计算实际路径长度
    path_length = 0.0
    for i in range(1, len(positions)):
        path_length += np.linalg.norm(positions[i] - positions[i-1])
    
    # 计算直线距离
    straight_distance = np.linalg.norm(positions[-1] - positions[0])
    
    # 路径效率：直线距离 / 实际路径长度
    if path_length > 0 and straight_distance > 0:
        path_efficiency = straight_distance / path_length
        
        # 额外考虑总体进展（移动距离）
        progress_bonus = min(straight_distance / 5.0, 1.0)  # 归一化到5米
        
        # 综合效率
        total_efficiency = 0.7 * path_efficiency + 0.3 * progress_bonus
    else:
        total_efficiency = 0.0
    
    return float(np.clip(total_efficiency, 0.0, 1.0))

def maze_success_rate(trajectory, goal=None):
    """计算maze任务的成功率：基于实际奖励函数的success_bar和检查点完成情况
    
    基于maze.py中的实际实现：
    - success_bar = 1200
    - 成功标准：完成所有检查点并维持足够长的时间
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return 0.1
    
    # 检查轨迹长度（对应success_bar）
    trajectory_length = len(states)
    success_length = 1200
    
    length_score = min(trajectory_length / success_length, 1.0)
    
    # 检查检查点完成情况
    checkpoints = [
        np.array([0, 0, 1]),
        np.array([3, 0, 1]),
        np.array([3, 6, 1]),
        np.array([6, 6, 1]),
        np.array([6, 6, 1]),
    ]
    
    # 计算到达的检查点数量
    reached_checkpoints = 0
    
    for checkpoint in checkpoints[:-1]:  # 排除最后一个重复的检查点
        min_distance = float('inf')
        
        for state in states:
            if isinstance(state, dict):
                obs = state["obs"]
            else:
                obs = state
            
            obs = np.array(obs)
            if len(obs) >= 2:
                robot_pos = obs[:2]
                distance = np.linalg.norm(robot_pos - checkpoint[:2])
                min_distance = min(min_distance, distance)
        
        # 如果曾经接近过这个检查点（距离<0.5m），认为到达了
        if min_distance < 0.5:
            reached_checkpoints += 1
    
    # 检查点完成分数
    checkpoint_score = reached_checkpoints / (len(checkpoints) - 1)
    
    # 综合成功率
    success_rate = 0.4 * length_score + 0.6 * checkpoint_score
    
    return float(np.clip(success_rate, 0.0, 1.0))

def maze_overall_performance(trajectory, goal=None):
    """计算maze任务的整体性能：基于实际奖励函数的综合评分
    
    基于maze.py中的实际实现：
    reward = (0.2 * (stand_reward * small_control) + 0.4 * move + 0.4 * checkpoint_proximity_reward) * wall_collision_discount + stage_convert_reward
    
    综合考虑：
    1. 站立稳定性和控制效率（20%权重）
    2. 移动效率（40%权重）
    3. 检查点接近度（40%权重）
    4. 墙壁碰撞避免（乘法因子）
    5. 导航效率（额外奖励）
    """
    try:
        # 获取各项指标
        standing = maze_standing_stability(trajectory, goal)
        control = maze_control_efficiency(trajectory, goal)
        movement = maze_movement_efficiency(trajectory, goal)
        checkpoint = maze_checkpoint_progress(trajectory, goal)
        collision = maze_wall_collision_avoidance(trajectory, goal)
        navigation = maze_navigation_efficiency(trajectory, goal)
        
        # 按照maze.py的奖励函数结构计算
        # reward = (0.2 * (stand_reward * small_control) + 0.4 * move + 0.4 * checkpoint_proximity_reward) * wall_collision_discount
        base_performance = (0.2 * (standing * control) + 0.4 * movement + 0.4 * checkpoint) * collision
        
        # 添加导航效率作为额外奖励
        enhanced_performance = base_performance + 0.1 * navigation
        
        return float(np.clip(enhanced_performance, 0.0, 1.0))
    except Exception:
        return 0.5

# ===== BALANCE_SIMPLE 任务专用指标函数 =====

def balance_simple_standing_stability(trajectory, goal=None):
    """评估balance_simple任务的站立稳定性：基于头部高度和躯干直立度
    
    基于balance.py中的实际奖励函数：
    - standing = tolerance(head_height, bounds=(_STAND_HEIGHT + 0.37, inf), margin=_STAND_HEIGHT/4)
    - upright = tolerance(torso_upright, bounds=(0.9, inf), margin=1.9)
    - stand_reward = standing * upright
    
    其中_STAND_HEIGHT = 1.65，所以头部高度阈值为2.02m
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    standing_scores = []
    upright_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 提取头部高度（通常是Z坐标）
        if len(obs) >= 3:
            head_height = obs[2]  # Z坐标
        else:
            head_height = 1.0
        
        # 计算站立得分：头部高度 > 2.02m (1.65 + 0.37)
        target_height = 2.02
        margin = 1.65 / 4  # 0.4125
        
        if head_height >= target_height:
            standing_score = 1.0
        elif head_height >= target_height - margin:
            # 线性插值
            standing_score = (head_height - (target_height - margin)) / margin
        else:
            standing_score = 0.0
        
        standing_scores.append(max(0.0, standing_score))
        
        # 提取四元数并计算躯干直立度
        if len(obs) >= 7:
            quaternion = obs[3:7]  # [w,x,y,z]
            
            # 计算躯干直立度（Z轴向上分量）
            w, x, y, z = quaternion
            # 旋转矩阵的(2,2)元素表示Z轴在世界坐标系中的Z分量
            z_up = 1 - 2 * (x*x + y*y)
            
            # 直立得分：z_up > 0.9
            if z_up >= 0.9:
                upright_score = 1.0
            elif z_up >= 0.9 - 1.9:  # margin = 1.9
                upright_score = max(0.0, (z_up - (0.9 - 1.9)) / 1.9)
            else:
                upright_score = 0.0
        else:
            upright_score = 0.5
        
        upright_scores.append(max(0.0, upright_score))
    
    # 计算平均站立稳定性（standing * upright）
    if standing_scores and upright_scores:
        combined_scores = [s * u for s, u in zip(standing_scores, upright_scores)]
        avg_stability = np.mean(combined_scores)
    else:
        avg_stability = 0.0
    
    return float(np.clip(avg_stability, 0.0, 1.0))

def balance_simple_control_smoothness(trajectory, goal=None):
    """评估balance_simple任务的控制平滑性：基于执行器力的平滑程度
    
    基于balance.py中的实际奖励函数：
    - small_control = tolerance(actuator_forces, margin=10, sigmoid="quadratic").mean()
    - small_control = (4 + small_control) / 5
    
    控制力越平滑，得分越高
    """
    actions = getattr(trajectory, "actions", [])
    if len(actions) < 3:
        return 1.0
    
    # 计算动作序列的平滑性
    action_forces = []
    for action in actions:
        if isinstance(action, dict):
            act = np.array(action["action"])
        else:
            act = np.array(action)
        
        # 计算动作力的L2范数
        force_magnitude = np.linalg.norm(act)
        action_forces.append(force_magnitude)
    
    # 计算力的变化率
    force_changes = np.diff(action_forces)
    
    # 平滑性评估：变化越小越平滑
    if len(force_changes) > 0:
        avg_change = np.mean(np.abs(force_changes))
        # 使用与balance.py相似的tolerance函数，margin=10
        margin = 10.0
        smoothness_score = 1.0 / (1.0 + (avg_change / margin) ** 2)  # quadratic sigmoid
        # 应用与原始奖励相同的变换
        smoothness_score = (4 + smoothness_score) / 5
    else:
        smoothness_score = 1.0
    
    return float(np.clip(smoothness_score, 0.0, 1.0))

def balance_simple_static_stability(trajectory, goal=None):
    """评估balance_simple任务的静止稳定性：基于水平速度控制
    
    基于balance.py中的实际奖励函数：
    - horizontal_velocity = center_of_mass_velocity()[[0, 1]]
    - dont_move = tolerance(horizontal_velocity, margin=2).mean()
    
    水平速度越接近0，静止稳定性越好
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    velocity_scores = []
    
    for i in range(1, len(states)):
        # 计算位置变化来估算速度
        if isinstance(states[i], dict):
            pos_curr = np.array(states[i]["obs"][:3])
            pos_prev = np.array(states[i-1]["obs"][:3])
        else:
            pos_curr = np.array(states[i][:3])
            pos_prev = np.array(states[i-1][:3])
        
        # 计算水平速度（X和Y方向）
        horizontal_velocity = pos_curr[:2] - pos_prev[:2]
        velocity_magnitude = np.linalg.norm(horizontal_velocity)
        
        # 静止得分：速度越小越好，margin=2
        margin = 2.0
        static_score = 1.0 / (1.0 + velocity_magnitude / margin)
        velocity_scores.append(static_score)
    
    # 计算平均静止稳定性
    if velocity_scores:
        avg_static = np.mean(velocity_scores)
    else:
        avg_static = 0.0
    
    return float(np.clip(avg_static, 0.0, 1.0))

def balance_simple_collision_avoidance(trajectory, goal=None):
    """评估balance_simple任务的碰撞避免：基于球体碰撞检测
    
    基于balance.py中的get_terminated()逻辑：
    - 检测pivot_sphere_collision与其他物体的碰撞
    - 避免与地面的不当接触
    
    由于无法直接访问碰撞数据，通过间接指标评估
    """
    states = getattr(trajectory, "states", [])
    actions = getattr(trajectory, "actions", [])
    
    if len(states) < 5:
        return 1.0  # 默认无碰撞
    
    collision_risk_scores = []
    
    for i, state in enumerate(states):
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 通过高度和姿态变化评估碰撞风险
        if len(obs) >= 3:
            height = obs[2]
            
            # 高度过低表示可能发生碰撞
            if height < 0.8:  # 与get_terminated()中的阈值一致
                height_risk = 1.0
            else:
                height_risk = 0.0
        else:
            height_risk = 0.0
        
        # 通过动作幅度评估碰撞风险
        if i < len(actions):
            if isinstance(actions[i], dict):
                action = np.array(actions[i]["action"])
            else:
                action = np.array(actions[i])
            
            # 动作幅度过大可能导致失控和碰撞
            action_magnitude = np.linalg.norm(action)
            action_risk = min(1.0, action_magnitude / 5.0)  # 归一化到[0,1]
        else:
            action_risk = 0.0
        
        # 综合碰撞风险
        total_risk = max(height_risk, action_risk * 0.5)
        collision_avoidance_score = 1.0 - total_risk
        
        collision_risk_scores.append(max(0.0, collision_avoidance_score))
    
    # 计算平均碰撞避免得分
    if collision_risk_scores:
        avg_avoidance = np.mean(collision_risk_scores)
    else:
        avg_avoidance = 1.0
    
    return float(np.clip(avg_avoidance, 0.0, 1.0))

def balance_simple_height_maintenance(trajectory, goal=None):
    """评估balance_simple任务的高度维持：基于质心高度的稳定性
    
    基于balance.py中的get_terminated()条件：
    - qpos[2] < 0.8 时任务终止
    
    维持足够的高度是任务成功的基本要求
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    height_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 提取质心高度
        if len(obs) >= 3:
            height = obs[2]
        else:
            height = 1.0
        
        # 高度维持得分
        min_height = 0.8  # 最低安全高度
        target_height = 1.4  # 理想高度
        
        if height >= target_height:
            height_score = 1.0
        elif height >= min_height:
            # 线性插值
            height_score = (height - min_height) / (target_height - min_height)
        else:
            height_score = 0.0  # 低于安全高度
        
        height_scores.append(max(0.0, height_score))
    
    # 计算平均高度维持得分
    if height_scores:
        avg_height = np.mean(height_scores)
        
        # 考虑高度的稳定性（变化越小越好）
        if len(height_scores) > 1:
            height_variance = np.var(height_scores)
            stability_bonus = 1.0 / (1.0 + height_variance * 20.0)
            total_height_score = 0.8 * avg_height + 0.2 * stability_bonus
        else:
            total_height_score = avg_height
    else:
        total_height_score = 0.0
    
    return float(np.clip(total_height_score, 0.0, 1.0))

def balance_simple_overall_performance(trajectory, goal=None):
    """计算balance_simple任务的综合性能：基于所有子指标的加权组合
    
    基于balance.py中的实际奖励公式：
    reward = small_control * stand_reward * dont_move
    
    权重分配反映原始奖励函数的乘性结构
    """
    # 计算各项子指标
    standing = balance_simple_standing_stability(trajectory, goal)
    control = balance_simple_control_smoothness(trajectory, goal)
    static = balance_simple_static_stability(trajectory, goal)
    collision = balance_simple_collision_avoidance(trajectory, goal)
    height = balance_simple_height_maintenance(trajectory, goal)
    
    # 基于原始奖励函数的乘性结构计算综合性能
    # reward = small_control * stand_reward * dont_move
    # 其中 stand_reward = standing * upright（已在standing_stability中计算）
    core_performance = control * standing * static
    
    # 添加安全性和稳定性因子
    safety_factor = 0.7 * collision + 0.3 * height
    
    # 综合性能：核心性能 * 安全因子
    overall_performance = core_performance * safety_factor
    
    return float(np.clip(overall_performance, 0.0, 1.0))

# ===== POLE 任务专用指标函数 =====
def pole_standing_stability(trajectory, goal=None):
    """评估pole任务中的站立稳定性：基于实际奖励函数的standing和upright组合
    
    基于pole.py中的实际实现：
    - standing = tolerance(head_height, bounds=(_STAND_HEIGHT, inf), margin=_STAND_HEIGHT/4)
    - upright = tolerance(torso_upright, bounds=(0.9, inf), sigmoid="linear", margin=1.9)
    - stand_reward = standing * upright
    - _STAND_HEIGHT = 1.65
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    # 提取头部高度和躯干姿态信息
    standing_scores = []
    upright_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 头部高度（假设在质心高度基础上加上头部偏移）
        if len(obs) >= 3:
            # 质心高度 + 头部偏移（约0.2m）
            head_height = obs[2] + 0.2
            
            # 站立奖励：基于头部高度的tolerance函数
            # tolerance(head_height, bounds=(1.65, inf), margin=1.65/4)
            stand_height = 1.65
            margin = stand_height / 4
            if head_height >= stand_height:
                standing_score = 1.0
            elif head_height >= stand_height - margin:
                # 线性插值
                standing_score = (head_height - (stand_height - margin)) / margin
            else:
                standing_score = 0.0
            
            standing_scores.append(standing_score)
        
        # 躯干直立程度
        if len(obs) >= 7:
            # 四元数在位置[x,y,z]之后的[w,x,y,z]位置
            quaternion = obs[3:7]
            
            # 计算躯干直立程度（基于四元数）
            w, x, y, z = quaternion
            # 计算Z轴向上的分量
            z_up = 2 * (w * z + x * y)
            # 直立程度：z_up接近1表示完全直立
            torso_upright = abs(z_up)
            
            # 直立奖励：基于torso_upright的tolerance函数
            # tolerance(torso_upright, bounds=(0.9, inf), sigmoid="linear", margin=1.9)
            if torso_upright >= 0.9:
                upright_score = 1.0
            elif torso_upright >= 0.9 - 1.9:  # 实际下界为负数，所以设为0
                upright_score = max(0.0, torso_upright / 0.9)
            else:
                upright_score = 0.0
            
            upright_scores.append(upright_score)
    
    # 计算综合站立稳定性
    if standing_scores and upright_scores:
        # 站立奖励 = standing * upright（与pole.py一致）
        combined_scores = [s * u for s, u in zip(standing_scores, upright_scores)]
        avg_stability = np.mean(combined_scores)
    elif standing_scores:
        avg_stability = np.mean(standing_scores) * 0.5  # 缺少upright信息时降权
    elif upright_scores:
        avg_stability = np.mean(upright_scores) * 0.5  # 缺少standing信息时降权
    else:
        avg_stability = 0.0
    
    return float(np.clip(avg_stability, 0.0, 1.0))

def pole_movement_efficiency(trajectory, goal=None):
    """评估pole任务中的移动效率：基于实际奖励函数的move奖励
    
    基于pole.py中的实际实现：
    - com_velocity = center_of_mass_velocity()[0]  # x方向速度
    - move = tolerance(com_velocity, bounds=(_WALK_SPEED, inf), margin=_WALK_SPEED)
    - move = (5 * move + 1) / 6
    - _WALK_SPEED = 0.5
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 3:
        return 0.0
    
    # 计算质心速度序列
    velocities = []
    
    for i in range(1, len(states)):
        if isinstance(states[i], dict):
            pos1 = np.array(states[i-1]["obs"][:3])
            pos2 = np.array(states[i]["obs"][:3])
        else:
            pos1 = np.array(states[i-1][:3])
            pos2 = np.array(states[i][:3])
        
        # 计算x方向的速度（前向速度）
        dt = 0.002  # pole.py中的timestep
        velocity_x = (pos2[0] - pos1[0]) / dt
        velocities.append(velocity_x)
    
    if not velocities:
        return 0.0
    
    # 计算移动奖励
    walk_speed = 0.5
    move_scores = []
    
    for vel in velocities:
        # tolerance(com_velocity, bounds=(0.5, inf), margin=0.5)
        if vel >= walk_speed:
            move_score = 1.0
        elif vel >= 0:  # 前向移动
            move_score = vel / walk_speed
        else:  # 后向移动惩罚
            move_score = 0.0
        
        # 应用pole.py中的变换：(5 * move + 1) / 6
        adjusted_score = (5 * move_score + 1) / 6
        move_scores.append(adjusted_score)
    
    # 计算平均移动效率
    avg_efficiency = np.mean(move_scores)
    return float(np.clip(avg_efficiency, 0.0, 1.0))

def pole_collision_avoidance(trajectory, goal=None):
    """评估pole任务中的碰撞避免：基于实际奖励函数的collision_discount
    
    基于pole.py中的实际实现：
    - 检查所有接触对中是否包含"pole_r"几何体
    - 如果发生碰撞，collision_discount = 0.1
    - 如果无碰撞，collision_discount = 1.0
    
    由于轨迹数据中可能没有直接的碰撞信息，
    我们通过间接指标来评估碰撞避免能力：
    1. 动作平滑性（剧烈动作可能导致碰撞）
    2. 姿态稳定性（不稳定姿态容易碰撞）
    3. 速度合理性（过快速度容易失控碰撞）
    """
    states = getattr(trajectory, "states", [])
    actions = getattr(trajectory, "actions", [])
    
    if len(states) < 5:
        return 1.0  # 默认无碰撞
    
    # 1. 动作平滑性评估
    action_smoothness = 1.0
    if len(actions) >= 3:
        action_changes = []
        for i in range(1, len(actions)):
            if isinstance(actions[i], dict):
                action1 = np.array(actions[i-1]["action"])
                action2 = np.array(actions[i]["action"])
            else:
                action1 = np.array(actions[i-1])
                action2 = np.array(actions[i])
            
            action_change = np.linalg.norm(action2 - action1)
            action_changes.append(action_change)
        
        if action_changes:
            # 动作变化越大，碰撞风险越高
            avg_change = np.mean(action_changes)
            action_smoothness = 1.0 / (1.0 + avg_change * 2.0)
    
    # 2. 姿态稳定性评估
    posture_stability = 1.0
    orientations = []
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        if len(obs) >= 7:
            orientations.append(obs[3:7])  # 四元数
    
    if len(orientations) >= 2:
        orientation_changes = []
        for i in range(1, len(orientations)):
            q1, q2 = orientations[i-1], orientations[i]
            # 计算四元数角度差
            dot_product = np.abs(np.dot(q1, q2))
            angle_change = 2 * np.arccos(np.clip(dot_product, 0, 1))
            orientation_changes.append(angle_change)
        
        if orientation_changes:
            # 姿态变化越大，碰撞风险越高
            avg_orientation_change = np.mean(orientation_changes)
            posture_stability = 1.0 / (1.0 + avg_orientation_change * 5.0)
    
    # 3. 速度合理性评估
    velocity_reasonableness = 1.0
    if len(states) >= 3:
        velocities = []
        for i in range(1, len(states)):
            if isinstance(states[i], dict):
                pos1 = np.array(states[i-1]["obs"][:3])
                pos2 = np.array(states[i]["obs"][:3])
            else:
                pos1 = np.array(states[i-1][:3])
                pos2 = np.array(states[i][:3])
            
            velocity = np.linalg.norm(pos2 - pos1) / 0.002  # timestep
            velocities.append(velocity)
        
        if velocities:
            max_velocity = np.max(velocities)
            # 速度过快（>2m/s）增加碰撞风险
            if max_velocity > 2.0:
                velocity_reasonableness = 1.0 / (1.0 + (max_velocity - 2.0) * 0.5)
    
    # 综合碰撞避免评分
    collision_avoidance = 0.4 * action_smoothness + 0.4 * posture_stability + 0.2 * velocity_reasonableness
    return float(np.clip(collision_avoidance, 0.0, 1.0))

def pole_control_efficiency(trajectory, goal=None):
    """评估pole任务中的控制效率：基于实际奖励函数的small_control
    
    基于pole.py中的实际实现：
    - small_control = tolerance(actuator_forces(), margin=10, value_at_margin=0, sigmoid="quadratic").mean()
    - small_control = (4 + small_control) / 5
    
    由于轨迹数据中可能没有直接的执行器力信息，
    我们通过动作幅度来近似评估控制效率
    """
    actions = getattr(trajectory, "actions", [])
    if len(actions) < 2:
        return 1.0
    
    # 计算动作幅度（近似执行器力）
    action_magnitudes = []
    
    for action in actions:
        if isinstance(action, dict):
            action_array = np.array(action["action"])
        else:
            action_array = np.array(action)
        
        # 计算动作的L2范数作为控制力的近似
        magnitude = np.linalg.norm(action_array)
        action_magnitudes.append(magnitude)
    
    if not action_magnitudes:
        return 1.0
    
    # 基于tolerance函数计算控制效率
    # tolerance(forces, margin=10, value_at_margin=0, sigmoid="quadratic")
    control_scores = []
    margin = 10.0
    
    for magnitude in action_magnitudes:
        if magnitude <= margin:
            # 二次sigmoid：score = 1 - (magnitude/margin)^2
            score = 1.0 - (magnitude / margin) ** 2
        else:
            score = 0.0
        
        control_scores.append(max(0.0, score))
    
    # 计算平均控制效率
    avg_control = np.mean(control_scores)
    
    # 应用pole.py中的变换：(4 + small_control) / 5
    adjusted_control = (4 + avg_control) / 5
    
    return float(np.clip(adjusted_control, 0.0, 1.0))

def pole_success_rate(trajectory, goal=None):
    """计算pole任务的成功率：基于实际奖励函数的success_bar
    
    基于pole.py中的实际实现：
    - success_bar = 700
    - get_terminated(): terminated = qpos[2] < 0.5
    
    成功标准：
    1. 机器人没有倒下（质心高度 > 0.5m）
    2. 维持足够长的时间（类似success_bar的概念）
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return 0.0
    
    # 检查是否倒下
    min_height = float('inf')
    for state in states:
        if isinstance(state, dict):
            height = state["obs"][2]
        else:
            height = state[2]
        
        min_height = min(min_height, height)
    
    # 倒下阈值：0.5m（与pole.py一致）
    fall_threshold = 0.5
    
    if min_height < fall_threshold:
        return 0.1  # 倒下了，任务失败
    
    # 基于轨迹长度评估成功程度
    # success_bar = 700，假设对应700个时间步
    trajectory_length = len(states)
    success_length = 700
    
    if trajectory_length >= success_length:
        return 1.0  # 完全成功
    else:
        # 部分成功：基于完成的比例
        return float(trajectory_length / success_length)

def pole_overall_performance(trajectory, goal=None):
    """计算pole任务的整体性能：基于实际奖励函数的综合评分
    
    基于pole.py中的实际实现：
    reward = (0.5 * (small_control * stand_reward) + 0.5 * move) * collision_discount
    
    综合考虑：
    1. 站立稳定性（50%权重的一部分）
    2. 控制效率（50%权重的一部分）
    3. 移动效率（50%权重）
    4. 碰撞避免（乘法因子）
    """
    # 获取各项指标
    standing = pole_standing_stability(trajectory, goal)
    control = pole_control_efficiency(trajectory, goal)
    movement = pole_movement_efficiency(trajectory, goal)
    collision = pole_collision_avoidance(trajectory, goal)
    
    # 按照pole.py的奖励函数结构计算
    # reward = (0.5 * (small_control * stand_reward) + 0.5 * move) * collision_discount
    performance = (0.5 * (control * standing) + 0.5 * movement) * collision
    
    return float(np.clip(performance, 0.0, 1.0))

# ===== PUSH 任务专用指标函数 =====
def hand_to_box_distance(trajectory, goal=None):
    """计算手部到箱子的距离：push任务的核心指标
    
    基于push.py中的实际实现：
    - 手部位置：robot.left_hand_position()
    - 箱子位置：_env.named.data.qpos["free_object"][:3]
    - 距离计算：np.sqrt(np.square(left_hand - box).sum())
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return float("inf")
    
    last_state = states[-1]
    
    # 提取观测数据
    if isinstance(last_state, dict):
        obs = last_state["obs"]
    else:
        obs = last_state
    
    obs = np.array(obs)
    
    # 根据push.py的观测结构提取手部位置和箱子位置
    # 观测结构：[robot_state, box_position, goal_position]
    if len(obs) >= 6:
        # 对于push任务，箱子位置通常在观测的特定位置
        # 根据push.py，箱子位置是free_object的前3维
        if len(obs) > 76:  # h1hand的情况
            # 手部位置在观测中的位置（需要根据实际观测结构调整）
            hand_pos = obs[-9:-6]  # 假设手部位置在倒数第9到第6个位置
            box_pos = obs[-6:-3]   # 箱子位置在倒数第6到第3个位置
        elif len(obs) > 26:  # h1的情况
            hand_pos = obs[-9:-6]
            box_pos = obs[-6:-3]
        else:
            # 简化情况
            hand_pos = obs[:3] if len(obs) >= 3 else np.array([0.0, 0.0, 1.0])
            box_pos = obs[3:6] if len(obs) >= 6 else np.array([1.0, 0.0, 1.0])
    else:
        # 如果观测维度不足，使用默认值
        hand_pos = obs[:3] if len(obs) >= 3 else np.array([0.0, 0.0, 1.0])
        box_pos = np.array([1.0, 0.0, 1.0])
    
    # 计算欧几里得距离（与push.py中的计算方式一致）
    distance = np.sqrt(np.square(hand_pos - box_pos).sum())
    return float(distance)

def box_to_goal_distance(trajectory, goal=None):
    """计算箱子到目标的距离：push任务的目标指标
    
    基于push.py中的实际实现：
    - 箱子位置：_env.data.qpos.flat.copy()[-7:-4]
    - 目标位置：self.goal
    - 距离计算：np.sqrt(np.square(box - self.goal).sum())
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return float("inf")
    
    last_state = states[-1]
    
    # 提取观测数据
    if isinstance(last_state, dict):
        obs = last_state["obs"]
    else:
        obs = last_state
    
    obs = np.array(obs)
    
    # 根据push.py的观测结构提取箱子位置和目标位置
    if len(obs) >= 6:
        # 箱子位置在观测的倒数第6到第3个位置
        box_pos = obs[-6:-3]
        # 目标位置在观测的最后3个位置
        target_pos = obs[-3:]
    else:
        # 如果观测维度不足，使用默认值
        box_pos = obs[:3] if len(obs) >= 3 else np.array([1.0, 0.0, 1.0])
        if goal is not None and isinstance(goal, (list, tuple, np.ndarray)) and len(goal) >= 3:
            target_pos = np.array(goal[:3])
        else:
            target_pos = np.array([1.0, 0.0, 1.0])  # push.py中的默认目标
    
    # 计算欧几里得距离（与push.py中的计算方式一致）
    distance = np.sqrt(np.square(box_pos - target_pos).sum())
    return float(distance)

def push_success_rate(trajectory, goal=None):
    """计算push任务的成功率：基于实际奖励函数的成功标准
    
    实际奖励函数中：
    - reward_success = 1000 if goal_dist < 0.05 else 0
    - get_terminated(): terminated = goal_dist < 0.05
    
    成功阈值设为0.05m（5cm），与实际奖励函数一致
    """
    final_distance = box_to_goal_distance(trajectory, goal)
    
    # 成功阈值：0.05m（与push.py中reward_success的阈值一致）
    success_threshold = 0.05
    
    # 返回成功率：1.0表示成功，0.1表示失败
    return float(1.0 if final_distance < success_threshold else 0.1)

def push_efficiency(trajectory, goal=None):
    """计算push任务的效率：基于手部到箱子距离和箱子到目标距离的综合评估
    
    实际奖励函数中：
    - hand_penalty = 0.1 * hand_dist
    - penalty_dist = 1 * goal_dist
    
    效率越高，总距离惩罚越低
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return 0.0
    
    # 计算最终的手部到箱子距离和箱子到目标距离
    hand_box_dist = hand_to_box_distance(trajectory, goal)
    box_goal_dist = box_to_goal_distance(trajectory, goal)
    
    # 根据push.py的奖励权重计算总惩罚
    # hand_penalty = 0.1 * hand_dist, penalty_dist = 1 * goal_dist
    total_penalty = 0.1 * hand_box_dist + 1.0 * box_goal_dist
    
    # 效率 = 1 / (1 + 总惩罚)
    efficiency = 1.0 / (1.0 + total_penalty)
    
    return float(efficiency)

def push_smoothness(trajectory, goal=None):
    """评估push任务的平滑性：基于动作序列的平滑程度
    
    平滑的推动策略应该产生连续、稳定的动作序列，
    避免剧烈的动作变化，这有助于提高推动效果和控制质量
    """
    actions = getattr(trajectory, "actions", [])
    if len(actions) < 3:
        return 1.0  # 如果动作太少，认为是平滑的
    
    # 计算动作序列的变化率
    action_changes = []
    for i in range(1, len(actions)):
        if isinstance(actions[i], dict):
            action1 = np.array(actions[i-1]["action"])
            action2 = np.array(actions[i]["action"])
        else:
            action1 = np.array(actions[i-1])
            action2 = np.array(actions[i])
        
        # 计算动作变化的L2范数
        action_change = np.linalg.norm(action2 - action1)
        action_changes.append(action_change)
    
    # 计算动作变化的二阶导数（加速度）
    if len(action_changes) > 1:
        action_accelerations = np.diff(action_changes)
        # 平滑性：加速度变化越小越平滑
        smoothness_score = 1.0 / (1.0 + np.mean(np.abs(action_accelerations)) * 5.0)
    else:
        # 如果只有一个变化，基于变化大小评估
        smoothness_score = 1.0 / (1.0 + np.mean(action_changes) * 2.0)
    
    return float(np.clip(smoothness_score, 0.0, 1.0))

def push_stability(trajectory, goal=None):
    """评估push任务中的身体稳定性：在推动过程中保持身体平衡
    
    推动任务需要在施力的同时保持身体稳定，
    避免因为推动动作而失去平衡
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    # 提取躯干姿态信息
    stability_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 提取四元数姿态（通常在位置之后）
        if len(obs) >= 7:
            # 四元数在位置[x,y,z]之后的[w,x,y,z]位置
            quaternion = obs[3:7]
            
            # 将四元数转换为旋转矩阵的Z轴分量
            # 对于四元数[w,x,y,z]，Z轴向上的分量可以通过以下公式计算：
            # z_up = 2*(x*z + w*y)
            w, x, y, z = quaternion
            z_up_component = 2 * (x * z + w * y)
            
            # 直立程度：z_up_component接近0表示直立
            upright_score = 1.0 - abs(z_up_component)
            stability_scores.append(max(0.0, upright_score))
        else:
            # 如果观测维度不足，使用默认稳定性
            stability_scores.append(0.5)
    
    # 计算平均稳定性
    if stability_scores:
        avg_stability = np.mean(stability_scores)
        
        # 额外考虑稳定性的一致性（变化越小越稳定）
        if len(stability_scores) > 1:
            stability_variance = np.var(stability_scores)
            consistency_bonus = 1.0 / (1.0 + stability_variance * 10.0)
            total_stability = 0.7 * avg_stability + 0.3 * consistency_bonus
        else:
            total_stability = avg_stability
    else:
        total_stability = 0.0
    
    return float(np.clip(total_stability, 0.0, 1.0))

def enhanced_climbing_efficiency(trajectory, goal=None):
    """增强版攀爬效率：考虑高度增益、能耗和路径优化"""
    try:
        states = getattr(trajectory, "states", [])
        actions = getattr(trajectory, "actions", [])
        
        if len(states) < 10 or len(actions) < 5:
            return climbing_efficiency(trajectory, goal)
        
        # 提取观测和动作序列
        observations = []
        for state in states:
            if isinstance(state, dict) and "obs" in state:
                observations.append(np.array(state["obs"]))
            elif hasattr(state, "obs"):
                observations.append(np.array(state.obs))
            else:
                return climbing_efficiency(trajectory, goal)
        
        action_sequence = []
        for action in actions:
            if isinstance(action, dict) and "action" in action:
                action_sequence.append(np.array(action["action"]))
            elif hasattr(action, "action"):
                action_sequence.append(np.array(action.action))
            else:
                action_sequence.append(np.array(action))
        
        observations = np.array(observations)
        action_sequence = np.array(action_sequence)
        
        # 高度增益
        height_gain_val = observations[-1, 2] - observations[0, 2]
        
        # 能耗计算（基于动作幅度）
        energy_cost = np.sum(np.linalg.norm(action_sequence, axis=1))
        
        # 路径效率（垂直vs总路径）
        total_path = 0.0
        for i in range(1, len(observations)):
            total_path += np.linalg.norm(observations[i, :3] - observations[i-1, :3])
        
        vertical_efficiency = abs(height_gain_val) / total_path if total_path > 0 else 0
        
        # 攀爬平滑性
        height_changes = np.diff(observations[:, 2])
        climbing_smoothness = 1.0 / (1.0 + np.std(height_changes))
        
        # 综合效率
        if energy_cost > 0:
            base_efficiency = height_gain_val / energy_cost
            enhanced_efficiency = base_efficiency * (0.6 + 0.2 * vertical_efficiency + 0.2 * climbing_smoothness)
        else:
            enhanced_efficiency = 0.0
        
        return float(enhanced_efficiency)
        
    except Exception:
        return climbing_efficiency(trajectory, goal)

def enhanced_step_coordination(trajectory, goal=None):
    """增强版步伐协调性：基于关节协调和步态模式"""
    try:
        states = getattr(trajectory, "states", [])
        if len(states) < 20:
            return step_coordination(trajectory, goal)
        
        # 提取观测序列
        observations = []
        for state in states:
            if isinstance(state, dict) and "obs" in state:
                observations.append(np.array(state["obs"]))
            elif hasattr(state, "obs"):
                observations.append(np.array(state.obs))
            else:
                return step_coordination(trajectory, goal)
        
        observations = np.array(observations)
        
        # 质心高度变化（步态周期性）
        heights = observations[:, 2]
        
        # 使用自相关分析周期性
        if len(heights) > 30:
            # 去除趋势
            detrended_heights = heights - np.linspace(heights[0], heights[-1], len(heights))
            
            # 计算自相关
            autocorr = np.correlate(detrended_heights, detrended_heights, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # 归一化
            
            # 寻找周期性峰值
            peaks = []
            for i in range(5, min(len(autocorr)-5, 50)):
                if (autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and 
                    autocorr[i] > autocorr[i-2] and autocorr[i] > autocorr[i+2]):
                    peaks.append((i, autocorr[i]))
            
            if peaks:
                # 选择最强的周期性信号
                best_peak = max(peaks, key=lambda x: x[1])
                periodicity_score = best_peak[1]
            else:
                periodicity_score = 0.3
        else:
            periodicity_score = 0.3
        
        # 步态规律性（高度变化的一致性）
        height_diff = np.diff(heights)
        regularity_score = 1.0 / (1.0 + np.std(height_diff))
        
        # 综合协调性评分
        coordination_score = 0.6 * periodicity_score + 0.4 * regularity_score
        return float(np.clip(coordination_score, 0.0, 1.0))
        
    except Exception:
        return step_coordination(trajectory, goal)

def enhanced_energy_efficiency(trajectory, goal=None):
    """增强版能量效率：综合考虑动作平滑性和任务完成度"""
    try:
        states = getattr(trajectory, "states", [])
        actions = getattr(trajectory, "actions", [])
        
        if len(actions) < 5:
            return energy_usage(trajectory, goal)
        
        # 提取动作序列
        action_sequence = []
        for action in actions:
            if isinstance(action, dict) and "action" in action:
                action_sequence.append(np.array(action["action"]))
            elif hasattr(action, "action"):
                action_sequence.append(np.array(action.action))
            else:
                action_sequence.append(np.array(action))
        
        action_sequence = np.array(action_sequence)
        
        # 基础能耗
        base_energy = np.sum(np.linalg.norm(action_sequence, axis=1))
        
        # 动作平滑性（减少急剧变化）
        action_changes = np.diff(action_sequence, axis=0)
        smoothness_penalty = np.sum(np.linalg.norm(action_changes, axis=1))
        
        # 动作效率（避免冗余动作）
        action_variance = np.var(action_sequence, axis=0).mean()
        efficiency_bonus = 1.0 / (1.0 + action_variance)
        
        # 综合能耗评估
        total_energy_cost = base_energy + 0.3 * smoothness_penalty
        energy_efficiency_score = efficiency_bonus / (1.0 + total_energy_cost)
        
        return float(energy_efficiency_score)
        
    except Exception:
        return 1.0 / (1.0 + energy_usage(trajectory, goal))

# ===== RUN 任务专用指标函数 =====
def running_speed_consistency(trajectory, goal=None):
    """评估跑步速度一致性：跑步任务需要保持高速且稳定的前向速度"""
    try:
        states = getattr(trajectory, "states", [])
        if len(states) < 10:
            return 0.0
        
        # 提取观测序列
        observations = []
        for state in states:
            if isinstance(state, dict) and "obs" in state:
                observations.append(np.array(state["obs"]))
            elif hasattr(state, "obs"):
                observations.append(np.array(state.obs))
            else:
                return 0.0
        
        observations = np.array(observations)
        
        # 计算前向速度序列
        positions = observations[:, 0]  # x坐标
        velocities = np.diff(positions)  # 前向速度
        
        if len(velocities) < 5:
            return 0.0
        
        # 目标跑步速度（基于_RUN_SPEED=5）
        target_speed = 5.0
        
        # 速度一致性：与目标速度的偏差
        speed_deviations = np.abs(velocities - target_speed)
        speed_consistency = 1.0 / (1.0 + np.mean(speed_deviations))
        
        # 速度稳定性：速度变化的标准差
        speed_stability = 1.0 / (1.0 + np.std(velocities))
        
        # 综合评分
        consistency_score = 0.6 * speed_consistency + 0.4 * speed_stability
        return float(np.clip(consistency_score, 0.0, 1.0))
        
    except Exception:
        return 0.0

def running_efficiency(trajectory, goal=None):
    """评估跑步效率：前向距离与能耗的比值，针对高速移动优化"""
    try:
        # 前向进展
        forward_dist = enhanced_forward_progress(trajectory, goal)
        
        # 能耗
        actions = getattr(trajectory, "actions", [])
        if len(actions) < 5:
            return 0.0
        
        # 提取动作序列
        action_sequence = []
        for action in actions:
            if isinstance(action, dict) and "action" in action:
                action_sequence.append(np.array(action["action"]))
            elif hasattr(action, "action"):
                action_sequence.append(np.array(action.action))
            else:
                action_sequence.append(np.array(action))
        
        action_sequence = np.array(action_sequence)
        
        # 计算能耗（考虑跑步的高强度特点）
        energy_cost = np.sum(np.linalg.norm(action_sequence, axis=1))
        
        # 跑步效率：距离/能耗，但对高速移动给予奖励
        if energy_cost > 0 and forward_dist > 0:
            base_efficiency = forward_dist / energy_cost
            # 对高速移动给予额外奖励
            speed_bonus = min(forward_dist / 50.0, 2.0)  # 最多2倍奖励
            running_efficiency_score = base_efficiency * (1.0 + speed_bonus)
        else:
            running_efficiency_score = 0.0
        
        return float(running_efficiency_score)
        
    except Exception:
        return 0.0

# ===== SIT_SIMPLE 任务专用指标函数 =====
def sitting_posture_quality(trajectory, goal=None):
    """评估坐姿质量：坐姿应该保持在合适的高度范围内，且姿势稳定"""
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 提取质心高度（z坐标）
    heights = []
    for state in states:
        if isinstance(state, dict):
            heights.append(state["obs"][2])
        else:
            heights.append(state[2])
    
    heights = np.array(heights)
    
    # 理想坐姿高度范围：0.68-0.72
    ideal_height_min = 0.68
    ideal_height_max = 0.72
    
    # 计算高度的偏离程度
    height_deviations = []
    for h in heights:
        if h < ideal_height_min:
            height_deviations.append(ideal_height_min - h)
        elif h > ideal_height_max:
            height_deviations.append(h - ideal_height_max)
        else:
            height_deviations.append(0.0)  # 在理想范围内
    
    avg_deviation = np.mean(height_deviations)
    
    # 计算高度的稳定性（标准差）
    height_stability = 1.0 / (1.0 + np.std(heights))
    
    # 综合评分：高度合适性 + 稳定性
    posture_score = 1.0 / (1.0 + avg_deviation) * 0.7 + height_stability * 0.3
    return float(np.clip(posture_score, 0.0, 1.0))

def sitting_position_accuracy(trajectory, goal=None):
    """评估坐姿位置准确性：坐在椅子上的位置是否准确"""
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 假设椅子位置在(-0.25, 0, 0)
    chair_position = np.array([-0.25, 0.0, 0.0])
    
    # 提取质心位置
    positions = []
    for state in states:
        if isinstance(state, dict):
            positions.append(np.array(state["obs"][:3]))
        else:
            positions.append(np.array(state[:3]))
    
    positions = np.array(positions)
    
    # 计算与椅子位置的距离（只考虑x和y坐标）
    horizontal_distances = []
    for pos in positions:
        # 计算水平距离
        horizontal_dist = np.sqrt((pos[0] - chair_position[0])**2 + (pos[1] - chair_position[1])**2)
        horizontal_distances.append(horizontal_dist)
    
    avg_distance = np.mean(horizontal_distances)
    
    # 位置准确性评分：距离越近得分越高
    position_score = 1.0 / (1.0 + avg_distance * 10.0)  # 乘以10使得距离差异更明显
    return float(np.clip(position_score, 0.0, 1.0))

def sitting_stability(trajectory, goal=None):
    """评估坐姿稳定性：保持上身直立且稳定"""
    states = getattr(trajectory, "states", [])
    if len(states) < 10:
        return 0.0
    
    # 提取姿态信息（假设四元数在obs的3:7位置）
    orientations = []
    for state in states:
        if isinstance(state, dict):
            orientations.append(np.array(state["obs"][3:7]))
        else:
            orientations.append(np.array(state[3:7]))
    
    orientations = np.array(orientations)
    
    # 计算姿态变化
    orientation_changes = []
    for i in range(1, len(orientations)):
        # 计算四元数之间的角度差
        q1, q2 = orientations[i-1], orientations[i]
        dot_product = np.abs(np.dot(q1, q2))
        angle_change = 2 * np.arccos(np.clip(dot_product, 0, 1))
        orientation_changes.append(angle_change)
    
    # 姿态稳定性：变化越小越稳定
    if orientation_changes:
        orientation_stability = 1.0 / (1.0 + np.mean(orientation_changes) * 10.0)  # 乘以10使得变化更明显
    else:
        orientation_stability = 1.0
    
    # 上身直立程度（假设理想的上身四元数是[1,0,0,0]）
    # 简化计算，使用四元数的第一个分量（w）作为直立程度的近似
    upright_scores = []
    for orientation in orientations:
        # w接近1表示接近理想姿态
        upright_score = orientation[0]  # 假设w在第一个位置
        upright_scores.append(upright_score)
    
    avg_upright = np.mean(upright_scores)
    
    # 综合评分：姿态稳定性 + 上身直立程度
    stability_score = orientation_stability * 0.5 + avg_upright * 0.5
    return float(np.clip(stability_score, 0.0, 1.0))

def sitting_minimal_movement(trajectory, goal=None):
    """评估坐姿最小运动：坐姿应该保持稳定，减少不必要的移动"""
    states = getattr(trajectory, "states", [])
    if len(states) < 2:
        return 1.0
    
    # 计算位置变化
    total_movement = 0.0
    for i in range(1, len(states)):
        if isinstance(states[i], dict):
            pos1 = np.array(states[i-1]["obs"][:3])
            pos2 = np.array(states[i]["obs"][:3])
        else:
            pos1 = np.array(states[i-1][:3])
            pos2 = np.array(states[i][:3])
        
        # 计算三维空间中的移动
        movement = np.linalg.norm(pos2 - pos1)
        total_movement += movement
    
    # 转换为奖励：移动越少越好
    return float(1.0 / (1.0 + total_movement * 5.0))  # 乘以5使得移动更明显

# ===== REACH 任务专用指标函数 =====
def hand_to_goal_distance(trajectory, goal=None):
    """计算手部到目标的距离：reach任务的核心指标
    
    基于reach.py中的实际实现：
    - 观测空间：(robot.dof * 2 - 1) + 6 维度
    - 观测结构：[position, velocity, left_hand, target]
    - left_hand位置在观测的倒数第6到第3个位置
    - target位置在观测的最后3个位置
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 1:
        return float("inf")
    
    last_state = states[-1]
    
    # 提取观测数据
    if isinstance(last_state, dict):
        obs = last_state["obs"]
    else:
        obs = last_state
    
    obs = np.array(obs)
    
    # 根据reach.py的get_obs()实现提取手部位置和目标位置
    if len(obs) >= 6:
        # 手部位置在观测的倒数第6到第3个位置
        hand_pos = obs[-6:-3]
        # 目标位置在观测的最后3个位置
        target_pos = obs[-3:]
    else:
        # 如果观测维度不足，使用默认值
        hand_pos = obs[:3] if len(obs) >= 3 else np.array([0.0, 0.0, 1.0])
        if goal is not None and isinstance(goal, (list, tuple, np.ndarray)) and len(goal) >= 3:
            target_pos = np.array(goal[:3])
        else:
            target_pos = np.array([1.0, 0.0, 1.0])
    
    # 计算欧几里得距离（与reach.py中的计算方式一致）
    distance = np.sqrt(np.square(hand_pos - target_pos).sum())
    return float(distance)

def reach_efficiency(trajectory, goal=None):
    """计算reach任务的效率：基于实际奖励函数的motion_penalty反向优化
    
    实际奖励函数中的motion_penalty = qvel^2 * 0.0001
    效率越高，motion_penalty越低
    """
    states = getattr(trajectory, "states", [])
    actions = getattr(trajectory, "actions", [])
    
    if len(states) < 2:
        return 0.0
    
    # 计算运动惩罚（基于速度的平方和）
    total_motion_penalty = 0.0
    velocity_count = 0
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 根据reach.py，速度在观测中的位置
        # 观测结构：[position, velocity, left_hand, target]
        # 对于h1hand，dof=76，所以velocity维度是75
        if len(obs) > 76:  # h1hand的情况
            velocity = obs[76:151]  # position后面的75维是velocity
        elif len(obs) > 26:  # h1的情况
            velocity = obs[26:51]  # position后面的25维是velocity
        else:
            # 简化情况，使用前几维作为速度近似
            velocity = obs[3:6] if len(obs) > 6 else obs[:3]
        
        # 计算速度的平方和（对应motion_penalty）
        motion_penalty = np.square(velocity).sum()
        total_motion_penalty += motion_penalty
        velocity_count += 1
    
    # 平均运动惩罚
    if velocity_count > 0:
        avg_motion_penalty = total_motion_penalty / velocity_count
        # 效率 = 1 / (1 + 归一化的运动惩罚)
        # 乘以0.0001对应实际奖励函数中的系数
        normalized_penalty = avg_motion_penalty * 0.0001
        efficiency = 1.0 / (1.0 + normalized_penalty)
    else:
        efficiency = 0.0
    
    return float(efficiency)

def reach_success_rate(trajectory, goal=None):
    """计算reach任务的成功率：基于实际奖励函数的成功标准
    
    实际奖励函数中：
    - reward_success = 10 if hand_dist < 0.05 else 0
    - reward_close = 5 if hand_dist < 1 else 0
    
    成功阈值设为0.05m（5cm），与实际奖励函数一致
    """
    final_distance = hand_to_goal_distance(trajectory, goal)
    
    # 成功阈值：0.05m（与reach.py中reward_success的阈值一致）
    success_threshold = 0.05
    
    # 返回成功率：1.0表示成功，0.1表示失败
    return float(1.0 if final_distance < success_threshold else 0.1)

def reach_smoothness(trajectory, goal=None):
    """评估reach任务的平滑性：基于动作序列的平滑程度
    
    平滑的控制策略应该产生连续、稳定的动作序列，
    避免剧烈的动作变化，这有助于提高控制质量和减少机械磨损
    """
    actions = getattr(trajectory, "actions", [])
    if len(actions) < 3:
        return 1.0  # 如果动作太少，认为是平滑的
    
    # 计算动作序列的变化率
    action_changes = []
    for i in range(1, len(actions)):
        if isinstance(actions[i], dict):
            action1 = np.array(actions[i-1]["action"])
            action2 = np.array(actions[i]["action"])
        else:
            action1 = np.array(actions[i-1])
            action2 = np.array(actions[i])
        
        # 计算动作变化的L2范数
        action_change = np.linalg.norm(action2 - action1)
        action_changes.append(action_change)
    
    # 计算动作变化的二阶导数（加速度）
    if len(action_changes) > 1:
        action_accelerations = np.diff(action_changes)
        # 平滑性：加速度变化越小越平滑
        smoothness_score = 1.0 / (1.0 + np.mean(np.abs(action_accelerations)) * 5.0)
    else:
        # 如果只有一个变化，基于变化大小评估
        smoothness_score = 1.0 / (1.0 + np.mean(action_changes) * 2.0)
    
    return float(np.clip(smoothness_score, 0.0, 1.0))

def reach_stability(trajectory, goal=None):
    """评估reach任务中的身体稳定性：基于实际奖励函数的healthy_reward
    
    实际奖励函数中：
    healthy_reward = xmat[1, -1] * 5.0
    xmat[1, -1]表示躯干的Z轴方向分量，反映身体的直立程度
    """
    states = getattr(trajectory, "states", [])
    if len(states) < 5:
        return 0.0
    
    # 提取躯干姿态信息
    stability_scores = []
    
    for state in states:
        if isinstance(state, dict):
            obs = state["obs"]
        else:
            obs = state
        
        obs = np.array(obs)
        
        # 提取四元数姿态（通常在位置之后）
        if len(obs) >= 7:
            # 四元数在位置[x,y,z]之后的[w,x,y,z]位置
            quaternion = obs[3:7]
            
            # 将四元数转换为旋转矩阵的Z轴分量
            # 对于四元数[w,x,y,z]，Z轴向上的分量可以通过以下公式计算：
            # z_up = 2*(x*z + w*y)
            w, x, y, z = quaternion
            z_up_component = 2 * (x * z + w * y)
            
            # 直立程度：z_up_component接近0表示直立
            upright_score = 1.0 - abs(z_up_component)
            stability_scores.append(max(0.0, upright_score))
        else:
            # 如果观测维度不足，使用默认稳定性
            stability_scores.append(0.5)
    
    # 计算平均稳定性
    if stability_scores:
        avg_stability = np.mean(stability_scores)
        
        # 额外考虑稳定性的一致性（变化越小越稳定）
        if len(stability_scores) > 1:
            stability_variance = np.var(stability_scores)
            consistency_bonus = 1.0 / (1.0 + stability_variance * 10.0)
            total_stability = 0.7 * avg_stability + 0.3 * consistency_bonus
        else:
            total_stability = avg_stability
    else:
        total_stability = 0.0
    
    return float(np.clip(total_stability, 0.0, 1.0))