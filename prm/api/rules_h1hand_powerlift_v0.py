import numpy as np
from dm_control.utils import rewards


from .common_utils import tolerance

def compute_powerlift_reward_components(observations, actions):
    """
    计算powerlift任务的奖励组件 - 基于humanoid_bench/envs/powerlift.py的真实奖励函数
    
    基于powerlift.py中的实际实现：
    - stand_reward = standing * upright
    - standing = rewards.tolerance(head_height, bounds=(_STAND_HEIGHT, inf), margin=_STAND_HEIGHT/3)
    - upright = rewards.tolerance(torso_upright, bounds=(0.8, inf), sigmoid="linear", margin=1.9, value_at_margin=0)
    - small_control = rewards.tolerance(actuator_forces, margin=10, value_at_margin=0, sigmoid="quadratic").mean()
    - reward_dumbbell_lifted = rewards.tolerance(dumbbell_height, bounds=(1.9, 2.1), margin=2)
    - reward = 0.2 * (small_control * stand_reward) + 0.8 * reward_dumbbell_lifted
    
    Args:
        observations: 观测序列 [T, obs_dim]
        actions: 动作序列 [T, act_dim]
    
    Returns:
        dict: 包含各个奖励组件的字典
    """
    if len(observations) == 0:
        return {
            'stand_reward': 0.0,
            'small_control': 0.0,
            'reward_dumbbell_lifted': 0.0,
            'standing': 0.0,
            'upright': 0.0
        }
    
    # 常量定义（基于powerlift.py）
    _STAND_HEIGHT = 1.65  # h1hand的站立高度
    
    stand_rewards = []
    small_controls = []
    dumbbell_rewards = []
    standings = []
    uprights = []
    
    for i, (obs, action) in enumerate(zip(observations, actions)):
        obs = np.array(obs)
        action = np.array(action)
        
        # 提取头部高度（通常是观测的z坐标）
        if len(obs) >= 3:
            head_height = obs[2]  # z坐标表示高度
        else:
            head_height = 0.0
        
        # 计算standing奖励
        # tolerance(head_height, bounds=(_STAND_HEIGHT, inf), margin=_STAND_HEIGHT/3)
        if head_height >= _STAND_HEIGHT:
            standing = 1.0
        else:
            margin = _STAND_HEIGHT / 3
            standing = max(0.0, (head_height - (_STAND_HEIGHT - margin)) / margin)
        
        # 提取躯干直立度（基于四元数）
        if len(obs) >= 7:
            quaternion = obs[3:7]
            w, x, y, z = quaternion
            # 计算躯干直立度（简化计算）
            torso_upright = 2 * (w * w + z * z) - 1
        else:
            torso_upright = 0.0
        
        # 计算upright奖励
        # tolerance(torso_upright, bounds=(0.8, inf), sigmoid="linear", margin=1.9, value_at_margin=0)
        if torso_upright >= 0.8:
            upright = 1.0
        else:
            # 线性插值从value_at_margin=0到1.0
            upright = max(0.0, (torso_upright - (0.8 - 1.9)) / 1.9)
        
        # 计算站立奖励
        stand_reward = standing * upright
        
        # 计算小控制奖励
        # tolerance(actuator_forces, margin=10, value_at_margin=0, sigmoid="quadratic").mean()
        actuator_forces = np.abs(action)  # 简化为动作的绝对值
        control_costs = []
        for force in actuator_forces:
            if force <= 10.0:
                # 二次函数衰减
                control_cost = 1.0 - (force / 10.0) ** 2
            else:
                control_cost = 0.0
            control_costs.append(control_cost)
        
        small_control = np.mean(control_costs)
        small_control = (4 + small_control) / 5  # 基于powerlift.py的调整
        
        # 计算哑铃高度奖励（从观测中提取哑铃位置）
        if len(obs) > 50:  # h1hand的情况，假设哑铃位置在观测的特定位置
            dumbbell_height = obs[-1] if len(obs) > 0 else 0.0  # 假设哑铃高度在最后一个观测维度
        else:
            # 简化情况，使用观测的某个维度作为哑铃高度
            dumbbell_height = obs[-1] if len(obs) > 0 else 0.0
        
        # 计算哑铃举起奖励
        # tolerance(dumbbell_height, bounds=(1.9, 2.1), margin=2)
        if 1.9 <= dumbbell_height <= 2.1:
            reward_dumbbell_lifted = 1.0
        else:
            # 计算到目标区间的距离
            if dumbbell_height < 1.9:
                distance = 1.9 - dumbbell_height
            else:
                distance = dumbbell_height - 2.1
            
            # 基于margin=2的tolerance计算
            reward_dumbbell_lifted = max(0.0, 1.0 - distance / 2.0)
        
        # 存储各个组件
        stand_rewards.append(stand_reward)
        small_controls.append(small_control)
        dumbbell_rewards.append(reward_dumbbell_lifted)
        standings.append(standing)
        uprights.append(upright)
    
    return {
        'stand_reward': float(np.mean(stand_rewards)),
        'small_control': float(np.mean(small_controls)),
        'reward_dumbbell_lifted': float(np.mean(dumbbell_rewards)),
        'standing': float(np.mean(standings)),
        'upright': float(np.mean(uprights))
    }


def compute_trajectory_score(observations, actions):
    """
    计算轨迹的综合得分 - 基于humanoid_bench/envs/powerlift.py的真实奖励函数
    
    基于powerlift.py中的实际实现：
    reward = 0.2 * (small_control * stand_reward) + 0.8 * reward_dumbbell_lifted
    
    Args:
        observations: 观测序列 [T, obs_dim]
        actions: 动作序列 [T, act_dim]
    
    Returns:
        float: 轨迹综合得分
    """
    if len(observations) == 0 or len(actions) == 0:
        return 0.0
    
    # 计算各个奖励组件
    components = compute_powerlift_reward_components(observations, actions)
    
    # 基于powerlift.py的权重计算综合得分
    trajectory_score = (
        0.2 * (components['small_control'] * components['stand_reward']) +
        0.8 * components['reward_dumbbell_lifted']
    )
    
    return float(trajectory_score)


def evaluate_powerlift_trajectory(observations, actions, info=None):
    """
    评估powerlift轨迹的整体表现
    
    Args:
        observations: 观测序列 [T, obs_dim]
        actions: 动作序列 [T, act_dim]
        info: 额外信息（可选）
    
    Returns:
        dict: 包含评估结果的字典
    """
    if len(observations) == 0 or len(actions) == 0:
        return {
            'trajectory_score': 0.0,
            'components': {
                'stand_reward': 0.0,
                'small_control': 0.0,
                'reward_dumbbell_lifted': 0.0,
                'standing': 0.0,
                'upright': 0.0
            },
            'success': False,
            'episode_length': 0
        }
    
    # 计算轨迹得分和组件
    trajectory_score = compute_trajectory_score(observations, actions)
    components = compute_powerlift_reward_components(observations, actions)
    
    # 判断任务成功（基于powerlift.py的success_bar = 800和哑铃举起情况）
    success = (
        components['reward_dumbbell_lifted'] > 0.8 and  # 哑铃举起到合适高度
        components['stand_reward'] > 0.7 and            # 保持站立稳定
        len(observations) >= 100                        # 维持足够长的时间
    )
    
    return {
        'trajectory_score': trajectory_score,
        'components': components,
        'success': success,
        'episode_length': len(observations)
    }

def compare_h1hand_powerlift_v0_trajectories(trajectory_a, trajectory_b, config=None):
    """
    比较两个powerlift任务轨迹，返回更好的轨迹和较差的轨迹
    
    Args:
        trajectory_a: 第一个轨迹数据（包含observations和actions）
        trajectory_b: 第二个轨迹数据（包含observations和actions）
        config: 配置参数
        
    Returns:
        (better_trajectory, worse_trajectory): 更好的轨迹和较差的轨迹
    """
    try:
        if config is None:
            config = {}
        
        # 提取轨迹数据
        obs_a = trajectory_a.get('observations', trajectory_a.get('obs', []))
        act_a = trajectory_a.get('actions', trajectory_a.get('acts', []))
        obs_b = trajectory_b.get('observations', trajectory_b.get('obs', []))
        act_b = trajectory_b.get('actions', trajectory_b.get('acts', []))
        
        # 计算两个轨迹的得分
        score_a = compute_trajectory_score(obs_a, act_a)
        score_b = compute_trajectory_score(obs_b, act_b)
        
        # 确保分数为标量值
        if hasattr(score_a, 'item'):
            score_a = score_a.item()
        if hasattr(score_b, 'item'):
            score_b = score_b.item()
        
        # 显式转换为float进行比较
        score_a = float(score_a)
        score_b = float(score_b)
        
        # 返回更好的轨迹和较差的轨迹
        if score_a > score_b:
            return trajectory_a, trajectory_b
        else:
            return trajectory_b, trajectory_a
            
    except Exception as e:
        # 如果比较失败，返回原始顺序
        return trajectory_a, trajectory_b

def evaluate_dpo_preference(trajectory_a, trajectory_b, config=None):
    """
    评估两个轨迹的DPO偏好
    
    Args:
        trajectory_a: 第一个轨迹数据
        trajectory_b: 第二个轨迹数据
        config: 配置参数
        
    Returns:
        (preference, confidence): 偏好结果('A'或'B')和置信度(0-1)
    """
    try:
        if config is None:
            config = {}
        
        # 提取轨迹数据
        obs_a = trajectory_a.get('observations', trajectory_a.get('obs', []))
        act_a = trajectory_a.get('actions', trajectory_a.get('acts', []))
        obs_b = trajectory_b.get('observations', trajectory_b.get('obs', []))
        act_b = trajectory_b.get('actions', trajectory_b.get('acts', []))
        
        # 计算两个轨迹的得分
        score_a = compute_trajectory_score(obs_a, act_a)
        score_b = compute_trajectory_score(obs_b, act_b)
        
        # 确保分数为标量值
        if hasattr(score_a, 'item'):
            score_a = score_a.item()
        if hasattr(score_b, 'item'):
            score_b = score_b.item()
        
        # 计算偏好和置信度
        score_diff = abs(float(score_a) - float(score_b))
        max_possible_diff = 10.0  # 假设最大可能差异
        confidence = min(score_diff / max_possible_diff, 1.0)
        
        # 确定偏好
        preference = 'A' if float(score_a) > float(score_b) else 'B'
        
        return preference, confidence
        
    except Exception as e:
        # 如果评估失败，返回默认值
        return 'A', 0.1