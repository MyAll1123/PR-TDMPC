import numpy as np
import torch
import os
import uuid
from datetime import datetime

# Global registry for comparison rules
global_compare_rules = {}

def register_compare_rule(task_name, rule_fn):
    """Register a comparison rule for a given task."""
    global_compare_rules[task_name.lower()] = rule_fn

class PreferenceDataEngine:
    def __init__(self, env, agent, task_name, max_trajectory_length=1000, num_trajectories_per_policy=10, rule_data_save_path=None, collect_data_save_path=None):
        self.env = env
        self.agent = agent
        self.task_name = task_name.lower()
        self.max_trajectory_length = max_trajectory_length
        self.num_trajectories_per_policy = num_trajectories_per_policy
        self.rule_data_save_path = rule_data_save_path
        self.collect_data_save_path = collect_data_save_path
        self.trajectories = []
        self.compare_rule = global_compare_rules.get(self.task_name)

        if self.compare_rule is None:
            raise ValueError(f"任务 '{self.task_name}' 没有对应的比较规则，系统终止")

    def collect_trajectories(self):
        """Collects trajectories by interacting with the environment."""
        print(f"Collecting {self.num_trajectories_per_policy} trajectories...")
        trajectories = []
        for _ in range(self.num_trajectories_per_policy):
            obs, _ = self.env.reset()
            done = False
            trajectory = {'obs': [], 'action': [], 'reward': [], 'done': []}
            steps = 0
            while not done and steps < self.max_trajectory_length:
                action = self.agent.act(obs, t0=False, eval_mode=True)
                step_result = self.env.step(action)
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

                trajectory['obs'].append(obs)
                trajectory['action'].append(action)
                trajectory['reward'].append(reward)
                trajectory['done'].append(done)
                steps += 1
            trajectories.append(trajectory)
        self.trajectories = trajectories
        return trajectories
        if self.compare_rule is None:
            raise ValueError(f"任务 '{self.task_name}' 没有对应的比较规则，无法生成偏好对")

        preference_pairs = []
        rule_based_pairs = []
        num_trajectories = len(trajectories)

        for i in range(num_trajectories):
            for j in range(i + 1, num_trajectories):
                traj1 = trajectories[i]
                traj2 = trajectories[j]

                rule_preference = self.compare_rule(traj1, traj2)
                if rule_preference is not None:
                    if rule_preference == 1:
                        rule_based_pairs.append((traj1, traj2))
                    elif rule_preference == -1:
                        rule_based_pairs.append((traj2, traj1))

        print(f"Generated {len(rule_based_pairs)} rule-based preference pairs.")
        preference_pairs.extend(rule_based_pairs)

        return preference_pairs, rule_based_pairs

    def save_preferences(self, preference_pairs, is_rule_based):
        """Saves the generated preference pairs to disk."""
        if not preference_pairs:
            return

        if is_rule_based:
            save_path = self.rule_data_save_path
            file_prefix = "rule_pairs"
            if not save_path:
                print("Warning: rule_data_save_path is not set. Cannot save rule-based preferences.")
                return
        else:
            save_path = self.collect_data_save_path
            file_prefix = "preference_pairs"
            if not save_path:
                print("Warning: collect_data_save_path is not set. Cannot save collected preferences.")
                return

        os.makedirs(save_path, exist_ok=True) # Ensure directory exists

        # Convert to the format expected by collate_fn
        formatted_pairs = []
        for traj1, traj2 in preference_pairs:
            obs1 = np.array(traj1['obs'], dtype=np.float32)
            act1 = np.array(traj1['action'], dtype=np.float32)
            obs2 = np.array(traj2['obs'], dtype=np.float32)
            act2 = np.array(traj2['action'], dtype=np.float32)
            formatted_pairs.append(((obs1, act1), (obs2, act2)))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_path, f"{file_prefix}_{timestamp}.pt")
        torch.save(formatted_pairs, filename)
        print(f"Saved {len(formatted_pairs)} pairs to {filename}")

    def run_iteration(self):
        """Runs one iteration of trajectory collection and preference generation."""
        trajectories = self.collect_trajectories()
        preference_pairs, rule_based_pairs = self.generate_preference_pairs(trajectories)
        if rule_based_pairs:
            self.save_preferences(rule_based_pairs, is_rule_based=True)
        # Here you can decide if you want to save the collected pairs as well
        # self.save_preferences(preference_pairs, is_rule_based=False)

# 注意：所有任务特定的比较规则现在都在 prm/api/ 目录下定义
# 如果任务没有对应的规则文件，系统将抛出错误