import torch
import torch.nn as nn
import torch.nn.functional as F

class PreferenceRewardModel(nn.Module):
    """
    偏好奖励模型，支持目标条件化。
    输入：(states, actions, goals)
    """
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim

        # 简单MLP编码器，先拼接(s, a, g)，再池化
        self.traj_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 轨迹池化后输出奖励分数
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions, goals):
        """
        states: [B, T, state_dim], actions: [B, T, action_dim], goals: [B, goal_dim]
        返回: [B]，轨迹分数
        """
        B, T, _ = states.shape
        goals_expanded = goals.unsqueeze(1).expand(-1, T, -1)  # 扩展 goals 维度
        x = torch.cat([states, actions, goals_expanded], dim=-1)  # 拼接 states, actions 和 goals
        h = self.traj_encoder(x)                  # [B, T, hidden]
        h = h.mean(dim=1)                         # [B, hidden]
        reward = self.reward_head(h).squeeze(-1)      # [B]
        return reward

    @staticmethod
    def grpo_loss(model, traj_chosen, traj_rejected):
        """
        GRPO/DPO pairwise loss:
        让模型对偏好轨迹打高分，对被拒绝轨迹打低分。
        traj_chosen, traj_rejected: [(states, actions, rewards), ...]
        """
        # 批量处理
        scores_chosen = []
        scores_rejected = []
        for (s1, a1, _), (s2, a2, _) in zip(traj_chosen, traj_rejected):
            scores_chosen.append(model(s1, a1))
            scores_rejected.append(model(s2, a2))
        scores_chosen = torch.stack(scores_chosen)
        scores_rejected = torch.stack(scores_rejected)
        # GRPO/DPO 损失
        loss = -F.logsigmoid(scores_chosen - scores_rejected).mean()
        return loss