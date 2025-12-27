import random

class PreferencePairDataset:
    def __init__(self, traj_pairs, model, device, task_idx=0):
        self.traj_pairs = traj_pairs  # [(traj_chosen, traj_rejected), ...]
        self.model = model
        self.device = device
        self.task_idx = task_idx

    def sample(self, batch_size):
        batch = []
        for _ in range(batch_size):
            traj_c, traj_r = random.choice(self.traj_pairs)
            t_c = random.randint(0, len(traj_c["obs"]) - 1)
            t_r = random.randint(0, len(traj_r["obs"]) - 1)
            obs_c = torch.tensor(traj_c["obs"][t_c], dtype=torch.float32, device=self.device).unsqueeze(0)
            act_c = torch.tensor(traj_c["action"][t_c], dtype=torch.float32, device=self.device).unsqueeze(0)
            obs_r = torch.tensor(traj_r["obs"][t_r], dtype=torch.float32, device=self.device).unsqueeze(0)
            act_r = torch.tensor(traj_r["action"][t_r], dtype=torch.float32, device=self.device).unsqueeze(0)
            # encode obs to latent z
            z_c = self.model.encode(obs_c, torch.tensor([self.task_idx], device=self.device)).squeeze(0)
            z_r = self.model.encode(obs_r, torch.tensor([self.task_idx], device=self.device)).squeeze(0)
            batch.append((z_c, act_c.squeeze(0), z_r, act_r.squeeze(0), torch.tensor(self.task_idx, device=self.device)))
        return batch