from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from natron.models.transformer import NatronTransformer
from natron.utils.torch_utils import get_device


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    batch_size: int = 128
    epochs: int = 10
    horizon: int = 1024
    use_amp: bool = True


class ActorCritic(nn.Module):
    def __init__(self, backbone: NatronTransformer, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = backbone.d_model
        self.policy = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )
        self.value = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.encode(obs)
        pooled = features.mean(dim=1)
        logits = self.policy(pooled)
        values = self.value(pooled).squeeze(-1)
        return logits, values

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, values


class PPOAgent:
    def __init__(self, model: ActorCritic, config: PPOConfig) -> None:
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def collect_rollout(self, env) -> Dict[str, List]:
        obs, _ = env.reset()
        storage = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

        for _ in range(self.config.horizon):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action, log_prob, value = self.model.act(obs_tensor)

            next_obs, reward, done, _, info = env.step(action.item())

            storage["obs"].append(obs_tensor.squeeze(0).cpu().numpy())
            storage["actions"].append(action.cpu().numpy())
            storage["log_probs"].append(log_prob.cpu().numpy())
            storage["rewards"].append(reward)
            storage["values"].append(value.cpu().numpy())
            storage["dones"].append(done)

            obs = next_obs
            if done:
                obs, _ = env.reset()

        return storage

    def compute_advantages(self, storage: Dict[str, List]) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = storage["rewards"]
        values = storage["values"] + [0.0]
        dones = storage["dones"]

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        return advantages, returns

    def update(self, storage: Dict[str, List]) -> None:
        obs = torch.tensor(np.array(storage["obs"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(storage["actions"]), dtype=torch.int64).to(self.device).squeeze()
        old_log_probs = torch.tensor(np.array(storage["log_probs"]), dtype=torch.float32).to(self.device).squeeze()
        values = torch.tensor(np.array(storage["values"]), dtype=torch.float32).to(self.device).squeeze()

        advantages, returns = self.compute_advantages(storage)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = obs.size(0)
        for _ in range(self.config.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.config.batch_size):
                idx = indices[start : start + self.config.batch_size]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_returns = returns[idx]
                batch_adv = advantages[idx]
                batch_old_log_probs = old_log_probs[idx]

                logits, new_values = self.model(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, batch_returns)

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

    def train(self, env, total_steps: int) -> None:
        steps = 0
        while steps < total_steps:
            storage = self.collect_rollout(env)
            self.update(storage)
            steps += self.config.horizon
