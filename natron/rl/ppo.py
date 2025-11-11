"""Proximal Policy Optimization implementation for Natron RL phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from natron.config import RLConfig, TrainingConfig


@dataclass
class PPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class PPOAgent(nn.Module):
    def __init__(self, backbone, action_dim: int, train_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.action_dim = action_dim
        self.actor_head = nn.Sequential(
            nn.Linear(backbone.config.d_model, backbone.config.d_model // 2),
            nn.GELU(),
            nn.Linear(backbone.config.d_model // 2, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(backbone.config.d_model, backbone.config.d_model // 2),
            nn.GELU(),
            nn.Linear(backbone.config.d_model // 2, 1),
        )
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        hidden, pooled = self.backbone.encode(obs)
        logits = self.actor_head(pooled)
        dist = Categorical(logits=logits)
        value = self.value_head(pooled).squeeze(-1)
        return dist, value

    def act(self, observation: np.ndarray, device: torch.device) -> Tuple[int, float, float, np.ndarray]:
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(device=device, dtype=torch.float32)
        dist, value = self.forward(obs_tensor)
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.item()),
            obs_tensor.squeeze(0).cpu().numpy(),
        )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages[step] = gae
        next_value = values[step]
    returns = advantages + values
    return returns, advantages


class PPOTrainer:
    def __init__(
        self,
        agent: PPOAgent,
        rl_cfg: RLConfig,
        train_cfg: TrainingConfig,
        logger,
        device: torch.device,
    ):
        self.agent = agent.to(device)
        self.rl_cfg = rl_cfg
        self.train_cfg = train_cfg
        self.logger = logger
        self.device = device
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.agent.parameters()),
            lr=rl_cfg.policy_lr,
        )

    def rollout(self, env, num_steps: int) -> PPOBatch:
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        obs = env.reset()
        for _ in range(num_steps):
            action, log_prob, value, obs_tensor = self.agent.act(obs, self.device)
            next_state = env.step(action)

            observations.append(obs_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(next_state.reward)
            dones.append(float(next_state.done))
            values.append(value)

            obs = next_state.observation if not next_state.done else env.reset()

        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, next_value = self.agent.forward(torch.from_numpy(obs).unsqueeze(0).to(self.device, dtype=torch.float32))
        returns, advantages = compute_gae(
            rewards,
            values,
            dones,
            next_value.squeeze(0),
            self.rl_cfg.gamma,
            self.rl_cfg.lam,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return PPOBatch(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
            values=values,
        )

    def update(self, batch: PPOBatch) -> Dict[str, float]:
        total_loss = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        num_updates = 0

        for _ in range(self.rl_cfg.num_epochs):
            indices = torch.randperm(len(batch.actions), device=self.device)
            for start in range(0, len(indices), self.rl_cfg.minibatch_size):
                batch_idx = indices[start : start + self.rl_cfg.minibatch_size]
                obs = batch.observations[batch_idx]
                actions = batch.actions[batch_idx]
                old_log_probs = batch.log_probs[batch_idx]
                returns = batch.returns[batch_idx]
                advantages = batch.advantages[batch_idx]

                dist, values = self.agent.forward(obs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.rl_cfg.clip_ratio, 1.0 + self.rl_cfg.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)

                loss = (
                    policy_loss
                    + self.rl_cfg.value_coef * value_loss
                    - self.rl_cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.train_cfg.gradient_clip_norm)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.item()
                num_updates += 1

        return {
            "loss": total_loss / max(num_updates, 1),
            "policy_loss": policy_loss_total / max(num_updates, 1),
            "value_loss": value_loss_total / max(num_updates, 1),
            "entropy": entropy_total / max(num_updates, 1),
        }
