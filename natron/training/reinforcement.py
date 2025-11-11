"""Phase 3 — Reinforcement learning for trading policy optimization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from natron.data.datamodule import DataModuleConfig, NatronDataModule
from natron.models.transformer import NatronEncoder, NatronTransformer
from natron.utils.config import NatronConfig
from natron.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class RLHyperParams:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 5e-5
    alpha: float = 0.001
    beta: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    horizon: int = 3
    train_encoder: bool = False


class RLSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, returns: np.ndarray) -> None:
        self.inputs = torch.from_numpy(sequences).float()
        self.returns = torch.from_numpy(returns).float()

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {"inputs": self.inputs[idx], "returns": self.returns[idx]}


class NatronRLPolicy(nn.Module):
    def __init__(self, encoder: NatronEncoder, train_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.train_encoder = train_encoder
        self.action_dim = 3  # long, flat, short

        d_model = encoder.config.d_model
        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(self.train_encoder):
            hidden_seq = self.encoder(inputs, return_sequence=True)
        pooled = hidden_seq[:, -1]
        logits = self.actor(pooled)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.critic(pooled).squeeze(-1)
        return dist, value, pooled


def run_reinforcement(
    config: NatronConfig,
    device: torch.device,
    policy_model: NatronTransformer,
    data_module: Optional[NatronDataModule] = None,
) -> NatronRLPolicy:
    """Execute optional reinforcement learning fine-tuning."""
    phase_cfg = config.phases.get("reinforcement", {})
    if not phase_cfg.get("enabled", False):
        logger.info("Reinforcement learning phase disabled. Skipping.")
        return NatronRLPolicy(policy_model.get_encoder(), train_encoder=False)

    reward_cfg = phase_cfg.get("reward", {})
    hyper = RLHyperParams(
        batch_size=phase_cfg.get("batch_size", 128),
        epochs=phase_cfg.get("epochs", 5),
        lr=phase_cfg.get("lr", 5e-5),
        alpha=reward_cfg.get("alpha", 0.001),
        beta=reward_cfg.get("beta", 0.01),
        entropy_coef=phase_cfg.get("entropy_coef", 0.01),
        value_coef=phase_cfg.get("value_coef", 0.5),
        horizon=phase_cfg.get("horizon", 3),
        train_encoder=phase_cfg.get("train_encoder", False),
    )

    if data_module is None:
        dm_config = DataModuleConfig(batch_size=hyper.batch_size)
        data_module = NatronDataModule(Path(config.data_path), dm_config)
        data_module.prepare()
    elif data_module.sequences is None:
        data_module.prepare()

    returns = _compute_future_returns(data_module, hyper.horizon)
    train_idx = data_module.get_split_indices("train")
    train_sequences = data_module.sequences[train_idx]
    train_returns = returns[train_idx]

    dataset = RLSequenceDataset(train_sequences, train_returns)
    dataloader = DataLoader(dataset, batch_size=hyper.batch_size, shuffle=True, drop_last=True)

    encoder = policy_model.get_encoder()
    rl_policy = NatronRLPolicy(encoder, train_encoder=hyper.train_encoder).to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, rl_policy.parameters()), lr=hyper.lr)

    action_map = torch.tensor([1.0, 0.0, -1.0], device=device)

    for epoch in range(1, hyper.epochs + 1):
        epoch_loss = 0.0
        epoch_reward = 0.0
        batches = 0
        for batch in tqdm(dataloader, desc="rl", leave=False):
            inputs = batch["inputs"].to(device)
            future_returns = batch["returns"].to(device)

            dist, value, _ = rl_policy(inputs)
            actions = dist.sample()
            action_sign = action_map[actions]

            profit = action_sign * future_returns
            turnover_penalty = hyper.alpha * action_sign.abs()
            drawdown_penalty = hyper.beta * torch.clamp(-profit, min=0.0)
            reward = profit - turnover_penalty - drawdown_penalty

            advantage = reward - value.detach()
            log_prob = dist.log_prob(actions)
            policy_loss = -(log_prob * advantage).mean()
            value_loss = F.mse_loss(value, reward)
            entropy_loss = -hyper.entropy_coef * dist.entropy().mean()
            loss = policy_loss + hyper.value_coef * value_loss + entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rl_policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_reward += reward.mean().item()
            batches += 1

        logger.info(
            "RL epoch %s | loss=%.4f | avg_reward=%.5f",
            epoch,
            epoch_loss / max(batches, 1),
            epoch_reward / max(batches, 1),
        )

    _save_policy(config, rl_policy)
    return rl_policy


def _compute_future_returns(data_module: NatronDataModule, horizon: int) -> np.ndarray:
    if data_module.dataframe is None:
        raise RuntimeError("DataModule must be prepared before computing returns.")
    close_series = data_module.dataframe["close"]
    future = close_series.shift(-horizon)
    raw_returns = (future - close_series) / close_series
    seq_index = np.array(data_module.sequence_timestamps)
    returns = raw_returns.reindex(seq_index).to_numpy(dtype=np.float32)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns


def _save_policy(config: NatronConfig, policy: NatronRLPolicy) -> None:
    path = Path(config.checkpoint_dir) / f"{config.run_name}_rl.pt"
    torch.save(policy.state_dict(), path)
    logger.info("Saved RL policy checkpoint to %s", path)


__all__ = ["run_reinforcement", "NatronRLPolicy"]
