#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from natron.config.loader import load_natron_config
from natron.models.transformer import NatronTransformer
from natron.rl.env import TradingEnv
from natron.rl.ppo import ActorCritic, PPOAgent, PPOConfig
from natron.utils.logging import configure_logging
from natron.utils.seed import seed_everything
from natron.utils.torch_utils import get_device, load_checkpoint, save_checkpoint


def main(config_path: str | None) -> None:
    configure_logging()
    cfg = load_natron_config(Path(config_path) if config_path else None)
    paths = cfg.get("paths")
    hardware = cfg.get("hardware")
    if hardware and "seed" in hardware:
        seed_everything(hardware["seed"])
    artifacts = Path(paths["artifacts"])

    sequences = np.load(artifacts / "sequences.npy")
    close_path = artifacts / "close_prices.csv"
    close_prices = (
        np.loadtxt(close_path, delimiter=",", skiprows=1)
        if close_path.exists()
        else None
    )
    if close_prices is None:
        raise FileNotFoundError("close_prices.csv not found. Save closes during data prep for RL.")

    model = NatronTransformer(
        input_dim=sequences.shape[-1],
        model_cfg=cfg.get("model"),
    )
    checkpoint, _ = load_checkpoint(Path(paths["supervised_model"]), map_location=get_device())
    model.load_state_dict(checkpoint["model_state"])

    env = TradingEnv(
        sequences=sequences.astype(np.float32),
        closes=close_prices.astype(np.float32),
        alpha=cfg.get("rl")["reward"]["alpha"],
        beta=cfg.get("rl")["reward"]["beta"],
    )

    actor_critic = ActorCritic(model)
    agent = PPOAgent(actor_critic, PPOConfig())
    agent.train(env, total_steps=cfg.get("rl")["total_steps"])

    save_checkpoint(
        {
            "policy_state": actor_critic.state_dict(),
        },
        Path(paths["model_dir"]) / "natron_ppo.pt",
        is_best=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO fine-tuning for Natron")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)
