"""CLI entrypoint for Natron PPO reinforcement learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from natron.config import load_config
from natron.data.loader import DataModule
from natron.models.transformer import NatronTransformer
from natron.rl.env import TradingEnvironment
from natron.rl.ppo import PPOAgent, PPOTrainer
from natron.utils import create_logger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Natron PPO Reinforcement Learning")
    parser.add_argument("--config", type=str, default="natron/configs/natron_base.yaml")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to fine-tuned weights for backbone")
    parser.add_argument("--iterations", type=int, default=100, help="Number of PPO update iterations")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="artifacts/model/natron_rl.pt")
    return parser.parse_args()


def build_sequences(features_scaled: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    sequences = []
    for idx in indices:
        start = idx - seq_len + 1
        sequences.append(features_scaled[start : idx + 1])
    return np.asarray(sequences, dtype=np.float32)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not cfg.rl.enabled:
        raise RuntimeError("RL phase disabled in configuration. Set rl.enabled=true to proceed.")

    device = torch.device(args.device or cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.training.seed)

    logger = create_logger("natron.rl", cfg.training.log_dir)
    logger.info("Preparing data for RL...")
    module = DataModule(cfg)
    data_bundle = module.prepare()

    features_scaled = data_bundle["features_scaled"]
    seq_len = cfg.data.sequence_length
    train_indices = data_bundle["indices"]["train"]

    sequences = build_sequences(features_scaled, train_indices, seq_len)
    close = data_bundle["dataframe"]["close"].values
    returns = np.zeros_like(close, dtype=np.float32)
    returns[:-1] = np.diff(close) / (close[:-1] + 1e-9)
    returns = returns[train_indices]

    env = TradingEnvironment(
        sequences=sequences,
        returns=returns,
        alpha=cfg.rl.reward_alpha,
        beta=cfg.rl.reward_beta,
    )

    feature_dim = data_bundle["features"].shape[1]
    backbone = NatronTransformer(feature_dim=feature_dim, config=cfg.model)
    if args.pretrained:
        logger.info("Loading backbone weights from %s", args.pretrained)
        state = torch.load(args.pretrained, map_location="cpu")
        incompat = backbone.load_state_dict(state, strict=False)
        if incompat.missing_keys:
            logger.warning("Missing keys: %s", incompat.missing_keys)
        if incompat.unexpected_keys:
            logger.warning("Unexpected keys: %s", incompat.unexpected_keys)

    agent = PPOAgent(backbone, action_dim=3, train_backbone=False)
    trainer = PPOTrainer(agent, cfg.rl, cfg.training, logger, device)

    logger.info("Starting PPO training for %d iterations...", args.iterations)
    for iteration in range(1, args.iterations + 1):
        batch = trainer.rollout(env, cfg.rl.num_steps)
        metrics = trainer.update(batch)
        logger.info(
            "[PPO][Iter %03d] loss=%.4f policy=%.4f value=%.4f entropy=%.4f",
            iteration,
            metrics["loss"],
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), args.output)
    logger.info("RL agent saved to %s", args.output)


if __name__ == "__main__":
    main()
