from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ..data.dataset import NatronDataModule
from ..models import NatronTransformer
from ..utils.logging_utils import get_logger
from .environment import TradingEnv, TradingEnvConfig


logger = get_logger(__name__)


@dataclass(slots=True)
class PPOConfig:
    total_steps: int = 100_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-5
    batch_size: int = 256
    turnover_penalty: float = 0.001
    drawdown_penalty: float = 0.005
    max_episode_length: int = 500
    policy_hidden_sizes: tuple[int, ...] = (256, 256)


class NatronRLTrainer:
    def __init__(
        self,
        model: NatronTransformer,
        datamodule: NatronDataModule,
        config: PPOConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.datamodule = datamodule
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, output_dir: Path) -> Path:
        if not getattr(self.datamodule, "_prepared", False):
            self.datamodule.setup("fit")
        dataset = self.datamodule.train_dataset
        if dataset is None:
            raise RuntimeError("Datamodule not prepared")
        sequences = dataset.sequences.to(self.device)
        embeddings = self._encode_sequences(sequences)
        prices = self._extract_prices(len(embeddings))

        env_config = TradingEnvConfig(alpha=self.cfg.turnover_penalty, beta=self.cfg.drawdown_penalty, max_episode_length=self.cfg.max_episode_length)
        env = DummyVecEnv([lambda: TradingEnv(embeddings, prices, env_config)])

        policy_kwargs = dict(net_arch=[dict(pi=list(self.cfg.policy_hidden_sizes), vf=list(self.cfg.policy_hidden_sizes))])
        model = PPO(
            "MlpPolicy",
            env,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_range=self.cfg.clip_range,
            vf_coef=self.cfg.value_coef,
            ent_coef=self.cfg.entropy_coef,
            learning_rate=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

        logger.info("Starting PPO training for %d steps", self.cfg.total_steps)
        model.learn(total_timesteps=self.cfg.total_steps)
        output_dir.mkdir(parents=True, exist_ok=True)
        policy_path = output_dir / "ppo_policy.zip"
        model.save(policy_path)
        logger.info("Saved PPO policy to %s", policy_path)
        return policy_path

    def _encode_sequences(self, sequences: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        embeddings = []
        batch_size = 128
        with torch.no_grad():
            for start in range(0, sequences.size(0), batch_size):
                end = min(start + batch_size, sequences.size(0))
                batch = sequences[start:end].to(self.device)
                encoded = self.model.encode(batch)
                embeddings.append(encoded.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def _extract_prices(self, length: int) -> np.ndarray:
        if self.datamodule.dataframe is None:
            raise RuntimeError("Datamodule dataframe not available")
        close = self.datamodule.dataframe["close"].to_numpy(dtype=float)
        seq_len = self.datamodule.cfg.sequence_length
        prices = close[seq_len - 1 : seq_len - 1 + length]
        if len(prices) < length:
            padding = np.repeat(prices[-1], length - len(prices))
            prices = np.concatenate([prices, padding])
        return prices.astype(np.float32)
