from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Simple episodic trading environment for RL fine-tuning.
    Actions: 0 = flat, 1 = long, 2 = short.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sequences: np.ndarray,
        closes: np.ndarray,
        alpha: float = 0.001,
        beta: float = 0.001,
    ) -> None:
        super().__init__()
        self.sequences = sequences
        self.closes = closes
        self.alpha = alpha
        self.beta = beta

        seq_len, feat_dim = sequences.shape[1], sequences.shape[2]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, feat_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._index = 0
        self._position = 0
        self._equity_curve = [1.0]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._index = 0
        self._position = 0
        self._equity_curve = [1.0]
        return self.sequences[self._index], {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prev_position = self._position
        self._position = action - 1  # -1, 0, 1

        next_idx = self._index + self.sequences.shape[1]
        if next_idx >= len(self.closes):
            obs = np.zeros_like(self.sequences[0])
            info = {"equity": self._equity_curve[-1], "drawdown": 0.0}
            return obs, 0.0, True, False, info

        price_today = self.closes[next_idx - 1]
        price_next = self.closes[next_idx]
        ret = (price_next - price_today) / price_today

        profit = self._position * ret
        turnover = abs(self._position - prev_position)
        equity = self._equity_curve[-1] * (1 + profit)
        self._equity_curve.append(equity)
        peak = max(self._equity_curve)
        drawdown = (peak - equity) / peak

        reward = profit - self.alpha * turnover - self.beta * drawdown

        self._index += 1
        terminated = self._index >= len(self.sequences) - 1
        truncated = False
        info = {"equity": equity, "drawdown": drawdown}

        obs = self.sequences[self._index] if not terminated else np.zeros_like(self.sequences[0])
        return obs, float(reward), terminated, truncated, info
