"""Trading environment for reinforcement learning phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TradingState:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class TradingEnvironment:
    """
    Lightweight environment for PPO/SAC style training.

    Observations are normalized feature sequences. Actions are discrete:
        0 -> short
        1 -> flat
        2 -> long
    """

    def __init__(
        self,
        sequences: np.ndarray,
        returns: np.ndarray,
        alpha: float,
        beta: float,
    ):
        assert len(sequences) == len(returns), "Sequences and returns must match"
        self.sequences = sequences
        self.returns = returns
        self.alpha = alpha
        self.beta = beta
        self.position = 0
        self.ptr = 0
        self.equity = 1.0
        self.max_equity = 1.0

    def reset(self) -> np.ndarray:
        self.position = 0
        self.ptr = 0
        self.equity = 1.0
        self.max_equity = 1.0
        return self.sequences[self.ptr]

    def step(self, action: int) -> TradingState:
        assert action in (0, 1, 2)
        current_return = self.returns[self.ptr]
        position_map = {0: -1, 1: 0, 2: 1}
        action_value = position_map[action]
        turnover = abs(action_value - self.position)
        profit = action_value * current_return
        self.position = action_value
        self.equity *= (1 + profit)
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = (self.max_equity - self.equity) / (self.max_equity + 1e-9)

        reward = profit - self.alpha * turnover - self.beta * drawdown
        self.ptr += 1
        done = self.ptr >= len(self.sequences) - 1
        next_obs = self.sequences[self.ptr] if not done else self.reset()

        return TradingState(
            observation=next_obs,
            reward=float(reward),
            done=done,
            info={"equity": self.equity, "drawdown": drawdown},
        )
