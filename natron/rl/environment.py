from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(slots=True)
class TradingEnvConfig:
    alpha: float = 0.001  # turnover penalty
    beta: float = 0.005   # drawdown penalty
    max_episode_length: int = 500


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, embeddings: np.ndarray, prices: np.ndarray, config: TradingEnvConfig | None = None) -> None:
        super().__init__()
        if len(embeddings) != len(prices):
            raise ValueError("Embeddings and prices must have the same length")
        self.cfg = config or TradingEnvConfig()
        self.embeddings = embeddings.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.action_space = spaces.Discrete(3)  # 0 hold, 1 long, 2 short
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embeddings.shape[1],),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.current_step = int(self._rng.integers(0, max(1, len(self.embeddings) - self.cfg.max_episode_length)))
        self.episode_length = 0
        self.position = 0  # -1 short, 0 flat, 1 long
        self.equity = 1.0
        self.max_equity = 1.0
        self.initial_price = float(self.prices[self.current_step])
        self.trade_history: list[Dict[str, Any]] = []
        observation = self.embeddings[self.current_step]
        return observation, {}

    def step(self, action: int):
        if action not in {0, 1, 2}:
            raise ValueError("Invalid action")

        next_step = self.current_step + 1
        done = False
        truncated = False

        if next_step >= len(self.embeddings):
            done = True
            next_step = len(self.embeddings) - 1

        price_now = float(self.prices[self.current_step])
        price_next = float(self.prices[next_step])
        price_return = (price_next - price_now) / (price_now + 1e-9)

        target_position = self.position
        if action == 1:
            target_position = 1
        elif action == 2:
            target_position = -1
        elif action == 0:
            target_position = self.position

        turnover = abs(target_position - self.position)
        self.position = target_position
        pnl = self.position * price_return
        self.equity += pnl
        self.max_equity = max(self.max_equity, self.equity)
        drawdown = max(0.0, self.max_equity - self.equity)

        reward = pnl - self.cfg.alpha * turnover - self.cfg.beta * drawdown
        self.trade_history.append(
            {
                "step": self.current_step,
                "action": action,
                "position": self.position,
                "price_now": price_now,
                "price_next": price_next,
                "reward": reward,
                "pnl": pnl,
                "equity": self.equity,
                "drawdown": drawdown,
            }
        )

        self.current_step = next_step
        self.episode_length += 1
        if self.episode_length >= self.cfg.max_episode_length:
            truncated = True

        observation = self.embeddings[self.current_step]
        info = {"equity": self.equity, "drawdown": drawdown}
        return observation, reward, done, truncated, info

    def render(self):  # pragma: no cover - optional visualization hook
        if not self.trade_history:
            print("No trades yet")
            return
        last_trade = self.trade_history[-1]
        print(
            f"Step {last_trade['step']}: action={last_trade['action']} position={last_trade['position']} "
            f"reward={last_trade['reward']:.5f} equity={last_trade['equity']:.4f}"
        )

    def close(self):  # pragma: no cover
        pass
