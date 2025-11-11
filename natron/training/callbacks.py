from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 1e-4
    best_score: float = field(default=float("inf"), init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def step(self, metric: float) -> None:
        if metric + self.min_delta < self.best_score:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
