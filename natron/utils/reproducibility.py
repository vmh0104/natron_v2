from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SeedContext:
    seed: int
    deterministic: bool = True
    benchmark: bool = False

    def __enter__(self) -> None:
        set_seed(self.seed, self.deterministic, self.benchmark)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        return False


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
