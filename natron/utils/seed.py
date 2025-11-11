import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed all deterministic libraries to ensure reproducible experiments.

    Parameters
    ----------
    seed:
        Random seed value.
    deterministic:
        If True, enforces deterministic behavior for CUDA operations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_seed(seed: Optional[int] = None) -> int:
    """
    Return a valid seed value, generating one if not supplied.
    """
    if seed is not None:
        return seed
    return random.SystemRandom().randint(0, 2**32 - 1)
