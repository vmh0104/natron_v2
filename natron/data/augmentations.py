from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SequenceAugmentations:
    noise_std: float = 0.01
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    time_mask_ratio: float = 0.1

    def jitter(self, sequence: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, size=sequence.shape)
        return sequence + noise

    def scaling(self, sequence: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(*self.scaling_range)
        return sequence * factor

    def time_mask(self, sequence: np.ndarray) -> np.ndarray:
        seq = sequence.copy()
        length = seq.shape[0]
        num_mask = max(1, int(length * self.time_mask_ratio))
        mask_indices = np.random.choice(length, num_mask, replace=False)
        seq[mask_indices] = 0
        return seq

    def augment_pair(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        aug1 = self.jitter(self.scaling(sequence))
        aug2 = self.time_mask(self.jitter(sequence))
        return aug1, aug2
