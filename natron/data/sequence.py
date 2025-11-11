from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SequenceCreatorConfig:
    sequence_length: int = 96
    stride: int = 1
    drop_incomplete: bool = True


class SequenceCreator:
    def __init__(self, config: SequenceCreatorConfig | None = None) -> None:
        self.cfg = config or SequenceCreatorConfig()

    def create(self, features: pd.DataFrame, labels: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same length")
        seq_len = self.cfg.sequence_length
        stride = self.cfg.stride
        num_samples = (len(features) - seq_len) // stride + 1
        if num_samples <= 0:
            raise ValueError("Not enough data to create sequences")

        feature_values = features.to_numpy(dtype=np.float32)
        label_values = {col: labels[col].to_numpy() for col in labels.columns}

        sequences = np.zeros((num_samples, seq_len, feature_values.shape[1]), dtype=np.float32)
        targets: Dict[str, np.ndarray] = {col: np.zeros(num_samples, dtype=label_values[col].dtype) for col in labels.columns}

        idx = 0
        for start in range(0, len(features) - seq_len + 1, stride):
            end = start + seq_len
            sequences[idx] = feature_values[start:end]
            target_idx = end - 1
            for col in labels.columns:
                targets[col][idx] = label_values[col][target_idx]
            idx += 1

        return sequences, targets
