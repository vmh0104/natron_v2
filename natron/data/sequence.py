from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceCreator:
    sequence_length: int = 96

    def create(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        valid = features.dropna().index
        features = features.loc[valid].reset_index(drop=True)
        labels = labels.loc[valid].reset_index(drop=True)

        num_samples = len(features) - self.sequence_length + 1
        if num_samples <= 0:
            raise ValueError("Not enough samples to create sequences.")

        feature_array = features.values.astype(np.float32)
        sequences = np.lib.stride_tricks.sliding_window_view(
            feature_array,
            window_shape=(self.sequence_length, feature_array.shape[1]),
        )
        sequences = sequences.reshape(-1, self.sequence_length, feature_array.shape[1])
        sequences = sequences[:num_samples]

        targets = {
            "buy": labels["buy"].values[self.sequence_length - 1 :],
            "sell": labels["sell"].values[self.sequence_length - 1 :],
            "direction": labels["direction"].values[self.sequence_length - 1 :],
            "regime": labels["regime"].values[self.sequence_length - 1 :],
        }
        return sequences, targets


class NatronDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: Dict[str, np.ndarray]) -> None:
        self.sequences = torch.from_numpy(sequences).float()
        self.targets = {
            key: torch.from_numpy(value)
            if key in {"direction", "regime"}
            else torch.from_numpy(value.astype(np.float32))
            for key, value in targets.items()
        }

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        x = self.sequences[idx]
        y = {
            "buy": self.targets["buy"][idx].float(),
            "sell": self.targets["sell"][idx].float(),
            "direction": self.targets["direction"][idx].long(),
            "regime": self.targets["regime"][idx].long(),
        }
        return x, y
