"""Dataset utilities for Natron sequence modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from natron.config import DataConfig


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray) -> "FeatureScaler":
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std < 1e-5] = 1e-5
        return cls(mean=mean, std=std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class SequenceDataset(Dataset):
    """Supervised dataset returning sequences and multi-task targets."""

    def __init__(
        self,
        features: np.ndarray,
        labels: Dict[str, np.ndarray],
        sequence_length: int,
        indices: Iterable[int],
        scaler: FeatureScaler | None = None,
    ):
        self.features = features.astype(np.float32)
        self.labels = labels
        self.sequence_length = sequence_length
        self.indices = np.array(list(indices))
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = self.indices[idx]
        start = anchor - self.sequence_length + 1
        seq = self.features[start : anchor + 1]
        if self.scaler:
            seq = self.scaler.transform(seq)
        seq_tensor = torch.from_numpy(seq)

        targets = {
            "buy": torch.tensor(self.labels["buy"][anchor], dtype=torch.float32),
            "sell": torch.tensor(self.labels["sell"][anchor], dtype=torch.float32),
            "direction": torch.tensor(self.labels["direction"][anchor], dtype=torch.long),
            "regime": torch.tensor(self.labels["regime"][anchor], dtype=torch.long),
        }
        return {"inputs": seq_tensor, "targets": targets}


class MaskedSequenceDataset(Dataset):
    """Dataset for masked modeling and contrastive pretraining."""

    def __init__(
        self,
        features: np.ndarray,
        sequence_length: int,
        indices: Iterable[int],
        masking_ratio: float = 0.15,
        scaler: FeatureScaler | None = None,
    ):
        self.features = features.astype(np.float32)
        self.sequence_length = sequence_length
        self.indices = np.array(list(indices))
        self.masking_ratio = masking_ratio
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = self.indices[idx]
        start = anchor - self.sequence_length + 1
        seq = self.features[start : anchor + 1]
        if self.scaler:
            seq = self.scaler.transform(seq)
        seq_tensor = torch.from_numpy(seq)

        mask = torch.zeros(seq_tensor.shape[:2], dtype=torch.bool)
        num_mask = max(1, int(self.masking_ratio * mask.numel()))
        flat_indices = torch.randperm(mask.numel())[:num_mask]
        mask.view(-1)[flat_indices] = True
        masked_seq = seq_tensor.clone()
        masked_seq[mask] = 0.0

        # Augmented view for contrastive learning
        noise = torch.randn_like(seq_tensor) * 0.01
        aug_seq = seq_tensor + noise

        return {
            "inputs": seq_tensor,
            "masked_inputs": masked_seq,
            "mask": mask,
            "augmented": aug_seq,
        }


def build_indices(num_samples: int, sequence_length: int) -> np.ndarray:
    """Return anchor indices for valid sequences."""
    return np.arange(sequence_length - 1, num_samples)


def split_indices(num_samples: int, cfg: DataConfig, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = build_indices(num_samples, sequence_length)
    total = len(indices)
    train_end = int(total * cfg.train_split)
    val_end = train_end + int(total * cfg.val_split)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx
