"""Data preparation utilities and PyTorch datasets for Natron."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from natron.data.feature_engineer import FeatureEngine
from natron.data.label_generator import LabelGeneratorV2
from natron.data.sequence_builder import SequenceConfig, SequenceCreator
from natron.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DataModuleConfig:
    window_length: int = 96
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    num_workers: int = 0
    drop_last: bool = True


class NatronPretrainDataset(Dataset):
    def __init__(self, sequences: np.ndarray) -> None:
        self.inputs = torch.from_numpy(sequences).float()

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.inputs[idx]


class NatronSupervisedDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: Dict[str, np.ndarray]) -> None:
        self.inputs = torch.from_numpy(sequences).float()
        self.buy = torch.from_numpy(labels["buy"]).float()
        self.sell = torch.from_numpy(labels["sell"]).float()
        self.direction = torch.from_numpy(labels["direction"]).long()
        self.regime = torch.from_numpy(labels["regime"]).long()

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "buy": self.buy[idx],
            "sell": self.sell[idx],
            "direction": self.direction[idx],
            "regime": self.regime[idx],
        }


class NatronDataModule:
    """Loads OHLCV data, produces features/labels, and builds DataLoaders."""

    def __init__(self, csv_path: Path, config: DataModuleConfig | None = None) -> None:
        self.csv_path = Path(csv_path)
        self.config = config or DataModuleConfig()
        self.sequence_creator = SequenceCreator(SequenceConfig(window_length=self.config.window_length))
        self.feature_engine = FeatureEngine()
        self.label_generator = LabelGeneratorV2()
        self.scaler = StandardScaler()

        self.dataframe: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.labels: pd.DataFrame | None = None
        self.sequences: np.ndarray | None = None
        self.sequence_labels: Dict[str, np.ndarray] | None = None
        self.sequence_index: np.ndarray | None = None

        self.train_idx: np.ndarray | None = None
        self.val_idx: np.ndarray | None = None
        self.test_idx: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Preparation and splitting
    # ------------------------------------------------------------------ #
    def prepare(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.csv_path}")

        logger.info("Loading OHLCV data from %s", self.csv_path)
        df = pd.read_csv(self.csv_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").set_index("time")
        else:
            df = df.sort_index()
        self.dataframe = df

        logger.info("Computing engineered features (~100 dimensions)")
        features = self.feature_engine.transform(df)
        scaled_values = self.scaler.fit_transform(features)
        scaled_features = pd.DataFrame(scaled_values, index=features.index, columns=features.columns)
        self.features = scaled_features

        logger.info("Generating institutional labels (buy/sell/direction/regime)")
        labels = self.label_generator.transform(scaled_features, df)
        self.labels = labels

        logger.info("Constructing fixed-length sequences (window=%s)", self.config.window_length)
        sequences, label_dict = self.sequence_creator.create_sequences(scaled_features, labels)
        sequence_index = label_dict.pop("index")

        self.sequences = sequences
        self.sequence_labels = label_dict
        self.sequence_index = sequence_index

        self._split_sequences()

    def _split_sequences(self) -> None:
        assert self.sequences is not None
        total = self.sequences.shape[0]
        train_end = max(int(total * self.config.train_ratio), 1)
        val_end = train_end + max(int(total * self.config.val_ratio), 1)
        val_end = min(val_end, total)
        if val_end == total:
            val_end = total - 1
        val_end = max(val_end, train_end)
        self.train_idx = np.arange(0, train_end)
        self.val_idx = np.arange(train_end, val_end)
        self.test_idx = np.arange(val_end, total)

    # ------------------------------------------------------------------ #
    # DataLoader builders
    # ------------------------------------------------------------------ #
    def get_pretrain_dataloader(self, split: str, batch_size: int | None = None, shuffle: bool | None = None) -> DataLoader:
        sequences = self._subset_sequences(split)
        dataset = NatronPretrainDataset(sequences)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle if shuffle is not None else split == "train",
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
        )

    def get_supervised_dataloader(self, split: str, batch_size: int | None = None, shuffle: bool | None = None) -> DataLoader:
        sequences = self._subset_sequences(split)
        labels = self._subset_labels(split)
        dataset = NatronSupervisedDataset(sequences, labels)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle if shuffle is not None else split == "train",
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
        )

    def _subset_sequences(self, split: str) -> np.ndarray:
        if self.sequences is None:
            raise RuntimeError("DataModule not prepared. Call prepare() first.")
        indices = self._split_indices(split)
        return self.sequences[indices]

    def _subset_labels(self, split: str) -> Dict[str, np.ndarray]:
        if self.sequence_labels is None:
            raise RuntimeError("DataModule not prepared. Call prepare() first.")
        indices = self._split_indices(split)
        return {key: value[indices] for key, value in self.sequence_labels.items() if key != "index"}

    def get_split_indices(self, split: str) -> np.ndarray:
        return self._split_indices(split)

    def _split_indices(self, split: str) -> np.ndarray:
        if self.train_idx is None or self.val_idx is None or self.test_idx is None:
            raise RuntimeError("Splits not initialized. Call prepare().")
        if split == "train":
            return self.train_idx
        if split == "val":
            return self.val_idx
        if split == "test":
            return self.test_idx
        raise ValueError(f"Unknown split: {split}")

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def feature_dim(self) -> int:
        if self.features is None:
            raise RuntimeError("Features have not been computed yet.")
        return self.features.shape[1]

    @property
    def sequence_count(self) -> int:
        if self.sequences is None:
            raise RuntimeError("Sequences not computed yet.")
        return self.sequences.shape[0]

    @property
    def sequence_timestamps(self) -> np.ndarray:
        if self.sequence_index is None:
            raise RuntimeError("Sequence indices not available.")
        return self.sequence_index


__all__ = [
    "DataModuleConfig",
    "NatronDataModule",
    "NatronPretrainDataset",
    "NatronSupervisedDataset",
]
