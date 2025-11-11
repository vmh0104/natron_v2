from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore

from .feature_engine import FeatureEngine, FeatureEngineConfig
from .labeling import LabelConfig, LabelGeneratorV2
from .sequence import SequenceCreator, SequenceCreatorConfig
from ..utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(slots=True)
class DataModuleConfig:
    csv_path: str
    sequence_length: int = 96
    batch_size: int = 64
    num_workers: int = 4
    val_split: float = 0.1
    test_split: float = 0.1
    cache_features: Optional[str] = None
    cache_labels: Optional[str] = None
    feature_config: FeatureEngineConfig = FeatureEngineConfig()
    label_config: LabelConfig = LabelConfig()
    sequence_config: SequenceCreatorConfig = SequenceCreatorConfig()


class NatronDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, targets: Dict[str, torch.Tensor]) -> None:
        self.sequences = sequences
        self.targets = targets

    def __len__(self) -> int:
        return self.sequences.size(0)

    def __getitem__(self, idx: int):
        x = self.sequences[idx]
        y = {key: value[idx] for key, value in self.targets.items()}
        return x, y


class NatronDataModule(pl.LightningDataModule if pl else object):  # type: ignore[misc]
    def __init__(self, config: DataModuleConfig) -> None:
        if pl is None:
            raise ImportError("pytorch-lightning is required for NatronDataModule")
        super().__init__()
        self.cfg = config
        self.feature_engine = FeatureEngine(self.cfg.feature_config)
        self.label_generator = LabelGeneratorV2(self.cfg.label_config)
        self.sequence_creator = SequenceCreator(self.cfg.sequence_config)

        self.train_dataset: Optional[NatronDataset] = None
        self.val_dataset: Optional[NatronDataset] = None
        self.test_dataset: Optional[NatronDataset] = None
        self.dataframe: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.DataFrame] = None
        self._prepared = False

    def setup(self, stage: Optional[str] = None) -> None:  # pragma: no cover - orchestrates heavy IO
        if self._prepared:
            return
        df = self._load_csv(self.cfg.csv_path)
        features = self._load_or_compute_features(df)
        labels = self._load_or_compute_labels(df, features)
        self.dataframe = df
        self.features = features
        self.labels = labels
        sequences, targets = self.sequence_creator.create(features, labels)

        # Convert to tensors
        sequences_tensor = torch.from_numpy(sequences)
        targets_tensor: Dict[str, torch.Tensor] = {}
        for key, values in targets.items():
            if key in {"direction", "regime"}:
                targets_tensor[key] = torch.as_tensor(values, dtype=torch.long)
            else:
                targets_tensor[key] = torch.as_tensor(values, dtype=torch.float32)

        dataset = NatronDataset(sequences_tensor, targets_tensor)
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset(dataset)
        self._prepared = True

    def train_dataloader(self) -> DataLoader:
        if not self._prepared:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if not self._prepared:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if not self._prepared:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    # --- helpers -----------------------------------------------------------------

    def _load_csv(self, path: str) -> pd.DataFrame:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV data not found at {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["time"]).sort_values("time")
        logger.info("Loaded %d rows from %s", len(df), csv_path)
        return df

    def _load_or_compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.cache_features and Path(self.cfg.cache_features).exists():
            logger.info("Loading cached features from %s", self.cfg.cache_features)
            return pd.read_parquet(self.cfg.cache_features)
        features = self.feature_engine.transform(df)
        if self.cfg.cache_features:
            Path(self.cfg.cache_features).parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(self.cfg.cache_features, index=False)
        logger.info("Generated features with shape %s", features.shape)
        return features

    def _load_or_compute_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.cache_labels and Path(self.cfg.cache_labels).exists():
            logger.info("Loading cached labels from %s", self.cfg.cache_labels)
            return pd.read_parquet(self.cfg.cache_labels)
        labels = self.label_generator.generate(df, features)
        if self.cfg.cache_labels:
            Path(self.cfg.cache_labels).parent.mkdir(parents=True, exist_ok=True)
            labels.to_parquet(self.cfg.cache_labels, index=False)
        return labels

    def _split_dataset(self, dataset: NatronDataset) -> tuple[NatronDataset, NatronDataset, NatronDataset]:
        length = len(dataset)
        if length < 3:
            raise ValueError("Dataset too small for train/val/test split")

        val_len = max(1, int(length * self.cfg.val_split))
        test_len = max(1, int(length * self.cfg.test_split))
        train_len = length - val_len - test_len
        if train_len <= 0:
            raise ValueError("Not enough samples for training after splits")

        generator = torch.Generator().manual_seed(42)
        permutation = torch.randperm(length, generator=generator)

        train_idx = permutation[:train_len]
        val_idx = permutation[train_len : train_len + val_len]
        test_idx = permutation[train_len + val_len :]

        return (
            self._subset(dataset, train_idx),
            self._subset(dataset, val_idx),
            self._subset(dataset, test_idx),
        )

    def _subset(self, dataset: NatronDataset, indices: torch.Tensor) -> NatronDataset:
        sequences = dataset.sequences.index_select(0, indices)
        targets = {key: value.index_select(0, indices) for key, value in dataset.targets.items()}
        return NatronDataset(sequences, targets)
