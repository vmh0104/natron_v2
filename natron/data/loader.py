"""Data loading and preprocessing orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from natron.config import NatronConfig
from natron.datasets.sequences import (
    FeatureScaler,
    MaskedSequenceDataset,
    SequenceDataset,
    split_indices,
)
from natron.features.engine import FeatureEngine
from natron.labels.generator import LabelGeneratorV2
from natron.utils.seed import seed_worker


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from CSV and ensure required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=["time"])
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.sort_values("time").reset_index(drop=True)
    return df


@dataclass
class DataModule:
    cfg: NatronConfig

    def prepare(self) -> Dict[str, Any]:
        df = load_raw_data(self.cfg.data.data_path)

        feature_engine = FeatureEngine(self.cfg.features)
        features = feature_engine.transform(df)

        label_generator = LabelGeneratorV2(self.cfg.labels)
        labels = label_generator.generate(df.set_index("time"), features)

        features_np = features.values.astype("float32")
        label_dict = {col: labels[col].values for col in labels.columns}

        train_idx, val_idx, test_idx = split_indices(
            num_samples=len(features_np),
            cfg=self.cfg.data,
            sequence_length=self.cfg.data.sequence_length,
        )

        scaler = FeatureScaler.fit(features_np[train_idx])

        datasets = {
            "pretrain_train": MaskedSequenceDataset(
                features_np,
                sequence_length=self.cfg.data.sequence_length,
                indices=train_idx,
                masking_ratio=self.cfg.model.masking_ratio,
                scaler=scaler,
            ),
            "pretrain_val": MaskedSequenceDataset(
                features_np,
                sequence_length=self.cfg.data.sequence_length,
                indices=val_idx,
                masking_ratio=self.cfg.model.masking_ratio,
                scaler=scaler,
            ),
            "supervised_train": SequenceDataset(
                features_np,
                labels=label_dict,
                sequence_length=self.cfg.data.sequence_length,
                indices=train_idx,
                scaler=scaler,
            ),
            "supervised_val": SequenceDataset(
                features_np,
                labels=label_dict,
                sequence_length=self.cfg.data.sequence_length,
                indices=val_idx,
                scaler=scaler,
            ),
            "supervised_test": SequenceDataset(
                features_np,
                labels=label_dict,
                sequence_length=self.cfg.data.sequence_length,
                indices=test_idx,
                scaler=scaler,
            ),
        }

        scaled_features = scaler.transform(features_np)

        return {
            "dataframe": df,
            "features": features,
            "features_np": features_np,
            "features_scaled": scaled_features,
            "labels": labels,
            "datasets": datasets,
            "scaler": scaler,
            "label_dict": label_dict,
            "indices": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
        }


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
