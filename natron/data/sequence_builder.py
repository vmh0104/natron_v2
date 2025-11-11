"""Sequence construction utilities for Natron training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from natron.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SequenceConfig:
    window_length: int = 96


class SequenceCreator:
    """Slice engineered features and labels into fixed-length sequences."""

    def __init__(self, config: SequenceConfig | None = None) -> None:
        self.config = config or SequenceConfig()

    def create_sequences(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> Tuple[np.ndarray, dict]:
        """Create arrays of sequences and aligned labels."""
        window = self.config.window_length
        if len(features) != len(labels):
            raise ValueError("Features and labels must share the same length.")
        if len(features) < window:
            raise ValueError(
                f"Insufficient rows ({len(features)}) for window length {window}."
            )

        feature_array = features.to_numpy(dtype=np.float32)
        label_columns = ["buy", "sell", "direction", "regime"]
        for col in label_columns:
            if col not in labels.columns:
                raise ValueError(f"Label column '{col}' is missing.")

        sequences: list[np.ndarray] = []
        label_targets = {col: [] for col in label_columns}
        indices: list = []

        num_samples = len(features) - window + 1
        for start in range(num_samples):
            end = start + window
            window_slice = feature_array[start:end]
            if np.isnan(window_slice).any():
                continue

            sequences.append(window_slice)
            label_row = labels.iloc[end - 1]
            for col in label_columns:
                label_targets[col].append(label_row[col])
            indices.append(labels.index[end - 1])

        if not sequences:
            logger.warning("No sequences could be generated; returning empty arrays.")
            empty = np.empty((0, window, feature_array.shape[1]), dtype=np.float32)
            outputs = {col: np.array([], dtype=np.int64) for col in label_columns}
            outputs["index"] = np.array([], dtype=object)
            return empty, outputs

        X = np.stack(sequences).astype(np.float32)
        y = {
            "buy": np.asarray(label_targets["buy"], dtype=np.float32),
            "sell": np.asarray(label_targets["sell"], dtype=np.float32),
            "direction": np.asarray(label_targets["direction"], dtype=np.int64),
            "regime": np.asarray(label_targets["regime"], dtype=np.int64),
            "index": np.asarray(indices),
        }

        logger.info(
            "Created %s sequences with shape %s from %s rows.",
            X.shape[0],
            X.shape[1:],
            len(features),
        )

        return X, y


__all__ = ["SequenceCreator", "SequenceConfig"]
