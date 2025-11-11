from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def temporal_train_val_test_split(
    sequences: np.ndarray,
    targets: dict,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[Tuple[np.ndarray, dict], Tuple[np.ndarray, dict], Tuple[np.ndarray, dict]]:
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1
    total = len(sequences)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)

    train_end = total - val_size - test_size
    val_end = total - test_size

    x_train = sequences[:train_end]
    x_val = sequences[train_end:val_end]
    x_test = sequences[val_end:]

    y_train = {k: v[:train_end] for k, v in targets.items()}
    y_val = {k: v[train_end:val_end] for k, v in targets.items()}
    y_test = {k: v[val_end:] for k, v in targets.items()}

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def balance_binary_labels(labels: pd.Series, target_ratio: float = 0.35) -> pd.Series:
    ones = labels[labels == 1]
    zeros = labels[labels == 0]

    if len(ones) == 0 or len(zeros) == 0:
        return labels

    desired_ones = int(len(labels) * target_ratio)
    desired_zeros = len(labels) - desired_ones

    if len(ones) > desired_ones:
        ones = ones.sample(desired_ones, random_state=42)
    if len(zeros) > desired_zeros:
        zeros = zeros.sample(desired_zeros, random_state=42)

    balanced = pd.concat([ones, zeros]).sort_index()
    return balanced.reindex(labels.index).fillna(method="bfill").fillna(method="ffill")


def sliding_window(
    array: np.ndarray,
    window_size: int,
    step: int = 1,
) -> Iterable[np.ndarray]:
    for start in range(0, len(array) - window_size + 1, step):
        yield array[start : start + window_size]
