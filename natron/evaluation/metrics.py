from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def multilabel_bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights)
    return loss_fn(predictions, targets)


def cross_entropy_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    return loss_fn(predictions, targets)
