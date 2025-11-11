"""Evaluation metrics for Natron multi-task outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

__all__ = [
    "compute_classification_metrics",
    "multitask_loss_dict",
]


@dataclass
class MetricResult:
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    auc_roc: float | None = None


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_prob: Iterable[Iterable[float]] | None = None,
    average: str = "macro",
) -> MetricResult:
    """Compute common classification metrics."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    auc = None
    if y_prob is not None:
        y_prob = np.asarray(list(y_prob))
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                auc = roc_auc_score(y_true, y_prob, average=average)
            else:
                auc = roc_auc_score(y_true, y_prob, average=average, multi_class="ovo")
        except ValueError:
            auc = None

    return MetricResult(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1_macro=f1,
        precision_macro=precision,
        recall_macro=recall,
        auc_roc=auc,
    )


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def multitask_loss_dict(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Dict[str, float] | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-task loss for Natron outputs.

    Parameters
    ----------
    outputs:
        Dictionary containing tensors for keys 'buy', 'sell', 'direction', 'regime'.
        The buy/sell tensors are logits for binary classification, while direction/regime
        are logits for multi-class softmax.
    targets:
        Dictionary containing ground-truth tensors with matching keys.
    weights:
        Optional dictionary specifying per-task weights.
    """
    weights = weights or {"buy": 1.0, "sell": 1.0, "direction": 1.0, "regime": 1.0}
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_ce = torch.nn.CrossEntropyLoss()

    buy_loss = criterion_bce(outputs["buy"].squeeze(-1), targets["buy"].float())
    sell_loss = criterion_bce(outputs["sell"].squeeze(-1), targets["sell"].float())
    direction_loss = criterion_ce(outputs["direction"], targets["direction"].long())
    regime_loss = criterion_ce(outputs["regime"], targets["regime"].long())

    loss_dict = {
        "buy": buy_loss.item(),
        "sell": sell_loss.item(),
        "direction": direction_loss.item(),
        "regime": regime_loss.item(),
    }

    total_loss = (
        weights["buy"] * buy_loss
        + weights["sell"] * sell_loss
        + weights["direction"] * direction_loss
        + weights["regime"] * regime_loss
    )

    return total_loss, loss_dict
