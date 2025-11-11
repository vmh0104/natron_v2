from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (logits.sigmoid() > 0.5).float()
    correct = (preds == targets.float()).float().mean().item()
    return correct


def multiclass_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().mean().item()
    return correct


def compute_metrics(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = {
        "buy_acc": binary_accuracy(outputs["buy_logits"], targets["buy"]),
        "sell_acc": binary_accuracy(outputs["sell_logits"], targets["sell"]),
        "direction_acc": multiclass_accuracy(outputs["direction_logits"], targets["direction"]),
        "regime_acc": multiclass_accuracy(outputs["regime_logits"], targets["regime"]),
    }
    return metrics
