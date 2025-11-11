from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    pred_masked = pred[mask]
    target_masked = target[mask]
    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred_masked, target_masked)


def info_nce_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    logits = query @ key.t() / temperature
    labels = torch.arange(query.size(0), device=query.device)
    loss_qk = F.cross_entropy(logits, labels)
    loss_kq = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_qk + loss_kq)


def multitask_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], weights: Dict[str, float] | None = None) -> torch.Tensor:
    weights = weights or {"buy": 1.0, "sell": 1.0, "direction": 1.0, "regime": 1.0}
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    loss_buy = bce(outputs["buy_logits"], targets["buy"].float()) * weights.get("buy", 1.0)
    loss_sell = bce(outputs["sell_logits"], targets["sell"].float()) * weights.get("sell", 1.0)
    loss_direction = ce(outputs["direction_logits"], targets["direction"].long()) * weights.get("direction", 1.0)
    loss_regime = ce(outputs["regime_logits"], targets["regime"].long()) * weights.get("regime", 1.0)

    return loss_buy + loss_sell + loss_direction + loss_regime
