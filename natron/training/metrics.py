from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    class_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    loss_buy = F.binary_cross_entropy_with_logits(
        outputs["buy"],
        targets["buy"],
        weight=class_weights["buy"],
    )
    loss_sell = F.binary_cross_entropy_with_logits(
        outputs["sell"],
        targets["sell"],
        weight=class_weights["sell"],
    )
    loss_direction = F.cross_entropy(
        outputs["direction"],
        targets["direction"],
        weight=class_weights["direction"],
    )
    loss_regime = F.cross_entropy(
        outputs["regime"],
        targets["regime"],
        weight=class_weights["regime"],
    )
    total_loss = loss_buy + loss_sell + loss_direction + loss_regime
    return {
        "total": total_loss,
        "buy": loss_buy,
        "sell": loss_sell,
        "direction": loss_direction,
        "regime": loss_regime,
    }
