from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from natron.models.heads import build_task_heads
from natron.models.modules import ProjectionHead, TransformerBackbone


class NatronTransformer(nn.Module):
    def __init__(self, input_dim: int, model_cfg: dict) -> None:
        super().__init__()
        self.d_model = model_cfg["d_model"]
        self.backbone = TransformerBackbone(
            input_dim=input_dim,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            num_layers=model_cfg["num_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            max_len=model_cfg["max_len"],
            embedding_cfg=model_cfg.get("embedding", {}),
        )
        self.task_heads = build_task_heads(self.d_model)
        self.pretrain_head = ProjectionHead(self.d_model, out_dim=input_dim)
        self.contrastive_head = ProjectionHead(self.d_model, out_dim=self.d_model)

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.backbone(x, mask=mask)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoded = self.encode(x, mask=mask)
        pooled = encoded.mean(dim=1)

        outputs = {
            "buy": self.task_heads["buy"](pooled).squeeze(-1),
            "sell": self.task_heads["sell"](pooled).squeeze(-1),
            "direction": self.task_heads["direction"](pooled),
            "regime": self.task_heads["regime"](pooled),
        }

        if return_sequence:
            outputs["sequence"] = encoded

        return outputs, encoded

    def masked_reconstruction(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x, mask=None)
        reconstructed = self.pretrain_head(encoded)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(reconstructed)
        return reconstructed[mask].view(-1, reconstructed.size(-1))

    def contrastive_projection(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x, mask=None)
        pooled = encoded.mean(dim=1)
        return nn.functional.normalize(self.contrastive_head(pooled), dim=-1)
