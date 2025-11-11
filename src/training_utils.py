"""
Training utilities and loss functions for Natron Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self,
                 buy_weight: float = 1.0,
                 sell_weight: float = 1.0,
                 direction_weight: float = 1.0,
                 regime_weight: float = 1.0,
                 use_class_weights: bool = True):
        super().__init__()
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        self.use_class_weights = use_class_weights
        
        # Individual loss functions
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dict with 'buy_prob', 'sell_prob', 'direction', 'regime'
            targets: (batch_size, 4) tensor [buy, sell, direction, regime]
        """
        batch_size = targets.shape[0]
        
        # Extract targets
        buy_target = targets[:, 0]
        sell_target = targets[:, 1]
        direction_target = targets[:, 2].long()
        regime_target = targets[:, 3].long()
        
        # Calculate losses
        buy_loss = self.bce_loss(predictions['buy_prob'], buy_target)
        sell_loss = self.bce_loss(predictions['sell_prob'], sell_target)
        direction_loss = self.ce_loss(predictions['direction'], direction_target)
        regime_loss = self.ce_loss(predictions['regime'], regime_target)
        
        # Weighted combination
        total_loss = (self.buy_weight * buy_loss +
                     self.sell_weight * sell_loss +
                     self.direction_weight * direction_loss +
                     self.regime_weight * regime_loss)
        
        return {
            'total_loss': total_loss,
            'buy_loss': buy_loss,
            'sell_loss': sell_loss,
            'direction_loss': direction_loss,
            'regime_loss': regime_loss
        }


class PretrainingLoss(nn.Module):
    """Loss for pretraining (masked reconstruction + contrastive)"""
    
    def __init__(self,
                 reconstruction_weight: float = 0.5,
                 contrastive_weight: float = 0.5,
                 temperature: float = 0.07):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self,
                reconstructed_features: torch.Tensor,
                original_features: torch.Tensor,
                masked_positions: torch.Tensor,
                embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            reconstructed_features: Reconstructed features at masked positions
            original_features: Original features (batch_size, seq_len, feature_dim)
            masked_positions: Boolean mask (batch_size, seq_len)
            embeddings: Encoder embeddings (batch_size, seq_len, d_model)
        """
        # Reconstruction loss (MSE on masked positions)
        masked_original = original_features[masked_positions]
        recon_loss = self.mse_loss(reconstructed_features, masked_original)
        
        # Contrastive loss (InfoNCE)
        # Use last timestep embeddings as representations
        batch_size = embeddings.shape[0]
        representations = embeddings[:, -1, :]  # (batch_size, d_model)
        
        # Normalize
        representations = F.normalize(representations, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels (each sample is positive to itself)
        labels = torch.arange(batch_size, device=embeddings.device)
        
        # InfoNCE loss
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        # Combined loss
        total_loss = (self.reconstruction_weight * recon_loss +
                     self.contrastive_weight * contrastive_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'contrastive_loss': contrastive_loss
        }


def create_optimizer(model: nn.Module, 
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-5,
                    optimizer_type: str = "adamw") -> torch.optim.Optimizer:
    """Create optimizer"""
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = "reduce_on_plateau",
                    patience: int = 5,
                    factor: float = 0.5) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    if scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor, verbose=True
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
