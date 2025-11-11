"""
Natron Transformer - Multi-Task Financial Trading Model
Core model architecture with multi-task heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-Task Transformer for Financial Trading.
    
    Inputs: (batch, seq_len, n_features)
    Outputs:
        - buy_prob: (batch,) binary classification
        - sell_prob: (batch,) binary classification
        - direction_logits: (batch, 3) three-class classification
        - regime_logits: (batch, 6) six-class classification
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_seq_length: int = 96
    ):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Global pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Task-specific heads
        
        # Buy head (binary classification)
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Sell head (binary classification)
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Direction head (3-class classification)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)
        )
        
        # Regime head (6-class classification)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            mask: Optional attention mask
            return_embeddings: If True, return embeddings instead of predictions
            
        Returns:
            Dictionary with predictions or embeddings
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding (need to transpose for pos_encoder)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)  # (batch, seq_len, d_model)
        
        # Global pooling
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        if return_embeddings:
            return {'embeddings': pooled}
        
        # Task-specific predictions
        buy_prob = self.buy_head(pooled).squeeze(-1)  # (batch,)
        sell_prob = self.sell_head(pooled).squeeze(-1)  # (batch,)
        direction_logits = self.direction_head(pooled)  # (batch, 3)
        regime_logits = self.regime_head(pooled)  # (batch, 6)
        
        return {
            'buy': buy_prob,
            'sell': sell_prob,
            'direction': direction_logits,
            'regime': regime_logits
        }
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for pretraining or visualization"""
        return self.forward(x, return_embeddings=True)['embeddings']


class NatronPretrainer(nn.Module):
    """
    Pretraining module with masked reconstruction and contrastive learning.
    """
    
    def __init__(
        self,
        encoder: NatronTransformer,
        mask_ratio: float = 0.15,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.GELU(),
            nn.Linear(encoder.d_model, encoder.n_features)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.GELU(),
            nn.Linear(encoder.d_model, 128)
        )
    
    def mask_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask input features.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            
        Returns:
            masked_x: Masked input
            mask: Boolean mask indicating masked positions
        """
        batch_size, seq_len, n_features = x.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, device=x.device) < self.mask_ratio
        
        # Clone input and mask selected positions
        masked_x = x.clone()
        masked_x[mask] = 0.0
        
        return masked_x, mask
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Pretraining forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            
        Returns:
            Dictionary with reconstruction and contrastive outputs
        """
        # Masked reconstruction
        masked_x, mask = self.mask_input(x)
        embeddings = self.encoder.get_embeddings(masked_x)
        reconstructed = self.reconstruction_head(embeddings)
        
        # Contrastive learning (create two augmented views)
        x_aug1, _ = self.mask_input(x)
        x_aug2, _ = self.mask_input(x)
        
        z1 = self.projection_head(self.encoder.get_embeddings(x_aug1))
        z2 = self.projection_head(self.encoder.get_embeddings(x_aug2))
        
        # Normalize for contrastive loss
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        return {
            'reconstructed': reconstructed,
            'original': x,
            'mask': mask,
            'z1': z1,
            'z2': z2
        }


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss.
    
    Args:
        z1, z2: Normalized embeddings (batch, dim)
        temperature: Temperature parameter
        
    Returns:
        Contrastive loss
    """
    batch_size = z1.shape[0]
    
    # Concatenate z1 and z2
    z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
    
    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2*batch, 2*batch)
    
    # Create labels (positive pairs)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    # Mask out self-similarities
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    
    # Cross entropy loss
    loss = F.cross_entropy(sim, labels)
    
    return loss


def masked_reconstruction_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruction loss only on masked positions.
    
    Args:
        reconstructed: Reconstructed features (batch, n_features)
        original: Original features (batch, seq_len, n_features)
        mask: Boolean mask (batch, seq_len)
        
    Returns:
        MSE loss on masked positions
    """
    # Average original over sequence length
    original_avg = original.mean(dim=1)  # (batch, n_features)
    
    # MSE loss
    loss = F.mse_loss(reconstructed, original_avg, reduction='none')
    
    # Weight by mask (approximate - using mask frequency)
    mask_weight = mask.float().mean(dim=1, keepdim=True)  # (batch, 1)
    loss = (loss * mask_weight).mean()
    
    return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with weighted combination.
    """
    
    def __init__(
        self,
        buy_weight: float = 1.0,
        sell_weight: float = 1.0,
        direction_weight: float = 1.5,
        regime_weight: float = 1.2
    ):
        super().__init__()
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Buy loss (binary cross-entropy)
        buy_loss = F.binary_cross_entropy(
            predictions['buy'],
            targets['buy'],
            reduction='mean'
        )
        
        # Sell loss (binary cross-entropy)
        sell_loss = F.binary_cross_entropy(
            predictions['sell'],
            targets['sell'],
            reduction='mean'
        )
        
        # Direction loss (cross-entropy)
        direction_loss = F.cross_entropy(
            predictions['direction'],
            targets['direction'],
            reduction='mean'
        )
        
        # Regime loss (cross-entropy)
        regime_loss = F.cross_entropy(
            predictions['regime'],
            targets['regime'],
            reduction='mean'
        )
        
        # Total weighted loss
        total_loss = (
            self.buy_weight * buy_loss +
            self.sell_weight * sell_loss +
            self.direction_weight * direction_loss +
            self.regime_weight * regime_loss
        )
        
        loss_dict = {
            'buy': buy_loss.item(),
            'sell': sell_loss.item(),
            'direction': direction_loss.item(),
            'regime': regime_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing NatronTransformer...")
    
    # Model parameters
    n_features = 100
    batch_size = 16
    seq_len = 96
    
    # Create model
    model = NatronTransformer(
        n_features=n_features,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"✅ Model created")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, n_features)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Buy output shape: {outputs['buy'].shape}")
    print(f"   Sell output shape: {outputs['sell'].shape}")
    print(f"   Direction output shape: {outputs['direction'].shape}")
    print(f"   Regime output shape: {outputs['regime'].shape}")
    
    # Test pretrainer
    print(f"\n🔧 Testing pretrainer...")
    pretrainer = NatronPretrainer(model)
    
    with torch.no_grad():
        pretrain_outputs = pretrainer(x)
    
    print(f"✅ Pretraining forward pass successful")
    print(f"   Reconstructed shape: {pretrain_outputs['reconstructed'].shape}")
    print(f"   z1 shape: {pretrain_outputs['z1'].shape}")
    print(f"   z2 shape: {pretrain_outputs['z2'].shape}")
