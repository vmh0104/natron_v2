"""
NatronTransformer: Multi-task Transformer model for financial trading
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-task Transformer model for financial trading.
    
    Architecture:
    - Transformer Encoder (learns temporal patterns)
    - Multi-task heads: buy, sell, direction, regime
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_length: int = 96,
        num_features: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Multi-task heads
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 3)  # Up, Down, Neutral
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 6)  # 6 regime classes
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)
            mask: Optional attention mask
            
        Returns:
            Dictionary with predictions: buy, sell, direction, regime
        """
        batch_size, seq_length, num_features = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Use last timestep for prediction
        x_last = x[:, -1, :]  # (batch_size, d_model)
        
        # Multi-task predictions
        buy_logit = self.buy_head(x_last).squeeze(-1)
        sell_logit = self.sell_head(x_last).squeeze(-1)
        direction_logits = self.direction_head(x_last)
        regime_logits = self.regime_head(x_last)
        
        return {
            'buy': buy_logit,
            'sell': sell_logit,
            'direction': direction_logits,
            'regime': regime_logits
        }
