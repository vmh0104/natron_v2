"""
Natron Transformer Model - Multi-task Transformer for Financial Trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-task Transformer for financial trading
    
    Architecture:
    - Input projection: features -> d_model
    - Positional encoding
    - Transformer encoder (6 layers)
    - Multi-task heads: buy, sell, direction, regime
    """
    
    def __init__(self,
                 num_features: int = 100,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 96,
                 activation: str = "gelu"):
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling (mean + max)
        self.pooling = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task heads
        # Buy/Sell heads (binary classification)
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Direction head (3-class: Down, Up, Neutral)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
            nn.LogSoftmax(dim=1)
        )
        
        # Regime head (6-class)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6),
            nn.LogSoftmax(dim=1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features)
            mask: Optional attention mask (batch_size, seq_len)
        
        Returns:
            Dictionary with predictions for all tasks
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if needed
        src_mask = None
        if mask is not None:
            src_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            src_mask = src_mask.expand(batch_size, 1, seq_len, seq_len)
            src_mask = src_mask.masked_fill(src_mask == 0, float('-inf'))
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (batch_size, seq_len, d_model)
        
        # Global pooling (mean + max)
        x_mean = x.mean(dim=1)  # (batch_size, d_model)
        x_max = x.max(dim=1)[0]  # (batch_size, d_model)
        x_pooled = torch.cat([x_mean, x_max], dim=1)  # (batch_size, 2*d_model)
        x_pooled = self.pooling(x_pooled)  # (batch_size, d_model)
        
        # Multi-task predictions
        buy_prob = self.buy_head(x_pooled).squeeze(-1)  # (batch_size,)
        sell_prob = self.sell_head(x_pooled).squeeze(-1)  # (batch_size,)
        direction_logits = self.direction_head(x_pooled)  # (batch_size, 3)
        regime_logits = self.regime_head(x_pooled)  # (batch_size, 6)
        
        return {
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'direction_logits': direction_logits,
            'regime_logits': regime_logits,
            'hidden_state': x_pooled  # For pretraining
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hidden representations for pretraining"""
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        return self.pooling(x_pooled)


def create_model(config: dict) -> NatronTransformer:
    """Create model from config dictionary"""
    return NatronTransformer(
        num_features=config['num_features'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length'],
        activation=config.get('activation', 'gelu')
    )
