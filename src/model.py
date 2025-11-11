"""
NatronTransformer: Multi-task Transformer model for financial trading
Architecture: Transformer Encoder + Multi-task Heads (Buy/Sell/Direction/Regime)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """Multi-task Transformer for financial trading"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 max_seq_length: int = 96):
        """
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
            nn.Linear(dim_feedforward // 2, 3),  # Up, Down, Neutral
            nn.Softmax(dim=-1)
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 6),  # 6 regime classes
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_len, input_dim)
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with predictions and optionally embeddings
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Use last timestep for prediction
        last_hidden = encoded[:, -1, :]  # (batch_size, d_model)
        
        # Multi-task predictions
        buy_prob = self.buy_head(last_hidden).squeeze(-1)
        sell_prob = self.sell_head(last_hidden).squeeze(-1)
        direction_probs = self.direction_head(last_hidden)
        regime_probs = self.regime_head(last_hidden)
        
        output = {
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'direction': direction_probs,
            'regime': regime_probs
        }
        
        if return_embeddings:
            output['embeddings'] = encoded
            output['last_hidden'] = last_hidden
        
        return output
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for pretraining"""
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        return encoded


class MaskedLanguageModelHead(nn.Module):
    """Head for masked token reconstruction (pretraining)"""
    
    def __init__(self, d_model: int, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim)
        )
    
    def forward(self, hidden_states: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            masked_positions: (batch_size, seq_len) boolean mask
        """
        # Extract masked positions
        masked_hidden = hidden_states[masked_positions]
        reconstructed = self.head(masked_hidden)
        return reconstructed


class ContrastiveHead(nn.Module):
    """Head for contrastive learning (pretraining)"""
    
    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model) or (batch_size, d_model)
        """
        if len(hidden_states.shape) == 3:
            # Use last timestep
            hidden_states = hidden_states[:, -1, :]
        # Normalize
        projected = self.projection(hidden_states)
        return F.normalize(projected, p=2, dim=1)
