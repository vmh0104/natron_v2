"""
Natron Multi-Task Transformer - Core Architecture
Optimized for financial time series with multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FeatureEmbedding(nn.Module):
    """Projects input features to model dimension"""
    
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.projection(x)
        x = self.layer_norm(x)
        return self.dropout(x)


class MultiTaskHead(nn.Module):
    """Multi-task prediction heads"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Shared representation
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model // 2)
        )
        
        # Buy head (binary classification)
        self.buy_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)
        )
        
        # Sell head (binary classification)
        self.sell_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)
        )
        
        # Direction head (3-class: Down, Up, Neutral)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 3)
        )
        
        # Regime head (6-class market states)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 6)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, d_model) - aggregated sequence representation
            
        Returns:
            Dict with keys: 'buy', 'sell', 'direction', 'regime'
        """
        shared = self.shared_fc(x)
        
        return {
            'buy': self.buy_head(shared),
            'sell': self.sell_head(shared),
            'direction': self.direction_head(shared),
            'regime': self.regime_head(shared)
        }


class NatronTransformer(nn.Module):
    """
    Natron Multi-Task Transformer Model
    
    Architecture:
    1. Feature Embedding: n_features → d_model
    2. Positional Encoding
    3. Transformer Encoder (N layers)
    4. Sequence Aggregation (mean pooling + attention)
    5. Multi-task Heads (Buy, Sell, Direction, Regime)
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_length: int = 96
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        
        # Input embedding
        self.feature_embedding = FeatureEmbedding(n_features, d_model, dropout)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # (seq, batch, features)
            norm_first=True  # Pre-LN for training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Attention pooling for sequence aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=0)
        )
        
        # Multi-task heads
        self.multi_task_head = MultiTaskHead(d_model, dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, n_features)
            src_mask: Optional attention mask
            return_features: If True, return encoder features
            
        Returns:
            Dict with task predictions and optionally features
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Feature embedding
        x = self.feature_embedding(x)  # (batch, seq, d_model)
        
        # 2. Reshape for transformer (seq, batch, d_model)
        x = x.transpose(0, 1)
        
        # 3. Add positional encoding
        x = self.pos_encoder(x)
        
        # 4. Transformer encoding
        encoded = self.transformer_encoder(x, mask=src_mask)  # (seq, batch, d_model)
        
        # 5. Sequence aggregation using attention pooling
        attention_weights = self.attention_pool(encoded)  # (seq, batch, 1)
        aggregated = (encoded * attention_weights).sum(dim=0)  # (batch, d_model)
        
        # 6. Multi-task predictions
        outputs = self.multi_task_head(aggregated)
        
        if return_features:
            outputs['features'] = aggregated
            outputs['attention_weights'] = attention_weights.squeeze(-1).transpose(0, 1)
        
        return outputs
    
    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract encoder features for pretraining or analysis
        
        Args:
            x: (batch, seq_len, n_features)
            
        Returns:
            (batch, seq_len, d_model) - encoded sequence
        """
        # Feature embedding
        x = self.feature_embedding(x)
        x = x.transpose(0, 1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Back to (batch, seq, d_model)
        return encoded.transpose(0, 1)


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with configurable weights.
    Uses label smoothing for better generalization.
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.loss_weights = loss_weights or {
            'buy': 1.0,
            'sell': 1.0,
            'direction': 1.5,
            'regime': 1.2
        }
        
        self.label_smoothing = label_smoothing
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dict with keys 'buy', 'sell', 'direction', 'regime'
            targets: Dict with target labels
            
        Returns:
            total_loss, loss_dict (for logging)
        """
        losses = {}
        
        # Buy loss
        losses['buy'] = self.ce_loss(predictions['buy'], targets['buy'])
        
        # Sell loss
        losses['sell'] = self.ce_loss(predictions['sell'], targets['sell'])
        
        # Direction loss (3-class)
        losses['direction'] = self.ce_loss(predictions['direction'], targets['direction'])
        
        # Regime loss (6-class)
        losses['regime'] = self.ce_loss(predictions['regime'], targets['regime'])
        
        # Weighted total loss
        total_loss = sum(
            self.loss_weights[task] * loss
            for task, loss in losses.items()
        )
        
        # Convert to float for logging
        loss_dict = {f'loss_{k}': v.item() for k, v in losses.items()}
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict


def create_natron_model(config: Dict, n_features: int) -> NatronTransformer:
    """
    Factory function to create Natron model from config
    
    Args:
        config: Model configuration dict
        n_features: Number of input features
        
    Returns:
        NatronTransformer instance
    """
    model_config = config.get('model', {})
    
    model = NatronTransformer(
        n_features=n_features,
        d_model=model_config.get('d_model', 256),
        nhead=model_config.get('nhead', 8),
        num_encoder_layers=model_config.get('num_encoder_layers', 6),
        dim_feedforward=model_config.get('dim_feedforward', 1024),
        dropout=model_config.get('dropout', 0.1),
        activation=model_config.get('activation', 'gelu'),
        max_seq_length=model_config.get('max_seq_length', 96)
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Natron Transformer Model Created")
    print(f"  ✅ Parameters: {n_params:,}")
    print(f"  ✅ Input features: {n_features}")
    print(f"  ✅ Model dimension: {model_config.get('d_model', 256)}")
    print(f"  ✅ Encoder layers: {model_config.get('num_encoder_layers', 6)}")
    print(f"  ✅ Attention heads: {model_config.get('nhead', 8)}\n")
    
    return model
