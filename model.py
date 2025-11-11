"""
Natron Transformer - Model Architecture
Multi-Task Transformer for Financial Trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-Task Transformer for Financial Trading
    
    Architecture:
    - Input projection layer
    - Positional encoding
    - Transformer encoder
    - Multi-task prediction heads:
        * Buy classifier (binary)
        * Sell classifier (binary)
        * Direction classifier (3-class)
        * Regime classifier (6-class)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.model.input_dim, config.model.d_model)
        
        # Positional encoding
        if config.model.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                config.model.d_model,
                config.model.max_seq_length,
                config.model.dropout
            )
        else:
            self.pos_encoder = nn.Dropout(config.model.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.num_encoder_layers
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Task-specific heads
        hidden_dim = config.model.d_model
        
        # Buy head
        self.buy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        # Sell head
        self.sell_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Direction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_dim // 2, config.model.direction_classes)
        )
        
        # Regime head
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_dim // 2, config.model.regime_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_dim)
            return_embeddings: Whether to return encoder embeddings
        
        Returns:
            Dictionary with logits for each task
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global pooling
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Multi-task predictions
        outputs = {
            'buy': self.buy_head(pooled),
            'sell': self.sell_head(pooled),
            'direction': self.direction_head(pooled),
            'regime': self.regime_head(pooled)
        }
        
        if return_embeddings:
            outputs['embeddings'] = pooled
            outputs['sequence_embeddings'] = encoded
        
        return outputs
    
    def get_encoder(self) -> nn.Module:
        """Return the encoder for pretraining"""
        return nn.Sequential(
            self.input_projection,
            self.pos_encoder,
            self.transformer_encoder
        )


class PretrainingModel(nn.Module):
    """
    Model for Phase 1 unsupervised pretraining
    - Masked token reconstruction
    - Contrastive learning (InfoNCE)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder (shared with main model)
        self.input_projection = nn.Linear(config.model.input_dim, config.model.d_model)
        
        if config.model.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                config.model.d_model,
                config.model.max_seq_length,
                config.model.dropout
            )
        else:
            self.pos_encoder = nn.Dropout(config.model.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            activation=config.model.activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.num_encoder_layers
        )
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.model.d_model, config.model.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.dim_feedforward, config.model.input_dim)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.model.d_model, config.model.d_model),
            nn.GELU(),
            nn.Linear(config.model.d_model, config.model.d_model // 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pretraining
        
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            reconstructed: (batch, seq_len, input_dim)
            projected: (batch, d_model//2) for contrastive learning
        """
        # Encode
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.transformer_encoder(x_pos)
        
        # Reconstruction
        reconstructed = self.reconstruction_head(encoded)
        
        # Projection for contrastive learning
        pooled = encoded.mean(dim=1)
        projected = self.projection_head(pooled)
        projected = F.normalize(projected, dim=-1)
        
        return reconstructed, projected
    
    def transfer_weights_to(self, model: NatronTransformer):
        """Transfer pretrained weights to supervised model"""
        model.input_projection.load_state_dict(self.input_projection.state_dict())
        model.transformer_encoder.load_state_dict(self.transformer_encoder.state_dict())
        if hasattr(model, 'pos_encoder') and hasattr(self, 'pos_encoder'):
            if hasattr(model.pos_encoder, 'pe'):
                model.pos_encoder.pe = self.pos_encoder.pe
        print("✅ Transferred pretrained weights to supervised model")


def create_model(config, pretrained_path: Optional[str] = None) -> NatronTransformer:
    """
    Create NatronTransformer model
    
    Args:
        config: Configuration
        pretrained_path: Path to pretrained checkpoint (optional)
    
    Returns:
        model: NatronTransformer instance
    """
    model = NatronTransformer(config)
    
    if pretrained_path:
        print(f"📂 Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            # Load from pretraining checkpoint
            pretrain_model = PretrainingModel(config)
            pretrain_model.load_state_dict(checkpoint['model_state_dict'])
            pretrain_model.transfer_weights_to(model)
        else:
            # Load full model checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded full model checkpoint")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    from config import load_config
    
    config = load_config()
    
    # Test supervised model
    print("🧠 Testing NatronTransformer...")
    model = create_model(config)
    
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 96
    input_dim = 100
    
    x = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(x)
    
    print(f"\n✅ Output shapes:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Test pretraining model
    print(f"\n🧠 Testing PretrainingModel...")
    pretrain_model = PretrainingModel(config)
    
    print(f"  Total parameters: {count_parameters(pretrain_model):,}")
    
    reconstructed, projected = pretrain_model(x)
    print(f"\n✅ Pretraining output shapes:")
    print(f"  Reconstructed: {reconstructed.shape}")
    print(f"  Projected: {projected.shape}")
