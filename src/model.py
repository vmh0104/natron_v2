"""
Natron Transformer - Multi-Task Financial Trading Model
Architecture: Transformer Encoder + Multiple Prediction Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-Task Transformer for Financial Trading.
    
    Architecture:
        Input (96, num_features) 
        → Linear projection to d_model
        → Positional Encoding
        → Transformer Encoder (6 layers)
        → [CLS] token aggregation
        → Multiple task heads:
            - Buy head (sigmoid)
            - Sell head (sigmoid)
            - Direction head (softmax 3-class)
            - Regime head (softmax 6-class)
    """
    
    def __init__(self,
                 num_features: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 96):
        """
        Args:
            num_features: Number of input features per timestep
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # [CLS] token for sequence aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Multi-task prediction heads
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3 classes: Down, Up, Neutral
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 6)  # 6 regimes
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len, num_features)
            
        Returns:
            Dictionary with predictions:
                - buy_prob: (batch_size, 1)
                - sell_prob: (batch_size, 1)
                - direction_logits: (batch_size, 3)
                - regime_logits: (batch_size, 6)
        """
        batch_size = x.size(0)
        
        # Project input to d_model
        x = self.input_projection(x)  # (B, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, seq_len+1, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, seq_len+1, d_model)
        
        # Extract CLS token representation
        cls_output = x[:, 0, :]  # (B, d_model)
        
        # Multi-task predictions
        buy_prob = self.buy_head(cls_output)  # (B, 1)
        sell_prob = self.sell_head(cls_output)  # (B, 1)
        direction_logits = self.direction_head(cls_output)  # (B, 3)
        regime_logits = self.regime_head(cls_output)  # (B, 6)
        
        return {
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'direction_logits': direction_logits,
            'regime_logits': regime_logits
        }
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get transformer embeddings (for analysis or RL).
        
        Args:
            x: (batch_size, seq_len, num_features)
            
        Returns:
            CLS token embeddings: (batch_size, d_model)
        """
        batch_size = x.size(0)
        
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer_encoder(x)
        
        return x[:, 0, :]  # CLS token


class PretrainModel(nn.Module):
    """
    Pretraining model for Phase 1: Masked Token Reconstruction + Contrastive Learning
    """
    
    def __init__(self, base_model: NatronTransformer, temperature: float = 0.07):
        """
        Args:
            base_model: NatronTransformer instance
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.encoder = base_model
        self.temperature = temperature
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(base_model.d_model, base_model.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(base_model.d_model * 2, base_model.num_features)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.d_model, base_model.d_model),
            nn.GELU(),
            nn.Linear(base_model.d_model, 128)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pretraining.
        
        Args:
            x: (batch_size, seq_len, num_features)
            mask: (batch_size, seq_len) boolean mask
            
        Returns:
            reconstructed: (batch_size, seq_len, num_features)
            embeddings: (batch_size, d_model)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Get transformer encoding
        x_proj = self.encoder.input_projection(x)
        x_enc = self.encoder.pos_encoder(x_proj)
        
        cls_tokens = self.encoder.cls_token.expand(batch_size, -1, -1)
        x_enc = torch.cat([cls_tokens, x_enc], dim=1)
        
        encoded = self.encoder.transformer_encoder(x_enc)
        
        # Separate CLS and sequence tokens
        cls_output = encoded[:, 0, :]
        seq_output = encoded[:, 1:, :]
        
        # Reconstruct masked tokens
        reconstructed = self.reconstruction_head(seq_output)
        
        # Get contrastive embeddings
        embeddings = self.projection_head(cls_output)
        
        return reconstructed, embeddings
    
    def contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss.
        
        Args:
            embeddings: (batch_size, embedding_dim)
            
        Returns:
            loss: scalar
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive pairs (adjacent sequences are considered positive)
        batch_size = embeddings.size(0)
        labels = torch.arange(batch_size, device=embeddings.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


def create_model(config: Dict, num_features: int, pretrain: bool = False) -> nn.Module:
    """
    Factory function to create model.
    
    Args:
        config: Configuration dictionary
        num_features: Number of input features
        pretrain: If True, return PretrainModel
        
    Returns:
        Model instance
    """
    base_model = NatronTransformer(
        num_features=num_features,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        max_seq_length=config['model']['max_seq_length']
    )
    
    if pretrain:
        model = PretrainModel(
            base_model,
            temperature=config['pretrain']['temperature']
        )
    else:
        model = base_model
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🧠 Model Architecture:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing NatronTransformer...")
    
    batch_size = 4
    seq_len = 96
    num_features = 100
    
    # Create model
    model = NatronTransformer(
        num_features=num_features,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, num_features)
    outputs = model(x)
    
    print(f"\n✅ Model test complete!")
    print(f"Buy prob shape: {outputs['buy_prob'].shape}")
    print(f"Sell prob shape: {outputs['sell_prob'].shape}")
    print(f"Direction logits shape: {outputs['direction_logits'].shape}")
    print(f"Regime logits shape: {outputs['regime_logits'].shape}")
    
    # Test pretrain model
    print("\n\nTesting PretrainModel...")
    pretrain_model = PretrainModel(model)
    
    mask = torch.rand(batch_size, seq_len) < 0.15
    reconstructed, embeddings = pretrain_model(x, mask)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test contrastive loss
    loss = pretrain_model.contrastive_loss(embeddings)
    print(f"Contrastive loss: {loss.item():.4f}")
