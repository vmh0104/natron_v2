"""
Natron Transformer Model: Multi-task Transformer for financial trading
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    - Transformer Encoder (learns temporal patterns)
    - Multi-task heads (Buy/Sell, Direction, Regime)
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        sequence_length: int = 96
    ):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length, dropout=dropout)
        
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
        
        # Global pooling (mean over sequence)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Task-specific heads
        # Buy/Sell heads (binary classification)
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()
        )
        
        # Direction head (3-class: Up=1, Down=0, Neutral=2)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 3),
            nn.Softmax(dim=-1)
        )
        
        # Regime head (6-class classification)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 6),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, n_features)
        
        Returns:
            dict with keys: 'buy', 'sell', 'direction', 'regime'
        """
        batch_size, seq_len, n_features = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Global pooling (mean over sequence dimension)
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Task-specific predictions
        buy_prob = self.buy_head(pooled).squeeze(-1)  # (batch_size,)
        sell_prob = self.sell_head(pooled).squeeze(-1)  # (batch_size,)
        direction_probs = self.direction_head(pooled)  # (batch_size, 3)
        regime_probs = self.regime_head(pooled)  # (batch_size, 6)
        
        return {
            'buy': buy_prob,
            'sell': sell_prob,
            'direction': direction_probs,
            'regime': regime_probs
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation (for pretraining)
        
        Args:
            x: (batch_size, sequence_length, n_features)
        
        Returns:
            encoded: (batch_size, sequence_length, d_model)
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        return encoded


class MaskedTokenModel(nn.Module):
    """Masked token reconstruction model for pretraining"""
    
    def __init__(self, encoder: NatronTransformer, mask_prob: float = 0.15):
        super().__init__()
        self.encoder = encoder
        self.mask_prob = mask_prob
        self.reconstruction_head = nn.Linear(encoder.d_model, encoder.input_projection.in_features)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_features)
            mask: (batch_size, seq_len) boolean mask (True = masked)
        
        Returns:
            reconstructed: (batch_size, seq_len, n_features)
        """
        if mask is None:
            # Random masking
            mask = torch.rand(x.shape[:2], device=x.device) < self.mask_prob
        
        # Encode
        encoded = self.encoder.encode(x)
        
        # Reconstruct
        reconstructed = self.reconstruction_head(encoded)
        
        return reconstructed, mask


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss
    
    Args:
        z1, z2: (batch_size, d_model) encoded representations
        temperature: temperature parameter
    
    Returns:
        loss: scalar
    """
    batch_size = z1.shape[0]
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similarity matrix
    similarity_matrix = torch.matmul(z1, z2.T) / temperature
    
    # Positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=z1.device)
    
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss
