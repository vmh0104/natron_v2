"""
Phase 1: Pretraining - Masked Token Reconstruction + Contrastive Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
from model import NatronTransformer


class MaskedTokenPredictor(nn.Module):
    """
    Head for masked token prediction during pretraining
    """
    def __init__(self, d_model: int, num_features: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


class PretrainingTrainer:
    """
    Phase 1: Pretraining with masked token reconstruction and contrastive learning
    """
    
    def __init__(
        self,
        model: NatronTransformer,
        device: torch.device,
        mask_probability: float = 0.15,
        contrastive_temperature: float = 0.07,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.mask_probability = mask_probability
        self.contrastive_temperature = contrastive_temperature
        
        # Add masked token predictor
        self.masked_predictor = MaskedTokenPredictor(
            model.d_model,
            model.num_features
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.masked_predictor.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = nn.CrossEntropyLoss()
        
    def create_mask(self, batch_size: int, seq_length: int) -> torch.Tensor:
        """
        Create random mask for masked token prediction
        """
        mask = torch.rand(batch_size, seq_length) < self.mask_probability
        return mask.to(self.device)
    
    def masked_token_loss(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked token reconstruction loss
        """
        batch_size, seq_length, num_features = sequences.shape
        
        # Create masked sequences
        masked_sequences = sequences.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(masked_sequences)
        masked_sequences[mask_expanded] = 0  # Mask tokens
        
        # Get encoder output
        x = self.model.input_projection(masked_sequences)
        x = self.model.pos_encoder(x)
        x = self.model.transformer_encoder(x)
        
        # Predict masked tokens
        masked_positions = mask.unsqueeze(-1).expand(-1, -1, self.model.d_model)
        masked_features = x[masked_positions].view(-1, self.model.d_model)
        
        # Predict original features
        predicted_features = self.masked_predictor(masked_features)
        
        # Get original features at masked positions
        original_features = sequences[mask_expanded].view(-1, num_features)
        
        # Reconstruction loss
        loss = self.mse_loss(predicted_features, original_features)
        
        return loss
    
    def contrastive_loss_fn(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss - learn similar representations for similar sequences
        """
        batch_size = sequences.shape[0]
        
        # Get representations
        x = self.model.input_projection(sequences)
        x = self.model.pos_encoder(x)
        x = self.model.transformer_encoder(x)
        
        # Use mean pooling
        representations = x.mean(dim=1)  # (batch_size, d_model)
        
        # Normalize
        representations = torch.nn.functional.normalize(representations, p=2, dim=1)
        
        # Create positive pairs (augmented versions)
        # Simple augmentation: add small noise
        noise = torch.randn_like(representations) * 0.01
        augmented = torch.nn.functional.normalize(representations + noise, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, augmented.t()) / self.contrastive_temperature
        
        # Positive pairs are on diagonal
        labels = torch.arange(batch_size).to(self.device)
        
        loss = self.contrastive_loss(similarity_matrix, labels)
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch
        """
        self.model.train()
        self.masked_predictor.train()
        
        total_loss = 0.0
        total_mask_loss = 0.0
        total_contrastive_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Pretraining"):
            sequences = batch['sequence'].to(self.device)
            batch_size, seq_length, num_features = sequences.shape
            
            self.optimizer.zero_grad()
            
            # Masked token loss
            mask = self.create_mask(batch_size, seq_length)
            mask_loss = self.masked_token_loss(sequences, mask)
            
            # Contrastive loss
            contrastive_loss = self.contrastive_loss_fn(sequences)
            
            # Total loss
            loss = mask_loss + contrastive_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.masked_predictor.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mask_loss += mask_loss.item()
            total_contrastive_loss += contrastive_loss.item()
        
        return {
            'loss': total_loss / len(dataloader),
            'mask_loss': total_mask_loss / len(dataloader),
            'contrastive_loss': total_contrastive_loss / len(dataloader)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        save_path: Optional[str] = None
    ):
        """
        Full pretraining loop
        """
        print(f"\n🚀 Starting Phase 1: Pretraining ({num_epochs} epochs)")
        print("="*60)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Total Loss: {metrics['loss']:.6f}")
            print(f"  Mask Loss:  {metrics['mask_loss']:.6f}")
            print(f"  Contrastive Loss: {metrics['contrastive_loss']:.6f}")
            
            # Save best model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                if save_path:
                    self.save_checkpoint(save_path)
                    print(f"  ✓ Saved checkpoint to {save_path}")
        
        print("\n✅ Pretraining complete!")
        return self.model
    
    def save_checkpoint(self, path: str):
        """Save pretrained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'predictor_state_dict': self.masked_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
