"""
Natron Phase 1: Pretraining (Unsupervised)
Learns latent market structure through:
1. Masked token reconstruction (like BERT)
2. Contrastive learning (InfoNCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
import os


class MaskedReconstructionLoss(nn.Module):
    """Reconstruct masked tokens"""
    
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features)
        )
    
    def forward(
        self,
        encoded: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            encoded: (batch, seq, d_model)
            original: (batch, seq, n_features)
            mask: (batch, seq) - 1 where masked, 0 otherwise
            
        Returns:
            Reconstruction loss (MSE on masked positions)
        """
        # Decode to original feature space
        reconstructed = self.decoder(encoded)
        
        # Compute loss only on masked positions
        mask_expanded = mask.unsqueeze(-1).expand_as(original)
        loss = F.mse_loss(
            reconstructed[mask_expanded == 1],
            original[mask_expanded == 1]
        )
        
        return loss


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for learning representations"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, d_model)
            
        Returns:
            Contrastive loss
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create positive pairs (adjacent samples in batch)
        labels = torch.arange(batch_size, device=features.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class PretrainModel(nn.Module):
    """Wrapper for pretraining tasks"""
    
    def __init__(self, encoder, n_features: int, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.reconstruction_head = MaskedReconstructionLoss(d_model, n_features)
        self.contrastive_head = ContrastiveLoss()
    
    def forward(
        self,
        x: torch.Tensor,
        x_masked: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Original sequence (batch, seq, n_features)
            x_masked: Masked sequence
            mask: Mask indicator (batch, seq)
            
        Returns:
            reconstruction_loss, contrastive_loss
        """
        # Get encoded features
        encoded = self.encoder.get_encoder_features(x_masked)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_head(encoded, x, mask)
        
        # Contrastive loss (on aggregated features)
        # Use attention pooling
        attention_weights = F.softmax(
            torch.mean(encoded, dim=-1, keepdim=True),
            dim=1
        )
        aggregated = (encoded * attention_weights).sum(dim=1)
        contrast_loss = self.contrastive_head(aggregated)
        
        return recon_loss, contrast_loss


def create_random_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.15,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create random mask for sequences
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_ratio: Fraction of tokens to mask
        device: Device
        
    Returns:
        Boolean mask (batch, seq)
    """
    mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
    return mask


def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply mask to sequence by replacing with zeros
    
    Args:
        x: (batch, seq, features)
        mask: (batch, seq)
        
    Returns:
        Masked sequence
    """
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked


class PretrainTrainer:
    """Handles Phase 1 pretraining"""
    
    def __init__(
        self,
        model,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        pretrain_config = config.get('training', {}).get('pretrain', {})
        self.epochs = pretrain_config.get('epochs', 50)
        self.mask_ratio = pretrain_config.get('mask_ratio', 0.15)
        self.contrastive_weight = pretrain_config.get('contrastive_weight', 0.3)
        self.reconstruction_weight = pretrain_config.get('reconstruction_weight', 0.7)
        
        # Get n_features from model
        self.n_features = model.n_features
        self.d_model = model.d_model
        
        # Create pretrain wrapper
        self.pretrain_model = PretrainModel(
            model,
            self.n_features,
            self.d_model
        ).to(device)
        
        # Optimizer
        lr = pretrain_config.get('lr', 1e-4)
        weight_decay = pretrain_config.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.AdamW(
            self.pretrain_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Mixed precision
        self.use_amp = config.get('system', {}).get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Logging
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.pretrain_model.train()
        
        total_recon_loss = 0
        total_contrast_loss = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc="Pretraining")
        
        for batch_idx, (sequences, _) in enumerate(pbar):
            sequences = sequences.to(self.device)
            batch_size, seq_len, n_features = sequences.shape
            
            # Create random mask
            mask = create_random_mask(batch_size, seq_len, self.mask_ratio, self.device)
            
            # Apply mask
            sequences_masked = apply_mask(sequences, mask)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    recon_loss, contrast_loss = self.pretrain_model(
                        sequences,
                        sequences_masked,
                        mask
                    )
                    
                    # Combined loss
                    loss = (
                        self.reconstruction_weight * recon_loss +
                        self.contrastive_weight * contrast_loss
                    )
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon_loss, contrast_loss = self.pretrain_model(
                    sequences,
                    sequences_masked,
                    mask
                )
                
                loss = (
                    self.reconstruction_weight * recon_loss +
                    self.contrastive_weight * contrast_loss
                )
                
                loss.backward()
                self.optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_contrast_loss += contrast_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'recon': f'{recon_loss.item():.4f}',
                'contrast': f'{contrast_loss.item():.4f}',
                'total': f'{loss.item():.4f}'
            })
        
        avg_recon = total_recon_loss / n_batches
        avg_contrast = total_contrast_loss / n_batches
        avg_total = (
            self.reconstruction_weight * avg_recon +
            self.contrastive_weight * avg_contrast
        )
        
        return {
            'recon_loss': avg_recon,
            'contrast_loss': avg_contrast,
            'total_loss': avg_total
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate pretraining"""
        self.pretrain_model.eval()
        
        total_recon_loss = 0
        total_contrast_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for sequences, _ in dataloader:
                sequences = sequences.to(self.device)
                batch_size, seq_len, n_features = sequences.shape
                
                # Create mask
                mask = create_random_mask(batch_size, seq_len, self.mask_ratio, self.device)
                sequences_masked = apply_mask(sequences, mask)
                
                # Forward
                recon_loss, contrast_loss = self.pretrain_model(
                    sequences,
                    sequences_masked,
                    mask
                )
                
                total_recon_loss += recon_loss.item()
                total_contrast_loss += contrast_loss.item()
                n_batches += 1
        
        avg_recon = total_recon_loss / n_batches
        avg_contrast = total_contrast_loss / n_batches
        avg_total = (
            self.reconstruction_weight * avg_recon +
            self.contrastive_weight * avg_contrast
        )
        
        return {
            'recon_loss': avg_recon,
            'contrast_loss': avg_contrast,
            'total_loss': avg_total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = 'model'
    ):
        """Full pretraining loop"""
        print("\n" + "="*60)
        print("🔥 PHASE 1: PRETRAINING (Unsupervised)")
        print("="*60)
        print(f"Epochs: {self.epochs}")
        print(f"Mask ratio: {self.mask_ratio}")
        print(f"Reconstruction weight: {self.reconstruction_weight}")
        print(f"Contrastive weight: {self.contrastive_weight}")
        print("="*60 + "\n")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.epochs):
            print(f"\n📍 Epoch {epoch+1}/{self.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['total_loss'])
            
            # Print metrics
            print(f"\n  Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Recon: {train_metrics['recon_loss']:.4f}, "
                  f"Contrast: {train_metrics['contrast_loss']:.4f})")
            print(f"  Val Loss:   {val_metrics['total_loss']:.4f} "
                  f"(Recon: {val_metrics['recon_loss']:.4f}, "
                  f"Contrast: {val_metrics['contrast_loss']:.4f})")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_loss:
                self.best_loss = val_metrics['total_loss']
                checkpoint_path = f"{checkpoint_dir}/pretrained_encoder.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_loss,
                    'config': self.config
                }, checkpoint_path)
                print(f"  ✅ Saved best model to {checkpoint_path}")
        
        print("\n" + "="*60)
        print("✅ PRETRAINING COMPLETE")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print("="*60 + "\n")
        
        return self.model
