"""
Natron Transformer - Phase 1: Unsupervised Pretraining
Masked token reconstruction + Contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Tuple
import wandb

from model import PretrainingModel, count_parameters
from dataset import create_pretraining_dataloader
from feature_engine import load_and_prepare_data


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (batch, dim) - anchor embeddings
            z2: (batch, dim) - positive embeddings
        
        Returns:
            loss: scalar
        """
        batch_size = z1.shape[0]
        
        # Compute similarity matrix
        z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2*batch, 2*batch)
        
        # Create labels: positive pairs are (i, i+batch) and (i+batch, i)
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class PreTrainer:
    """Trainer for Phase 1 unsupervised pretraining"""
    
    def __init__(self, model: PretrainingModel, config, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss functions
        self.reconstruction_criterion = nn.MSELoss()
        self.contrastive_criterion = InfoNCELoss(config.pretrain.temperature)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.pretrain.learning_rate,
            weight_decay=config.pretrain.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Logging
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Weights
        self.recon_weight = config.pretrain.reconstruction_weight
        self.contrast_weight = config.pretrain.contrastive_weight
        
        print(f"🔥 PreTrainer initialized on {device}")
        print(f"   Model parameters: {count_parameters(model):,}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            warmup_steps = self.config.pretrain.warmup_epochs * 1000  # Approximate
            if step < warmup_steps:
                return step / warmup_steps
            return max(0.1, 1.0 - (step - warmup_steps) / 
                       (self.config.pretrain.epochs * 1000))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_contrast_loss = 0
        
        pbar = tqdm(dataloader, desc="Pretraining")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            original = batch['original'].to(self.device)
            masked = batch['masked'].to(self.device)
            mask = batch['mask'].to(self.device)
            positive = batch['positive'].to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    # Reconstruction
                    recon, z_masked = self.model(masked)
                    recon_loss = self.reconstruction_criterion(
                        recon[mask], original[mask]
                    )
                    
                    # Contrastive learning
                    _, z_positive = self.model(positive)
                    contrast_loss = self.contrastive_criterion(z_masked, z_positive)
                    
                    # Combined loss
                    loss = (self.recon_weight * recon_loss + 
                            self.contrast_weight * contrast_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Reconstruction
                recon, z_masked = self.model(masked)
                recon_loss = self.reconstruction_criterion(
                    recon[mask], original[mask]
                )
                
                # Contrastive learning
                _, z_positive = self.model(positive)
                contrast_loss = self.contrastive_criterion(z_masked, z_positive)
                
                # Combined loss
                loss = (self.recon_weight * recon_loss + 
                        self.contrast_weight * contrast_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrast_loss += contrast_loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'contrast': f'{contrast_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Average losses
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'contrast_loss': total_contrast_loss / n_batches
        }
    
    def train(self, dataloader) -> PretrainingModel:
        """Full pretraining loop"""
        print(f"\n🚀 Starting Phase 1: Unsupervised Pretraining")
        print(f"   Epochs: {self.config.pretrain.epochs}")
        print(f"   Mask ratio: {self.config.pretrain.mask_ratio}")
        print(f"   Learning rate: {self.config.pretrain.learning_rate}")
        
        # Create checkpoint directory
        os.makedirs(self.config.pretrain.checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, self.config.pretrain.epochs + 1):
            print(f"\n📍 Epoch {epoch}/{self.config.pretrain.epochs}")
            
            # Train
            metrics = self.train_epoch(dataloader)
            
            # Log
            print(f"   Loss: {metrics['loss']:.4f} "
                  f"(Recon: {metrics['recon_loss']:.4f}, "
                  f"Contrast: {metrics['contrast_loss']:.4f})")
            
            if self.config.wandb_enabled:
                wandb.log({
                    'pretrain/loss': metrics['loss'],
                    'pretrain/recon_loss': metrics['recon_loss'],
                    'pretrain/contrast_loss': metrics['contrast_loss'],
                    'pretrain/epoch': epoch,
                    'pretrain/lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            if epoch % self.config.pretrain.save_every == 0:
                self._save_checkpoint(epoch, metrics['loss'])
            
            # Save best model
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self._save_checkpoint(epoch, metrics['loss'], is_best=True)
        
        print(f"\n✅ Pretraining completed!")
        print(f"   Best loss: {self.best_loss:.4f}")
        
        return self.model
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.config.pretrain.checkpoint_dir, 'best_pretrain.pt')
            print(f"   💾 Saved best checkpoint (loss: {loss:.4f})")
        else:
            path = os.path.join(self.config.pretrain.checkpoint_dir, f'pretrain_epoch_{epoch}.pt')
            print(f"   💾 Saved checkpoint at epoch {epoch}")
        
        torch.save(checkpoint, path)


def run_pretraining(config):
    """Run Phase 1 pretraining"""
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load data
    print("\n📊 Loading data for pretraining...")
    raw_df, features_df = load_and_prepare_data(config.data.csv_path)
    
    # Create dataloader
    dataloader, scaler = create_pretraining_dataloader(features_df, config)
    
    # Save scaler
    from dataset import save_scaler
    scaler_path = os.path.join(config.output_dir, 'pretrain_scaler.pkl')
    os.makedirs(config.output_dir, exist_ok=True)
    save_scaler(scaler, scaler_path)
    
    # Create model
    model = PretrainingModel(config)
    
    # Initialize wandb
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name='pretrain',
            config=config.__dict__
        )
    
    # Train
    trainer = PreTrainer(model, config, config.device)
    trained_model = trainer.train(dataloader)
    
    if config.wandb_enabled:
        wandb.finish()
    
    return trained_model


if __name__ == "__main__":
    from config import load_config
    import sys
    
    # Load config
    config = load_config()
    
    # Override with command line args
    if len(sys.argv) > 1:
        config.data.csv_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        config.pretrain.epochs = int(sys.argv[2])
    
    # Run pretraining
    run_pretraining(config)
