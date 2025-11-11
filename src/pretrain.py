"""
Phase 1: Pretraining
Unsupervised pretraining with masked reconstruction and contrastive learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict
import os

from model import NatronTransformer, NatronPretrainer, info_nce_loss, masked_reconstruction_loss


class PretrainEngine:
    """
    Pretraining engine for Natron Transformer.
    """
    
    def __init__(
        self,
        model: NatronTransformer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        
        # Create pretrainer
        self.pretrainer = NatronPretrainer(
            encoder=model,
            mask_ratio=config['pretrain']['mask_ratio'],
            temperature=config['pretrain']['contrastive_temperature']
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.pretrainer.parameters(),
            lr=config['pretrain']['learning_rate'],
            weight_decay=config['pretrain']['weight_decay']
        )
        
        # Loss weights
        self.reconstruction_weight = config['pretrain']['reconstruction_weight']
        self.contrastive_weight = config['pretrain']['contrastive_weight']
        
        # Mixed precision
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(config['pretrain']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.best_loss = float('inf')
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.pretrainer.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_contrast_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_x, _ in pbar:
            batch_x = batch_x.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.pretrainer(batch_x)
                    
                    # Reconstruction loss
                    recon_loss = masked_reconstruction_loss(
                        outputs['reconstructed'],
                        outputs['original'],
                        outputs['mask']
                    )
                    
                    # Contrastive loss
                    contrast_loss = info_nce_loss(
                        outputs['z1'],
                        outputs['z2'],
                        temperature=self.pretrainer.temperature
                    )
                    
                    # Combined loss
                    loss = (
                        self.reconstruction_weight * recon_loss +
                        self.contrastive_weight * contrast_loss
                    )
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.pretrainer(batch_x)
                
                recon_loss = masked_reconstruction_loss(
                    outputs['reconstructed'],
                    outputs['original'],
                    outputs['mask']
                )
                
                contrast_loss = info_nce_loss(
                    outputs['z1'],
                    outputs['z2'],
                    temperature=self.pretrainer.temperature
                )
                
                loss = (
                    self.reconstruction_weight * recon_loss +
                    self.contrastive_weight * contrast_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrast_loss += contrast_loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'contrast': contrast_loss.item()
            })
        
        metrics = {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'contrast_loss': total_contrast_loss / n_batches
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate"""
        self.pretrainer.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_contrast_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.pretrainer(batch_x)
                        
                        recon_loss = masked_reconstruction_loss(
                            outputs['reconstructed'],
                            outputs['original'],
                            outputs['mask']
                        )
                        
                        contrast_loss = info_nce_loss(
                            outputs['z1'],
                            outputs['z2'],
                            temperature=self.pretrainer.temperature
                        )
                        
                        loss = (
                            self.reconstruction_weight * recon_loss +
                            self.contrastive_weight * contrast_loss
                        )
                else:
                    outputs = self.pretrainer(batch_x)
                    
                    recon_loss = masked_reconstruction_loss(
                        outputs['reconstructed'],
                        outputs['original'],
                        outputs['mask']
                    )
                    
                    contrast_loss = info_nce_loss(
                        outputs['z1'],
                        outputs['z2'],
                        temperature=self.pretrainer.temperature
                    )
                    
                    loss = (
                        self.reconstruction_weight * recon_loss +
                        self.contrastive_weight * contrast_loss
                    )
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_contrast_loss += contrast_loss.item()
                n_batches += 1
        
        metrics = {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'contrast_loss': total_contrast_loss / n_batches
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.pretrainer.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f'pretrain_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'pretrain_best.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint: {best_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full pretraining loop"""
        n_epochs = self.config['pretrain']['epochs']
        
        print("\n" + "="*60)
        print("🚀 Phase 1: Pretraining")
        print("="*60)
        print(f"Epochs: {n_epochs}")
        print(f"Mask ratio: {self.config['pretrain']['mask_ratio']}")
        print(f"Reconstruction weight: {self.reconstruction_weight}")
        print(f"Contrastive weight: {self.contrastive_weight}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Print metrics
            print(f"\n📊 Epoch {epoch}/{n_epochs}")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"Contrast: {train_metrics['contrast_loss']:.4f}")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"Contrast: {val_metrics['contrast_loss']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['loss']
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
        
        print("\n✅ Pretraining complete!")
        print(f"   Best validation loss: {self.best_loss:.4f}")
        print(f"   Checkpoints saved to: {self.checkpoint_dir}")


def pretrain_from_config(config_path: str):
    """
    Run pretraining from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"🖥️  Using device: {device}")
    
    # Set seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # Load data (placeholder - actual data loading happens in main.py)
    print("⚠️  Note: This script expects preprocessed data.")
    print("   Run main.py for full end-to-end pipeline.")
    
    return config, device


if __name__ == "__main__":
    print("Phase 1: Pretraining Module")
    print("Run main.py for full training pipeline")
