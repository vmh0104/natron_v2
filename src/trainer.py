"""
Training scripts for Natron Transformer
Phase 1: Pretraining (Unsupervised)
Phase 2: Supervised Fine-tuning
Phase 3: Reinforcement Learning (Optional)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import yaml
from typing import Dict, Optional
import json

from src.model import NatronTransformer, MaskedLanguageModelHead, ContrastiveHead
from src.training_utils import (
    PretrainingLoss, MultiTaskLoss,
    create_optimizer, create_scheduler
)


class PretrainingTrainer:
    """Phase 1: Pretraining with masked token reconstruction + contrastive learning"""
    
    def __init__(self, model: NatronTransformer, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Add pretraining heads
        self.mlm_head = MaskedLanguageModelHead(
            config['model']['d_model'],
            config['features']['num_features']
        ).to(self.device)
        
        self.contrastive_head = ContrastiveHead(
            config['model']['d_model']
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = PretrainingLoss(
            reconstruction_weight=config['pretraining']['reconstruction_weight'],
            contrastive_weight=config['pretraining']['contrastive_weight'],
            temperature=config['pretraining']['contrastive_temperature']
        )
        
        self.optimizer = create_optimizer(
            list(self.model.parameters()) + list(self.mlm_head.parameters()) + list(self.contrastive_head.parameters()),
            config['training']['learning_rate'],
            config['training']['weight_decay'],
            config['training']['optimizer']
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            config['training']['scheduler'],
            config['training']['scheduler_patience'],
            config['training']['scheduler_factor']
        )
    
    def mask_tokens(self, x: torch.Tensor, mask_prob: float = 0.15) -> tuple:
        """Randomly mask tokens for pretraining"""
        batch_size, seq_len, feature_dim = x.shape
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_prob
        masked_x = x.clone()
        masked_x[mask] = 0.0  # Simple zero masking
        return masked_x, mask
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        self.mlm_head.train()
        self.contrastive_head.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_contrast_loss = 0.0
        
        for batch_idx, (x, _) in enumerate(tqdm(dataloader, desc="Pretraining")):
            x = x.to(self.device)
            
            # Mask tokens
            masked_x, mask = self.mask_tokens(x, self.config['pretraining']['mask_probability'])
            
            # Forward pass
            embeddings = self.model.encode(masked_x)
            
            # Reconstruction
            reconstructed = self.mlm_head(embeddings, mask)
            
            # Contrastive
            contrastive_repr = self.contrastive_head(embeddings)
            
            # Loss
            loss_dict = self.criterion(reconstructed, x, mask, embeddings)
            loss = loss_dict['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_contrast_loss += loss_dict['contrastive_loss'].item()
        
        return {
            'loss': total_loss / len(dataloader),
            'reconstruction_loss': total_recon_loss / len(dataloader),
            'contrastive_loss': total_contrast_loss / len(dataloader)
        }
    
    def train(self, train_loader: DataLoader, num_epochs: int, save_dir: str):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics['loss'])
            else:
                self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {metrics['loss']:.4f} "
                  f"(Recon: {metrics['reconstruction_loss']:.4f}, "
                  f"Contrast: {metrics['contrastive_loss']:.4f})")
            
            # Save checkpoint
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': metrics['loss']
                }, os.path.join(save_dir, 'best_pretrain.pt'))


class SupervisedTrainer:
    """Phase 2: Supervised fine-tuning"""
    
    def __init__(self, model: NatronTransformer, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = MultiTaskLoss(
            buy_weight=config['supervised']['buy_weight'],
            sell_weight=config['supervised']['sell_weight'],
            direction_weight=config['supervised']['direction_weight'],
            regime_weight=config['supervised']['regime_weight'],
            use_class_weights=config['supervised']['class_weights']
        )
        
        self.optimizer = create_optimizer(
            self.model.parameters(),
            config['training']['learning_rate'],
            config['training']['weight_decay'],
            config['training']['optimizer']
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            config['training']['scheduler'],
            config['training']['scheduler_patience'],
            config['training']['scheduler_factor']
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        metrics_dict = {'buy_loss': 0.0, 'sell_loss': 0.0, 
                       'direction_loss': 0.0, 'regime_loss': 0.0}
        
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Supervised Training")):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            predictions = self.model(x)
            
            # Loss
            loss_dict = self.criterion(predictions, y)
            loss = loss_dict['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for key in metrics_dict:
                if key in loss_dict:
                    metrics_dict[key] += loss_dict[key].item()
        
        metrics_dict['total_loss'] = total_loss / len(dataloader)
        for key in metrics_dict:
            if key != 'total_loss':
                metrics_dict[key] /= len(dataloader)
        
        return metrics_dict
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate"""
        self.model.eval()
        
        total_loss = 0.0
        metrics_dict = {'buy_loss': 0.0, 'sell_loss': 0.0,
                       'direction_loss': 0.0, 'regime_loss': 0.0}
        
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Validation"):
                x = x.to(self.device)
                y = y.to(self.device)
                
                predictions = self.model(x)
                loss_dict = self.criterion(predictions, y)
                
                total_loss += loss_dict['total_loss'].item()
                for key in metrics_dict:
                    if key in loss_dict:
                        metrics_dict[key] += loss_dict[key].item()
        
        metrics_dict['total_loss'] = total_loss / len(dataloader)
        for key in metrics_dict:
            if key != 'total_loss':
                metrics_dict[key] /= len(dataloader)
        
        return metrics_dict
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, save_dir: str):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['total_loss'])
            else:
                self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Save checkpoint
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['total_loss']
                }, os.path.join(save_dir, 'best_supervised.pt'))
                
                # Save final model (state_dict only for easier loading)
                model_path = os.path.join(save_dir, 'natron_v2.pt')
                torch.save(self.model.state_dict(), model_path)
                
                # Also save full checkpoint
                checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['total_loss'],
                    'config': self.config
                }, checkpoint_path)
