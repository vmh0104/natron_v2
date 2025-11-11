"""
Phase 2: Supervised Fine-Tuning - Multi-task learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os


class SupervisedTrainer:
    """Supervised fine-tuning for multi-task learning"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5),
            verbose=True
        )
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.nll_loss = nn.NLLLoss()
    
    def compute_loss(self, outputs: dict, labels: dict) -> dict:
        """
        Compute multi-task loss
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
        
        Returns:
            Dictionary with individual and total losses
        """
        # Buy/Sell losses (binary classification)
        buy_loss = self.bce_loss(outputs['buy_prob'], labels['buy'])
        sell_loss = self.bce_loss(outputs['sell_prob'], labels['sell'])
        
        # Direction loss (3-class classification)
        direction_loss = self.nll_loss(outputs['direction_logits'], labels['direction'])
        
        # Regime loss (6-class classification)
        regime_loss = self.nll_loss(outputs['regime_logits'], labels['regime'])
        
        # Total loss (equal weighting)
        total_loss = buy_loss + sell_loss + direction_loss + regime_loss
        
        return {
            'buy_loss': buy_loss.item(),
            'sell_loss': sell_loss.item(),
            'direction_loss': direction_loss.item(),
            'regime_loss': regime_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train one epoch"""
        self.model.train()
        
        total_losses = {
            'buy_loss': 0,
            'sell_loss': 0,
            'direction_loss': 0,
            'regime_loss': 0,
            'total_loss': 0
        }
        
        for batch in tqdm(dataloader, desc="Supervised Training"):
            sequences = batch['sequence'].to(self.device)
            labels = {
                'buy': batch['buy'].to(self.device),
                'sell': batch['sell'].to(self.device),
                'direction': batch['direction'].to(self.device),
                'regime': batch['regime'].to(self.device)
            }
            
            # Forward pass
            outputs = self.model(sequences)
            
            # Compute loss
            losses = self.compute_loss(outputs, labels)
            total_loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            # Convert back to tensor for backward
            buy_loss = self.bce_loss(outputs['buy_prob'], labels['buy'])
            sell_loss = self.bce_loss(outputs['sell_prob'], labels['sell'])
            direction_loss = self.nll_loss(outputs['direction_logits'], labels['direction'])
            regime_loss = self.nll_loss(outputs['regime_logits'], labels['regime'])
            total_loss_tensor = buy_loss + sell_loss + direction_loss + regime_loss
            
            total_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
            self.optimizer.step()
            
            for key in total_losses:
                total_losses[key] += losses[key]
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate model"""
        self.model.eval()
        
        total_losses = {
            'buy_loss': 0,
            'sell_loss': 0,
            'direction_loss': 0,
            'regime_loss': 0,
            'total_loss': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                sequences = batch['sequence'].to(self.device)
                labels = {
                    'buy': batch['buy'].to(self.device),
                    'sell': batch['sell'].to(self.device),
                    'direction': batch['direction'].to(self.device),
                    'regime': batch['regime'].to(self.device)
                }
                
                # Forward pass
                outputs = self.model(sequences)
                
                # Compute loss
                losses = self.compute_loss(outputs, labels)
                
                for key in total_losses:
                    total_losses[key] += losses[key]
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, save_path: str):
        """Full training loop"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.get('scheduler_patience', 5) * 2
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Buy: {train_metrics['buy_loss']:.4f}, "
                  f"Sell: {train_metrics['sell_loss']:.4f}, "
                  f"Direction: {train_metrics['direction_loss']:.4f}, "
                  f"Regime: {train_metrics['regime_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"  Buy: {val_metrics['buy_loss']:.4f}, "
                  f"Sell: {val_metrics['sell_loss']:.4f}, "
                  f"Direction: {val_metrics['direction_loss']:.4f}, "
                  f"Regime: {val_metrics['regime_loss']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'config': self.config
                }, save_path)
                print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n🎉 Supervised training complete! Best val loss: {best_val_loss:.4f}")
