"""
Phase 2: Supervised Fine-Tuning - Multi-task Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
from model import NatronTransformer


class SupervisedTrainer:
    """
    Phase 2: Supervised fine-tuning for multi-task prediction
    """
    
    def __init__(
        self,
        model: NatronTransformer,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss weights
        self.loss_weights = loss_weights or {
            'buy': 1.0,
            'sell': 1.0,
            'direction': 1.0,
            'regime': 1.0
        }
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        """
        losses = {}
        
        # Buy/Sell losses (binary classification)
        buy_loss = self.bce_loss(predictions['buy'], targets['buy'])
        sell_loss = self.bce_loss(predictions['sell'], targets['sell'])
        
        # Direction loss (3-class classification)
        direction_loss = self.ce_loss(predictions['direction'], targets['direction'])
        
        # Regime loss (6-class classification)
        regime_loss = self.ce_loss(predictions['regime'], targets['regime'])
        
        # Weighted total loss
        total_loss = (
            self.loss_weights['buy'] * buy_loss +
            self.loss_weights['sell'] * sell_loss +
            self.loss_weights['direction'] * direction_loss +
            self.loss_weights['regime'] * regime_loss
        )
        
        losses = {
            'total': total_loss,
            'buy': buy_loss,
            'sell': sell_loss,
            'direction': direction_loss,
            'regime': regime_loss
        }
        
        return losses
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'buy': 0.0,
            'sell': 0.0,
            'direction': 0.0,
            'regime': 0.0
        }
        
        for batch in tqdm(dataloader, desc="Training"):
            sequences = batch['sequence'].to(self.device)
            
            # Targets
            targets = {
                'buy': batch['buy'].to(self.device),
                'sell': batch['sell'].to(self.device),
                'direction': batch['direction'].to(self.device),
                'regime': batch['regime'].to(self.device)
            }
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Compute loss
            losses = self.compute_loss(predictions, targets)
            
            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'buy': 0.0,
            'sell': 0.0,
            'direction': 0.0,
            'regime': 0.0
        }
        
        correct = {
            'buy': 0,
            'sell': 0,
            'direction': 0,
            'regime': 0
        }
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                sequences = batch['sequence'].to(self.device)
                
                # Targets
                targets = {
                    'buy': batch['buy'].to(self.device),
                    'sell': batch['sell'].to(self.device),
                    'direction': batch['direction'].to(self.device),
                    'regime': batch['regime'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                losses = self.compute_loss(predictions, targets)
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += losses[key].item()
                
                # Compute accuracy
                batch_size = sequences.shape[0]
                total += batch_size
                
                # Buy/Sell accuracy (threshold 0.5)
                correct['buy'] += ((predictions['buy'] > 0.5).float() == targets['buy']).sum().item()
                correct['sell'] += ((predictions['sell'] > 0.5).float() == targets['sell']).sum().item()
                
                # Direction/Regime accuracy
                correct['direction'] += (predictions['direction'].argmax(dim=1) == targets['direction']).sum().item()
                correct['regime'] += (predictions['regime'].argmax(dim=1) == targets['regime']).sum().item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        # Compute accuracies
        accuracies = {
            'buy': correct['buy'] / total,
            'sell': correct['sell'] / total,
            'direction': correct['direction'] / total,
            'regime': correct['regime'] / total
        }
        
        return total_losses, accuracies
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Full supervised training loop
        """
        print(f"\n🚀 Starting Phase 2: Supervised Fine-Tuning ({num_epochs} epochs)")
        print("="*60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics, val_accuracies = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['total'])
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['total']:.6f}")
            print(f"  Val Loss:   {val_metrics['total']:.6f}")
            print(f"  Val Accuracies:")
            print(f"    Buy:       {val_accuracies['buy']:.4f}")
            print(f"    Sell:      {val_accuracies['sell']:.4f}")
            print(f"    Direction: {val_accuracies['direction']:.4f}")
            print(f"    Regime:    {val_accuracies['regime']:.4f}")
            
            # Early stopping
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                if save_path:
                    self.save_checkpoint(save_path)
                    print(f"  ✓ Saved checkpoint to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⏹ Early stopping triggered (patience: {early_stopping_patience})")
                    break
        
        print("\n✅ Supervised fine-tuning complete!")
        return self.model
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
