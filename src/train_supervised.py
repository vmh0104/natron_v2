"""
Phase 2: Supervised Fine-Tuning
Train multi-task model on labeled data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, Tuple
import os

from model import NatronTransformer, MultiTaskLoss


class SupervisedTrainer:
    """
    Supervised training engine for Natron Transformer.
    """
    
    def __init__(
        self,
        model: NatronTransformer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.device = device
        self.config = config
        self.model = model.to(device)
        
        # Loss function
        self.criterion = MultiTaskLoss(
            buy_weight=config['supervised']['loss_weights']['buy'],
            sell_weight=config['supervised']['loss_weights']['sell'],
            direction_weight=config['supervised']['loss_weights']['direction'],
            regime_weight=config['supervised']['loss_weights']['regime']
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['supervised']['learning_rate'],
            weight_decay=config['supervised']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['supervised']['scheduler']['factor'],
            patience=config['supervised']['scheduler']['patience'],
            min_lr=config['supervised']['scheduler']['min_lr'],
            verbose=True
        )
        
        # Mixed precision
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(config['supervised']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = config['supervised']['early_stopping_patience']
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_buy_loss = 0.0
        total_sell_loss = 0.0
        total_direction_loss = 0.0
        total_regime_loss = 0.0
        
        # Accuracy tracking
        buy_correct = 0
        sell_correct = 0
        direction_correct = 0
        regime_correct = 0
        n_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y_cuda = {k: v.to(self.device) for k, v in batch_y.items()}
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch_x)
                    loss, loss_dict = self.criterion(outputs, batch_y_cuda)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_x)
                loss, loss_dict = self.criterion(outputs, batch_y_cuda)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total']
            total_buy_loss += loss_dict['buy']
            total_sell_loss += loss_dict['sell']
            total_direction_loss += loss_dict['direction']
            total_regime_loss += loss_dict['regime']
            
            # Calculate accuracy
            with torch.no_grad():
                buy_pred = (outputs['buy'] > 0.5).float()
                sell_pred = (outputs['sell'] > 0.5).float()
                direction_pred = outputs['direction'].argmax(dim=1)
                regime_pred = outputs['regime'].argmax(dim=1)
                
                buy_correct += (buy_pred == batch_y_cuda['buy']).sum().item()
                sell_correct += (sell_pred == batch_y_cuda['sell']).sum().item()
                direction_correct += (direction_pred == batch_y_cuda['direction']).sum().item()
                regime_correct += (regime_pred == batch_y_cuda['regime']).sum().item()
                n_samples += len(batch_x)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'buy_acc': buy_correct / n_samples,
                'dir_acc': direction_correct / n_samples
            })
        
        n_batches = len(train_loader)
        
        metrics = {
            'loss': total_loss / n_batches,
            'buy_loss': total_buy_loss / n_batches,
            'sell_loss': total_sell_loss / n_batches,
            'direction_loss': total_direction_loss / n_batches,
            'regime_loss': total_regime_loss / n_batches,
            'buy_acc': buy_correct / n_samples,
            'sell_acc': sell_correct / n_samples,
            'direction_acc': direction_correct / n_samples,
            'regime_acc': regime_correct / n_samples
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate"""
        self.model.eval()
        
        total_loss = 0.0
        total_buy_loss = 0.0
        total_sell_loss = 0.0
        total_direction_loss = 0.0
        total_regime_loss = 0.0
        
        buy_correct = 0
        sell_correct = 0
        direction_correct = 0
        regime_correct = 0
        n_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y_cuda = {k: v.to(self.device) for k, v in batch_y.items()}
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch_x)
                        loss, loss_dict = self.criterion(outputs, batch_y_cuda)
                else:
                    outputs = self.model(batch_x)
                    loss, loss_dict = self.criterion(outputs, batch_y_cuda)
                
                # Update metrics
                total_loss += loss_dict['total']
                total_buy_loss += loss_dict['buy']
                total_sell_loss += loss_dict['sell']
                total_direction_loss += loss_dict['direction']
                total_regime_loss += loss_dict['regime']
                
                # Calculate accuracy
                buy_pred = (outputs['buy'] > 0.5).float()
                sell_pred = (outputs['sell'] > 0.5).float()
                direction_pred = outputs['direction'].argmax(dim=1)
                regime_pred = outputs['regime'].argmax(dim=1)
                
                buy_correct += (buy_pred == batch_y_cuda['buy']).sum().item()
                sell_correct += (sell_pred == batch_y_cuda['sell']).sum().item()
                direction_correct += (direction_pred == batch_y_cuda['direction']).sum().item()
                regime_correct += (regime_pred == batch_y_cuda['regime']).sum().item()
                n_samples += len(batch_x)
        
        n_batches = len(val_loader)
        
        metrics = {
            'loss': total_loss / n_batches,
            'buy_loss': total_buy_loss / n_batches,
            'sell_loss': total_sell_loss / n_batches,
            'direction_loss': total_direction_loss / n_batches,
            'regime_loss': total_regime_loss / n_batches,
            'buy_acc': buy_correct / n_samples,
            'sell_acc': sell_correct / n_samples,
            'direction_acc': direction_correct / n_samples,
            'regime_acc': regime_correct / n_samples
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f'supervised_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'supervised_best.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint: {best_path}")
    
    def load_pretrained(self, pretrain_path: str):
        """Load pretrained encoder weights"""
        print(f"📂 Loading pretrained weights from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ Pretrained weights loaded")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full supervised training loop"""
        n_epochs = self.config['supervised']['epochs']
        
        print("\n" + "="*60)
        print("🚀 Phase 2: Supervised Fine-Tuning")
        print("="*60)
        print(f"Epochs: {n_epochs}")
        print(f"Learning rate: {self.config['supervised']['learning_rate']}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        for epoch in range(1, n_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Print metrics
            print(f"\n📊 Epoch {epoch}/{n_epochs}")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Buy: {train_metrics['buy_acc']:.3f}, "
                  f"Sell: {train_metrics['sell_acc']:.3f}, "
                  f"Dir: {train_metrics['direction_acc']:.3f}, "
                  f"Regime: {train_metrics['regime_acc']:.3f}")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Buy: {val_metrics['buy_acc']:.3f}, "
                  f"Sell: {val_metrics['sell_acc']:.3f}, "
                  f"Dir: {val_metrics['direction_acc']:.3f}, "
                  f"Regime: {val_metrics['regime_acc']:.3f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
        
        print("\n✅ Supervised training complete!")
        print(f"   Best validation loss: {self.best_loss:.4f}")
        print(f"   Checkpoints saved to: {self.checkpoint_dir}")


if __name__ == "__main__":
    print("Phase 2: Supervised Training Module")
    print("Run main.py for full training pipeline")
