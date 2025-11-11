"""
Natron Phase 2: Supervised Fine-Tuning
Multi-task learning for Buy/Sell/Direction/Regime prediction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report


class MetricsCalculator:
    """Calculate and track training metrics"""
    
    @staticmethod
    def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy"""
        pred_labels = torch.argmax(predictions, dim=1)
        correct = (pred_labels == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def calculate_metrics(predictions: Dict, targets: Dict) -> Dict[str, float]:
        """Calculate metrics for all tasks"""
        metrics = {}
        
        for task in ['buy', 'sell', 'direction', 'regime']:
            pred = predictions[task]
            tgt = targets[task]
            
            # Accuracy
            acc = MetricsCalculator.calculate_accuracy(pred, tgt)
            metrics[f'{task}_acc'] = acc
            
            # F1 score (weighted for multi-class)
            pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
            tgt_labels = tgt.cpu().numpy()
            
            if task in ['buy', 'sell']:
                # Binary classification
                f1 = f1_score(tgt_labels, pred_labels, average='binary', zero_division=0)
            else:
                # Multi-class
                f1 = f1_score(tgt_labels, pred_labels, average='weighted', zero_division=0)
            
            metrics[f'{task}_f1'] = f1
        
        return metrics


class SupervisedTrainer:
    """Handles Phase 2 supervised fine-tuning"""
    
    def __init__(
        self,
        model,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        supervised_config = config.get('training', {}).get('supervised', {})
        self.epochs = supervised_config.get('epochs', 100)
        
        # Loss function
        from ..models.transformer import MultiTaskLoss
        loss_weights = supervised_config.get('loss_weights', {})
        label_smoothing = supervised_config.get('label_smoothing', 0.1)
        self.criterion = MultiTaskLoss(loss_weights, label_smoothing)
        
        # Optimizer
        lr = supervised_config.get('lr', 1e-4)
        weight_decay = supervised_config.get('weight_decay', 1e-5)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        opt_config = config.get('optimization', {})
        scheduler_params = opt_config.get('scheduler_params', {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 5),
            min_lr=scheduler_params.get('min_lr', 1e-6)
        )
        
        # Gradient clipping
        self.gradient_clip = opt_config.get('gradient_clip', 1.0)
        
        # Mixed precision
        self.use_amp = config.get('system', {}).get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_metrics = []
        n_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    predictions = self.model(sequences)
                    
                    # Compute loss
                    loss, loss_dict = self.criterion(predictions, labels)
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                loss, loss_dict = self.criterion(predictions, labels)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = MetricsCalculator.calculate_metrics(predictions, labels)
                all_metrics.append(metrics)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'buy_acc': f'{metrics["buy_acc"]:.3f}',
                'dir_acc': f'{metrics["direction_acc"]:.3f}'
            })
        
        # Aggregate metrics
        avg_loss = total_loss / n_batches
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        return avg_loss, avg_metrics
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_metrics = []
        n_batches = 0
        
        with torch.no_grad():
            for sequences, labels in tqdm(dataloader, desc="Validating"):
                sequences = sequences.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                loss, _ = self.criterion(predictions, labels)
                
                # Calculate metrics
                metrics = MetricsCalculator.calculate_metrics(predictions, labels)
                all_metrics.append(metrics)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        return avg_loss, avg_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str = 'model',
        save_every: int = 10
    ):
        """Full supervised training loop"""
        print("\n" + "="*60)
        print("🎯 PHASE 2: SUPERVISED FINE-TUNING")
        print("="*60)
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Gradient clip: {self.gradient_clip}")
        print(f"Mixed precision: {self.use_amp}")
        print("="*60 + "\n")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.epochs):
            print(f"\n📍 Epoch {epoch+1}/{self.epochs}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track history
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['train_acc'].append(train_metrics['buy_acc'])
            self.metrics_history['val_acc'].append(val_metrics['buy_acc'])
            
            # Print metrics
            print(f"\n  📊 Training Metrics:")
            print(f"     Loss: {train_loss:.4f}")
            print(f"     Buy Acc:       {train_metrics['buy_acc']:.3f} | F1: {train_metrics['buy_f1']:.3f}")
            print(f"     Sell Acc:      {train_metrics['sell_acc']:.3f} | F1: {train_metrics['sell_f1']:.3f}")
            print(f"     Direction Acc: {train_metrics['direction_acc']:.3f} | F1: {train_metrics['direction_f1']:.3f}")
            print(f"     Regime Acc:    {train_metrics['regime_acc']:.3f} | F1: {train_metrics['regime_f1']:.3f}")
            
            print(f"\n  📊 Validation Metrics:")
            print(f"     Loss: {val_loss:.4f}")
            print(f"     Buy Acc:       {val_metrics['buy_acc']:.3f} | F1: {val_metrics['buy_f1']:.3f}")
            print(f"     Sell Acc:      {val_metrics['sell_acc']:.3f} | F1: {val_metrics['sell_f1']:.3f}")
            print(f"     Direction Acc: {val_metrics['direction_acc']:.3f} | F1: {val_metrics['direction_f1']:.3f}")
            print(f"     Regime Acc:    {val_metrics['regime_acc']:.3f} | F1: {val_metrics['regime_f1']:.3f}")
            
            print(f"\n  ⚙️  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = f"{checkpoint_dir}/natron_v2_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, checkpoint_path)
                print(f"  ✅ Saved best model (loss: {val_loss:.4f})")
            
            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f"{checkpoint_dir}/natron_v2_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, checkpoint_path)
                print(f"  💾 Saved checkpoint: epoch_{epoch+1}")
        
        # Save final model
        final_path = f"{checkpoint_dir}/natron_v2.pt"
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.config,
            'metrics_history': self.metrics_history
        }, final_path)
        
        print("\n" + "="*60)
        print("✅ SUPERVISED TRAINING COMPLETE")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to: {final_path}")
        print("="*60 + "\n")
        
        return self.model


def load_pretrained_encoder(model, checkpoint_path: str):
    """Load pretrained encoder weights"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded pretrained encoder from {checkpoint_path}")
    return model
