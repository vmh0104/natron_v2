"""
Natron Transformer - Phase 2: Supervised Fine-Tuning
Multi-task learning: Buy, Sell, Direction, Regime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import wandb

from model import NatronTransformer, create_model, count_parameters
from dataset import create_dataloaders, save_scaler
from feature_engine import load_and_prepare_data
from label_generator_v2 import create_labels


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes)
            targets: (batch,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self, config, class_weights: Optional[Dict] = None):
        super().__init__()
        self.config = config
        
        # Task weights
        self.buy_weight = config.train.buy_weight
        self.sell_weight = config.train.sell_weight
        self.direction_weight = config.train.direction_weight
        self.regime_weight = config.train.regime_weight
        
        # Loss functions
        if config.train.focal_loss:
            self.buy_criterion = FocalLoss(config.train.focal_alpha, config.train.focal_gamma)
            self.sell_criterion = FocalLoss(config.train.focal_alpha, config.train.focal_gamma)
            self.direction_criterion = FocalLoss(config.train.focal_alpha, config.train.focal_gamma)
            self.regime_criterion = FocalLoss(config.train.focal_alpha, config.train.focal_gamma)
        else:
            # Use class weights if provided
            buy_weights = torch.tensor(class_weights['buy']) if class_weights else None
            sell_weights = torch.tensor(class_weights['sell']) if class_weights else None
            dir_weights = torch.tensor(class_weights['direction']) if class_weights else None
            reg_weights = torch.tensor(class_weights['regime']) if class_weights else None
            
            self.buy_criterion = nn.CrossEntropyLoss(weight=buy_weights)
            self.sell_criterion = nn.CrossEntropyLoss(weight=sell_weights)
            self.direction_criterion = nn.CrossEntropyLoss(weight=dir_weights)
            self.regime_criterion = nn.CrossEntropyLoss(weight=reg_weights)
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss
        
        Returns:
            total_loss, loss_dict
        """
        buy_loss = self.buy_criterion(outputs['buy'], targets['buy'])
        sell_loss = self.sell_criterion(outputs['sell'], targets['sell'])
        direction_loss = self.direction_criterion(outputs['direction'], targets['direction'])
        regime_loss = self.regime_criterion(outputs['regime'], targets['regime'])
        
        total_loss = (self.buy_weight * buy_loss +
                      self.sell_weight * sell_loss +
                      self.direction_weight * direction_loss +
                      self.regime_weight * regime_loss)
        
        loss_dict = {
            'buy_loss': buy_loss.item(),
            'sell_loss': sell_loss.item(),
            'direction_loss': direction_loss.item(),
            'regime_loss': regime_loss.item()
        }
        
        return total_loss, loss_dict


class Trainer:
    """Trainer for Phase 2 supervised fine-tuning"""
    
    def __init__(self, model: NatronTransformer, config, 
                 class_weights: Optional[Dict] = None, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = MultiTaskLoss(config, class_weights).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        print(f"🔥 Trainer initialized on {device}")
        print(f"   Model parameters: {count_parameters(model):,}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.train.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.train.scheduler_factor,
                patience=self.config.train.scheduler_patience,
                verbose=True
            )
        elif self.config.train.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.train.epochs,
                eta_min=1e-6
            )
        else:
            return None
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        task_losses = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        
        all_preds = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        all_targets = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    loss, loss_dict = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.train.gradient_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.gradient_clip
                )
                
                self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            for task in task_losses.keys():
                task_losses[task] += loss_dict[f'{task}_loss']
            
            # Collect predictions
            for task in all_preds.keys():
                preds = outputs[task].argmax(dim=1).cpu().numpy()
                all_preds[task].extend(preds)
                all_targets[task].extend(targets[task].cpu().numpy())
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate metrics
        n_batches = len(dataloader)
        metrics = {
            'loss': total_loss / n_batches,
            'buy_loss': task_losses['buy'] / n_batches,
            'sell_loss': task_losses['sell'] / n_batches,
            'direction_loss': task_losses['direction'] / n_batches,
            'regime_loss': task_losses['regime'] / n_batches
        }
        
        # Calculate accuracies
        for task in all_preds.keys():
            acc = accuracy_score(all_targets[task], all_preds[task])
            metrics[f'{task}_acc'] = acc
        
        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        """Validation"""
        self.model.eval()
        
        total_loss = 0
        task_losses = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        
        all_preds = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        all_targets = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            outputs = self.model(inputs)
            loss, loss_dict = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            for task in task_losses.keys():
                task_losses[task] += loss_dict[f'{task}_loss']
            
            # Collect predictions
            for task in all_preds.keys():
                preds = outputs[task].argmax(dim=1).cpu().numpy()
                all_preds[task].extend(preds)
                all_targets[task].extend(targets[task].cpu().numpy())
        
        # Calculate metrics
        n_batches = len(dataloader)
        metrics = {
            'loss': total_loss / n_batches,
            'buy_loss': task_losses['buy'] / n_batches,
            'sell_loss': task_losses['sell'] / n_batches,
            'direction_loss': task_losses['direction'] / n_batches,
            'regime_loss': task_losses['regime'] / n_batches
        }
        
        # Calculate accuracies and detailed metrics
        for task in all_preds.keys():
            acc = accuracy_score(all_targets[task], all_preds[task])
            metrics[f'{task}_acc'] = acc
            
            # Precision, recall, F1
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_targets[task], all_preds[task], average='weighted', zero_division=0
            )
            metrics[f'{task}_precision'] = prec
            metrics[f'{task}_recall'] = rec
            metrics[f'{task}_f1'] = f1
        
        return metrics
    
    def train(self, train_loader, val_loader) -> NatronTransformer:
        """Full training loop"""
        print(f"\n🚀 Starting Phase 2: Supervised Fine-Tuning")
        print(f"   Epochs: {self.config.train.epochs}")
        print(f"   Learning rate: {self.config.train.learning_rate}")
        
        # Create checkpoint directory
        os.makedirs(self.config.train.checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, self.config.train.epochs + 1):
            print(f"\n📍 Epoch {epoch}/{self.config.train.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if self.config.train.save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics['loss'], is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.train.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                break
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        
        return self.model
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log training metrics"""
        print(f"\n   📊 Training:")
        print(f"      Loss: {train_metrics['loss']:.4f}")
        print(f"      Buy: {train_metrics['buy_acc']:.3f} | "
              f"Sell: {train_metrics['sell_acc']:.3f} | "
              f"Dir: {train_metrics['direction_acc']:.3f} | "
              f"Regime: {train_metrics['regime_acc']:.3f}")
        
        print(f"\n   📊 Validation:")
        print(f"      Loss: {val_metrics['loss']:.4f}")
        print(f"      Buy: {val_metrics['buy_acc']:.3f} (F1: {val_metrics['buy_f1']:.3f})")
        print(f"      Sell: {val_metrics['sell_acc']:.3f} (F1: {val_metrics['sell_f1']:.3f})")
        print(f"      Dir: {val_metrics['direction_acc']:.3f} (F1: {val_metrics['direction_f1']:.3f})")
        print(f"      Regime: {val_metrics['regime_acc']:.3f} (F1: {val_metrics['regime_f1']:.3f})")
        
        if self.config.wandb_enabled:
            log_dict = {}
            for key, val in train_metrics.items():
                log_dict[f'train/{key}'] = val
            for key, val in val_metrics.items():
                log_dict[f'val/{key}'] = val
            log_dict['epoch'] = epoch
            log_dict['lr'] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.config.train.checkpoint_dir, 'best_model.pt')
            print(f"   💾 Saved best model (val_loss: {loss:.4f})")
        else:
            path = os.path.join(self.config.train.checkpoint_dir, f'model_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)


def run_training(config, pretrained_path: Optional[str] = None):
    """Run Phase 2 supervised training"""
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load data
    print("\n📊 Loading data...")
    raw_df, features_df = load_and_prepare_data(config.data.csv_path)
    labels_df, class_weights = create_labels(raw_df, features_df,
                                              config.data.neutral_buffer,
                                              config.data.lookforward)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        features_df, labels_df, config
    )
    
    # Save scaler
    scaler_path = os.path.join(config.output_dir, 'scaler.pkl')
    os.makedirs(config.output_dir, exist_ok=True)
    save_scaler(scaler, scaler_path)
    
    # Create model
    if pretrained_path:
        print(f"\n🔄 Loading pretrained model from {pretrained_path}")
    model = create_model(config, pretrained_path)
    
    # Convert class weights to correct device
    if config.train.use_class_weights and not config.train.focal_loss:
        for key in class_weights:
            class_weights[key] = class_weights[key].astype(np.float32)
    else:
        class_weights = None
    
    # Initialize wandb
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name='supervised_training',
            config=config.__dict__
        )
    
    # Train
    trainer = Trainer(model, config, class_weights, config.device)
    trained_model = trainer.train(train_loader, val_loader)
    
    # Test
    print("\n🧪 Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    print(f"\n   📊 Test Results:")
    print(f"      Buy Acc: {test_metrics['buy_acc']:.3f} (F1: {test_metrics['buy_f1']:.3f})")
    print(f"      Sell Acc: {test_metrics['sell_acc']:.3f} (F1: {test_metrics['sell_f1']:.3f})")
    print(f"      Direction Acc: {test_metrics['direction_acc']:.3f} (F1: {test_metrics['direction_f1']:.3f})")
    print(f"      Regime Acc: {test_metrics['regime_acc']:.3f} (F1: {test_metrics['regime_f1']:.3f})")
    
    # Save final model
    final_path = os.path.join(config.model_dir, 'natron_v2.pt')
    os.makedirs(config.model_dir, exist_ok=True)
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'test_metrics': test_metrics
    }, final_path)
    print(f"\n💾 Final model saved to {final_path}")
    
    if config.wandb_enabled:
        wandb.log({'test/' + k: v for k, v in test_metrics.items()})
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
        pretrained_path = sys.argv[2]
    else:
        pretrained_path = None
    
    # Run training
    run_training(config, pretrained_path)
