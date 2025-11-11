"""
Natron Training Pipeline - All Three Training Phases
Phase 1: Pretraining (Unsupervised)
Phase 2: Supervised Fine-Tuning
Phase 3: Reinforcement Learning (Optional)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Tuple, Optional
import json


class Phase1Trainer:
    """Phase 1: Unsupervised Pretraining"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device):
        """
        Args:
            model: PretrainModel instance
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration dict
            device: torch device
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['pretrain']['learning_rate'],
            weight_decay=config['pretrain']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        self.reconstruction_loss_fn = nn.MSELoss()
        
        # Tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config['system']['log_dir'], 'pretrain')
        )
        
        # Checkpoint directory
        self.checkpoint_dir = config['pretrain']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        recon_loss_sum = 0
        contra_loss_sum = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (masked_seq, mask, original_seq) in enumerate(pbar):
            masked_seq = masked_seq.to(self.device)
            mask = mask.to(self.device)
            original_seq = original_seq.to(self.device)
            
            # Forward pass
            reconstructed, embeddings = self.model(masked_seq, mask)
            
            # Reconstruction loss (only on masked positions)
            mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)
            recon_loss = self.reconstruction_loss_fn(
                reconstructed[mask_expanded],
                original_seq[mask_expanded]
            )
            
            # Contrastive loss
            contra_loss = self.model.contrastive_loss(embeddings)
            
            # Combined loss
            loss = recon_loss + 0.5 * contra_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            contra_loss_sum += contra_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'contra': f'{contra_loss.item():.4f}'
            })
            
            # Tensorboard logging
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/recon_loss', recon_loss.item(), self.global_step)
                self.writer.add_scalar('train/contra_loss', contra_loss.item(), self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_recon = recon_loss_sum / len(self.train_loader)
        avg_contra = contra_loss_sum / len(self.train_loader)
        
        print(f"\n[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Contra: {avg_contra:.4f}")
        
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for masked_seq, mask, original_seq in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                masked_seq = masked_seq.to(self.device)
                mask = mask.to(self.device)
                original_seq = original_seq.to(self.device)
                
                reconstructed, embeddings = self.model(masked_seq, mask)
                
                mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)
                recon_loss = self.reconstruction_loss_fn(
                    reconstructed[mask_expanded],
                    original_seq[mask_expanded]
                )
                
                contra_loss = self.model.contrastive_loss(embeddings)
                loss = recon_loss + 0.5 * contra_loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        print(f"[Epoch {epoch}] Val Loss: {avg_loss:.4f}")
        
        # Tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 PHASE 1: PRETRAINING (Unsupervised)")
        print("="*70)
        
        epochs = self.config['pretrain']['epochs']
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_pretrain.pt', epoch, val_loss)
                print(f"✅ Best model saved (val_loss: {val_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'pretrain_epoch_{epoch}.pt', epoch, val_loss)
        
        print(f"\n✅ Pretraining complete! Best val loss: {self.best_val_loss:.4f}")
        self.writer.close()
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)


class Phase2Trainer:
    """Phase 2: Supervised Fine-Tuning"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device):
        """
        Args:
            model: NatronTransformer instance
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration dict
            device: torch device
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['supervised']['learning_rate'],
            weight_decay=config['supervised']['weight_decay']
        )
        
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
        
        # Loss weights
        self.loss_weights = config['supervised']['loss_weights']
        
        # Tensorboard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config['system']['log_dir'], 'supervised')
        )
        
        # Checkpoint directory
        self.checkpoint_dir = config['supervised']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute multi-task loss"""
        # Buy/Sell losses (BCE)
        buy_loss = self.bce_loss(outputs['buy_prob'], targets['buy'])
        sell_loss = self.bce_loss(outputs['sell_prob'], targets['sell'])
        
        # Direction loss (CE)
        direction_loss = self.ce_loss(
            outputs['direction_logits'],
            targets['direction'].squeeze()
        )
        
        # Regime loss (CE)
        regime_loss = self.ce_loss(
            outputs['regime_logits'],
            targets['regime'].squeeze()
        )
        
        # Weighted total loss
        total_loss = (
            self.loss_weights['buy'] * buy_loss +
            self.loss_weights['sell'] * sell_loss +
            self.loss_weights['direction'] * direction_loss +
            self.loss_weights['regime'] * regime_loss
        )
        
        losses = {
            'total': total_loss.item(),
            'buy': buy_loss.item(),
            'sell': sell_loss.item(),
            'direction': direction_loss.item(),
            'regime': regime_loss.item()
        }
        
        return total_loss, losses
    
    def compute_metrics(self, outputs: Dict, targets: Dict) -> Dict:
        """Compute accuracy metrics"""
        metrics = {}
        
        # Buy/Sell accuracy (threshold 0.5)
        buy_pred = (outputs['buy_prob'] > 0.5).float()
        sell_pred = (outputs['sell_prob'] > 0.5).float()
        
        metrics['buy_acc'] = (buy_pred == targets['buy']).float().mean().item()
        metrics['sell_acc'] = (sell_pred == targets['sell']).float().mean().item()
        
        # Direction accuracy
        direction_pred = outputs['direction_logits'].argmax(dim=1)
        metrics['direction_acc'] = (direction_pred == targets['direction'].squeeze()).float().mean().item()
        
        # Regime accuracy
        regime_pred = outputs['regime_logits'].argmax(dim=1)
        metrics['regime_acc'] = (regime_pred == targets['regime'].squeeze()).float().mean().item()
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """Train one epoch"""
        self.model.train()
        epoch_losses = {k: 0.0 for k in ['total', 'buy', 'sell', 'direction', 'regime']}
        epoch_metrics = {k: 0.0 for k in ['buy_acc', 'sell_acc', 'direction_acc', 'regime_acc']}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (sequences, targets) in enumerate(pbar):
            sequences = sequences.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            outputs = self.model(sequences)
            
            # Compute loss
            loss, losses = self.compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['supervised']['gradient_clip']
            )
            self.optimizer.step()
            
            # Compute metrics
            metrics = self.compute_metrics(outputs, targets)
            
            # Accumulate
            for k, v in losses.items():
                epoch_losses[k] += v
            for k, v in metrics.items():
                epoch_metrics[k] += v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses["total"]:.4f}',
                'dir_acc': f'{metrics["direction_acc"]:.3f}',
                'reg_acc': f'{metrics["regime_acc"]:.3f}'
            })
            
            # Tensorboard logging
            if batch_idx % 10 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}_loss', v, self.global_step)
                for k, v in metrics.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            self.global_step += 1
        
        # Average metrics
        n_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
        
        print(f"\n[Epoch {epoch}] Train Loss: {epoch_losses['total']:.4f} | "
              f"Dir Acc: {epoch_metrics['direction_acc']:.3f} | "
              f"Regime Acc: {epoch_metrics['regime_acc']:.3f}")
        
        return epoch_losses['total'], epoch_metrics
    
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """Validate"""
        self.model.eval()
        epoch_losses = {k: 0.0 for k in ['total', 'buy', 'sell', 'direction', 'regime']}
        epoch_metrics = {k: 0.0 for k in ['buy_acc', 'sell_acc', 'direction_acc', 'regime_acc']}
        
        with torch.no_grad():
            for sequences, targets in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                sequences = sequences.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(sequences)
                _, losses = self.compute_loss(outputs, targets)
                metrics = self.compute_metrics(outputs, targets)
                
                for k, v in losses.items():
                    epoch_losses[k] += v
                for k, v in metrics.items():
                    epoch_metrics[k] += v
        
        # Average
        n_batches = len(self.val_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
        
        print(f"[Epoch {epoch}] Val Loss: {epoch_losses['total']:.4f} | "
              f"Dir Acc: {epoch_metrics['direction_acc']:.3f} | "
              f"Regime Acc: {epoch_metrics['regime_acc']:.3f}")
        
        # Tensorboard
        for k, v in epoch_losses.items():
            self.writer.add_scalar(f'val/{k}_loss', v, epoch)
        for k, v in epoch_metrics.items():
            self.writer.add_scalar(f'val/{k}', v, epoch)
        
        return epoch_losses['total'], epoch_metrics
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🎯 PHASE 2: SUPERVISED FINE-TUNING")
        print("="*70)
        
        epochs = self.config['supervised']['epochs']
        patience = self.config['supervised']['early_stopping_patience']
        
        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_supervised.pt', epoch, val_loss, val_metrics)
                print(f"✅ Best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                    break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'supervised_epoch_{epoch}.pt', epoch, val_loss, val_metrics)
        
        print(f"\n✅ Supervised training complete! Best val loss: {self.best_val_loss:.4f}")
        self.writer.close()
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.config
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)


class Phase3Trainer:
    """Phase 3: Reinforcement Learning (PPO-based)"""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 device: torch.device):
        """
        Args:
            model: Trained NatronTransformer
            config: Configuration dict
            device: torch device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        print("\n" + "="*70)
        print("💪 PHASE 3: REINFORCEMENT LEARNING")
        print("="*70)
        print("\n⚠️  RL training requires trading environment setup.")
        print("   This is a placeholder for future integration with:")
        print("   - Gym trading environment")
        print("   - Stable-Baselines3 PPO/SAC")
        print("   - Custom reward function (profit - turnover - drawdown)")
        print("\n   For now, skipping RL phase...")
    
    def train(self):
        """RL training (placeholder)"""
        if not self.config['rl']['enabled']:
            print("\n⏭️  RL training disabled in config. Skipping Phase 3.")
            return
        
        print("\n🚧 RL training not yet implemented. Skipping...")


def load_pretrained_weights(model: nn.Module, pretrain_path: str) -> nn.Module:
    """Load pretrained encoder weights into supervised model"""
    print(f"\n📥 Loading pretrained weights from {pretrain_path}")
    
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    
    # Extract encoder weights from PretrainModel
    pretrain_state = checkpoint['model_state_dict']
    encoder_state = {}
    
    for k, v in pretrain_state.items():
        if k.startswith('encoder.'):
            new_key = k.replace('encoder.', '')
            encoder_state[new_key] = v
    
    # Load into model
    model.load_state_dict(encoder_state, strict=False)
    
    print("✅ Pretrained weights loaded successfully!")
    
    return model


if __name__ == "__main__":
    print("Training module ready. Use main.py to run full pipeline.")
