"""
Phase 2: Supervised Fine-Tuning - Multi-task prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from typing import Optional, Dict

from src.model import NatronTransformer


class SupervisedDataset(Dataset):
    """Dataset for supervised training"""
    
    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        self.X = torch.FloatTensor(X)
        self.y_buy = torch.FloatTensor(y['buy'])
        self.y_sell = torch.FloatTensor(y['sell'])
        self.y_direction = torch.LongTensor(y['direction'])
        self.y_regime = torch.LongTensor(y['regime'])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'buy': self.y_buy[idx],
            'sell': self.y_sell[idx],
            'direction': self.y_direction[idx],
            'regime': self.y_regime[idx]
        }


def supervised_train(
    model: NatronTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 10,
    save_path: Optional[str] = None
):
    """
    Supervised fine-tuning for multi-task prediction
    
    Args:
        model: NatronTransformer instance
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: torch device
        num_epochs: number of epochs
        learning_rate: learning rate
        weight_decay: weight decay
        scheduler_patience: patience for ReduceLROnPlateau
        scheduler_factor: factor for ReduceLROnPlateau
        early_stopping_patience: patience for early stopping
        save_path: path to save best model
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
    )
    
    # Loss functions
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            X = batch['X'].to(device)
            y_buy = batch['buy'].to(device)
            y_sell = batch['sell'].to(device)
            y_direction = batch['direction'].to(device)
            y_regime = batch['regime'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            
            # Compute losses
            loss_buy = bce_loss(outputs['buy'], y_buy)
            loss_sell = bce_loss(outputs['sell'], y_sell)
            loss_direction = ce_loss(outputs['direction'], y_direction)
            loss_regime = ce_loss(outputs['regime'], y_regime)
            
            # Total loss (equal weights)
            loss = loss_buy + loss_sell + loss_direction + loss_regime
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_metrics['buy'] += loss_buy.item()
            train_metrics['sell'] += loss_sell.item()
            train_metrics['direction'] += loss_direction.item()
            train_metrics['regime'] += loss_regime.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'buy': f'{loss_buy.item():.4f}',
                'sell': f'{loss_sell.item():.4f}'
            })
        
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                y_buy = batch['buy'].to(device)
                y_sell = batch['sell'].to(device)
                y_direction = batch['direction'].to(device)
                y_regime = batch['regime'].to(device)
                
                outputs = model(X)
                
                loss_buy = bce_loss(outputs['buy'], y_buy)
                loss_sell = bce_loss(outputs['sell'], y_sell)
                loss_direction = ce_loss(outputs['direction'], y_direction)
                loss_regime = ce_loss(outputs['regime'], y_regime)
                
                loss = loss_buy + loss_sell + loss_direction + loss_regime
                
                val_loss += loss.item()
                val_metrics['buy'] += loss_buy.item()
                val_metrics['sell'] += loss_sell.item()
                val_metrics['direction'] += loss_direction.item()
                val_metrics['regime'] += loss_regime.item()
        
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Metrics: Buy={train_metrics['buy']:.4f}, Sell={train_metrics['sell']:.4f}, "
              f"Dir={train_metrics['direction']:.4f}, Reg={train_metrics['regime']:.4f}")
        print(f"  Val Metrics: Buy={val_metrics['buy']:.4f}, Sell={val_metrics['sell']:.4f}, "
              f"Dir={val_metrics['direction']:.4f}, Reg={val_metrics['regime']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                }, save_path)
                print(f"✅ Saved best model to {save_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n🎯 Supervised training complete! Best validation loss: {best_val_loss:.4f}")
    return model
