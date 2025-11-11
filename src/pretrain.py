"""
Phase 1: Pretraining - Masked Token Reconstruction + Contrastive Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from typing import Optional

from src.model import NatronTransformer, MaskedTokenModel, contrastive_loss


class PretrainDataset(Dataset):
    """Dataset for pretraining"""
    
    def __init__(self, X: np.ndarray):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


def pretrain(
    model: MaskedTokenModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    reconstruction_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    save_path: Optional[str] = None
):
    """
    Pretrain model using masked token reconstruction + contrastive learning
    
    Args:
        model: MaskedTokenModel instance
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: torch device
        num_epochs: number of epochs
        learning_rate: learning rate
        weight_decay: weight decay
        reconstruction_weight: weight for reconstruction loss
        contrastive_weight: weight for contrastive loss
        save_path: path to save best model
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mask = model(batch)
            
            # Reconstruction loss (only on masked tokens)
            recon_loss = nn.MSELoss()(reconstructed[mask], batch[mask])
            
            # Contrastive loss (augment batch with noise)
            noise = torch.randn_like(batch) * 0.01
            batch_augmented = batch + noise
            encoded_orig = model.encoder.encode(batch).mean(dim=1)
            encoded_aug = model.encoder.encode(batch_augmented).mean(dim=1)
            cont_loss = contrastive_loss(encoded_orig, encoded_aug)
            
            # Total loss
            loss = reconstruction_weight * recon_loss + contrastive_weight * cont_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed, mask = model(batch)
                recon_loss = nn.MSELoss()(reconstructed[mask], batch[mask])
                
                noise = torch.randn_like(batch) * 0.01
                batch_augmented = batch + noise
                encoded_orig = model.encoder.encode(batch).mean(dim=1)
                encoded_aug = model.encoder.encode(batch_augmented).mean(dim=1)
                cont_loss = contrastive_loss(encoded_orig, encoded_aug)
                
                loss = reconstruction_weight * recon_loss + contrastive_weight * cont_loss
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"✅ Saved best model to {save_path}")
    
    print(f"\n🎯 Pretraining complete! Best validation loss: {best_val_loss:.4f}")
    return model
