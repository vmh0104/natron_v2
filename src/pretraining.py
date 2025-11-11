"""
Phase 1: Pretraining - Masked Token Reconstruction + Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os


class PretrainingTrainer:
    """Pretraining with masked token reconstruction and contrastive learning"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Masked token reconstruction head
        self.reconstruction_head = nn.Linear(config['d_model'], config['num_features']).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(self.reconstruction_head.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.mask_prob = config.get('mask_probability', 0.15)
        self.temperature = config.get('contrastive_temperature', 0.07)
        self.reconstruction_weight = config.get('reconstruction_weight', 0.5)
        self.contrastive_weight = config.get('contrastive_weight', 0.5)
    
    def mask_tokens(self, sequences: torch.Tensor) -> tuple:
        """
        Randomly mask tokens in sequences
        
        Returns:
            masked_sequences: Sequences with masked tokens
            mask: Boolean mask indicating which tokens were masked
        """
        batch_size, seq_len, num_features = sequences.shape
        mask = torch.rand(batch_size, seq_len, device=self.device) < self.mask_prob
        
        # Create masked sequences (replace with zeros)
        masked_sequences = sequences.clone()
        masked_sequences[mask] = 0
        
        return masked_sequences, mask
    
    def contrastive_loss(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss
        
        Args:
            hidden_states: (batch_size, d_model)
        
        Returns:
            contrastive_loss: Scalar loss
        """
        batch_size = hidden_states.shape[0]
        
        # Normalize
        hidden_states = F.normalize(hidden_states, p=2, dim=1)
        
        # Create positive pairs (augmentations of same sequence)
        # For simplicity, use different random masks as augmentations
        # In practice, you'd use actual data augmentations
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(hidden_states, hidden_states.t()) / self.temperature
        
        # Create labels (diagonal = positive pairs)
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def reconstruction_loss(self, 
                           sequences: torch.Tensor,
                           masked_sequences: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct masked tokens
        
        Args:
            sequences: Original sequences
            masked_sequences: Masked sequences
            mask: Boolean mask
        
        Returns:
            reconstruction_loss: MSE loss on masked tokens
        """
        # Forward pass through encoder
        x = self.model.input_projection(masked_sequences)
        x = self.model.pos_encoder(x)
        x = self.model.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Reconstruct features
        reconstructed = self.reconstruction_head(x)  # (batch_size, seq_len, num_features)
        
        # Compute loss only on masked tokens
        mask_expanded = mask.unsqueeze(-1).expand_as(sequences)
        masked_reconstructed = reconstructed[mask_expanded].view(-1, sequences.shape[-1])
        masked_original = sequences[mask_expanded].view(-1, sequences.shape[-1])
        
        if len(masked_reconstructed) == 0:
            # No masked tokens in this batch
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        loss = F.mse_loss(masked_reconstructed, masked_original)
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train one epoch"""
        self.model.train()
        self.reconstruction_head.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_contrastive_loss = 0
        
        for batch in tqdm(dataloader, desc="Pretraining"):
            sequences = batch['sequence'].to(self.device)
            
            # Mask tokens
            masked_sequences, mask = self.mask_tokens(sequences)
            
            # Forward pass
            outputs = self.model(masked_sequences)
            hidden_states = outputs['hidden_state']
            
            # Reconstruction loss
            recon_loss = self.reconstruction_loss(sequences, masked_sequences, mask)
            
            # Contrastive loss
            contrastive_loss = self.contrastive_loss(hidden_states)
            
            # Total loss
            loss = (self.reconstruction_weight * recon_loss + 
                   self.contrastive_weight * contrastive_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contrastive_loss.item()
        
        return {
            'loss': total_loss / len(dataloader),
            'reconstruction_loss': total_recon_loss / len(dataloader),
            'contrastive_loss': total_contrastive_loss / len(dataloader)
        }
    
    def train(self, train_loader: DataLoader, num_epochs: int, save_path: str):
        """Full training loop"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            metrics = self.train_epoch(train_loader)
            
            print(f"Loss: {metrics['loss']:.4f}")
            print(f"  Reconstruction: {metrics['reconstruction_loss']:.4f}")
            print(f"  Contrastive: {metrics['contrastive_loss']:.4f}")
            
            # Save best model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'reconstruction_head_state_dict': self.reconstruction_head.state_dict(),
                    'epoch': epoch,
                    'loss': metrics['loss']
                }, save_path)
                print(f"✅ Saved best model (loss: {best_loss:.4f})")
        
        print(f"\n🎉 Pretraining complete! Best loss: {best_loss:.4f}")
