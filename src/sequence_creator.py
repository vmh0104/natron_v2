"""
SequenceCreator: Constructs sequences of 96 consecutive candles for training
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional
import torch


class TradingSequenceDataset(Dataset):
    """PyTorch Dataset for trading sequences"""
    
    def __init__(self, 
                 features: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 sequence_length: int = 96):
        """
        Args:
            features: (N, num_features) array of features
            labels: (N, num_labels) array of labels [buy, sell, direction, regime]
            sequence_length: Number of consecutive candles per sample
        """
        self.sequence_length = sequence_length
        self.features = features
        self.labels = labels
        
        # Calculate valid indices (need sequence_length consecutive samples)
        self.valid_indices = list(range(len(features) - sequence_length + 1))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            X: (sequence_length, num_features) tensor
            y: (num_labels,) tensor or None
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract sequence
        X = self.features[start_idx:end_idx]
        X = torch.FloatTensor(X)
        
        # Extract label (use last timestep)
        if self.labels is not None:
            y = self.labels[end_idx - 1]  # Label at end of sequence
            y = torch.FloatTensor(y)
            return X, y
        else:
            return X, None


class SequenceCreator:
    """Creates sequences from features and labels"""
    
    def __init__(self, sequence_length: int = 96):
        self.sequence_length = sequence_length
    
    def create_sequences(self, 
                        features_df: pd.DataFrame,
                        labels_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences from features and labels
        
        Args:
            features_df: DataFrame with feature columns (exclude 'time')
            labels_df: DataFrame with label columns ['buy', 'sell', 'direction', 'regime']
            
        Returns:
            X: (N, sequence_length, num_features) array
            y: (N, num_labels) array or None
        """
        # Extract feature columns (exclude 'time')
        feature_cols = [col for col in features_df.columns if col != 'time']
        features = features_df[feature_cols].values
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Create sequences
        num_samples = len(features) - self.sequence_length + 1
        X = np.zeros((num_samples, self.sequence_length, features.shape[1]))
        
        for i in range(num_samples):
            X[i] = features[i:i + self.sequence_length]
        
        # Create labels if provided
        y = None
        if labels_df is not None:
            label_cols = ['buy', 'sell', 'direction', 'regime']
            available_labels = [col for col in label_cols if col in labels_df.columns]
            
            if available_labels:
                labels = labels_df[available_labels].values
                # Extract labels for sequences (use last timestep)
                y = labels[self.sequence_length - 1:]
                
                # Ensure y matches X length
                if len(y) != len(X):
                    min_len = min(len(y), len(X))
                    X = X[:min_len]
                    y = y[:min_len]
        
        return X, y
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score"""
        mean = np.nanmean(features, axis=0, keepdims=True)
        std = np.nanstd(features, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
        normalized = (features - mean) / std
        
        # Replace NaN and Inf with 0
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized
    
    def create_datasets(self,
                       features_df: pd.DataFrame,
                       labels_df: Optional[pd.DataFrame] = None,
                       train_split: float = 0.7,
                       val_split: float = 0.15,
                       test_split: float = 0.15,
                       random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train/val/test datasets
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # Create sequences
        X, y = self.create_sequences(features_df, labels_df)
        
        # Split indices
        np.random.seed(random_seed)
        indices = np.random.permutation(len(X))
        
        train_end = int(len(X) * train_split)
        val_end = train_end + int(len(X) * val_split)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create datasets
        train_dataset = TradingSequenceDataset(
            X[train_indices],
            y[train_indices] if y is not None else None,
            self.sequence_length
        )
        
        val_dataset = TradingSequenceDataset(
            X[val_indices],
            y[val_indices] if y is not None else None,
            self.sequence_length
        )
        
        test_dataset = TradingSequenceDataset(
            X[test_indices],
            y[test_indices] if y is not None else None,
            self.sequence_length
        )
        
        return train_dataset, val_dataset, test_dataset
