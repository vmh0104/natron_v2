"""
SequenceCreator: Constructs sequences of 96 consecutive candles for training
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional


class TradingSequenceDataset(Dataset):
    """
    PyTorch Dataset for trading sequences.
    Each sample: (96, N_features) → (buy, sell, direction, regime)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: pd.DataFrame,
        sequence_length: int = 96,
        normalize: bool = True
    ):
        """
        Args:
            features: Array of shape (N, N_features)
            labels: DataFrame with columns: buy, sell, direction, regime
            sequence_length: Number of consecutive candles per sample
            normalize: Whether to normalize features
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Remove NaN rows
        valid_mask = ~(np.isnan(features).any(axis=1) | 
                     labels[['buy', 'sell', 'direction', 'regime']].isna().any(axis=1))
        
        self.features = features[valid_mask]
        self.labels = labels[valid_mask].reset_index(drop=True)
        
        # Normalize features
        if self.normalize:
            self.feature_mean = np.nanmean(self.features, axis=0, keepdims=True)
            self.feature_std = np.nanstd(self.features, axis=0, keepdims=True) + 1e-8
            self.features = (self.features - self.feature_mean) / self.feature_std
        
        # Create sequences
        self.sequences = []
        self.sequence_labels = []
        
        for i in range(len(self.features) - sequence_length + 1):
            seq = self.features[i:i+sequence_length]
            label_idx = i + sequence_length - 1
            
            # Extract labels
            buy = self.labels.iloc[label_idx]['buy']
            sell = self.labels.iloc[label_idx]['sell']
            direction = int(self.labels.iloc[label_idx]['direction'])
            regime = int(self.labels.iloc[label_idx]['regime'])
            
            self.sequences.append(seq)
            self.sequence_labels.append({
                'buy': buy,
                'sell': sell,
                'direction': direction,
                'regime': regime
            })
        
        self.sequences = np.array(self.sequences)
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].astype(np.float32)
        labels = self.sequence_labels[idx]
        
        return {
            'sequence': sequence,
            'buy': np.float32(labels['buy']),
            'sell': np.float32(labels['sell']),
            'direction': np.int64(labels['direction']),
            'regime': np.int64(labels['regime'])
        }
    
    def get_feature_stats(self):
        """Return normalization statistics"""
        if self.normalize:
            return {
                'mean': self.feature_mean,
                'std': self.feature_std
            }
        return None


class SequenceCreator:
    """
    Creates sequences from features and labels for training
    """
    
    @staticmethod
    def create_datasets(
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        sequence_length: int = 96,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        normalize: bool = True
    ) -> Tuple[TradingSequenceDataset, TradingSequenceDataset, TradingSequenceDataset]:
        """
        Create train/val/test datasets
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        # Convert to numpy
        features = features_df.values
        
        # Calculate split indices
        n_samples = len(features) - sequence_length + 1
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        
        # Shuffle indices
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Create datasets
        train_dataset = TradingSequenceDataset(
            features, labels_df, sequence_length, normalize=False
        )
        
        # Normalize using training statistics
        if normalize:
            train_mean = np.nanmean(train_dataset.features, axis=0, keepdims=True)
            train_std = np.nanstd(train_dataset.features, axis=0, keepdims=True) + 1e-8
            
            # Apply normalization to all datasets
            train_dataset.features = (train_dataset.features - train_mean) / train_std
            train_dataset.feature_mean = train_mean
            train_dataset.feature_std = train_std
        
        # Create validation and test datasets with same normalization
        val_dataset = TradingSequenceDataset(
            features, labels_df, sequence_length, normalize=False
        )
        test_dataset = TradingSequenceDataset(
            features, labels_df, sequence_length, normalize=False
        )
        
        if normalize:
            val_dataset.features = (val_dataset.features - train_mean) / train_std
            val_dataset.feature_mean = train_mean
            val_dataset.feature_std = train_std
            
            test_dataset.features = (test_dataset.features - train_mean) / train_std
            test_dataset.feature_mean = train_mean
            test_dataset.feature_std = train_std
        
        # Filter by split indices
        train_dataset.sequences = train_dataset.sequences[train_indices]
        train_dataset.sequence_labels = [train_dataset.sequence_labels[i] for i in train_indices]
        
        val_dataset.sequences = val_dataset.sequences[val_indices]
        val_dataset.sequence_labels = [val_dataset.sequence_labels[i] for i in val_indices]
        
        test_dataset.sequences = test_dataset.sequences[test_indices]
        test_dataset.sequence_labels = [test_dataset.sequence_labels[i] for i in test_indices]
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
