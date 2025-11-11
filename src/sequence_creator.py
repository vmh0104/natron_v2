"""
SequenceCreator - Constructs 96-candle sequences for training
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence data"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, 
                 sequence_length: int = 96):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx].astype(np.float32),
            'buy': self.labels[idx, 0].astype(np.float32),
            'sell': self.labels[idx, 1].astype(np.float32),
            'direction': self.labels[idx, 2].astype(np.int64),
            'regime': self.labels[idx, 3].astype(np.int64)
        }


class SequenceCreator:
    """Creates sequences of consecutive candles for training"""
    
    def __init__(self, sequence_length: int = 96, normalize: bool = True):
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.scaler = StandardScaler()
        self.feature_scaler = None
        
    def create_sequences(self, 
                        features_df: pd.DataFrame,
                        labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from features and labels
        
        Args:
            features_df: DataFrame with features (N, num_features)
            labels_df: DataFrame with labels (N, 4) [buy, sell, direction, regime]
        
        Returns:
            sequences: (N_samples, sequence_length, num_features)
            labels: (N_samples, 4)
        """
        # Align indices
        common_idx = features_df.index.intersection(labels_df.index)
        features_df = features_df.loc[common_idx]
        labels_df = labels_df.loc[common_idx]
        
        # Convert to numpy
        features = features_df.values.astype(np.float32)
        labels = labels_df[['buy', 'sell', 'direction', 'regime']].values
        
        # Normalize features
        if self.normalize:
            if self.feature_scaler is None:
                # Fit on all data (in production, fit only on train)
                self.feature_scaler = StandardScaler()
                features = self.feature_scaler.fit_transform(features)
            else:
                features = self.feature_scaler.transform(features)
        
        # Create sequences
        sequences = []
        sequence_labels = []
        
        for i in range(len(features) - self.sequence_length + 1):
            seq = features[i:i + self.sequence_length]
            label_idx = i + self.sequence_length - 1  # Label at end of sequence
            
            sequences.append(seq)
            sequence_labels.append(labels[label_idx])
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        
        print(f"\n✅ Created {len(sequences)} sequences")
        print(f"   Sequence shape: {sequences.shape}")
        print(f"   Labels shape: {sequence_labels.shape}")
        
        return sequences, sequence_labels
    
    def split_data(self, 
                   sequences: np.ndarray,
                   labels: np.ndarray,
                   train_split: float = 0.7,
                   val_split: float = 0.15,
                   test_split: float = 0.15,
                   random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train/val/test sets
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(sequences))
        
        n_train = int(len(sequences) * train_split)
        n_val = int(len(sequences) * val_split)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        X_train = sequences[train_idx]
        y_train = labels[train_idx]
        X_val = sequences[val_idx]
        y_val = labels[val_idx]
        X_test = sequences[test_idx]
        y_test = labels[test_idx]
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train)} samples ({train_split*100:.1f}%)")
        print(f"   Val:   {len(X_val)} samples ({val_split*100:.1f}%)")
        print(f"   Test:  {len(X_test)} samples ({test_split*100:.1f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
