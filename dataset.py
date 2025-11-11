"""
Natron Transformer - Dataset Module
Creates sequences of 96 candles for training
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Dict, Optional
import pickle


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence-based multi-task learning
    Each sample: (96, n_features) -> (buy, sell, direction, regime)
    """
    
    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame, 
                 sequence_length: int = 96, normalize: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            features: Technical features (N, n_features)
            labels: Multi-task labels (N, 4)
            sequence_length: Number of candles per sequence
            normalize: Whether to normalize features
            scaler: Pre-fitted scaler (for val/test sets)
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Convert to numpy
        self.features = features.values.astype(np.float32)
        self.labels = labels.values.astype(np.int64)
        
        # Normalize features
        if normalize:
            if scaler is None:
                self.scaler = RobustScaler()  # More robust to outliers
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
        
        print(f"  Created {len(self.sequences)} sequences")
    
    def _create_sequences(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create overlapping sequences"""
        n_samples = len(self.features) - self.sequence_length
        
        sequences = np.zeros((n_samples, self.sequence_length, self.features.shape[1]), 
                             dtype=np.float32)
        targets = {
            'buy': np.zeros(n_samples, dtype=np.int64),
            'sell': np.zeros(n_samples, dtype=np.int64),
            'direction': np.zeros(n_samples, dtype=np.int64),
            'regime': np.zeros(n_samples, dtype=np.int64)
        }
        
        for i in range(n_samples):
            sequences[i] = self.features[i:i+self.sequence_length]
            targets['buy'][i] = self.labels[i+self.sequence_length-1, 0]
            targets['sell'][i] = self.labels[i+self.sequence_length-1, 1]
            targets['direction'][i] = self.labels[i+self.sequence_length-1, 2]
            targets['regime'][i] = self.labels[i+self.sequence_length-1, 3]
        
        return sequences, targets
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sequence = torch.from_numpy(self.sequences[idx])
        targets = {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.targets.items()
        }
        return sequence, targets
    
    def get_scaler(self) -> Optional[StandardScaler]:
        """Return the fitted scaler"""
        return self.scaler


def create_dataloaders(features: pd.DataFrame, labels: pd.DataFrame,
                       config) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create train/val/test dataloaders with proper splits
    
    Args:
        features: Technical features dataframe
        labels: Labels dataframe
        config: Configuration object
    
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    print("\n📦 Creating dataloaders...")
    
    # Calculate split indices
    n_samples = len(features)
    train_end = int(n_samples * config.data.train_split)
    val_end = int(n_samples * (config.data.train_split + config.data.val_split))
    
    # Split data
    train_features = features.iloc[:train_end]
    train_labels = labels.iloc[:train_end]
    
    val_features = features.iloc[train_end:val_end]
    val_labels = labels.iloc[train_end:val_end]
    
    test_features = features.iloc[val_end:]
    test_labels = labels.iloc[val_end:]
    
    print(f"  Train: {len(train_features)} samples")
    print(f"  Val:   {len(val_features)} samples")
    print(f"  Test:  {len(test_features)} samples")
    
    # Create datasets
    train_dataset = SequenceDataset(
        train_features, train_labels, 
        config.data.sequence_length, 
        config.data.normalize
    )
    
    val_dataset = SequenceDataset(
        val_features, val_labels,
        config.data.sequence_length,
        config.data.normalize,
        scaler=train_dataset.get_scaler()
    )
    
    test_dataset = SequenceDataset(
        test_features, test_labels,
        config.data.sequence_length,
        config.data.normalize,
        scaler=train_dataset.get_scaler()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, train_dataset.get_scaler()


def save_scaler(scaler: StandardScaler, path: str):
    """Save scaler to disk"""
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to {path}")


def load_scaler(path: str) -> StandardScaler:
    """Load scaler from disk"""
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"📂 Scaler loaded from {path}")
    return scaler


class PretrainingDataset(Dataset):
    """
    Dataset for Phase 1 unsupervised pretraining
    Applies masking and creates positive pairs for contrastive learning
    """
    
    def __init__(self, features: pd.DataFrame, sequence_length: int = 96,
                 mask_ratio: float = 0.15, normalize: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            features: Technical features
            sequence_length: Sequence length
            mask_ratio: Ratio of features to mask
            normalize: Whether to normalize
            scaler: Pre-fitted scaler
        """
        self.sequence_length = sequence_length
        self.mask_ratio = mask_ratio
        self.normalize = normalize
        
        # Convert and normalize
        self.features = features.values.astype(np.float32)
        
        if normalize:
            if scaler is None:
                self.scaler = RobustScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"  Created {len(self.sequences)} pretraining sequences")
    
    def _create_sequences(self) -> np.ndarray:
        """Create sequences without labels"""
        n_samples = len(self.features) - self.sequence_length
        sequences = np.zeros((n_samples, self.sequence_length, self.features.shape[1]),
                             dtype=np.float32)
        
        for i in range(n_samples):
            sequences[i] = self.features[i:i+self.sequence_length]
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            original: Original sequence
            masked: Masked sequence
            mask: Boolean mask indicating masked positions
            positive: Augmented positive pair for contrastive learning
        """
        sequence = self.sequences[idx].copy()
        
        # Create mask
        seq_len, n_features = sequence.shape
        n_mask = int(seq_len * self.mask_ratio)
        mask_indices = np.random.choice(seq_len, n_mask, replace=False)
        
        # Create masked sequence
        masked_sequence = sequence.copy()
        mask = np.zeros(seq_len, dtype=bool)
        mask[mask_indices] = True
        masked_sequence[mask] = 0  # Mask by setting to 0
        
        # Create positive pair (with different masking)
        n_mask_pos = int(seq_len * self.mask_ratio)
        mask_indices_pos = np.random.choice(seq_len, n_mask_pos, replace=False)
        positive_sequence = sequence.copy()
        positive_sequence[mask_indices_pos] = 0
        
        return {
            'original': torch.from_numpy(sequence),
            'masked': torch.from_numpy(masked_sequence),
            'mask': torch.from_numpy(mask),
            'positive': torch.from_numpy(positive_sequence)
        }
    
    def get_scaler(self) -> Optional[StandardScaler]:
        return self.scaler


def create_pretraining_dataloader(features: pd.DataFrame, config) -> Tuple[DataLoader, StandardScaler]:
    """
    Create dataloader for Phase 1 pretraining
    
    Args:
        features: Technical features
        config: Configuration
    
    Returns:
        dataloader, scaler
    """
    print("\n📦 Creating pretraining dataloader...")
    
    dataset = PretrainingDataset(
        features,
        config.data.sequence_length,
        config.pretrain.mask_ratio,
        config.data.normalize
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    return dataloader, dataset.get_scaler()


if __name__ == "__main__":
    # Test dataset creation
    from feature_engine import load_and_prepare_data
    from label_generator_v2 import create_labels
    from config import load_config
    import sys
    
    config = load_config()
    
    if len(sys.argv) > 1:
        config.data.csv_path = sys.argv[1]
    
    # Load data
    raw_df, features_df = load_and_prepare_data(config.data.csv_path)
    labels_df, _ = create_labels(raw_df, features_df)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        features_df, labels_df, config
    )
    
    # Test batch
    for batch_X, batch_y in train_loader:
        print(f"\n✅ Batch shapes:")
        print(f"  Input: {batch_X.shape}")
        print(f"  Targets:")
        for key, val in batch_y.items():
            print(f"    {key}: {val.shape}")
        break
