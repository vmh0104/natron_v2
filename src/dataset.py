"""
Natron Dataset Module - Creates 96-candle sequence windows for training
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import pickle


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for 96-candle sequence windows.
    Each sample: (96, num_features) → (buy, sell, direction, regime)
    """
    
    def __init__(self, 
                 features: pd.DataFrame,
                 labels: pd.DataFrame,
                 sequence_length: int = 96,
                 normalize: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            features: Feature dataframe (N, num_features)
            labels: Label dataframe with [buy, sell, direction, regime]
            sequence_length: Number of candles per sequence (default 96)
            normalize: Whether to normalize features
            scaler: Pre-fitted scaler (for validation/test sets)
        """
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Convert to numpy
        self.features = features.values
        self.labels = labels.values
        
        # Normalize features
        if normalize:
            if scaler is None:
                self.scaler = StandardScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
        
        print(f"✅ Created {len(self.sequences)} sequences")
        print(f"   Sequence shape: {self.sequences.shape}")
        print(f"   Target shape: {self.targets.shape}")
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences"""
        n_samples = len(self.features) - self.sequence_length
        n_features = self.features.shape[1]
        
        sequences = np.zeros((n_samples, self.sequence_length, n_features))
        targets = np.zeros((n_samples, 4))  # buy, sell, direction, regime
        
        for i in range(n_samples):
            sequences[i] = self.features[i:i + self.sequence_length]
            targets[i] = self.labels[i + self.sequence_length - 1]
        
        return sequences, targets
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            sequence: (sequence_length, num_features)
            labels: dict with {buy, sell, direction, regime}
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        
        labels = {
            'buy': torch.FloatTensor([self.targets[idx, 0]]),
            'sell': torch.FloatTensor([self.targets[idx, 1]]),
            'direction': torch.LongTensor([int(self.targets[idx, 2])]),
            'regime': torch.LongTensor([int(self.targets[idx, 3])])
        }
        
        return sequence, labels
    
    def save_scaler(self, path: str):
        """Save the fitted scaler"""
        if self.scaler is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Scaler saved to {path}")
    
    @staticmethod
    def load_scaler(path: str) -> StandardScaler:
        """Load a saved scaler"""
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {path}")
        return scaler


class PretrainDataset(Dataset):
    """
    Dataset for Phase 1 Pretraining (unsupervised).
    Supports masked token reconstruction.
    """
    
    def __init__(self,
                 features: pd.DataFrame,
                 sequence_length: int = 96,
                 mask_ratio: float = 0.15,
                 normalize: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Args:
            features: Feature dataframe
            sequence_length: Sequence length
            mask_ratio: Ratio of tokens to mask
            normalize: Whether to normalize
            scaler: Pre-fitted scaler
        """
        self.sequence_length = sequence_length
        self.mask_ratio = mask_ratio
        
        # Convert to numpy
        self.features = features.values
        
        # Normalize
        if normalize:
            if scaler is None:
                self.scaler = StandardScaler()
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"✅ Created {len(self.sequences)} pretrain sequences")
    
    def _create_sequences(self) -> np.ndarray:
        """Create sequences without labels"""
        n_samples = len(self.features) - self.sequence_length
        n_features = self.features.shape[1]
        
        sequences = np.zeros((n_samples, self.sequence_length, n_features))
        
        for i in range(n_samples):
            sequences[i] = self.features[i:i + self.sequence_length]
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            masked_seq: Sequence with some tokens masked
            mask: Boolean mask indicating masked positions
            original_seq: Original sequence for reconstruction
        """
        original = torch.FloatTensor(self.sequences[idx])
        
        # Create mask
        seq_len = original.shape[0]
        num_mask = int(seq_len * self.mask_ratio)
        mask_indices = np.random.choice(seq_len, size=num_mask, replace=False)
        
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[mask_indices] = True
        
        # Create masked sequence (replace with zeros)
        masked = original.clone()
        masked[mask] = 0
        
        return masked, mask, original


def create_dataloaders(features: pd.DataFrame,
                       labels: pd.DataFrame,
                       config: Dict,
                       pretrain: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with proper splits.
    
    Args:
        features: Feature dataframe
        labels: Label dataframe
        config: Configuration dict
        pretrain: If True, create PretrainDataset instead
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split ratios
    train_ratio = config['data']['train_split']
    val_ratio = config['data']['val_split']
    
    n_total = len(features)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split data chronologically
    train_features = features.iloc[:n_train]
    val_features = features.iloc[n_train:n_train + n_val]
    test_features = features.iloc[n_train + n_val:]
    
    if pretrain:
        # Pretrain datasets
        train_dataset = PretrainDataset(
            train_features,
            sequence_length=config['data']['sequence_length'],
            mask_ratio=config['pretrain']['mask_ratio'],
            normalize=config['features']['normalize']
        )
        
        val_dataset = PretrainDataset(
            val_features,
            sequence_length=config['data']['sequence_length'],
            mask_ratio=config['pretrain']['mask_ratio'],
            normalize=config['features']['normalize'],
            scaler=train_dataset.scaler
        )
        
        test_dataset = PretrainDataset(
            test_features,
            sequence_length=config['data']['sequence_length'],
            mask_ratio=config['pretrain']['mask_ratio'],
            normalize=config['features']['normalize'],
            scaler=train_dataset.scaler
        )
        
        batch_size = config['pretrain']['batch_size']
    else:
        # Supervised datasets
        train_labels = labels.iloc[:n_train]
        val_labels = labels.iloc[n_train:n_train + n_val]
        test_labels = labels.iloc[n_train + n_val:]
        
        train_dataset = SequenceDataset(
            train_features,
            train_labels,
            sequence_length=config['data']['sequence_length'],
            normalize=config['features']['normalize']
        )
        
        val_dataset = SequenceDataset(
            val_features,
            val_labels,
            sequence_length=config['data']['sequence_length'],
            normalize=config['features']['normalize'],
            scaler=train_dataset.scaler
        )
        
        test_dataset = SequenceDataset(
            test_features,
            test_labels,
            sequence_length=config['data']['sequence_length'],
            normalize=config['features']['normalize'],
            scaler=train_dataset.scaler
        )
        
        batch_size = config['supervised']['batch_size']
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=True
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(train_dataset)} sequences")
    print(f"   Val:   {len(val_dataset)} sequences")
    print(f"   Test:  {len(test_dataset)} sequences")
    
    return train_loader, val_loader, test_loader, train_dataset.scaler if not pretrain else train_dataset.scaler


if __name__ == "__main__":
    # Test dataset creation
    print("Testing SequenceDataset...")
    
    from feature_engine import FeatureEngine
    from label_generator import LabelGeneratorV2
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='1H')
    
    df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.5) + np.random.rand(n),
        'low': 100 + np.cumsum(np.random.randn(n) * 0.5) - np.random.rand(n),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Extract features and labels
    engine = FeatureEngine()
    features = engine.extract_all_features(df)
    
    label_gen = LabelGeneratorV2()
    labels = label_gen.generate_labels(df, features)
    
    # Create dataset
    dataset = SequenceDataset(features, labels, sequence_length=96)
    
    # Test sample
    seq, target = dataset[0]
    print(f"\n✅ Dataset test complete!")
    print(f"Sequence shape: {seq.shape}")
    print(f"Targets: {target}")
    
    # Test pretrain dataset
    print("\n\nTesting PretrainDataset...")
    pretrain_dataset = PretrainDataset(features, sequence_length=96, mask_ratio=0.15)
    masked, mask, original = pretrain_dataset[0]
    print(f"Masked sequence shape: {masked.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Num masked: {mask.sum().item()}")
