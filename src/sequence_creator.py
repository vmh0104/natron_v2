"""
Sequence Creator - Creates 96-candle sequences for Transformer input
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
import pickle


class SequenceCreator:
    """
    Creates sequences of 96 consecutive candles for training.
    Each sample: (96, n_features) -> (buy, sell, direction, regime)
    """
    
    def __init__(self, sequence_length: int = 96):
        """
        Args:
            sequence_length: Number of consecutive candles per sequence
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.fitted = False
        
    def create_sequences(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences from features and labels.
        
        Args:
            features: Feature DataFrame (n_samples, n_features)
            labels: Labels DataFrame with columns [buy, sell, direction, regime]
            fit_scaler: Whether to fit the scaler (True for train, False for val/test)
            
        Returns:
            X: Input sequences (n_sequences, 96, n_features)
            y: Dictionary of target arrays
        """
        print(f"📦 Creating sequences (length={self.sequence_length})...")
        
        # Convert to numpy
        features_np = features.values
        
        # Fit or transform scaler
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features_np)
            self.fitted = True
            print("✅ Scaler fitted")
        else:
            if not self.fitted:
                raise ValueError("Scaler must be fitted before transforming")
            features_scaled = self.scaler.transform(features_np)
        
        # Create sequences
        n_samples = len(features_scaled) - self.sequence_length
        n_features = features_scaled.shape[1]
        
        X = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)
        y_buy = np.zeros(n_samples, dtype=np.float32)
        y_sell = np.zeros(n_samples, dtype=np.float32)
        y_direction = np.zeros(n_samples, dtype=np.int64)
        y_regime = np.zeros(n_samples, dtype=np.int64)
        
        for i in range(n_samples):
            # Input: 96 consecutive candles
            X[i] = features_scaled[i:i + self.sequence_length]
            
            # Output: labels at the last candle
            last_idx = i + self.sequence_length - 1
            y_buy[i] = labels.iloc[last_idx]['buy']
            y_sell[i] = labels.iloc[last_idx]['sell']
            y_direction[i] = labels.iloc[last_idx]['direction']
            y_regime[i] = labels.iloc[last_idx]['regime']
        
        y = {
            'buy': y_buy,
            'sell': y_sell,
            'direction': y_direction,
            'regime': y_regime
        }
        
        print(f"✅ Created {n_samples} sequences")
        print(f"   X shape: {X.shape}")
        print(f"   Features per sequence: {n_features}")
        
        return X, y
    
    def save_scaler(self, path: str):
        """Save the fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before saving")
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Scaler saved to {path}")
    
    def load_scaler(self, path: str):
        """Load a fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.fitted = True
        print(f"📂 Scaler loaded from {path}")


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading sequences.
    """
    
    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Dictionary of target arrays
        """
        self.X = torch.from_numpy(X)
        self.y_buy = torch.from_numpy(y['buy'])
        self.y_sell = torch.from_numpy(y['sell'])
        self.y_direction = torch.from_numpy(y['direction'])
        self.y_regime = torch.from_numpy(y['regime'])
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            x: Input sequence (seq_len, n_features)
            y: Dictionary of targets
        """
        x = self.X[idx]
        y = {
            'buy': self.y_buy[idx],
            'sell': self.y_sell[idx],
            'direction': self.y_direction[idx],
            'regime': self.y_regime[idx]
        }
        return x, y


def create_dataloaders(
    X_train: np.ndarray,
    y_train: Dict[str, np.ndarray],
    X_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"📊 DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def split_data(
    X: np.ndarray,
    y: Dict[str, np.ndarray],
    test_split: float = 0.2,
    val_split: float = 0.1,
    shuffle: bool = False
) -> Tuple:
    """
    Split data into train/val/test sets.
    
    Args:
        X: Input sequences
        y: Target dictionary
        test_split: Fraction for test set
        val_split: Fraction for validation set (from train)
        shuffle: Whether to shuffle before splitting
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    n_samples = len(X)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = {k: v[indices] for k, v in y.items()}
    
    # Test split
    n_test = int(n_samples * test_split)
    X_test = X[-n_test:]
    y_test = {k: v[-n_test:] for k, v in y.items()}
    
    X_trainval = X[:-n_test]
    y_trainval = {k: v[:-n_test] for k, v in y.items()}
    
    # Validation split
    n_val = int(len(X_trainval) * val_split)
    X_val = X_trainval[-n_val:]
    y_val = {k: v[-n_val:] for k, v in y_trainval.items()}
    
    X_train = X_trainval[:-n_val]
    y_train = {k: v[:-n_val] for k, v in y_trainval.items()}
    
    print(f"📊 Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Test sequence creation
    print("Testing SequenceCreator...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    
    labels = pd.DataFrame({
        'buy': np.random.randint(0, 2, n_samples),
        'sell': np.random.randint(0, 2, n_samples),
        'direction': np.random.randint(0, 3, n_samples),
        'regime': np.random.randint(0, 6, n_samples)
    })
    
    # Create sequences
    creator = SequenceCreator(sequence_length=96)
    X, y = creator.create_sequences(features, labels, fit_scaler=True)
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
    
    # Test a batch
    for batch_x, batch_y in train_loader:
        print(f"\n✅ Batch shape: {batch_x.shape}")
        print(f"   Buy shape: {batch_y['buy'].shape}")
        print(f"   Direction shape: {batch_y['direction'].shape}")
        break
