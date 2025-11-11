"""
Natron Sequence Creator - Constructs 96-timestep sequences for training
Handles data splitting, normalization, and PyTorch dataset creation
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, Tuple, List
import pickle


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence data"""
    
    def __init__(self, sequences: np.ndarray, labels: Dict[str, np.ndarray]):
        """
        Args:
            sequences: (N, sequence_length, n_features) array
            labels: Dict with keys 'buy', 'sell', 'direction', 'regime'
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = {
            'buy': torch.LongTensor(labels['buy']),
            'sell': torch.LongTensor(labels['sell']),
            'direction': torch.LongTensor(labels['direction']),
            'regime': torch.LongTensor(labels['regime'])
        }
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            {
                'buy': self.labels['buy'][idx],
                'sell': self.labels['sell'][idx],
                'direction': self.labels['direction'][idx],
                'regime': self.labels['regime'][idx]
            }
        )


class SequenceCreator:
    """
    Creates sequences from features and labels.
    Handles normalization, splitting, and DataLoader creation.
    """
    
    def __init__(self, config: Dict):
        self.sequence_length = config.get('sequence_length', 96)
        self.train_split = config.get('train_split', 0.7)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        self.normalize = config.get('normalize', True)
        self.scaler = None
        
    def create_sequences(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sequences from features and labels.
        
        Args:
            features: (N, n_features) DataFrame
            labels: (N, 4) DataFrame with columns [buy, sell, direction, regime]
            
        Returns:
            sequences: (N - sequence_length, sequence_length, n_features)
            labels_dict: Dict of label arrays
        """
        print(f"\n📦 Creating sequences (length={self.sequence_length})...")
        
        # Handle NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Normalize features
        if self.normalize:
            print("  🔄 Normalizing features...")
            self.scaler = RobustScaler()  # Robust to outliers
            features_normalized = self.scaler.fit_transform(features)
            features_normalized = pd.DataFrame(
                features_normalized,
                columns=features.columns,
                index=features.index
            )
        else:
            features_normalized = features
        
        # Create sequences
        n_samples = len(features) - self.sequence_length
        n_features = features.shape[1]
        
        sequences = np.zeros((n_samples, self.sequence_length, n_features))
        labels_dict = {
            'buy': np.zeros(n_samples, dtype=np.int64),
            'sell': np.zeros(n_samples, dtype=np.int64),
            'direction': np.zeros(n_samples, dtype=np.int64),
            'regime': np.zeros(n_samples, dtype=np.int64)
        }
        
        print(f"  🔨 Building {n_samples} sequences...")
        for i in range(n_samples):
            # Input: 96 consecutive candles
            sequences[i] = features_normalized.iloc[i:i+self.sequence_length].values
            
            # Label: from the last candle in the sequence
            label_idx = i + self.sequence_length - 1
            labels_dict['buy'][i] = labels.iloc[label_idx]['buy']
            labels_dict['sell'][i] = labels.iloc[label_idx]['sell']
            labels_dict['direction'][i] = labels.iloc[label_idx]['direction']
            labels_dict['regime'][i] = labels.iloc[label_idx]['regime']
        
        print(f"  ✅ Sequences shape: {sequences.shape}")
        print(f"  ✅ Labels: {list(labels_dict.keys())}")
        
        return sequences, labels_dict
    
    def split_data(
        self,
        sequences: np.ndarray,
        labels_dict: Dict[str, np.ndarray]
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split data into train/val/test sets.
        
        Returns:
            train_data, val_data, test_data: Each is a dict with 'sequences' and 'labels'
        """
        print(f"\n✂️  Splitting data (train={self.train_split}, val={self.val_split}, test={self.test_split})...")
        
        n_samples = len(sequences)
        n_train = int(n_samples * self.train_split)
        n_val = int(n_samples * self.val_split)
        
        # Time-series split (no shuffling to preserve temporal order)
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:n_train+n_val]
        test_sequences = sequences[n_train+n_val:]
        
        train_labels = {k: v[:n_train] for k, v in labels_dict.items()}
        val_labels = {k: v[n_train:n_train+n_val] for k, v in labels_dict.items()}
        test_labels = {k: v[n_train+n_val:] for k, v in labels_dict.items()}
        
        print(f"  ✅ Train: {len(train_sequences)} samples")
        print(f"  ✅ Val:   {len(val_sequences)} samples")
        print(f"  ✅ Test:  {len(test_sequences)} samples")
        
        return (
            {'sequences': train_sequences, 'labels': train_labels},
            {'sequences': val_sequences, 'labels': val_labels},
            {'sequences': test_sequences, 'labels': test_labels}
        )
    
    def create_dataloaders(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Returns:
            train_loader, val_loader, test_loader
        """
        print(f"\n🔄 Creating DataLoaders (batch_size={batch_size})...")
        
        train_dataset = SequenceDataset(train_data['sequences'], train_data['labels'])
        val_dataset = SequenceDataset(val_data['sequences'], val_data['labels'])
        test_dataset = SequenceDataset(test_data['sequences'], test_data['labels'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training
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
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        print(f"  ✅ Train batches: {len(train_loader)}")
        print(f"  ✅ Val batches:   {len(val_loader)}")
        print(f"  ✅ Test batches:  {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def save_scaler(self, path: str):
        """Save the fitted scaler for inference"""
        if self.scaler is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  💾 Scaler saved to {path}")
    
    def load_scaler(self, path: str):
        """Load a fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"  📂 Scaler loaded from {path}")
    
    def get_data_statistics(self, train_data: Dict, val_data: Dict, test_data: Dict):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("📊 DATASET STATISTICS")
        print("="*50)
        
        for split_name, data in [("TRAIN", train_data), ("VAL", val_data), ("TEST", test_data)]:
            print(f"\n▶ {split_name} Set:")
            print(f"  Samples: {len(data['sequences'])}")
            print(f"  Shape:   {data['sequences'].shape}")
            
            for label_name, label_values in data['labels'].items():
                unique, counts = np.unique(label_values, return_counts=True)
                print(f"\n  {label_name.upper()} distribution:")
                for val, count in zip(unique, counts):
                    pct = count / len(label_values) * 100
                    print(f"    Class {val}: {count:6d} ({pct:5.2f}%)")
        
        print("\n" + "="*50 + "\n")


def prepare_data_pipeline(
    csv_path: str,
    config: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader, SequenceCreator]:
    """
    Complete data preparation pipeline.
    
    Args:
        csv_path: Path to data_export.csv
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader, sequence_creator
    """
    from .feature_engine import FeatureEngine
    from .label_generator import LabelGeneratorV2
    
    print("="*60)
    print("🚀 NATRON DATA PREPARATION PIPELINE")
    print("="*60)
    
    # 1. Load data
    print(f"\n📁 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  ✅ Loaded {len(df)} rows")
    print(f"  ✅ Columns: {list(df.columns)}")
    
    # 2. Generate features
    feature_engine = FeatureEngine()
    features = feature_engine.generate_all_features(df)
    
    # 3. Generate labels
    label_config = config.get('labels', {})
    label_config['neutral_buffer'] = config.get('data', {}).get('neutral_buffer', 0.001)
    label_generator = LabelGeneratorV2(label_config)
    labels = label_generator.generate_all_labels(df, features)
    
    # 4. Create sequences
    data_config = config.get('data', {})
    sequence_creator = SequenceCreator(data_config)
    sequences, labels_dict = sequence_creator.create_sequences(features, labels)
    
    # 5. Split data
    train_data, val_data, test_data = sequence_creator.split_data(sequences, labels_dict)
    
    # 6. Print statistics
    sequence_creator.get_data_statistics(train_data, val_data, test_data)
    
    # 7. Create DataLoaders
    training_config = config.get('training', {}).get('supervised', {})
    batch_size = training_config.get('batch_size', 64)
    system_config = config.get('system', {})
    
    train_loader, val_loader, test_loader = sequence_creator.create_dataloaders(
        train_data,
        val_data,
        test_data,
        batch_size=batch_size,
        num_workers=system_config.get('num_workers', 4),
        pin_memory=system_config.get('pin_memory', True)
    )
    
    # 8. Save scaler
    checkpoint_dir = system_config.get('checkpoint_dir', 'model')
    sequence_creator.save_scaler(f"{checkpoint_dir}/scaler.pkl")
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader, sequence_creator
